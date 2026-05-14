import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from transformer_lab.data import make_batch, make_decoder_input
from transformer_lab.models import build_model


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, args, device, batches=20):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    correct_sequences = 0
    total_sequences = 0
    for _ in range(batches):
        x, y = make_batch(args.batch_size, args.seq_len, args.vocab_size, args.task, device)
        if getattr(model, "autoregressive", False):
            logits = model(x, make_decoder_input(y))
        else:
            logits = model(x)
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
        pred = logits.argmax(dim=-1)
        total_loss += loss.item()
        correct_tokens += (pred == y).sum().item()
        total_tokens += y.numel()
        correct_sequences += (pred == y).all(dim=1).sum().item()
        total_sequences += y.size(0)
    return {
        "loss": total_loss / batches,
        "token_acc": correct_tokens / total_tokens,
        "seq_acc": correct_sequences / total_sequences,
    }


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9
    )

    log_rows = []
    for step in range(1, args.steps + 1):
        model.train()
        x, y = make_batch(args.batch_size, args.seq_len, args.vocab_size, args.task, device)
        if getattr(model, "autoregressive", False):
            logits = model(x, make_decoder_input(y))
        else:
            logits = model(x)
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, args, device, batches=args.eval_batches)
            row = {
                "step": step,
                "train_loss": loss.item(),
                **metrics,
            }
            log_rows.append(row)
            print(
                f"step={step:04d} train_loss={loss.item():.4f} "
                f"val_loss={metrics['loss']:.4f} "
                f"token_acc={metrics['token_acc']:.4f} seq_acc={metrics['seq_acc']:.4f}",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.task}_{args.model}_L{args.layers}_seed{args.seed}"
    csv_path = args.output_dir / f"{stem}.csv"
    json_path = args.output_dir / f"{stem}.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    summary = {
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "final": log_rows[-1],
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved {csv_path}")
    print(f"saved {json_path}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["transformer", "qk_only", "cnn"], default="transformer")
    parser.add_argument("--task", choices=["copy", "reverse", "sort"], default="reverse")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
