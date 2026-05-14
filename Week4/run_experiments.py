import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from train import train


def make_config(base, model):
    cfg = vars(base).copy()
    cfg["model"] = model
    cfg["output_dir"] = Path(cfg["output_dir"])
    return SimpleNamespace(**cfg)


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--models",
        nargs="+",
        default=["transformer", "qk_only", "cnn"],
        choices=["transformer", "qk_only", "cnn"],
    )
    args = parser.parse_args()

    summaries = []
    for model_name in args.models:
        print(f"\n=== running {model_name} ===", flush=True)
        summaries.append(train(make_config(args, model_name)))

    out = args.output_dir / f"{args.task}_summary.json"
    out.write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

