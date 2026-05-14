import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


SERIES = {
    "Transformer L2": "reverse_transformer_L2_seed42.csv",
    "QK-only L2": "reverse_qk_only_L2_seed42.csv",
    "CNN L2": "reverse_cnn_L2_seed42.csv",
    "CNN L6": "reverse_cnn_L6_seed42.csv",
}


def read_csv(path):
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def plot_metric(series, metric, title, ylabel, out_path):
    plt.figure(figsize=(8, 5), dpi=150)
    for label, rows in series.items():
        steps = [row["step"] for row in rows]
        values = [row[metric] for row in rows]
        plt.plot(steps, values, marker="o", linewidth=2, label=label)
    plt.title(title)
    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_bar(series, metric, title, ylabel, out_path):
    labels = list(series.keys())
    values = [series[label][-1][metric] for label in labels]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    plt.figure(figsize=(8, 5), dpi=150)
    bars = plt.bar(labels, values, color=colors[: len(labels)])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs_final"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_series = {
        label: read_csv(args.runs_dir / filename)
        for label, filename in SERIES.items()
        if (args.runs_dir / filename).exists()
    }

    reproduction = {
        label: all_series[label]
        for label in ["Transformer L2", "CNN L2", "CNN L6"]
        if label in all_series
    }
    qkv_ablation = {
        label: all_series[label]
        for label in ["Transformer L2", "QK-only L2"]
        if label in all_series
    }
    cnn_ablation = {
        label: all_series[label]
        for label in ["Transformer L2", "CNN L2", "CNN L6"]
        if label in all_series
    }

    plot_metric(
        reproduction,
        "token_acc",
        "Small Reproduction: Token Accuracy",
        "Token accuracy",
        args.out_dir / "reproduction_token_accuracy.png",
    )
    plot_metric(
        qkv_ablation,
        "seq_acc",
        "Ablation 2.2: Q/K/V vs K=V",
        "Sequence accuracy",
        args.out_dir / "qkv_ablation_sequence_accuracy.png",
    )
    plot_metric(
        cnn_ablation,
        "seq_acc",
        "Ablation 2.4: Positional Encoding + CNN vs Transformer",
        "Sequence accuracy",
        args.out_dir / "cnn_ablation_sequence_accuracy.png",
    )
    plot_bar(
        all_series,
        "seq_acc",
        "Final Sequence Accuracy",
        "Sequence accuracy",
        args.out_dir / "final_sequence_accuracy_bar.png",
    )

    print(f"saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
