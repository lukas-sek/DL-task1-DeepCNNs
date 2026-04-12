"""
Results Plotting Script
-----------------------
Reads results.json files from experiment run directories and generates:
  1. Train/val accuracy & loss curves per experiment
  2. A comparison table across all experiments
  3. A single overlay plot comparing val accuracy of all experiments

Usage:
    # Plot a single experiment
    python plot_results.py --runs runs/std_regularized

    # Compare multiple experiments
    python plot_results.py --runs runs/std_underfit runs/std_overfit runs/std_regularized

    # Auto-discover all runs/ subdirectories
    python plot_results.py --runs runs/

    # Save figures to a folder instead of displaying
    python plot_results.py --runs runs/ --out_dir figures/
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for BSC
import matplotlib.pyplot as plt
import numpy as np


# ─── Load ────────────────────────────────────────────────────────────────────

def load_results(run_dir: Path) -> dict | None:
    result_file = run_dir / "results.json"
    if not result_file.exists():
        print(f"  WARNING: no results.json in {run_dir}, skipping.")
        return None
    with open(result_file) as f:
        data = json.load(f)
    data["run_name"] = run_dir.name
    return data


def collect_runs(paths: list[str]) -> list[dict]:
    results = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"  WARNING: {path} does not exist, skipping.")
            continue
        if (path / "results.json").exists():
            # It's a direct run directory
            r = load_results(path)
            if r:
                results.append(r)
        else:
            # Treat as a parent directory and search one level deep
            for sub in sorted(path.iterdir()):
                if sub.is_dir():
                    r = load_results(sub)
                    if r:
                        results.append(r)
    return results


# ─── Per-run curves ──────────────────────────────────────────────────────────

def plot_curves(result: dict, out_dir: Path | None):
    h    = result["history"]
    name = result["run_name"]
    epochs = range(len(h["train_loss"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(name, fontsize=13)

    # Accuracy
    axes[0].plot(epochs, h["train_accuracy"], label="train")
    axes[0].plot(epochs, h["val_accuracy"],   label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(epochs, h["train_loss"], label="train")
    axes[1].plot(epochs, h["val_loss"],   label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{name}_curves.pdf", bbox_inches="tight")
        print(f"  Saved {name}_curves.pdf")
    else:
        plt.show()
    plt.close(fig)


# ─── Comparison overlay ──────────────────────────────────────────────────────

def plot_val_comparison(results: list[dict], out_dir: Path | None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Validation accuracy & loss — all experiments", fontsize=13)

    for r in results:
        h      = r["history"]
        name   = r["run_name"]
        epochs = range(len(h["val_accuracy"]))
        axes[0].plot(epochs, h["val_accuracy"], label=name)
        axes[1].plot(epochs, h["val_loss"],     label=name)

    for ax, ylabel, title in zip(
        axes,
        ["Accuracy (%)", "Loss"],
        ["Val Accuracy", "Val Loss"],
    ):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "comparison_val.pdf", bbox_inches="tight")
        print("  Saved comparison_val.pdf")
    else:
        plt.show()
    plt.close(fig)


# ─── Summary table ───────────────────────────────────────────────────────────

def print_summary_table(results: list[dict]):
    header = (
        f"{'Experiment':<25} {'Model':<16} {'Params':>10} "
        f"{'BestValAcc':>11} {'TestAcc':>9} {'Epochs':>7} {'Aug':>5} "
        f"{'Drop':>6} {'WD':>8} {'LR':>8} {'Sched':>7}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for r in results:
        cfg = r.get("config", {})
        print(
            f"{r['run_name']:<25} "
            f"{cfg.get('model','?'):<16} "
            f"{r.get('n_parameters', 0):>10,} "
            f"{r.get('best_val_acc', 0):>10.2f}% "
            f"{r.get('test_acc', 0):>8.2f}% "
            f"{cfg.get('epochs', '?'):>7} "
            f"{'Y' if cfg.get('augment') else 'N':>5} "
            f"{cfg.get('dropout', 0):>6.2f} "
            f"{cfg.get('weight_decay', 0):>8.0e} "
            f"{cfg.get('lr', 0):>8.0e} "
            f"{cfg.get('scheduler','none'):>7}"
        )
    print("=" * len(header) + "\n")


# ─── Confusion matrix ────────────────────────────────────────────────────────

def plot_confusion_matrix(result: dict, out_dir: Path | None):
    name        = result["run_name"]
    matrix      = np.array(result["confusion_matrix"])
    class_names = result.get("class_names", [str(i) for i in range(len(matrix))])

    # Normalize rows → recall per class
    row_sums = matrix.sum(axis=1, keepdims=True).clip(min=1)
    norm     = matrix / row_sums

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (row-normalised) — {name}")

    plt.tight_layout()
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{name}_confusion.pdf", bbox_inches="tight")
        # Also save raw matrix as JSON
        cm_data = {"class_names": class_names, "matrix": matrix.tolist()}
        with open(out_dir / f"{name}_confusion.json", "w") as f:
            json.dump(cm_data, f, indent=2)
        print(f"  Saved {name}_confusion.pdf + .json")
    else:
        plt.show()
    plt.close(fig)


# ─── CSV comparison export ────────────────────────────────────────────────────

def export_comparison_csv(results: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "comparison_table.csv"
    fieldnames = [
        "experiment", "model", "n_parameters", "best_val_acc", "test_acc",
        "epochs", "augment", "dropout", "weight_decay", "lr", "scheduler",
        "time_per_epoch_s"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            cfg = r.get("config", {})
            writer.writerow({
                "experiment":       r["run_name"],
                "model":            cfg.get("model", ""),
                "n_parameters":     r.get("n_parameters", ""),
                "best_val_acc":     round(r.get("best_val_acc", 0), 2),
                "test_acc":         round(r.get("test_acc", 0), 2),
                "epochs":           cfg.get("epochs", ""),
                "augment":          cfg.get("augment", False),
                "dropout":          cfg.get("dropout", 0),
                "weight_decay":     cfg.get("weight_decay", 0),
                "lr":               cfg.get("lr", ""),
                "scheduler":        cfg.get("scheduler", "none"),
                "time_per_epoch_s": round(r.get("time_per_epoch_s", 0), 1),
            })
    print(f"  Saved comparison_table.csv")


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Plot MAMe experiment results.")
    p.add_argument("--runs",    nargs="+", required=True,
                   help="Run directories (or a parent dir containing them).")
    p.add_argument("--out_dir", type=str,  default="figures",
                   help="Directory to save figures (default: figures/). "
                        "Pass 'show' to display interactively.")
    return p.parse_args()


def main():
    args  = parse_args()
    out   = None if args.out_dir == "show" else Path(args.out_dir)
    results = collect_runs(args.runs)

    if not results:
        print("No results found. Have the training jobs finished?")
        return

    print(f"\nFound {len(results)} experiment(s):")
    for r in results:
        print(f"  {r['run_name']}  "
              f"(best_val={r.get('best_val_acc',0):.2f}%  "
              f"test={r.get('test_acc',0):.2f}%)")

    print_summary_table(results)

    print("Generating per-experiment curve plots ...")
    for r in results:
        plot_curves(r, out)

    if len(results) > 1:
        print("Generating comparison overlay ...")
        plot_val_comparison(results, out)

    print("Generating confusion matrices for runs that have them ...")
    for r in results:
        if "confusion_matrix" in r and r["confusion_matrix"]:
            plot_confusion_matrix(r, out)

    if out:
        export_comparison_csv(results, out)
        print(f"\nAll figures saved to {out}/")


if __name__ == "__main__":
    main()
