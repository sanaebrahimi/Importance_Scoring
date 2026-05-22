"""
Plot the impact of sample size on margin of error (95% CI width).

Two views per metric:
  - Top row:  mean ± 95% CI band (shaded) for each model
  - Bottom row: CI half-width = 1.96 × SEM vs sample size

Usage:
    python3 plot_sample_size_margin_of_error.py
    python3 plot_sample_size_margin_of_error.py --input-json results/sample_prefix_citation_metrics_all_models.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

MODEL_COLORS = {
    "llama3.2:1b":  "#4e79a7",
    "llama3.2:3b":  "#f28e2b",
    "gemma2:2b":    "#e15759",
    "gemma3:4b":    "#76b7b2",
    "qwen3:1.7b":   "#59a14f",
    "phi3:medium":  "#edc948",
    "qwen2.5:3b":   "#b07aa1",
}

MODEL_MARKERS = {
    "llama3.2:1b":  "o",
    "llama3.2:3b":  "s",
    "gemma2:2b":    "^",
    "gemma3:4b":    "D",
    "qwen3:1.7b":   "P",
    "phi3:medium":  "X",
    "qwen2.5:3b":   "v",
}

METRICS = [
    ("mean_overlap_at_k",                   "std_overlap_at_k",                   "sem_overlap_at_k",                   "Overlap@4"),
    ("mean_recall_at_k",                    "std_recall_at_k",                    "sem_recall_at_k",                    "Recall@4"),
    ("mean_hit_at_k",                       "std_hit_at_k",                       "sem_hit_at_k",                       "Hit@4"),
    ("mean_mrr_human_rank1_in_model_top_k", "std_mrr_human_rank1_in_model_top_k", "sem_mrr_human_rank1_in_model_top_k", "MRR"),
]

Z95 = 1.96


def load_series(data: dict):
    """Returns {model: {sample_size: {mean, std, sem, moe}}} for each metric."""
    series = {}
    for model, payload in data["models"].items():
        agg = payload.get("aggregate_by_sample_size", {})
        sizes = sorted(int(s) for s in agg)
        model_data = {"sizes": sizes}
        for mean_k, std_k, sem_k, label in METRICS:
            model_data[mean_k] = [float(agg[str(s)]["w"].get(mean_k) or 0) for s in sizes]
            model_data[std_k]  = [float(agg[str(s)]["w"].get(std_k)  or 0) for s in sizes]
            model_data[sem_k]  = [float(agg[str(s)]["w"].get(sem_k)  or 0) for s in sizes]
            moe_k = mean_k.replace("mean_", "moe_")
            model_data[moe_k]  = [Z95 * v for v in model_data[sem_k]]
        series[model] = model_data
    return series


def _style(ax, title, ylabel, xticks):
    ax.set_title(title, fontsize=11, fontweight="semibold", pad=7)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    ax.grid(True, axis="y", color="#D7DBE0", lw=0.8, alpha=0.9)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8.5)


def make_combined_figure(series: dict, out_dir: Path) -> None:
    """
    2-row × 4-col figure.
    Top row:    mean ± 95% CI shaded, per metric.
    Bottom row: 95% CI half-width (MoE = 1.96×SEM) per metric.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=False)

    for col, (mean_k, std_k, sem_k, label) in enumerate(METRICS):
        moe_k = mean_k.replace("mean_", "moe_")
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        for model, md in series.items():
            color  = MODEL_COLORS.get(model, "#333333")
            marker = MODEL_MARKERS.get(model, "o")
            xs     = md["sizes"]
            ys     = md[mean_k]
            moes   = md[moe_k]

            # Top: mean line + shaded 95% CI
            ax_top.plot(xs, ys, color=color, marker=marker, linewidth=2,
                        markersize=6, markeredgecolor="white", markeredgewidth=0.6,
                        label=model, zorder=3)
            ax_top.fill_between(xs,
                                [y - e for y, e in zip(ys, moes)],
                                [y + e for y, e in zip(ys, moes)],
                                color=color, alpha=0.12, zorder=1)

            # Bottom: MoE line
            ax_bot.plot(xs, moes, color=color, marker=marker, linewidth=2,
                        markersize=6, markeredgecolor="white", markeredgewidth=0.6,
                        label=model, zorder=3)

        _style(ax_top, label,           label,                          xs)
        _style(ax_bot, f"{label} — MoE", "95% CI half-width (1.96×SEM)", xs)
        ax_top.set_xlabel("")
        ax_bot.set_xlabel("Number of samples", fontsize=9.5)

    # Shared legend above all panels
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.01), columnspacing=1.2)
    fig.suptitle(
        "Impact of sample size on metric mean and margin of error (95% CI, n=28 papers)",
        fontsize=12, fontweight="semibold", y=1.045,
    )
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.08, top=0.90,
                        wspace=0.30, hspace=0.42)

    for ext in ("png", "pdf"):
        p = out_dir / f"sample_size_margin_of_error_combined.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  saved {p}")
    plt.close(fig)


def make_moe_only_figure(series: dict, out_dir: Path) -> None:
    """Single-row figure showing only MoE for each metric."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), constrained_layout=False)

    for ax, (mean_k, std_k, sem_k, label) in zip(axes, METRICS):
        moe_k = mean_k.replace("mean_", "moe_")
        for model, md in series.items():
            color  = MODEL_COLORS.get(model, "#333333")
            marker = MODEL_MARKERS.get(model, "o")
            ax.plot(md["sizes"], md[moe_k], color=color, marker=marker,
                    linewidth=2, markersize=6, markeredgecolor="white",
                    markeredgewidth=0.6, label=model, zorder=3)
        _style(ax, f"{label}", "95% CI half-width (1.96×SEM)", md["sizes"])
        ax.set_xlabel("Number of samples", fontsize=9.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.04), columnspacing=1.2)
    fig.suptitle("Margin of error vs. sample size  (95% CI, n=28 papers)",
                 fontsize=12, fontweight="semibold", y=1.08)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.14, top=0.86, wspace=0.30)

    for ext in ("png", "pdf"):
        p = out_dir / f"sample_size_margin_of_error.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  saved {p}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json",  default="results/sample_prefix_citation_metrics_all_models.json")
    parser.add_argument("--output-dir",  default="results/plots")
    args = parser.parse_args()

    data = json.loads(Path(args.input_json).read_text())
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    series = load_series(data)
    make_combined_figure(series, out)
    make_moe_only_figure(series, out)
    print("Done.")


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    main()
