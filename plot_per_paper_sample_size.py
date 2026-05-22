"""
Per-paper trajectory plots: how each paper's metric changes as sample size
increases from 1 → 3 → 5.

Each subplot (one per metric) shows:
  - One thin line per paper, coloured green if the metric improves
    (n=5 > n=1), red if it degrades, grey if it stays the same.
  - A thick black line for the cross-paper mean.

Usage:
    python3 plot_per_paper_sample_size.py
    python3 plot_per_paper_sample_size.py \
        --input-json results/sample_prefix_citation_metrics_all_models.json \
        --output-dir results/plots
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = [
    ("overlap_at_k",                   "Overlap@4"),
    ("recall_at_k",                    "Recall@4"),
    ("hit_at_k",                       "Hit@4"),
    ("mrr_human_rank1_in_model_top_k", "MRR"),
]

C_IMPROVE  = "#2ca02c"   # green
C_DEGRADE  = "#d62728"   # red
C_NEUTRAL  = "#aaaaaa"   # grey
C_MEAN     = "#000000"   # black
ALPHA_LINE = 0.35
LW_PAPER   = 0.9
LW_MEAN    = 2.4


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def extract_per_paper(data: dict) -> dict:
    """
    Returns {model: {paper_id: {sample_size: {metric: value}}}}
    """
    out = {}
    for model, payload in data["models"].items():
        agg = payload.get("aggregate_by_sample_size", {})
        paper_data: dict[str, dict] = {}
        for size_str, size_payload in agg.items():
            size = int(size_str)
            per_paper = size_payload.get("per_paper", {})
            for pid, metrics in per_paper.items():
                paper_data.setdefault(pid, {})[size] = metrics
        out[model] = paper_data
    return out


def paper_color(values: list[float]) -> str:
    """Green if improves, red if degrades, grey if flat."""
    if not values or len(values) < 2:
        return C_NEUTRAL
    delta = values[-1] - values[0]
    if delta > 1e-6:
        return C_IMPROVE
    if delta < -1e-6:
        return C_DEGRADE
    return C_NEUTRAL


def plot_model(model: str, paper_data: dict, sizes: list[int], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), constrained_layout=False)

    for ax, (mkey, mlabel) in zip(axes, METRICS):
        all_vals_by_size: dict[int, list[float]] = {s: [] for s in sizes}

        for pid, size_metrics in paper_data.items():
            ys = []
            xs_used = []
            for s in sizes:
                if s in size_metrics and mkey in size_metrics[s]:
                    xs_used.append(s)
                    ys.append(float(size_metrics[s][mkey]))

            if len(ys) < 2:
                continue

            color = paper_color(ys)
            ax.plot(xs_used, ys, color=color, alpha=ALPHA_LINE,
                    linewidth=LW_PAPER, zorder=2)

            for s, v in zip(xs_used, ys):
                all_vals_by_size[s].append(v)

        # Mean line
        mean_xs, mean_ys = [], []
        for s in sizes:
            vs = all_vals_by_size[s]
            if vs:
                mean_xs.append(s)
                mean_ys.append(sum(vs) / len(vs))
        if mean_xs:
            ax.plot(mean_xs, mean_ys, color=C_MEAN, linewidth=LW_MEAN,
                    zorder=4, label="Mean")

        ax.set_title(mlabel, fontsize=11, fontweight="semibold")
        ax.set_xlabel("Number of samples", fontsize=9.5)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.grid(True, axis="y", color="#D7DBE0", lw=0.8, alpha=0.9)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8.5)

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color=C_IMPROVE, lw=1.5, label="Improves (n=5 > n=1)"),
        Line2D([0], [0], color=C_DEGRADE, lw=1.5, label="Degrades (n=5 < n=1)"),
        Line2D([0], [0], color=C_NEUTRAL,  lw=1.5, label="No change"),
        Line2D([0], [0], color=C_MEAN,     lw=2.4, label="Cross-paper mean"),
    ]
    fig.legend(handles=legend_elems, loc="upper center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.03), columnspacing=1.5)

    safe_name = model.replace(":", "_").replace(".", "_")
    fig.suptitle(f"Per-paper metric trajectory  —  {model}  (sample sizes 1 → 3 → 5)",
                 fontsize=12, fontweight="semibold", y=1.07)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.14, top=0.88, wspace=0.30)

    for ext in ("png", "pdf"):
        p = out_dir / f"per_paper_trajectory_{safe_name}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  saved {p}")
    plt.close(fig)


def plot_all_models_grid(per_model: dict, sizes: list[int], out_dir: Path) -> None:
    """
    One combined figure: rows = models, cols = metrics.
    """
    models = list(per_model.keys())
    n_models = len(models)
    n_metrics = len(METRICS)

    fig, axes = plt.subplots(n_models, n_metrics,
                             figsize=(n_metrics * 3.8, n_models * 2.8),
                             constrained_layout=False)

    for row, model in enumerate(models):
        paper_data = per_model[model]
        for col, (mkey, mlabel) in enumerate(METRICS):
            ax = axes[row, col]
            all_vals_by_size: dict[int, list[float]] = {s: [] for s in sizes}

            for pid, size_metrics in paper_data.items():
                ys, xs_used = [], []
                for s in sizes:
                    if s in size_metrics and mkey in size_metrics[s]:
                        xs_used.append(s)
                        ys.append(float(size_metrics[s][mkey]))
                if len(ys) < 2:
                    continue
                color = paper_color(ys)
                ax.plot(xs_used, ys, color=color, alpha=ALPHA_LINE,
                        linewidth=LW_PAPER, zorder=2)
                for s, v in zip(xs_used, ys):
                    all_vals_by_size[s].append(v)

            mean_xs, mean_ys = [], []
            for s in sizes:
                vs = all_vals_by_size[s]
                if vs:
                    mean_xs.append(s)
                    mean_ys.append(sum(vs) / len(vs))
            if mean_xs:
                ax.plot(mean_xs, mean_ys, color=C_MEAN, linewidth=LW_MEAN, zorder=4)

            ax.set_xticks(sizes)
            ax.set_xticklabels([str(s) for s in sizes], fontsize=7)
            ax.grid(True, axis="y", color="#D7DBE0", lw=0.6, alpha=0.9)
            ax.grid(False, axis="x")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=7)

            if row == 0:
                ax.set_title(mlabel, fontsize=10, fontweight="semibold", pad=5)
            if col == 0:
                ax.set_ylabel(model, fontsize=8, labelpad=4)
            if row == n_models - 1:
                ax.set_xlabel("# samples", fontsize=8)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color=C_IMPROVE, lw=1.5, label="Improves"),
        Line2D([0], [0], color=C_DEGRADE, lw=1.5, label="Degrades"),
        Line2D([0], [0], color=C_NEUTRAL,  lw=1.5, label="No change"),
        Line2D([0], [0], color=C_MEAN,     lw=2.2, label="Mean"),
    ]
    fig.legend(handles=legend_elems, loc="upper center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.005), columnspacing=1.5)
    fig.suptitle("Per-paper metric trajectory across sample sizes  (1 → 3 → 5)",
                 fontsize=12, fontweight="semibold", y=1.015)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.05, top=0.95,
                        wspace=0.28, hspace=0.30)

    for ext in ("png", "pdf"):
        p = out_dir / f"per_paper_trajectory_all_models.{ext}"
        fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"  saved {p}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json",  default="results/sample_prefix_citation_metrics_all_models.json")
    parser.add_argument("--output-dir",  default="results/plots")
    parser.add_argument("--no-per-model", action="store_true",
                        help="Skip individual per-model figures")
    args = parser.parse_args()

    data = load_json(Path(args.input_json))
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_model = extract_per_paper(data)

    # Infer sizes from data
    sizes = sorted({
        int(s)
        for payload in data["models"].values()
        for s in payload.get("aggregate_by_sample_size", {})
    })
    print(f"Sample sizes found: {sizes}")

    if not args.no_per_model:
        for model, paper_data in per_model.items():
            print(f"\n{model} — {len(paper_data)} papers")
            plot_model(model, paper_data, sizes, out)

    print("\nGenerating combined grid …")
    plot_all_models_grid(per_model, sizes, out)
    print("Done.")


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    main()
