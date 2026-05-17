import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_LABELS = {
    "llama3.2_1b": "llama3.2:1b",
    "llama3_2_3": "ollama3.2:3b",
    "gemma2_2b_3": "gemma2:2b",
    "gemma3_4b": "gemma3:4b",
    "phi3_medium_run2": "phi3:medium",
    "qwen3_4b": "qwen3:4b",
    "qwen2_5_3b": "qwen2.5:3b",
}

# User-specified palette.
MODEL_COLORS = {
    "llama3.2_1b": "#9400D3",      # darkviolet
    "llama3_2_3": "#228B22",       # forestgreen
    "gemma2_2b_3": "#B22222",      # firebrick
    "gemma3_4b": "#008080",        # teal
    "phi3_medium_run2": "#FF8C00", # darkorange
    "qwen3_4b": "#000080",         # navy
    "qwen2_5_3b": "#00BFFF",       # deepskyblue
}

MODEL_MARKERS = {
    "llama3.2_1b": "X",
    "llama3_2_3": "o",
    "gemma2_2b_3": "s",
    "gemma3_4b": "v",
    "phi3_medium_run2": "^",
    "qwen3_4b": "D",
    "qwen2_5_3b": "P",
}

METRICS = [
    ("mean_overlap_at_k", "Overlap@4"),
    ("mean_recall_at_k", "Recall@4"),
    ("mean_hit_at_k", "Hit@4"),
    ("mean_mrr_human_rank1_in_model_top_k", "MRR"),
]


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_model_series(data: dict, score_key: str) -> Tuple[Dict[str, Dict[str, List[float]]], str]:
    models = data["models"]
    analyze_by = data.get("analyze_by", "samples")
    aggregate_key = "aggregate_by_attempt_budget" if analyze_by == "attempts" else "aggregate_by_sample_size"
    x_key = "attempt_budgets" if analyze_by == "attempts" else "sample_sizes"
    series: Dict[str, Dict[str, List[float]]] = {}
    for model_tag, payload in models.items():
        aggregate = payload.get(aggregate_key, {})
        x_values = sorted(int(key) for key in aggregate.keys())
        model_series = {x_key: x_values}
        for metric_key, _ in METRICS:
            model_series[metric_key] = [
                float(aggregate[str(x_value)][score_key][metric_key])
                for x_value in x_values
            ]
        series[model_tag] = model_series
    return series, analyze_by


def style_axes(ax: plt.Axes, title: str, ylabel: str, analyze_by: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="semibold", pad=10)
    ax.set_xlabel("Attempt budget" if analyze_by == "attempts" else "Number of samples", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(True, axis="y", color="#D7DBE0", linewidth=0.9, alpha=0.85)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8A9199")
    ax.spines["bottom"].set_color("#8A9199")
    ax.tick_params(axis="both", labelsize=9, colors="#1F2937")


def plot_metric(
    ax: plt.Axes,
    series: Dict[str, Dict[str, List[float]]],
    metric_key: str,
    metric_label: str,
    analyze_by: str,
) -> None:
    x_key = "attempt_budgets" if analyze_by == "attempts" else "sample_sizes"
    for model_tag, payload in series.items():
        ax.plot(
            payload[x_key],
            payload[metric_key],
            label=MODEL_LABELS.get(model_tag, model_tag),
            color=MODEL_COLORS.get(model_tag, "#374151"),
            marker=MODEL_MARKERS.get(model_tag, "o"),
            linewidth=2.4,
            markersize=7,
            markerfacecolor=MODEL_COLORS.get(model_tag, "#374151"),
            markeredgecolor="white",
            markeredgewidth=0.9,
        )
    style_axes(ax, metric_label, metric_label, analyze_by)


def save_combined_figure(
    series: Dict[str, Dict[str, List[float]]],
    output_dir: Path,
    score_key: str,
    analyze_by: str,
) -> List[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), constrained_layout=False)
    axes_flat = axes.flatten()
    for ax, (metric_key, metric_label) in zip(axes_flat, METRICS):
        plot_metric(ax, series, metric_key, metric_label, analyze_by)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.suptitle(
        (
            f"Attempt-budget sensitivity of top-4 citation metrics ({score_key})"
            if analyze_by == "attempts"
            else f"Sample-size sensitivity of top-4 citation metrics ({score_key})"
        ),
        fontsize=14,
        fontweight="semibold",
        y=0.985,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.945),
        columnspacing=1.6,
        handlelength=2.2,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.08,
        top=0.84,
        wspace=0.26,
        hspace=0.34,
    )

    stem_prefix = "attempt_budget_metrics" if analyze_by == "attempts" else "sample_prefix_metrics"
    png_path = output_dir / f"{stem_prefix}_{score_key}_combined.png"
    pdf_path = output_dir / f"{stem_prefix}_{score_key}_combined.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png_path, pdf_path]


def save_individual_figures(
    series: Dict[str, Dict[str, List[float]]],
    output_dir: Path,
    score_key: str,
    analyze_by: str,
) -> List[Path]:
    saved_paths: List[Path] = []
    for metric_key, metric_label in METRICS:
        fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=False)
        plot_metric(ax, series, metric_key, metric_label, analyze_by)
        ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(0.98, 0.05))
        fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.90)

        stem = metric_label.lower().replace("@", "at").replace("-", "_").replace(" ", "_")
        stem_prefix = "attempt_budget_metrics" if analyze_by == "attempts" else "sample_prefix_metrics"
        png_path = output_dir / f"{stem_prefix}_{score_key}_{stem}.png"
        pdf_path = output_dir / f"{stem_prefix}_{score_key}_{stem}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved_paths.extend([png_path, pdf_path])
    return saved_paths


def save_tsv(series: Dict[str, Dict[str, List[float]]], output_dir: Path, score_key: str, analyze_by: str) -> Path:
    x_name = "attempt_budget" if analyze_by == "attempts" else "sample_size"
    x_key = "attempt_budgets" if analyze_by == "attempts" else "sample_sizes"
    lines = [f"model_tag\t{x_name}\tmetric\tvalue"]
    for model_tag, payload in series.items():
        for metric_key, metric_label in METRICS:
            for x_value, value in zip(payload[x_key], payload[metric_key]):
                lines.append(f"{model_tag}\t{x_value}\t{metric_label}\t{value}")
    stem_prefix = "attempt_budget_metrics" if analyze_by == "attempts" else "sample_prefix_metrics"
    out_path = output_dir / f"{stem_prefix}_{score_key}.tsv"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sample-prefix or attempt-budget metric curves from saved JSON analysis.")
    parser.add_argument(
        "--input-json",
        default="results/sample_prefix_citation_metrics_all_models.json",
        help="Path to the saved analysis JSON.",
    )
    parser.add_argument(
        "--score-key",
        choices=["w", "w_tech"],
        default="w",
        help="Which score family to plot.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/plots",
        help="Directory to save the generated figures.",
    )
    args = parser.parse_args()

    data = load_json(Path(args.input_json))
    series, analyze_by = extract_model_series(data, args.score_key)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    saved.extend(save_combined_figure(series, output_dir, args.score_key, analyze_by))
    saved.extend(save_individual_figures(series, output_dir, args.score_key, analyze_by))
    saved.append(save_tsv(series, output_dir, args.score_key, analyze_by))

    print("Saved plots/data:")
    for path in saved:
        print(path)


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    main()
