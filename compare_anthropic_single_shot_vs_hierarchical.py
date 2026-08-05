from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

MPL_CACHE_DIR = Path("/private/tmp/codex_matplotlib")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from citation_resolver import CitationResolver
from evaluate_citation_jsd import (
    aggregate_scores_by_target,
    align_score_vectors,
    load_citation_score_map,
    normalize_citation_only_vector,
)
from evaluate_human_section_scores import (
    jensen_shannon_divergence,
    kendall_tau_b,
    kl_divergence,
    spearman_rho,
)


plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


LEFT_TAG = "anthropic_full_paper"
RIGHT_TAG = "single_shot_citation_anthropic"
LEFT_LABEL = "Hierarchical Anthropic"
RIGHT_LABEL = "Single-shot Anthropic"
DEFAULT_TOP_K = 4

LEFT_COLOR = "#1f9d55"
RIGHT_COLOR = "#c9472f"
SUMMARY_COLORS = {
    "jensen_shannon_divergence": "#4e79a7",
    "kl_divergence": "#f28e2b",
    "spearman": "#59a14f",
    "kendall_tau_b": "#9c755f",
    "top_k_overlap_fraction": "#7f7f7f",
}


def mean(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def stddev(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def bootstrap_mean_ci(
    values: Sequence[float],
    n_bootstrap: int,
    seed: int,
) -> Optional[Tuple[float, float]]:
    if not values:
        return None
    if len(values) == 1:
        only = float(values[0])
        return (only, only)

    rng = random.Random(seed)
    n = len(values)
    means: List[float] = []
    for _ in range(n_bootstrap):
        sample = [float(values[rng.randrange(n)]) for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    lo_index = max(0, int(0.025 * n_bootstrap))
    hi_index = min(n_bootstrap - 1, int(0.975 * n_bootstrap))
    return means[lo_index], means[hi_index]


def format_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def score_file_path(results_root: Path, paper_id: str, tag: str) -> Path:
    return results_root / paper_id / f"{paper_id}_{tag}_citation_scores.json"


def total_citation_mass(results_root: Path, paper_id: str, tag: str) -> float:
    score_map = load_citation_score_map(score_file_path(results_root, paper_id, tag))
    return sum(float(score) for score in score_map.values())


def discover_common_papers(results_root: Path, left_tag: str, right_tag: str) -> List[str]:
    common: List[str] = []
    for paper_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        paper_id = paper_dir.name
        if score_file_path(results_root, paper_id, left_tag).exists() and score_file_path(results_root, paper_id, right_tag).exists():
            common.append(paper_id)
    return common


def prepare_resolver_for_tags(
    results_root: Path,
    papers_dir: Path,
    paper_ids: Sequence[str],
    tags: Sequence[str],
) -> CitationResolver:
    resolver = CitationResolver()
    resolver.register_corpus_papers(results_root, papers_dir)

    for paper_id in paper_ids:
        keys: List[str] = []
        seen = set()
        for tag in tags:
            score_map = load_citation_score_map(score_file_path(results_root, paper_id, tag))
            for citation_key in score_map:
                normalized = " ".join(str(citation_key).split())
                if normalized in seen:
                    continue
                seen.add(normalized)
                keys.append(citation_key)
        resolver.parse_paper(paper_id, papers_dir / f"{paper_id}.pdf", keys)
    return resolver


def rank_grouped_citations(grouped_scores: Dict[str, dict]) -> List[dict]:
    ranked = [
        {
            "target_id": target_id,
            "citation": str(payload["display_key"]),
            "score": float(payload["score"]),
        }
        for target_id, payload in grouped_scores.items()
    ]
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["citation"])))
    return ranked


def top_k_overlap_report(
    left_grouped: Dict[str, dict],
    right_grouped: Dict[str, dict],
    k: int,
) -> dict:
    left_ranked = rank_grouped_citations(left_grouped)
    right_ranked = rank_grouped_citations(right_grouped)
    left_top_k = left_ranked[: max(0, k)]
    right_top_k = right_ranked[: max(0, k)]

    left_top_ids = {item["target_id"] for item in left_top_k}
    right_top_ids = {item["target_id"] for item in right_top_k}
    overlap_ids = sorted(left_top_ids.intersection(right_top_ids))
    denom = max(1, min(k, len(left_top_k), len(right_top_k)))

    left_lookup = {item["target_id"]: item for item in left_ranked}
    right_lookup = {item["target_id"]: item for item in right_ranked}
    return {
        "k": k,
        "overlap_count": len(overlap_ids),
        "overlap_fraction": len(overlap_ids) / denom,
        "overlapping_citations": [
            {
                "target_id": target_id,
                "left_citation": left_lookup[target_id]["citation"],
                "right_citation": right_lookup[target_id]["citation"],
                "left_score": left_lookup[target_id]["score"],
                "right_score": right_lookup[target_id]["score"],
            }
            for target_id in overlap_ids
        ],
    }


def top_absolute_differences(
    target_ids: Sequence[str],
    labels: Sequence[str],
    left_values: Sequence[float],
    right_values: Sequence[float],
    limit: int = 10,
) -> List[dict]:
    diffs = []
    for target_id, label, left_value, right_value in zip(target_ids, labels, left_values, right_values):
        diffs.append(
            {
                "target_id": target_id,
                "citation": label,
                "left_score": float(left_value),
                "right_score": float(right_value),
                "absolute_difference": abs(float(left_value) - float(right_value)),
            }
        )
    diffs.sort(key=lambda item: (-float(item["absolute_difference"]), str(item["citation"])))
    return diffs[: max(0, limit)]


def compare_pair_for_paper(
    paper_id: str,
    results_root: Path,
    left_tag: str,
    right_tag: str,
    resolver: CitationResolver,
    top_k: int,
) -> dict:
    left_scores = load_citation_score_map(score_file_path(results_root, paper_id, left_tag))
    right_scores = load_citation_score_map(score_file_path(results_root, paper_id, right_tag))
    left_grouped = aggregate_scores_by_target(paper_id, left_scores, resolver)
    right_grouped = aggregate_scores_by_target(paper_id, right_scores, resolver)
    target_ids, labels, left_values, right_values = align_score_vectors(left_grouped, right_grouped)

    left_vector, left_mass = normalize_citation_only_vector(left_values)
    right_vector, right_mass = normalize_citation_only_vector(right_values)

    metrics = {
        "kl_divergence": kl_divergence(left_vector, right_vector),
        "jensen_shannon_divergence": jensen_shannon_divergence(left_vector, right_vector),
        "spearman": spearman_rho(left_values, right_values),
        "kendall_tau_b": kendall_tau_b(left_values, right_values),
    }
    top_k_report = top_k_overlap_report(left_grouped, right_grouped, top_k)

    return {
        "paper_id": paper_id,
        "citation_components": len(target_ids),
        "left_raw_score_sum": left_mass,
        "right_raw_score_sum": right_mass,
        "metrics": metrics,
        "top_k_overlap": top_k_report,
        "top_absolute_differences": top_absolute_differences(
            target_ids=target_ids,
            labels=labels,
            left_values=left_values,
            right_values=right_values,
            limit=10,
        ),
    }


def summarize_results(
    per_paper: Sequence[dict],
    top_k: int,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict:
    metric_values = {
        "kl_divergence": [entry["metrics"]["kl_divergence"] for entry in per_paper if entry["metrics"]["kl_divergence"] is not None],
        "jensen_shannon_divergence": [entry["metrics"]["jensen_shannon_divergence"] for entry in per_paper if entry["metrics"]["jensen_shannon_divergence"] is not None],
        "spearman": [entry["metrics"]["spearman"] for entry in per_paper if entry["metrics"]["spearman"] is not None],
        "kendall_tau_b": [entry["metrics"]["kendall_tau_b"] for entry in per_paper if entry["metrics"]["kendall_tau_b"] is not None],
        "top_k_overlap_count": [entry["top_k_overlap"]["overlap_count"] for entry in per_paper],
        "top_k_overlap_fraction": [entry["top_k_overlap"]["overlap_fraction"] for entry in per_paper],
    }

    summary = {"papers_evaluated": len(per_paper), "top_k": top_k}
    seed_offsets = {
        "kl_divergence": 1,
        "jensen_shannon_divergence": 2,
        "spearman": 3,
        "kendall_tau_b": 4,
        "top_k_overlap_count": 5,
        "top_k_overlap_fraction": 6,
    }
    for metric_key, values in metric_values.items():
        summary[metric_key] = {
            "mean": mean(values),
            "mean_ci": bootstrap_mean_ci(values, n_bootstrap=n_bootstrap, seed=bootstrap_seed + seed_offsets[metric_key]),
            "median": median(values),
            "std": stddev(values),
        }
    return summary


def save_summary_tsv(path: Path, per_paper: Sequence[dict]) -> None:
    lines = [
        "\t".join(
            [
                "paper_id",
                "citation_components",
                "jsd",
                "kl",
                "spearman",
                "kendall_tau_b",
                "top4_overlap_count",
                "top4_overlap_fraction",
                "left_raw_score_sum",
                "right_raw_score_sum",
            ]
        )
    ]

    for entry in sorted(
        per_paper,
        key=lambda item: (
            -(item["metrics"]["jensen_shannon_divergence"] or float("-inf")),
            item["paper_id"],
        ),
    ):
        lines.append(
            "\t".join(
                [
                    entry["paper_id"],
                    str(entry["citation_components"]),
                    format_float(entry["metrics"]["jensen_shannon_divergence"]),
                    format_float(entry["metrics"]["kl_divergence"]),
                    format_float(entry["metrics"]["spearman"]),
                    format_float(entry["metrics"]["kendall_tau_b"]),
                    str(entry["top_k_overlap"]["overlap_count"]),
                    format_float(entry["top_k_overlap"]["overlap_fraction"]),
                    format_float(entry["left_raw_score_sum"]),
                    format_float(entry["right_raw_score_sum"]),
                ]
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary_distributions(path_prefix: Path, per_paper: Sequence[dict], summary: dict) -> None:
    metric_specs = [
        ("jensen_shannon_divergence", "JSD", "Per-paper divergence"),
        ("kl_divergence", "KL", "Per-paper divergence"),
        ("spearman", r"Spearman $\rho$", "Rank correlation"),
        ("kendall_tau_b", r"Kendall $\tau_b$", "Rank correlation"),
        ("top_k_overlap_fraction", f"Top-{summary['top_k']} overlap", "Overlap fraction"),
    ]

    fig, axes = plt.subplots(1, len(metric_specs), figsize=(16.0, 4.6), constrained_layout=False)
    rng = random.Random(7)

    for ax, (metric_key, title, y_label) in zip(axes, metric_specs):
        if metric_key.startswith("top_k_overlap"):
            values = [entry["top_k_overlap"]["overlap_fraction"] for entry in per_paper]
        else:
            values = [entry["metrics"][metric_key] for entry in per_paper if entry["metrics"][metric_key] is not None]
        if not values:
            ax.set_visible(False)
            continue

        violin = ax.violinplot([values], positions=[1], widths=0.72, showmeans=False, showmedians=True, showextrema=True)
        color = SUMMARY_COLORS[metric_key]
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("#2f2f2f")
            body.set_alpha(0.72)
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            artist = violin.get(key)
            if artist is not None:
                artist.set_color("#2f2f2f")
                artist.set_linewidth(1.0)

        x_positions = [1 + rng.uniform(-0.07, 0.07) for _ in values]
        ax.scatter(x_positions, values, s=18, color=color, alpha=0.78, edgecolors="white", linewidths=0.4, zorder=3)

        metric_summary = summary[metric_key]
        mean_value = metric_summary["mean"]
        mean_ci = metric_summary["mean_ci"]
        if mean_value is not None and mean_ci is not None:
            ax.plot([1, 1], [mean_ci[0], mean_ci[1]], color="#111111", linewidth=1.6, zorder=4)
            ax.scatter([1], [mean_value], s=34, color="#111111", zorder=5)

        ax.set_title(title, fontweight="semibold")
        ax.set_ylabel(y_label)
        ax.set_xticks([])
        ax.grid(True, axis="y", color="#d7dbe0", linewidth=0.8, alpha=0.9)
        ax.grid(False, axis="x")
        ax.spines["left"].set_color("#a0a0a0")
        ax.spines["bottom"].set_color("#a0a0a0")
        if metric_key in {"spearman", "kendall_tau_b", "top_k_overlap_fraction"}:
            ax.set_ylim(min(-0.05, min(values) - 0.05), max(1.0, max(values) + 0.05))
        else:
            ax.set_ylim(bottom=min(0.0, min(values) - 0.02))

    fig.suptitle(
        f"{RIGHT_LABEL} vs {LEFT_LABEL} Across {summary['papers_evaluated']} Papers",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.14, top=0.84, wspace=0.34)
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(path_prefix.with_suffix(suffix), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_per_paper_rankings(path_prefix: Path, per_paper: Sequence[dict], top_k: int) -> None:
    ranked = sorted(
        per_paper,
        key=lambda entry: (
            -(entry["metrics"]["jensen_shannon_divergence"] or float("-inf")),
            entry["paper_id"],
        ),
    )
    paper_ids = [entry["paper_id"] for entry in ranked]
    y = np.arange(len(ranked))

    jsd_values = np.array([float(entry["metrics"]["jensen_shannon_divergence"] or 0.0) for entry in ranked])
    spearman_values = np.array([float(entry["metrics"]["spearman"] or 0.0) for entry in ranked])
    overlap_values = np.array([float(entry["top_k_overlap"]["overlap_fraction"]) for entry in ranked])

    fig, axes = plt.subplots(1, 3, figsize=(15.8, max(8.0, 0.36 * len(ranked) + 1.8)), sharey=True, constrained_layout=False)
    panels = [
        (axes[0], jsd_values, "JSD", SUMMARY_COLORS["jensen_shannon_divergence"]),
        (axes[1], spearman_values, r"Spearman $\rho$", SUMMARY_COLORS["spearman"]),
        (axes[2], overlap_values, f"Top-{top_k} overlap", SUMMARY_COLORS["top_k_overlap_fraction"]),
    ]

    for ax, values, xlabel, color in panels:
        ax.hlines(y, xmin=0.0, xmax=values, color=color, alpha=0.30, linewidth=1.2, zorder=1)
        ax.scatter(values, y, color=color, s=34, edgecolors="white", linewidths=0.5, zorder=3)
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="x", color="#d7dbe0", linewidth=0.8, alpha=0.9)
        ax.grid(False, axis="y")
        ax.spines["left"].set_color("#a0a0a0")
        ax.spines["bottom"].set_color("#a0a0a0")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(paper_ids)
    axes[0].invert_yaxis()
    axes[1].set_yticks(y)
    axes[2].set_yticks(y)
    axes[0].set_xlim(left=0.0)
    axes[2].set_xlim(0.0, 1.02)

    fig.suptitle(
        f"Per-paper Citation Agreement: {RIGHT_LABEL} vs {LEFT_LABEL}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.22, right=0.99, bottom=0.06, top=0.92, wspace=0.18)
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(path_prefix.with_suffix(suffix), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def shorten_label(text: str, max_len: int = 34) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1] + "…"


def plot_detailed_paper_distribution(
    paper_id: str,
    rows: Sequence[dict],
    output_dir: Path,
    top_k: int,
) -> None:
    rows = list(rows)
    max_score = max([0.0] + [float(row["left_score"]) for row in rows] + [float(row["right_score"]) for row in rows])
    top_rows = sorted(
        rows,
        key=lambda item: (
            -max(float(item["left_score"]), float(item["right_score"])),
            -float(item["left_score"]),
            -float(item["right_score"]),
            str(item["citation"]),
        ),
    )[: max(1, top_k)]
    top_rows.reverse()

    fig = plt.figure(figsize=(13.2, 8.6), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.55], hspace=0.35, wspace=0.28)
    ax_budget = fig.add_subplot(grid[0, 0])
    ax_scatter = fig.add_subplot(grid[0, 1])
    ax_bar = fig.add_subplot(grid[1, :])

    left_mass = sum(float(row["left_score"]) for row in rows)
    right_mass = sum(float(row["right_score"]) for row in rows)
    ax_budget.bar(
        [LEFT_LABEL, RIGHT_LABEL],
        [left_mass, right_mass],
        color=[LEFT_COLOR, RIGHT_COLOR],
        width=0.62,
        edgecolor="white",
        linewidth=1.0,
    )
    ax_budget.axhline(1.0, color="#7a7a7a", linewidth=1.1, linestyle="--", alpha=0.85)
    ax_budget.set_title("Raw Citation Budget", pad=8)
    ax_budget.set_ylabel("Sum of Raw Citation Scores")
    ax_budget.set_ylim(0.0, max(1.08, left_mass, right_mass) * 1.08)

    x_vals = [float(row["left_score"]) for row in rows]
    y_vals = [float(row["right_score"]) for row in rows]
    diagonal_limit = max(0.02, max_score * 1.08)
    ax_scatter.scatter(x_vals, y_vals, s=34, color=RIGHT_COLOR, alpha=0.72, edgecolors="white", linewidths=0.5)
    ax_scatter.plot([0.0, diagonal_limit], [0.0, diagonal_limit], color="#7a7a7a", linestyle="--", linewidth=1.1)
    ax_scatter.set_xlim(0.0, diagonal_limit)
    ax_scatter.set_ylim(0.0, diagonal_limit)
    ax_scatter.set_title("Citation Scores by Aligned Target", pad=8)
    ax_scatter.set_xlabel(f"{LEFT_LABEL} score")
    ax_scatter.set_ylabel(f"{RIGHT_LABEL} score")

    bar_height = 0.34
    y_pos = list(range(len(top_rows)))
    ax_bar.barh(
        [value + bar_height / 2 for value in y_pos],
        [float(row["left_score"]) for row in top_rows],
        height=bar_height,
        color=LEFT_COLOR,
        label=LEFT_LABEL,
    )
    ax_bar.barh(
        [value - bar_height / 2 for value in y_pos],
        [float(row["right_score"]) for row in top_rows],
        height=bar_height,
        color=RIGHT_COLOR,
        label=RIGHT_LABEL,
    )
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([shorten_label(str(row["citation"])) for row in top_rows])
    ax_bar.set_xlabel("Citation Score")
    ax_bar.set_title(f"Top {min(top_k, len(top_rows))} Citations by Max Score Across Both Variants", pad=8)
    ax_bar.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, borderaxespad=0.0)
    ax_bar.grid(True, axis="x", color="#d7dbe0", linewidth=0.9, alpha=0.85)
    ax_bar.grid(False, axis="y")

    fig.suptitle(f"{paper_id}: {RIGHT_LABEL} vs {LEFT_LABEL}", fontsize=14, fontweight="semibold", y=0.98)
    for ax in (ax_budget, ax_scatter, ax_bar):
        ax.spines["left"].set_color("#8a9199")
        ax.spines["bottom"].set_color("#8a9199")

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{paper_id}_{RIGHT_TAG}_vs_{LEFT_TAG}_citation_distribution.png"
    pdf_path = output_dir / f"{paper_id}_{RIGHT_TAG}_vs_{LEFT_TAG}_citation_distribution.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_ordered_rows(
    paper_id: str,
    results_root: Path,
    left_tag: str,
    right_tag: str,
    resolver: CitationResolver,
) -> List[dict]:
    left_path = score_file_path(results_root, paper_id, left_tag)
    right_path = score_file_path(results_root, paper_id, right_tag)
    left_raw = load_citation_score_map(left_path)
    right_raw = load_citation_score_map(right_path)
    left_grouped = aggregate_scores_by_target(paper_id, left_raw, resolver)
    right_grouped = aggregate_scores_by_target(paper_id, right_raw, resolver)

    rows = []
    for target_id in sorted(set(left_grouped) | set(right_grouped)):
        left_bucket = left_grouped.get(target_id)
        right_bucket = right_grouped.get(target_id)
        label = target_id
        if left_bucket and str(left_bucket["display_key"]).strip():
            label = str(left_bucket["display_key"])
        elif right_bucket and str(right_bucket["display_key"]).strip():
            label = str(right_bucket["display_key"])
        rows.append(
            {
                "target_id": target_id,
                "citation": label,
                "left_score": float(left_bucket["score"]) if left_bucket else 0.0,
                "right_score": float(right_bucket["score"]) if right_bucket else 0.0,
            }
        )
    rows.sort(key=lambda item: (-item["left_score"], -item["right_score"], str(item["citation"])))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare single-shot Anthropic citation scores against hierarchical Anthropic citation scores."
    )
    parser.add_argument("--results-root", default="paper_results")
    parser.add_argument("--papers-dir", default="papers")
    parser.add_argument("--left-tag", default=LEFT_TAG)
    parser.add_argument("--right-tag", default=RIGHT_TAG)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument("--top-diff-plots", type=int, default=6)
    parser.add_argument(
        "--output-json",
        default="results/anthropic_single_shot_vs_hierarchical_comparison.json",
    )
    parser.add_argument(
        "--output-tsv",
        default="results/anthropic_single_shot_vs_hierarchical_summary.tsv",
    )
    parser.add_argument(
        "--plot-prefix",
        default="results/plots/anthropic_single_shot_vs_hierarchical",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    papers_dir = Path(args.papers_dir)
    common_papers = discover_common_papers(results_root, args.left_tag, args.right_tag)

    comparable_papers: List[str] = []
    excluded_papers: List[dict] = []
    for paper_id in common_papers:
        non_positive_tags = [
            tag for tag in (args.left_tag, args.right_tag)
            if total_citation_mass(results_root, paper_id, tag) <= 0.0
        ]
        if non_positive_tags:
            excluded_papers.append(
                {
                    "paper_id": paper_id,
                    "reason": "non_positive_citation_mass",
                    "tags": non_positive_tags,
                }
            )
            continue
        comparable_papers.append(paper_id)

    resolver = prepare_resolver_for_tags(
        results_root=results_root,
        papers_dir=papers_dir,
        paper_ids=comparable_papers,
        tags=[args.left_tag, args.right_tag],
    )

    per_paper = [
        compare_pair_for_paper(
            paper_id=paper_id,
            results_root=results_root,
            left_tag=args.left_tag,
            right_tag=args.right_tag,
            resolver=resolver,
            top_k=args.top_k,
        )
        for paper_id in comparable_papers
    ]

    summary = summarize_results(
        per_paper=per_paper,
        top_k=args.top_k,
        n_bootstrap=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    top_divergence = sorted(
        per_paper,
        key=lambda entry: (
            -(entry["metrics"]["jensen_shannon_divergence"] or float("-inf")),
            entry["paper_id"],
        ),
    )

    report = {
        "left_tag": args.left_tag,
        "right_tag": args.right_tag,
        "left_label": LEFT_LABEL,
        "right_label": RIGHT_LABEL,
        "top_k": args.top_k,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "shared_paper_count_before_validation": len(common_papers),
        "shared_paper_ids_before_validation": common_papers,
        "shared_paper_count": len(comparable_papers),
        "shared_paper_ids": comparable_papers,
        "excluded_papers": excluded_papers,
        "summary": summary,
        "top_divergence_papers": [
            {
                "paper_id": entry["paper_id"],
                "jensen_shannon_divergence": entry["metrics"]["jensen_shannon_divergence"],
                "spearman": entry["metrics"]["spearman"],
                "top_k_overlap_fraction": entry["top_k_overlap"]["overlap_fraction"],
            }
            for entry in top_divergence[: max(0, args.top_diff_plots)]
        ],
        "per_paper": per_paper,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_summary_tsv(Path(args.output_tsv), per_paper)

    plot_prefix = Path(args.plot_prefix)
    plot_summary_distributions(plot_prefix.with_name(plot_prefix.name + "_summary"), per_paper, summary)
    plot_per_paper_rankings(plot_prefix.with_name(plot_prefix.name + "_per_paper"), per_paper, args.top_k)

    detailed_dir = plot_prefix.with_name(plot_prefix.name + "_paper_distributions")
    for entry in top_divergence[: max(0, args.top_diff_plots)]:
        rows = build_ordered_rows(
            paper_id=entry["paper_id"],
            results_root=results_root,
            left_tag=args.left_tag,
            right_tag=args.right_tag,
            resolver=resolver,
        )
        plot_detailed_paper_distribution(
            paper_id=entry["paper_id"],
            rows=rows,
            output_dir=detailed_dir,
            top_k=args.top_k,
        )

    print(f"Shared paper count: {len(comparable_papers)}")
    print(f"Saved JSON report to {output_json}")
    print(f"Saved TSV summary to {args.output_tsv}")
    print(f"Saved summary plot to {plot_prefix.with_name(plot_prefix.name + '_summary')}.png/.pdf")
    print(f"Saved per-paper plot to {plot_prefix.with_name(plot_prefix.name + '_per_paper')}.png/.pdf")
    print(f"Saved detailed paper plots to {detailed_dir}")
    for entry in top_divergence[: max(0, min(args.top_diff_plots, 5))]:
        jsd_value = entry["metrics"]["jensen_shannon_divergence"]
        overlap = entry["top_k_overlap"]["overlap_fraction"]
        print(f"Top divergence: {entry['paper_id']} JSD={jsd_value:.6f} top{args.top_k} overlap={overlap:.3f}")


if __name__ == "__main__":
    main()
