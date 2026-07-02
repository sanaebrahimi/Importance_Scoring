from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_citation_jsd import (
    aggregate_scores_by_target,
    load_citation_score_map,
    prepare_resolver,
)


LEFT_COLOR = "#C44E52"
RIGHT_COLOR = "#4C78A8"
BUDGET_COLOR = "#7A7A7A"


def ordered_components_for_paper(
    paper_id: str,
    results_root: Path,
    left_tag: str,
    right_tag: str,
    resolver,
) -> List[dict]:
    left_path = results_root / paper_id / f"{paper_id}_{left_tag}_citation_scores.json"
    right_path = results_root / paper_id / f"{paper_id}_{right_tag}_citation_scores.json"

    left_raw = load_citation_score_map(left_path)
    right_raw = load_citation_score_map(right_path)
    left_grouped = aggregate_scores_by_target(paper_id, left_raw, resolver)
    right_grouped = aggregate_scores_by_target(paper_id, right_raw, resolver)

    all_target_ids = sorted(set(left_grouped) | set(right_grouped))
    rows = []
    for target_id in all_target_ids:
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


def shorten_label(text: str, max_len: int = 34) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1] + "…"


def augment_rows_with_raw_scores(rows: Sequence[dict]) -> tuple[List[dict], float, float]:
    left_mass = sum(float(row["left_score"]) for row in rows)
    right_mass = sum(float(row["right_score"]) for row in rows)
    augmented: List[dict] = []
    for row in rows:
        augmented.append(
            {
                **row,
                "label_short": shorten_label(str(row["citation"])),
            }
        )
    return augmented, left_mass, right_mass


def select_top_k_rows(rows: Sequence[dict], top_k: int) -> List[dict]:
    ranked = sorted(
        rows,
        key=lambda item: (
            -max(float(item["left_score"]), float(item["right_score"])),
            -float(item["left_score"]),
            -float(item["right_score"]),
            str(item["citation"]),
        ),
    )
    top = ranked[: max(1, top_k)]
    top.reverse()
    return top


def save_component_tsv(path: Path, rows: Sequence[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "rank",
                "citation",
                "target_id",
                "openai_score_raw",
                "promptv2_score_raw",
            ]
        )
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                [
                    rank,
                    row["citation"],
                    row["target_id"],
                    row["left_score"],
                    row["right_score"],
                ]
            )


def plot_paper_distribution(
    paper_id: str,
    rows: Sequence[dict],
    left_tag: str,
    right_tag: str,
    output_dir: Path,
    top_k: int,
) -> List[Path]:
    rows_augmented, left_mass, right_mass = augment_rows_with_raw_scores(rows)
    top_rows = select_top_k_rows(rows_augmented, top_k=top_k)
    x_citations = [float(row["left_score"]) for row in rows_augmented]
    y_citations = [float(row["right_score"]) for row in rows_augmented]
    max_score = max(x_citations + y_citations + [0.0])

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    fig = plt.figure(figsize=(13.2, 8.8), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.55], hspace=0.35, wspace=0.28)
    ax_budget = fig.add_subplot(grid[0, 0])
    ax_scatter = fig.add_subplot(grid[0, 1])
    ax_bar = fig.add_subplot(grid[1, :])

    ax_budget.bar(
        [left_tag, right_tag],
        [left_mass, right_mass],
        color=[LEFT_COLOR, RIGHT_COLOR],
        width=0.62,
        edgecolor="white",
        linewidth=1.0,
    )
    ax_budget.axhline(1.0, color=BUDGET_COLOR, linewidth=1.1, linestyle="--", alpha=0.8)
    ax_budget.set_title("Raw Citation Budget", pad=8)
    ax_budget.set_ylabel("Sum of Raw Citation Scores")
    ax_budget.set_ylim(0.0, max(1.08, left_mass, right_mass) * 1.08)
    for xpos, value in enumerate([left_mass, right_mass]):
        ax_budget.text(xpos, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    ax_scatter.scatter(
        x_citations,
        y_citations,
        s=34,
        color=RIGHT_COLOR,
        alpha=0.72,
        edgecolors="white",
        linewidths=0.5,
    )
    diagonal_limit = max(0.02, max_score * 1.08)
    ax_scatter.plot([0.0, diagonal_limit], [0.0, diagonal_limit], color=BUDGET_COLOR, linestyle="--", linewidth=1.1)
    ax_scatter.set_xlim(0.0, diagonal_limit)
    ax_scatter.set_ylim(0.0, diagonal_limit)
    ax_scatter.set_title("Raw Citation Scores", pad=8)
    ax_scatter.set_xlabel(f"{left_tag} raw score")
    ax_scatter.set_ylabel(f"{right_tag} raw score")
    ax_scatter.text(
        0.03,
        0.97,
        "Each point is one aligned citation",
        transform=ax_scatter.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        bbox={"facecolor": "white", "edgecolor": "#D0D7DE", "boxstyle": "round,pad=0.25", "alpha": 0.95},
    )

    y_positions = list(range(len(top_rows)))
    bar_height = 0.34
    ax_bar.barh(
        [y + bar_height / 2 for y in y_positions],
        [float(row["left_score"]) for row in top_rows],
        height=bar_height,
        color=LEFT_COLOR,
        label=left_tag,
    )
    ax_bar.barh(
        [y - bar_height / 2 for y in y_positions],
        [float(row["right_score"]) for row in top_rows],
        height=bar_height,
        color=RIGHT_COLOR,
        label=right_tag,
    )
    ax_bar.set_yticks(y_positions)
    ax_bar.set_yticklabels([row["label_short"] for row in top_rows])
    ax_bar.set_xlabel("Raw Citation Score")
    ax_bar.set_title(
        f"Top {min(top_k, len(top_rows))} Citations by Max Raw Score Across Both Models",
        pad=8,
    )
    ax_bar.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=2,
        borderaxespad=0.0,
        handlelength=1.8,
        columnspacing=1.6,
    )
    ax_bar.grid(True, axis="x", color="#D7DBE0", linewidth=0.9, alpha=0.85)
    ax_bar.grid(False, axis="y")

    summary = (
        f"Aligned citations: {len(rows_augmented)}\n"
        f"Top-left bars show total raw citation mass\n"
        f"Scatter and bars use raw citation scores"
    )
    ax_bar.text(
        0.995,
        0.02,
        summary,
        transform=ax_bar.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        bbox={"facecolor": "white", "edgecolor": "#D0D7DE", "boxstyle": "round,pad=0.35", "alpha": 0.95},
    )

    fig.suptitle(f"{paper_id}: Citation Comparison", fontsize=14, fontweight="semibold", y=0.98)

    for ax in (ax_budget, ax_scatter, ax_bar):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#8A9199")
        ax.spines["bottom"].set_color("#8A9199")
        ax.tick_params(axis="both", colors="#1F2937")

    fig.subplots_adjust(left=0.18, right=0.98, top=0.92, bottom=0.08, hspace=0.34, wspace=0.28)

    png_path = output_dir / f"{paper_id}_{left_tag}_vs_{right_tag}_citation_distribution.png"
    pdf_path = output_dir / f"{paper_id}_{left_tag}_vs_{right_tag}_citation_distribution.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    tsv_path = output_dir / f"{paper_id}_{left_tag}_vs_{right_tag}_citation_distribution.tsv"
    save_component_tsv(tsv_path, rows_augmented)
    return [png_path, pdf_path, tsv_path]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-paper citation score distributions for two model outputs."
    )
    parser.add_argument(
        "--results-root",
        default="paper_results",
        help="Directory containing per-paper result folders.",
    )
    parser.add_argument(
        "--papers-dir",
        default="papers",
        help="Directory containing the source PDFs used for citation resolution.",
    )
    parser.add_argument(
        "--paper-ids",
        nargs="+",
        default=["AFD", "requal"],
        help="Paper ids to plot.",
    )
    parser.add_argument(
        "--left-tag",
        default="openai_full_paper",
        help="Left citation-score model tag.",
    )
    parser.add_argument(
        "--right-tag",
        default="llama3_2_promptv2",
        help="Right citation-score model tag.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_results/distribution_plots",
        help="Directory to save plots and aligned TSV files.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of citations to include in the paired bar chart.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    papers_dir = Path(args.papers_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolver = prepare_resolver(
        results_root=results_root,
        papers_dir=papers_dir,
        paper_ids=args.paper_ids,
        left_tag=args.left_tag,
        right_tag=args.right_tag,
    )

    saved_paths: List[Path] = []
    for paper_id in args.paper_ids:
        rows = ordered_components_for_paper(
            paper_id=paper_id,
            results_root=results_root,
            left_tag=args.left_tag,
            right_tag=args.right_tag,
            resolver=resolver,
        )
        saved_paths.extend(
            plot_paper_distribution(
                paper_id=paper_id,
                rows=rows,
                left_tag=args.left_tag,
                right_tag=args.right_tag,
                output_dir=output_dir,
                top_k=max(1, args.top_k),
            )
        )

    print("Saved files:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
