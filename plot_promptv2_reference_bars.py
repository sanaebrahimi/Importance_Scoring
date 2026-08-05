from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from citation_resolver import CitationResolver
from evaluate_citation_jsd import aggregate_scores_by_target, load_citation_score_map
from evaluate_human_section_scores import ndcg_at_k, reciprocal_rank


plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
    }
)


DISPLAY_TO_TAG = OrderedDict(
    [
        ("qwen3:1.7b", "qwen3_1_7_promptv2"),
        ("qwen3:4b", "qwen3_4b_promptv2"),
        ("gemma2:2b", "gemma2_2b_promptv2"),
        ("gemma3:4b", "gemma3_4b_promptv2"),
        ("phi3:medium", "phi3_medium_promptv2"),
        ("qwen2.5:3b", "qwen2_5_3b_promptv2"),
        ("llama3.2:1b", "llama3_2_1b_promptv2"),
        ("llama3.2:3b", "llama3_2_promptv2"),
        ("Citation freq.", "citation_frequency"),
        ("Length-wtd. freq.", "length_weighted_frequency"),
    ]
)


MODEL_COLORS = {
    "qwen3:1.7b": "#59a14f",
    "qwen3:4b": "#9c755f",
    "gemma2:2b": "#e15759",
    "gemma3:4b": "#76b7b2",
    "phi3:medium": "#edc948",
    "qwen2.5:3b": "#b07aa1",
    "llama3.2:1b": "#4e79a7",
    "llama3.2:3b": "#f28e2b",
    "Citation freq.": "#aaaaaa",
    "Length-wtd. freq.": "#666666",
}


METRICS = ["Recall@4", "Hit@4", "MRR", "nDCG@4"]
LLM_COUNT = 8


def score_file_path(results_root: Path, paper_id: str, tag: str) -> Path:
    return results_root / paper_id / f"{paper_id}_{tag}_citation_scores.json"


def total_citation_mass(results_root: Path, paper_id: str, tag: str) -> float:
    return sum(load_citation_score_map(score_file_path(results_root, paper_id, tag)).values())


def discover_papers_for_tag(results_root: Path, tag: str) -> List[str]:
    paper_ids: List[str] = []
    for paper_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        if score_file_path(results_root, paper_dir.name, tag).exists():
            paper_ids.append(paper_dir.name)
    return paper_ids


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
            path = score_file_path(results_root, paper_id, tag)
            if not path.exists():
                continue
            score_map = load_citation_score_map(path)
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


def citation_top_k_metrics(
    paper_id: str,
    reference_tag: str,
    candidate_tag: str,
    results_root: Path,
    resolver: CitationResolver,
    k: int,
) -> dict:
    reference_scores = load_citation_score_map(score_file_path(results_root, paper_id, reference_tag))
    candidate_scores = load_citation_score_map(score_file_path(results_root, paper_id, candidate_tag))

    reference_grouped = aggregate_scores_by_target(paper_id, reference_scores, resolver)
    candidate_grouped = aggregate_scores_by_target(paper_id, candidate_scores, resolver)

    reference_ranked = rank_grouped_citations(reference_grouped)
    candidate_ranked = rank_grouped_citations(candidate_grouped)

    reference_top_k = reference_ranked[: max(0, k)]
    candidate_top_k = candidate_ranked[: max(0, k)]
    reference_top_keys = [item["target_id"] for item in reference_top_k]
    candidate_top_keys = [item["target_id"] for item in candidate_top_k]

    overlap_count = len(set(reference_top_keys).intersection(candidate_top_keys))
    denom = max(1, len(reference_top_k))
    recall_at_k = overlap_count / denom
    hit_at_k = 1.0 if overlap_count > 0 else 0.0
    mrr = reciprocal_rank(reference_top_keys[0] if reference_top_keys else None, candidate_top_keys)
    relevance_by_key = {
        key: float(len(reference_top_keys) - idx)
        for idx, key in enumerate(reference_top_keys)
    }
    ndcg = ndcg_at_k(relevance_by_key, candidate_top_keys, k)

    return {
        "paper_id": paper_id,
        "reference_tag": reference_tag,
        "candidate_tag": candidate_tag,
        "reference_top_k": reference_top_k,
        "candidate_top_k": candidate_top_k,
        "metrics": {
            "Recall@4": recall_at_k,
            "Hit@4": hit_at_k,
            "MRR": mrr,
            "nDCG@4": ndcg,
        },
    }


def aggregate_metrics(
    results_root: Path,
    papers_dir: Path,
    reference_tag: str,
    display_to_tag: OrderedDict[str, str],
    top_k: int,
) -> Tuple[Dict[str, Dict[str, float]], List[str], List[dict]]:
    reference_papers = set(discover_papers_for_tag(results_root, reference_tag))
    selected_tags = list(display_to_tag.values())

    common_papers = sorted(
        reference_papers.intersection(
            *[set(discover_papers_for_tag(results_root, tag)) for tag in selected_tags]
        )
    )

    excluded_papers: List[dict] = []
    comparable_papers: List[str] = []
    for paper_id in common_papers:
        non_positive_tags = [
            tag
            for tag in [reference_tag, *selected_tags]
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
        tags=[reference_tag, *selected_tags],
    )

    metrics_by_model: Dict[str, Dict[str, float]] = OrderedDict()
    for display_name, tag in display_to_tag.items():
        per_paper = [
            citation_top_k_metrics(
                paper_id=paper_id,
                reference_tag=reference_tag,
                candidate_tag=tag,
                results_root=results_root,
                resolver=resolver,
                k=top_k,
            )
            for paper_id in comparable_papers
        ]
        metrics_by_model[display_name] = {
            metric: float(np.mean([row["metrics"][metric] for row in per_paper]))
            for metric in METRICS
        }

    return metrics_by_model, comparable_papers, excluded_papers


def get_x_positions(n_models: int, n_llm: int, bar_width: float, gap: float, baseline_gap: float) -> List[float]:
    positions = []
    x = 0.0
    for i in range(n_models):
        if i == n_llm:
            x += baseline_gap
        positions.append(x + bar_width / 2.0)
        x += bar_width + gap
    midpoint = (positions[0] + positions[-1]) / 2.0
    return [p - midpoint for p in positions]


def build_metric_cfg(data: Dict[str, Dict[str, float]], metrics: Sequence[str]) -> Dict[str, dict]:
    metric_cfg: Dict[str, dict] = {}
    for metric in metrics:
        values = [payload[metric] for payload in data.values()]
        min_v = min(values)
        max_v = max(values)
        span = max(max_v - min_v, 0.10)
        ymin = max(0.0, min_v - 0.18 * span)
        ymax = min(1.05, max_v + 0.22 * span)
        yticks = np.linspace(ymin, ymax, 4)[1:-1]
        metric_cfg[metric] = {
            "ymin": ymin,
            "ymax": ymax,
            "yticks": list(yticks),
        }
    return metric_cfg


def metric_value_label(value: float) -> str:
    return f"{value:.2f}".lstrip("0") if value < 1 else f"{value:.2f}"


def plot_barchart(
    data: Dict[str, Dict[str, float]],
    reference_tag: str,
    output_prefix: Path,
) -> List[Path]:
    models = list(data.keys())
    best_model = {
        metric: max(models[:LLM_COUNT], key=lambda model: data[model][metric])
        for metric in METRICS
    }

    bar_width = 0.058
    gap = 0.010
    baseline_gap = 0.030
    x_positions = get_x_positions(len(models), LLM_COUNT, bar_width, gap, baseline_gap)
    metric_cfg = build_metric_cfg(data, METRICS)

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.4))
    fig.patch.set_facecolor("white")

    for ax, metric in zip(axes, METRICS):
        cfg = metric_cfg[metric]
        ax.set_facecolor("white")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(0.6)
            ax.spines[spine].set_color("#aaaaaa")

        ax.tick_params(axis="both", which="both", length=0)
        ax.set_ylim(cfg["ymin"], cfg["ymax"])
        ax.set_yticks(cfg["yticks"])
        ax.set_yticklabels([metric_value_label(v) for v in cfg["yticks"]], fontsize=8)
        ax.yaxis.grid(True, linestyle="-", linewidth=0.4, color="#e0e0e0", zorder=0)
        ax.set_axisbelow(True)
        ax.set_xticks([])
        ax.set_xlabel(metric, fontsize=10, labelpad=6)

        sep_x = (x_positions[LLM_COUNT - 1] + x_positions[LLM_COUNT]) / 2.0
        ax.axvline(sep_x, color="#cccccc", lw=1.0, ls="--", zorder=1)

        for i, model in enumerate(models):
            x = x_positions[i]
            value = data[model][metric]
            ax.bar(
                x,
                value - cfg["ymin"],
                bottom=cfg["ymin"],
                width=bar_width,
                color=MODEL_COLORS[model],
                zorder=3,
                linewidth=0,
                align="center",
            )
            ax.text(
                x,
                value + (cfg["ymax"] - cfg["ymin"]) * 0.018,
                metric_value_label(value),
                ha="center",
                va="bottom",
                fontsize=5.2,
                color="#333333",
            )

        best = best_model[metric]
        best_idx = models.index(best)
        best_x = x_positions[best_idx]
        ax.text(
            best_x,
            cfg["ymax"] * 0.985,
            "*",
            ha="center",
            va="top",
            fontsize=11,
            color="#222222",
            fontweight="bold",
        )

        pad = bar_width * 0.8
        ax.set_xlim(x_positions[0] - pad, x_positions[-1] + pad)

    handles = [mpatches.Patch(color=MODEL_COLORS[model], label=model) for model in models]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=LLM_COUNT,
        fontsize=7.2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
        handlelength=1.1,
        handleheight=0.9,
        columnspacing=0.7,
    )

    fig.suptitle(
        f"promptv2 top-4 vs. {reference_tag} reference   (* = best LLM)",
        fontsize=10.5,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.subplots_adjust(wspace=0.42)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    for suffix in (".png", ".pdf"):
        path = output_prefix.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", dpi=200, facecolor="white")
        saved_paths.append(path)
    plt.close(fig)
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot promptv2 top-4 citation metrics against a reference model."
    )
    parser.add_argument("--results-root", default="paper_results")
    parser.add_argument("--papers-dir", default="papers")
    parser.add_argument("--reference-tag", default="anthropic_full_paper")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--output-prefix",
        default="results/plots/anthropic_promptv2_top4_bars",
        help="Output file prefix for the bar chart.",
    )
    parser.add_argument(
        "--output-json",
        default="results/anthropic_promptv2_top4_bar_metrics.json",
        help="Path to save the computed metric summary.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    papers_dir = Path(args.papers_dir)

    metrics_by_model, comparable_papers, excluded_papers = aggregate_metrics(
        results_root=results_root,
        papers_dir=papers_dir,
        reference_tag=args.reference_tag,
        display_to_tag=DISPLAY_TO_TAG,
        top_k=args.top_k,
    )

    saved_plot_paths = plot_barchart(
        data=metrics_by_model,
        reference_tag=args.reference_tag,
        output_prefix=Path(args.output_prefix),
    )

    summary = {
        "reference_tag": args.reference_tag,
        "top_k": args.top_k,
        "papers_evaluated": len(comparable_papers),
        "paper_ids": comparable_papers,
        "excluded_papers": excluded_papers,
        "models": [
            {
                "display_name": display_name,
                "model_tag": DISPLAY_TO_TAG[display_name],
                "metrics": metrics,
            }
            for display_name, metrics in metrics_by_model.items()
        ],
        "plot_files": [str(path) for path in saved_plot_paths],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Reference: {args.reference_tag}")
    print(f"Papers evaluated: {len(comparable_papers)}")
    if excluded_papers:
        print(f"Excluded papers: {excluded_papers}")
    print("Metrics:")
    for display_name, metrics in metrics_by_model.items():
        metric_text = ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
        print(f"  {display_name}: {metric_text}")
    print("Saved files:")
    print(f"  {output_json}")
    for path in saved_plot_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
