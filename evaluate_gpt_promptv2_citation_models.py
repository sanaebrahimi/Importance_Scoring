from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from citation_resolver import CitationResolver
from evaluate_citation_jsd import (
    aggregate_scores_by_target,
    align_score_vectors,
    load_citation_score_map,
)
from evaluate_human_section_scores import (
    jensen_shannon_divergence,
    kendall_tau_b,
    kl_divergence,
    spearman_rho,
)


GPT_TAG = "openai_full_paper"
DEFAULT_TOP_K = 4

# These are the promptv2 local models already used in the repo's comparison plots.
DEFAULT_MODEL_TAGS: Dict[str, str] = {
    "llama3.2:1b": "llama3_2_1b_promptv2",
    "llama3.2:3b": "llama3_2_promptv2",
    "gemma2:2b": "gemma2_2b_promptv2",
    "gemma3:4b": "gemma3_4b_promptv2",
    "qwen3:1.7b": "qwen3_1_7_promptv2",
    "phi3:medium": "phi3_medium_promptv2",
    "qwen2.5:3b": "qwen2_5_3b_promptv2",
}

MODEL_COLORS: Dict[str, str] = {
    "llama3.2:1b": "#4e79a7",
    "llama3.2:3b": "#f28e2b",
    "gemma2:2b": "#e15759",
    "gemma3:4b": "#76b7b2",
    "qwen3:1.7b": "#59a14f",
    "phi3:medium": "#edc948",
    "qwen2.5:3b": "#b07aa1",
}


def mean(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def stddev(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return 0.0 if values else None
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
    means: List[float] = []
    n = len(values)
    for _ in range(n_bootstrap):
        sample = [float(values[rng.randrange(n)]) for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()

    lo_index = max(0, int(0.025 * n_bootstrap))
    hi_index = min(n_bootstrap - 1, int(0.975 * n_bootstrap))
    return (means[lo_index], means[hi_index])


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def model_display_name(tag: str) -> str:
    for display_name, model_tag in DEFAULT_MODEL_TAGS.items():
        if model_tag == tag:
            return display_name
    return tag


def score_file_path(results_root: Path, paper_id: str, tag: str) -> Path:
    return results_root / paper_id / f"{paper_id}_{tag}_citation_scores.json"


def total_citation_mass(results_root: Path, paper_id: str, tag: str) -> float:
    score_map = load_citation_score_map(score_file_path(results_root, paper_id, tag))
    return sum(float(score) for score in score_map.values())


def discover_papers_for_tag(results_root: Path, tag: str) -> List[str]:
    paper_ids: List[str] = []
    for paper_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        if score_file_path(results_root, paper_dir.name, tag).exists():
            paper_ids.append(paper_dir.name)
    return paper_ids


def discover_promptv2_tags(results_root: Path) -> Dict[str, int]:
    coverage: Dict[str, set] = {}
    suffix = "_citation_scores.json"
    for paper_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        paper_id = paper_dir.name
        prefix = f"{paper_id}_"
        for path in paper_dir.glob(f"{paper_id}_*_citation_scores.json"):
            name = path.name
            if not name.endswith(suffix) or not name.startswith(prefix):
                continue
            tag = name[len(prefix) : -len(suffix)]
            if not tag.endswith("promptv2"):
                continue
            coverage.setdefault(tag, set()).add(paper_id)
    return {tag: len(papers) for tag, papers in sorted(coverage.items())}


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
                normalized = re.sub(r"\s+", " ", citation_key).strip()
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
    gpt_grouped: Dict[str, dict],
    model_grouped: Dict[str, dict],
    k: int,
) -> dict:
    gpt_ranked = rank_grouped_citations(gpt_grouped)
    model_ranked = rank_grouped_citations(model_grouped)
    gpt_top_k = gpt_ranked[: max(0, k)]
    model_top_k = model_ranked[: max(0, k)]

    gpt_top_ids = {item["target_id"] for item in gpt_top_k}
    model_top_ids = {item["target_id"] for item in model_top_k}
    overlap_ids = sorted(gpt_top_ids.intersection(model_top_ids))

    gpt_lookup = {item["target_id"]: item for item in gpt_ranked}
    model_lookup = {item["target_id"]: item for item in model_ranked}
    denom = max(1, min(k, len(gpt_top_k), len(model_top_k)))

    return {
        "k": k,
        "overlap_count": len(overlap_ids),
        "overlap_fraction": len(overlap_ids) / denom,
        "gpt_top_k": gpt_top_k,
        "model_top_k": model_top_k,
        "overlapping_citations": [
            {
                "target_id": target_id,
                "gpt_citation": gpt_lookup[target_id]["citation"],
                "model_citation": model_lookup[target_id]["citation"],
                "gpt_score": gpt_lookup[target_id]["score"],
                "model_score": model_lookup[target_id]["score"],
            }
            for target_id in overlap_ids
        ],
    }


def compare_model_to_gpt_for_paper(
    paper_id: str,
    results_root: Path,
    gpt_tag: str,
    model_tag: str,
    resolver: CitationResolver,
    top_k: int,
) -> dict:
    gpt_scores = load_citation_score_map(score_file_path(results_root, paper_id, gpt_tag))
    model_scores = load_citation_score_map(score_file_path(results_root, paper_id, model_tag))
    gpt_grouped = aggregate_scores_by_target(paper_id, gpt_scores, resolver)
    model_grouped = aggregate_scores_by_target(paper_id, model_scores, resolver)
    target_ids, labels, gpt_values, model_values = align_score_vectors(gpt_grouped, model_grouped)

    metrics = {
        "kl_divergence": kl_divergence(gpt_values, model_values),
        "jensen_shannon_divergence": jensen_shannon_divergence(gpt_values, model_values),
        "spearman": spearman_rho(gpt_values, model_values),
        "kendall_tau_b": kendall_tau_b(gpt_values, model_values),
    }
    top_k_report = top_k_overlap_report(gpt_grouped, model_grouped, top_k)

    return {
        "paper_id": paper_id,
        "citation_components": len(target_ids),
        "gpt_total_citation_mass": sum(gpt_values),
        "model_total_citation_mass": sum(model_values),
        "metrics": metrics,
        "top_k_overlap": top_k_report,
        "top_absolute_differences": top_absolute_differences(
            target_ids=target_ids,
            labels=labels,
            left_values=gpt_values,
            right_values=model_values,
            limit=10,
        ),
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
                "gpt_score": float(left_value),
                "model_score": float(right_value),
                "absolute_difference": abs(float(left_value) - float(right_value)),
            }
        )
    diffs.sort(key=lambda item: (-float(item["absolute_difference"]), str(item["citation"])))
    return diffs[: max(0, limit)]


def summarize_model_results(
    per_paper: Sequence[dict],
    top_k: int,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict:
    kl_values = [entry["metrics"]["kl_divergence"] for entry in per_paper if entry["metrics"]["kl_divergence"] is not None]
    jsd_values = [entry["metrics"]["jensen_shannon_divergence"] for entry in per_paper if entry["metrics"]["jensen_shannon_divergence"] is not None]
    spearman_values = [entry["metrics"]["spearman"] for entry in per_paper if entry["metrics"]["spearman"] is not None]
    kendall_values = [entry["metrics"]["kendall_tau_b"] for entry in per_paper if entry["metrics"]["kendall_tau_b"] is not None]
    overlap_counts = [entry["top_k_overlap"]["overlap_count"] for entry in per_paper]
    overlap_fractions = [entry["top_k_overlap"]["overlap_fraction"] for entry in per_paper]

    return {
        "papers_evaluated": len(per_paper),
        "mean_kl_divergence": mean(kl_values),
        "mean_kl_divergence_ci": bootstrap_mean_ci(kl_values, n_bootstrap=n_bootstrap, seed=bootstrap_seed + 1),
        "median_kl_divergence": median(kl_values),
        "std_kl_divergence": stddev(kl_values),
        "mean_jensen_shannon_divergence": mean(jsd_values),
        "mean_jensen_shannon_divergence_ci": bootstrap_mean_ci(jsd_values, n_bootstrap=n_bootstrap, seed=bootstrap_seed + 2),
        "median_jensen_shannon_divergence": median(jsd_values),
        "std_jensen_shannon_divergence": stddev(jsd_values),
        "mean_spearman": mean(spearman_values),
        "mean_spearman_ci": bootstrap_mean_ci(spearman_values, n_bootstrap=n_bootstrap, seed=bootstrap_seed + 3),
        "median_spearman": median(spearman_values),
        "std_spearman": stddev(spearman_values),
        "mean_kendall_tau_b": mean(kendall_values),
        "mean_kendall_tau_b_ci": bootstrap_mean_ci(kendall_values, n_bootstrap=n_bootstrap, seed=bootstrap_seed + 4),
        "median_kendall_tau_b": median(kendall_values),
        "std_kendall_tau_b": stddev(kendall_values),
        f"mean_top_{top_k}_overlap_count": mean(overlap_counts),
        f"mean_top_{top_k}_overlap_count_ci": bootstrap_mean_ci(overlap_counts, n_bootstrap=n_bootstrap, seed=bootstrap_seed + 5),
        f"median_top_{top_k}_overlap_count": median(overlap_counts),
        f"mean_top_{top_k}_overlap_fraction": mean(overlap_fractions),
        f"mean_top_{top_k}_overlap_fraction_ci": bootstrap_mean_ci(overlap_fractions, n_bootstrap=n_bootstrap, seed=bootstrap_seed + 6),
        f"median_top_{top_k}_overlap_fraction": median(overlap_fractions),
        f"std_top_{top_k}_overlap_fraction": stddev(overlap_fractions),
    }


def sorted_metric_ranks(
    model_summaries: Dict[str, dict],
    metric_key: str,
    reverse: bool,
) -> Dict[str, int]:
    sortable = []
    for display_name, payload in model_summaries.items():
        value = payload.get(metric_key)
        fallback = float("-inf") if reverse else float("inf")
        sortable.append((display_name, fallback if value is None else float(value)))
    sortable.sort(key=lambda item: ((-item[1]) if reverse else item[1], item[0]))
    return {display_name: rank for rank, (display_name, _) in enumerate(sortable, start=1)}


def build_selection_summary(model_summaries: Dict[str, dict], top_k: int) -> dict:
    ranking_specs = [
        ("mean_kl_divergence", False),
        ("mean_jensen_shannon_divergence", False),
        ("std_kl_divergence", False),
        ("mean_spearman", True),
        ("mean_kendall_tau_b", True),
        (f"mean_top_{top_k}_overlap_fraction", True),
    ]

    rank_maps = {
        metric_key: sorted_metric_ranks(model_summaries, metric_key, reverse=reverse)
        for metric_key, reverse in ranking_specs
    }

    leaderboard = []
    for display_name, summary in sorted(model_summaries.items()):
        rank_sum = sum(rank_maps[metric_key][display_name] for metric_key, _ in ranking_specs)
        leaderboard.append(
            {
                "model": display_name,
                "model_tag": summary["model_tag"],
                "rank_sum": rank_sum,
                "metric_ranks": {
                    metric_key: rank_maps[metric_key][display_name]
                    for metric_key, _ in ranking_specs
                },
            }
        )

    leaderboard.sort(key=lambda item: (item["rank_sum"], item["model"]))
    return {
        "ranking_metrics": [metric_key for metric_key, _ in ranking_specs],
        "leaderboard": leaderboard,
        "recommended_model": leaderboard[0] if leaderboard else None,
    }


def save_summary_tsv(path: Path, model_summaries: Dict[str, dict], selection_summary: dict, top_k: int) -> None:
    recommended = selection_summary.get("recommended_model") or {}
    rank_by_model = {
        entry["model"]: entry
        for entry in selection_summary.get("leaderboard", [])
    }
    lines = [
        "\t".join(
            [
                "model",
                "model_tag",
                "papers_evaluated",
                "mean_kl_divergence",
                "mean_kl_divergence_ci_low",
                "mean_kl_divergence_ci_high",
                "median_kl_divergence",
                "std_kl_divergence",
                "mean_jensen_shannon_divergence",
                "mean_jensen_shannon_divergence_ci_low",
                "mean_jensen_shannon_divergence_ci_high",
                "mean_spearman",
                "mean_spearman_ci_low",
                "mean_spearman_ci_high",
                "mean_kendall_tau_b",
                "mean_kendall_tau_b_ci_low",
                "mean_kendall_tau_b_ci_high",
                f"mean_top_{top_k}_overlap_count",
                f"mean_top_{top_k}_overlap_count_ci_low",
                f"mean_top_{top_k}_overlap_count_ci_high",
                f"mean_top_{top_k}_overlap_fraction",
                f"mean_top_{top_k}_overlap_fraction_ci_low",
                f"mean_top_{top_k}_overlap_fraction_ci_high",
                "selection_rank_sum",
                "is_recommended",
            ]
        )
    ]

    for display_name, summary in sorted(
        model_summaries.items(),
        key=lambda item: (
            rank_by_model.get(item[0], {}).get("rank_sum", float("inf")),
            item[0],
        ),
    ):
        rank_entry = rank_by_model.get(display_name, {})
        lines.append(
            "\t".join(
                [
                    display_name,
                    summary["model_tag"],
                    str(summary["papers_evaluated"]),
                    format_float(summary["mean_kl_divergence"]),
                    format_ci_low(summary["mean_kl_divergence_ci"]),
                    format_ci_high(summary["mean_kl_divergence_ci"]),
                    format_float(summary["median_kl_divergence"]),
                    format_float(summary["std_kl_divergence"]),
                    format_float(summary["mean_jensen_shannon_divergence"]),
                    format_ci_low(summary["mean_jensen_shannon_divergence_ci"]),
                    format_ci_high(summary["mean_jensen_shannon_divergence_ci"]),
                    format_float(summary["mean_spearman"]),
                    format_ci_low(summary["mean_spearman_ci"]),
                    format_ci_high(summary["mean_spearman_ci"]),
                    format_float(summary["mean_kendall_tau_b"]),
                    format_ci_low(summary["mean_kendall_tau_b_ci"]),
                    format_ci_high(summary["mean_kendall_tau_b_ci"]),
                    format_float(summary[f"mean_top_{top_k}_overlap_count"]),
                    format_ci_low(summary[f"mean_top_{top_k}_overlap_count_ci"]),
                    format_ci_high(summary[f"mean_top_{top_k}_overlap_count_ci"]),
                    format_float(summary[f"mean_top_{top_k}_overlap_fraction"]),
                    format_ci_low(summary[f"mean_top_{top_k}_overlap_fraction_ci"]),
                    format_ci_high(summary[f"mean_top_{top_k}_overlap_fraction_ci"]),
                    str(rank_entry.get("rank_sum", "")),
                    "yes" if recommended.get("model") == display_name else "no",
                ]
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def format_ci_low(bounds: Optional[Tuple[float, float]]) -> str:
    if bounds is None:
        return ""
    return f"{float(bounds[0]):.6f}"


def format_ci_high(bounds: Optional[Tuple[float, float]]) -> str:
    if bounds is None:
        return ""
    return f"{float(bounds[1]):.6f}"


def plot_metric_violin(
    path_prefix: Path,
    per_model_results: Dict[str, Sequence[dict]],
    metric_key: str,
    title: str,
    y_label: str,
) -> None:
    labels = list(per_model_results)
    data = [
        [entry["metrics"][metric_key] for entry in per_model_results[label] if entry["metrics"][metric_key] is not None]
        for label in labels
    ]
    if not any(data):
        return

    fig, ax = plt.subplots(figsize=(11.2, 6.2), constrained_layout=False)
    violin = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=True)

    for body, label in zip(violin["bodies"], labels):
        body.set_facecolor(MODEL_COLORS.get(label, "#7f7f7f"))
        body.set_edgecolor("#2f2f2f")
        body.set_alpha(0.72)

    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        artist = violin.get(key)
        if artist is not None:
            artist.set_color("#2f2f2f")
            artist.set_linewidth(1.0)

    rng = random.Random(7)
    for index, label in enumerate(labels, start=1):
        values = data[index - 1]
        x_positions = [index + rng.uniform(-0.08, 0.08) for _ in values]
        ax.scatter(
            x_positions,
            values,
            s=18,
            color=MODEL_COLORS.get(label, "#7f7f7f"),
            alpha=0.7,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )

    ax.set_title(title, fontsize=16, fontweight="semibold")
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", color="#d7dbe0", linewidth=0.8, alpha=0.9)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.24, top=0.90)

    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(path_prefix.with_suffix(suffix), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_kl_violin(
    path_prefix: Path,
    per_model_results: Dict[str, Sequence[dict]],
) -> None:
    plot_metric_violin(
        path_prefix=path_prefix,
        per_model_results=per_model_results,
        metric_key="kl_divergence",
        title="KL Divergence to OpenAI Across Papers",
        y_label="KL Divergence",
    )


def plot_jsd_violin(
    path_prefix: Path,
    per_model_results: Dict[str, Sequence[dict]],
) -> None:
    plot_metric_violin(
        path_prefix=path_prefix,
        per_model_results=per_model_results,
        metric_key="jensen_shannon_divergence",
        title="JSD to OpenAI Across Papers",
        y_label="Jensen-Shannon Divergence",
    )


def collect_selected_models(
    results_root: Path,
    gpt_tag: str,
    requested_tags: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, str], Dict[str, dict], List[str], Dict[str, int]]:
    gpt_papers = set(discover_papers_for_tag(results_root, gpt_tag))
    discovered_promptv2 = discover_promptv2_tags(results_root)

    if requested_tags:
        selected_by_display = {model_display_name(tag): tag for tag in requested_tags}
    else:
        selected_by_display = dict(DEFAULT_MODEL_TAGS)

    coverage_summary: Dict[str, dict] = {}
    max_common_count = 0
    for display_name, tag in selected_by_display.items():
        common_papers = sorted(gpt_papers.intersection(discover_papers_for_tag(results_root, tag)))
        coverage_summary[display_name] = {
            "model_tag": tag,
            "gpt_common_papers": common_papers,
            "gpt_common_count": len(common_papers),
        }
        max_common_count = max(max_common_count, len(common_papers))

    fully_covered = {
        display_name: payload["model_tag"]
        for display_name, payload in coverage_summary.items()
        if payload["gpt_common_count"] == max_common_count and payload["gpt_common_count"] > 0
    }

    common_papers = sorted(
        set(gpt_papers).intersection(
            *(
                set(coverage_summary[display_name]["gpt_common_papers"])
                for display_name in fully_covered
            )
        )
    ) if fully_covered else []

    return fully_covered, coverage_summary, common_papers, discovered_promptv2


def build_report(
    results_root: Path,
    papers_dir: Path,
    gpt_tag: str,
    selected_models: Dict[str, str],
    coverage_summary: Dict[str, dict],
    common_papers: Sequence[str],
    discovered_promptv2: Dict[str, int],
    top_k: int,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict:
    comparable_papers: List[str] = []
    excluded_papers: List[dict] = []
    required_tags = [gpt_tag, *selected_models.values()]
    for paper_id in common_papers:
        non_positive_tags = [
            tag for tag in required_tags
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
        tags=required_tags,
    )

    per_model_results: Dict[str, List[dict]] = {}
    model_summaries: Dict[str, dict] = {}

    for display_name, model_tag in selected_models.items():
        per_paper = [
            compare_model_to_gpt_for_paper(
                paper_id=paper_id,
                results_root=results_root,
                gpt_tag=gpt_tag,
                model_tag=model_tag,
                resolver=resolver,
                top_k=top_k,
            )
            for paper_id in comparable_papers
        ]
        per_model_results[display_name] = per_paper
        summary = summarize_model_results(
            per_paper,
            top_k=top_k,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
        )
        summary["model_tag"] = model_tag
        model_summaries[display_name] = summary

    selection_summary = build_selection_summary(model_summaries, top_k=top_k)

    return {
        "gpt_tag": gpt_tag,
        "top_k": top_k,
        "bootstrap_samples": n_bootstrap,
        "bootstrap_seed": bootstrap_seed,
        "shared_paper_count_before_validation": len(common_papers),
        "shared_paper_ids_before_validation": list(common_papers),
        "shared_paper_count": len(comparable_papers),
        "shared_paper_ids": list(comparable_papers),
        "excluded_papers": excluded_papers,
        "selected_models": selected_models,
        "coverage_summary": coverage_summary,
        "discovered_promptv2_tags": discovered_promptv2,
        "notes": [
            "Citation keys are canonicalized and merged with CitationResolver before comparison.",
            "KL and JSD are computed on citation-only score distributions normalized within each paper.",
            "Spearman and Kendall tau-b are computed on aligned raw citation-score vectors after canonical aggregation.",
            f"Top-k overlap uses k={top_k} on canonicalized citation targets.",
            "Only models with the maximum OpenAI-overlap coverage are included so every selected model is evaluated on the same paper set.",
            "Papers with empty or non-positive citation mass for OpenAI or any selected local model are excluded from the final citation benchmark.",
        ],
        "model_summaries": model_summaries,
        "selection_summary": selection_summary,
        "per_model_results": per_model_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare promptv2 local citation-score distributions against OpenAI outputs."
    )
    parser.add_argument("--results-root", default="paper_results")
    parser.add_argument("--papers-dir", default="papers")
    parser.add_argument("--gpt-tag", default=GPT_TAG)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--model-tags",
        nargs="*",
        default=None,
        help="Optional explicit promptv2 model tags. Defaults to the repo's main promptv2 local models.",
    )
    parser.add_argument(
        "--output-json",
        default="results/openai_promptv2_citation_model_comparison.json",
    )
    parser.add_argument(
        "--output-tsv",
        default="results/openai_promptv2_citation_model_summary.tsv",
    )
    parser.add_argument(
        "--plot-prefix",
        default="results/plots/openai_promptv2_kl_violin",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    papers_dir = Path(args.papers_dir)

    selected_models, coverage_summary, common_papers, discovered_promptv2 = collect_selected_models(
        results_root=results_root,
        gpt_tag=args.gpt_tag,
        requested_tags=args.model_tags,
    )

    report = build_report(
        results_root=results_root,
        papers_dir=papers_dir,
        gpt_tag=args.gpt_tag,
        selected_models=selected_models,
        coverage_summary=coverage_summary,
        common_papers=common_papers,
        discovered_promptv2=discovered_promptv2,
        top_k=args.top_k,
        n_bootstrap=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    save_summary_tsv(
        path=Path(args.output_tsv),
        model_summaries=report["model_summaries"],
        selection_summary=report["selection_summary"],
        top_k=args.top_k,
    )
    plot_kl_violin(
        path_prefix=Path(args.plot_prefix),
        per_model_results=report["per_model_results"],
    )
    plot_jsd_violin(
        path_prefix=Path(str(args.plot_prefix).replace("_kl_violin", "_jsd_violin")),
        per_model_results=report["per_model_results"],
    )

    print(f"Saved JSON report to {output_json}")
    print(f"Shared paper count: {report['shared_paper_count']}")
    print("Selected models:")
    for display_name, model_tag in report["selected_models"].items():
        print(f"  {display_name}: {model_tag}")
    recommended = report["selection_summary"].get("recommended_model")
    if recommended:
        print(
            "Recommended model: "
            f"{recommended['model']} ({recommended['model_tag']}) "
            f"with rank_sum={recommended['rank_sum']}"
        )


if __name__ == "__main__":
    main()
