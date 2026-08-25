"""
Analyse how top-k citation metrics change as k varies,
either against ranked human/chatgpt annotations or against another model's
full citation-score output.

In reference-tag mode, we compare:
  model_top_k vs reference_top_k

Metrics:
  overlap@k = |model_top_k ∩ reference_top_k|
  recall@k  = overlap@k / |reference_top_k|
  hit@k     = 1 if overlap > 0 else 0
  nDCG@k    = graded ranking agreement against the reference top-k ordering

Usage:
    python3 plot_top_k_sensitivity.py \\
        --reference-tag anthropic_full_paper_claude_sonnet_4_6_direct \\
        --output-dir results/final_sonnet46_model_comparison
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from citation_resolver import CitationResolver
from evaluate_citation_jsd import aggregate_scores_by_target, load_citation_score_map
from evaluate_human_section_scores import (
    get_top_k_model_citations,
    load_json,
    ndcg_at_k,
    resolve_human_top_k_citations,
)

# ── The promptv2 (5-sample) results are used as the primary model scores ──
MODEL_TAGS: Dict[str, Sequence[str]] = {
    "llama3.2:1b":  ("llama3_2_1b_promptv2", "llama3_2_1b_promptv2_retry8"),
    "llama3.2:3b":  ("llama3_2_3b_promptv2", "llama3_2_3b_promptv2_retry8", "llama3_2_promptv2"),
    "gemma2:2b":    ("gemma2_2b_promptv2", "gemma2_2b_promptv2_retry8"),
    "gemma3:4b":    ("gemma3_4b_promptv2", "gemma3_4b_promptv2_retry8"),
    "qwen3:1.7b":   ("qwen3_1_7_promptv2", "qwen3_1_7b_promptv2_retry8"),
    "qwen3:4b":     ("qwen3_4b_promptv2", "qwen3_4b_promptv2_retry8"),
    "phi3:medium":  ("phi3_medium_promptv2", "phi3_med_promptv2", "phi3_medium_promptv2_retry8"),
    "qwen2.5:3b":   ("qwen2_5_3b_promptv2", "qwen2_5_3b_promptv2_retry8"),
}

MODEL_COLORS = {
    "llama3.2:1b":  "#4e79a7",
    "llama3.2:3b":  "#f28e2b",
    "gemma2:2b":    "#e15759",
    "gemma3:4b":    "#76b7b2",
    "qwen3:1.7b":   "#59a14f",
    "qwen3:4b":     "#9c755f",
    "phi3:medium":  "#edc948",
    "qwen2.5:3b":   "#b07aa1",
}

MODEL_MARKERS = {
    "llama3.2:1b":  "o",
    "llama3.2:3b":  "s",
    "gemma2:2b":    "^",
    "gemma3:4b":    "D",
    "qwen3:1.7b":   "P",
    "qwen3:4b":     "h",
    "phi3:medium":  "X",
    "qwen2.5:3b":   "v",
}

MAX_K = 8

METRICS = [
    ("overlap_at_k",                   "Overlap@k"),
    ("recall_at_k",                    "Recall@k"),
    ("hit_at_k",                       "Hit@k"),
    ("ndcg_at_k",                      "nDCG@k"),
]


def mean(vals: List[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


def sem(vals: List[float]) -> Optional[float]:
    if len(vals) < 2:
        return None
    mu = sum(vals) / len(vals)
    s = math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))
    return s / math.sqrt(len(vals))


def resolve_existing_tag(results_root: Path, paper_id: str, tags: Sequence[str]) -> Optional[str]:
    for tag in tags:
        path = results_root / paper_id / f"{paper_id}_{tag}_citation_scores.json"
        if path.exists():
            return tag
    return None


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


def compute_metrics_at_k(
    paper_id: str,
    chatgpt_top8: List[dict],
    citation_json: dict,
    resolver: CitationResolver,
    k: int,
) -> dict:
    # Model's top-k by aggregated citation score
    ranked_full = get_top_k_model_citations(
        citation_json, max(k, len(citation_json)), paper_id=paper_id, resolver=resolver
    )
    model_top_k_keys = [key for key, _ in ranked_full[:k]]

    # ChatGPT's top-k (by rank field)
    chatgpt_top_k = sorted(chatgpt_top8, key=lambda x: x.get("rank", 10**9))[:k]
    resolved_chatgpt = resolve_human_top_k_citations(
        paper_id=paper_id,
        human_items=chatgpt_top_k,
        model_ranked_full=ranked_full,
        resolver=resolver,
    )
    resolved_set = {key for key in resolved_chatgpt if key is not None}

    overlap = len(resolved_set & set(model_top_k_keys))
    denom = max(1, len(chatgpt_top_k))

    return {
        "overlap_at_k": overlap,
        "recall_at_k":  overlap / denom,
        "hit_at_k":     1.0 if overlap > 0 else 0.0,
        "ndcg_at_k":    ndcg_at_k(
            {
                key: float(len(chatgpt_top_k) - idx)
                for idx, key in enumerate(resolved_chatgpt)
                if key is not None
            },
            model_top_k_keys,
            k,
        ),
    }


def compute_metrics_at_k_against_reference_tag(
    paper_id: str,
    reference_tag: str,
    candidate_json: dict,
    results_root: Path,
    resolver: CitationResolver,
    k: int,
) -> dict:
    reference_path = results_root / paper_id / f"{paper_id}_{reference_tag}_citation_scores.json"
    reference_scores = load_citation_score_map(reference_path)
    candidate_scores = load_citation_score_map(candidate_json)

    reference_grouped = aggregate_scores_by_target(paper_id, reference_scores, resolver)
    candidate_grouped = aggregate_scores_by_target(paper_id, candidate_scores, resolver)

    reference_ranked = rank_grouped_citations(reference_grouped)
    candidate_ranked = rank_grouped_citations(candidate_grouped)

    reference_top_k = reference_ranked[: max(0, k)]
    candidate_top_k = candidate_ranked[: max(0, k)]
    reference_top_keys = [item["target_id"] for item in reference_top_k]
    candidate_top_keys = [item["target_id"] for item in candidate_top_k]

    overlap = len(set(reference_top_keys).intersection(candidate_top_keys))
    denom = max(1, len(reference_top_k))

    return {
        "overlap_at_k": overlap,
        "recall_at_k": overlap / denom,
        "hit_at_k": 1.0 if overlap > 0 else 0.0,
        "ndcg_at_k": ndcg_at_k(
            {
                key: float(len(reference_top_keys) - idx)
                for idx, key in enumerate(reference_top_keys)
            },
            candidate_top_keys,
            k,
        ),
    }


def analyze_model(
    model_name: str,
    tags: Sequence[str],
    annotations: Optional[dict],
    results_root: Path,
    resolver: CitationResolver,
    reference_tag: Optional[str],
) -> Dict[int, Dict[str, float]]:
    """Returns {k: {mean_metric: value, sem_metric: value}}"""
    results: Dict[int, Dict[str, float]] = {}

    for k in range(1, MAX_K + 1):
        rows: List[dict] = []
        paper_ids = sorted((annotations["papers"].keys() if annotations is not None else [p.name for p in results_root.iterdir() if p.is_dir()]))
        for paper_id in paper_ids:
            resolved_tag = resolve_existing_tag(results_root, paper_id, list(tags))
            if resolved_tag is None:
                continue
            cs_path = results_root / paper_id / f"{paper_id}_{resolved_tag}_citation_scores.json"
            if not cs_path.exists():
                continue
            citation_json = load_json(cs_path)

            if reference_tag is not None:
                ref_path = results_root / paper_id / f"{paper_id}_{reference_tag}_citation_scores.json"
                if not ref_path.exists():
                    continue
                m = compute_metrics_at_k_against_reference_tag(
                    paper_id=paper_id,
                    reference_tag=reference_tag,
                    candidate_json=cs_path,
                    results_root=results_root,
                    resolver=resolver,
                    k=k,
                )
            else:
                assert annotations is not None
                payload = annotations["papers"][paper_id]
                top8 = payload.get("important_papers_top_8", [])
                if not top8:
                    continue
                m = compute_metrics_at_k(paper_id, top8, citation_json, resolver, k)
            rows.append(m)

        n = len(rows)
        entry: Dict[str, float] = {"n_papers": n}
        for mkey, _ in METRICS:
            vals = [r[mkey] for r in rows]
            entry[f"mean_{mkey}"] = mean(vals) or 0.0
            entry[f"sem_{mkey}"]  = sem(vals)  or 0.0
        results[k] = entry

    return results


def plot_combined(series: Dict[str, Dict[int, Dict[str, float]]], out_dir: Path) -> None:
    ks = list(range(1, MAX_K + 1))
    fig, axes = plt.subplots(1, 4, figsize=(18.2, 5.4), constrained_layout=False)

    for ax, (mkey, mlabel) in zip(axes, METRICS):
        for model, k_data in series.items():
            color  = MODEL_COLORS.get(model, "#333333")
            marker = MODEL_MARKERS.get(model, "o")
            ys   = [k_data[k][f"mean_{mkey}"] for k in ks]
            sems = [k_data[k][f"sem_{mkey}"]  for k in ks]
            moes = [1.96 * s for s in sems]

            ax.plot(ks, ys, color=color, marker=marker, linewidth=2.8,
                    markersize=7.5, markeredgecolor="white", markeredgewidth=0.7,
                    label=model, zorder=3)
            ax.fill_between(ks,
                            [y - e for y, e in zip(ys, moes)],
                            [y + e for y, e in zip(ys, moes)],
                            color=color, alpha=0.14, zorder=1)

        ax.set_title(mlabel, fontsize=17, fontweight="bold", pad=8)
        ax.set_ylabel(mlabel, fontsize=15, fontweight="bold")
        ax.set_xticks(ks)
        ax.grid(True, axis="y", color="#D7DBE0", lw=0.8, alpha=0.9)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=13.5)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper center", ncol=8, frameon=False,
               fontsize=14.5, bbox_to_anchor=(0.5, 1.06), columnspacing=1.15)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.supxlabel("k  (top-k model citations vs top-k Sonnet-4.6)", fontsize=15, fontweight="bold", y=0.04)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.20, top=0.85, wspace=0.28)

    for ext in ("png", "pdf"):
        p = out_dir / f"top_k_sensitivity_combined.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  saved {p}")
    plt.close(fig)


def plot_per_metric(series: Dict[str, Dict[int, Dict[str, float]]], out_dir: Path) -> None:
    ks = list(range(1, MAX_K + 1))
    for mkey, mlabel in METRICS:
        fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=False)
        for model, k_data in series.items():
            color  = MODEL_COLORS.get(model, "#333333")
            marker = MODEL_MARKERS.get(model, "o")
            ys   = [k_data[k][f"mean_{mkey}"] for k in ks]
            sems = [k_data[k][f"sem_{mkey}"]  for k in ks]
            moes = [1.96 * s for s in sems]
            ax.plot(ks, ys, color=color, marker=marker, linewidth=2.8,
                    markersize=7.5, markeredgecolor="white", markeredgewidth=0.7,
                    label=model, zorder=3)
            ax.fill_between(ks,
                            [y - e for y, e in zip(ys, moes)],
                            [y + e for y, e in zip(ys, moes)],
                            color=color, alpha=0.14, zorder=1)

        ax.set_title(mlabel, fontsize=18, fontweight="bold")
        ax.set_xlabel("k", fontsize=15, fontweight="bold")
        ax.set_ylabel(mlabel, fontsize=15, fontweight="bold")
        ax.set_xticks(ks)
        legend = ax.legend(frameon=False, fontsize=13.5, loc="best")
        for text in legend.get_texts():
            text.set_fontweight("bold")
        ax.grid(True, axis="y", color="#D7DBE0", lw=0.8, alpha=0.9)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=13.5)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.14, top=0.90)

        safe = mlabel.lower().replace("@", "at").replace(" ", "_")
        for ext in ("png", "pdf"):
            p = out_dir / f"top_k_sensitivity_{safe}.{ext}"
            fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
            print(f"  saved {p}")
        plt.close(fig)


def save_tsv(series: Dict[str, Dict[int, Dict[str, float]]], out_dir: Path) -> None:
    lines = ["model\tk\tmetric\tmean\tsem"]
    for model, k_data in series.items():
        for k in sorted(k_data):
            for mkey, mlabel in METRICS:
                mu  = k_data[k][f"mean_{mkey}"]
                se  = k_data[k][f"sem_{mkey}"]
                lines.append(f"{model}\t{k}\t{mlabel}\t{mu:.4f}\t{se:.4f}")
    p = out_dir / "top_k_sensitivity.tsv"
    p.write_text("\n".join(lines) + "\n")
    print(f"  saved {p}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations",  default="chatgpt_baseline_annotations.json")
    parser.add_argument("--reference-tag", default="")
    parser.add_argument("--results-root", default="paper_results")
    parser.add_argument("--papers-dir",   default="papers")
    parser.add_argument("--output-dir",   default="results/plots")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    papers_dir   = Path(args.papers_dir)
    annotations  = load_json(Path(args.annotations)) if not args.reference_tag else None

    resolver = CitationResolver()
    resolver.parse_all(results_root, papers_dir)

    series: Dict[str, Dict[int, Dict[str, float]]] = {}
    for model_name, tags in MODEL_TAGS.items():
        print(f"  {model_name} ({', '.join(tags)}) …")
        series[model_name] = analyze_model(
            model_name=model_name,
            tags=tags,
            annotations=annotations,
            results_root=results_root,
            resolver=resolver,
            reference_tag=args.reference_tag or None,
        )
        n = series[model_name][1]["n_papers"]
        print(f"    → {n} papers evaluated")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\nPlotting …")
    plot_combined(series, out)
    plot_per_metric(series, out)
    save_tsv(series, out)
    print("Done.")


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        'font.family':      'STIXGeneral',
        'mathtext.fontset': 'stix',
        'font.size':        14,
        'font.weight':      'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
    })
    main()
