"""
Analyse how @k metrics change as k varies from 1 to 8,
using the ChatGPT top-8 annotations as ground truth.

For every k we ask: among the model's top-k citations and ChatGPT's
top-k annotations, how much do they agree?

  overlap@k   = |model_top_k ∩ chatgpt_top_k|
  recall@k    = overlap@k / k   (= precision@k here since both sets have size k)
  hit@k       = 1 if overlap > 0 else 0
  mrr@k       = reciprocal rank of ChatGPT's rank-1 paper in model's top-k list

One combined figure (4 panels, one per metric) with one line per model.
Plus a summary TSV.

Usage:
    python3 plot_top_k_sensitivity.py
    python3 plot_top_k_sensitivity.py \\
        --annotations chatgpt_baseline_annotations.json \\
        --output-dir results/plots
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
from evaluate_human_section_scores import (
    get_top_k_model_citations,
    load_json,
    reciprocal_rank,
    resolve_human_top_k_citations,
)

# ── The promptv2 (5-sample) results are used as the primary model scores ──
MODEL_TAGS: Dict[str, str] = {
    "llama3.2:1b":  "llama3_2_1b_promptv2",
    "llama3.2:3b":  "llama3_2_promptv2",
    "gemma2:2b":    "gemma2_2b_promptv2",
    "gemma3:4b":    "gemma3_4b_promptv2",
    "qwen3:1.7b":   "qwen3_1_7_promptv2",
    "phi3:medium":  "phi3_medium_promptv2",
    "qwen2.5:3b":   "qwen2_5_3b_promptv2",
}

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

MAX_K = 8

METRICS = [
    ("overlap_at_k",                   "Overlap@k"),
    ("recall_at_k",                    "Recall@k"),
    ("hit_at_k",                       "Hit@k"),
    ("mrr",                            "MRR@k"),
]


def mean(vals: List[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


def sem(vals: List[float]) -> Optional[float]:
    if len(vals) < 2:
        return None
    mu = sum(vals) / len(vals)
    s = math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))
    return s / math.sqrt(len(vals))


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
        "mrr":          reciprocal_rank(
            resolved_chatgpt[0] if resolved_chatgpt else None,
            model_top_k_keys,
        ),
    }


def analyze_model(
    model_name: str,
    tag: str,
    annotations: dict,
    results_root: Path,
    resolver: CitationResolver,
) -> Dict[int, Dict[str, float]]:
    """Returns {k: {mean_metric: value, sem_metric: value}}"""
    results: Dict[int, Dict[str, float]] = {}

    for k in range(1, MAX_K + 1):
        rows: List[dict] = []
        for paper_id, payload in annotations["papers"].items():
            top8 = payload.get("important_papers_top_8", [])
            if not top8:
                continue
            cs_path = results_root / paper_id / f"{paper_id}_{tag}_citation_scores.json"
            if not cs_path.exists():
                continue
            citation_json = load_json(cs_path)
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
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), constrained_layout=False)

    for ax, (mkey, mlabel) in zip(axes, METRICS):
        for model, k_data in series.items():
            color  = MODEL_COLORS.get(model, "#333333")
            marker = MODEL_MARKERS.get(model, "o")
            ys   = [k_data[k][f"mean_{mkey}"] for k in ks]
            sems = [k_data[k][f"sem_{mkey}"]  for k in ks]
            moes = [1.96 * s for s in sems]

            ax.plot(ks, ys, color=color, marker=marker, linewidth=2,
                    markersize=6, markeredgecolor="white", markeredgewidth=0.6,
                    label=model, zorder=3)
            ax.fill_between(ks,
                            [y - e for y, e in zip(ys, moes)],
                            [y + e for y, e in zip(ys, moes)],
                            color=color, alpha=0.12, zorder=1)

        ax.set_title(mlabel, fontsize=15, fontweight="semibold", pad=6)
        ax.set_xlabel("k  (top-k model citations vs top-k ChatGPT)", fontsize=13)
        ax.set_ylabel(mlabel, fontsize=13.5)
        ax.set_xticks(ks)
        ax.grid(True, axis="y", color="#D7DBE0", lw=0.8, alpha=0.9)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=12.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False,
               fontsize=13, bbox_to_anchor=(0.5, 1.04), columnspacing=1.2)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.14, top=0.88, wspace=0.32)

    for ext in ("png", "pdf"):
        p = out_dir / f"top_k_sensitivity_combined.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  saved {p}")
    plt.close(fig)


def plot_per_metric(series: Dict[str, Dict[int, Dict[str, float]]], out_dir: Path) -> None:
    ks = list(range(1, MAX_K + 1))
    for mkey, mlabel in METRICS:
        fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=False)
        for model, k_data in series.items():
            color  = MODEL_COLORS.get(model, "#333333")
            marker = MODEL_MARKERS.get(model, "o")
            ys   = [k_data[k][f"mean_{mkey}"] for k in ks]
            sems = [k_data[k][f"sem_{mkey}"]  for k in ks]
            moes = [1.96 * s for s in sems]
            ax.plot(ks, ys, color=color, marker=marker, linewidth=2,
                    markersize=6, markeredgecolor="white", markeredgewidth=0.6,
                    label=model, zorder=3)
            ax.fill_between(ks,
                            [y - e for y, e in zip(ys, moes)],
                            [y + e for y, e in zip(ys, moes)],
                            color=color, alpha=0.13, zorder=1)

        ax.set_title(mlabel, fontsize=16, fontweight="semibold")
        ax.set_xlabel("k", fontsize=14)
        ax.set_ylabel(mlabel, fontsize=14)
        ax.set_xticks(ks)
        ax.legend(frameon=False, fontsize=12.5, loc="best")
        ax.grid(True, axis="y", color="#D7DBE0", lw=0.8, alpha=0.9)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=13)
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
    parser.add_argument("--results-root", default="paper_results")
    parser.add_argument("--papers-dir",   default="papers")
    parser.add_argument("--output-dir",   default="results/plots")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    papers_dir   = Path(args.papers_dir)
    annotations  = load_json(Path(args.annotations))

    resolver = CitationResolver()
    resolver.parse_all(results_root, papers_dir)

    series: Dict[str, Dict[int, Dict[str, float]]] = {}
    for model_name, tag in MODEL_TAGS.items():
        print(f"  {model_name} ({tag}) …")
        series[model_name] = analyze_model(
            model_name, tag, annotations, results_root, resolver
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
    })
    main()
