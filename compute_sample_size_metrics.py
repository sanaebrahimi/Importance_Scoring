"""
Compute sample-size sensitivity metrics directly from the _s1_ / _s3_ / _promptv2_
result files (no debug-log replay needed).

Sample sizes: s1 → 1 sample, s3 → 3 samples, promptv2 → 5 samples.

Output JSON is compatible with plot_sample_prefix_metrics.py.

Usage:
    python3 compute_sample_size_metrics.py
    python3 compute_sample_size_metrics.py --output-json results/sample_prefix_citation_metrics_all_models.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from citation_resolver import CitationResolver
from evaluate_human_section_scores import (
    get_top_k_model_citations,
    load_json,
    reciprocal_rank,
    resolve_human_top_k_citations,
)

# ── Model → {sample_size: file_tag} mapping ───────────────────────────────
MODEL_TAGS: Dict[str, Dict[int, str]] = {
    "llama3.2:1b":  {1: "llama3_2_1b_s1",    3: "llama3_2_1b_s3",    5: "llama3_2_1b_promptv2"},
    "llama3.2:3b":  {1: "llama3_2_s1",        3: "llama3_2_s3",        5: "llama3_2_promptv2"},
    "gemma2:2b":    {1: "gemma2_2b_s1",        3: "gemma2_2b_s3",       5: "gemma2_2b_promptv2"},
    "gemma3:4b":    {1: "gemma3_4b_s1",        3: "gemma3_4b_s3",       5: "gemma3_4b_promptv2"},
    "qwen3:1.7b":   {1: "qwen3_1_7_s1",        3: "qwen3_1_7_s3",       5: "qwen3_1_7_promptv2"},
    "phi3:medium":  {1: "phi3_medium_s1",       3: "phi3_medium_s3",     5: "phi3_medium_promptv2"},
    "qwen2.5:3b":   {1: "qwen2_5_3b_s1",        3: "qwen2_5_3b_s3",      5: "qwen2_5_3b_promptv2"},
}

TOP_K = 4


def citation_report(
    paper_id: str,
    important_papers: List[dict],
    citation_json: dict,
    resolver: CitationResolver,
    k: int,
) -> dict:
    ranked_full = get_top_k_model_citations(
        citation_json, max(k, len(citation_json)), paper_id=paper_id, resolver=resolver
    )
    top_k_keys = [key for key, _ in ranked_full[:k]]
    human_ranked = sorted(important_papers, key=lambda x: x.get("rank", 10**9))[:k]
    resolved_human = resolve_human_top_k_citations(
        paper_id=paper_id,
        human_items=human_ranked,
        model_ranked_full=ranked_full,
        resolver=resolver,
    )
    resolved_set = {key for key in resolved_human if key is not None}
    overlap = len(resolved_set & set(top_k_keys))
    denom = max(1, len(human_ranked))
    return {
        "overlap_at_k": overlap,
        "recall_at_k": overlap / denom,
        "hit_at_k": 1.0 if overlap > 0 else 0.0,
        "mrr_human_rank1_in_model_top_k": reciprocal_rank(
            resolved_human[0] if resolved_human else None, top_k_keys
        ),
    }


def mean_metric(rows: List[dict], field: str) -> Optional[float]:
    if not rows:
        return None
    return sum(float(r[field]) for r in rows) / len(rows)


def std_metric(rows: List[dict], field: str) -> Optional[float]:
    if len(rows) < 2:
        return None
    import math
    vals = [float(r[field]) for r in rows]
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))


def sem_metric(rows: List[dict], field: str) -> Optional[float]:
    import math
    s = std_metric(rows, field)
    if s is None or len(rows) < 2:
        return None
    return s / math.sqrt(len(rows))


def analyze_model(
    model_name: str,
    annotations: dict,
    results_root: Path,
    resolver: CitationResolver,
    k: int,
) -> dict:
    size_to_tag = MODEL_TAGS[model_name]
    aggregate: Dict[int, dict] = {}

    for sample_size, tag in sorted(size_to_tag.items()):
        rows: List[dict] = []
        skipped = []

        for paper_id, payload in annotations["papers"].items():
            if not payload.get("important_papers"):
                continue
            cs_path = results_root / paper_id / f"{paper_id}_{tag}_citation_scores.json"
            if not cs_path.exists():
                skipped.append(paper_id)
                continue
            citation_json = load_json(cs_path)
            metrics = citation_report(
                paper_id, payload["important_papers"], citation_json, resolver, k
            )
            metrics["paper_id"] = paper_id
            rows.append(metrics)

        if skipped:
            print(f"  [{model_name}] n={sample_size}: skipped {len(skipped)} papers (missing files)")

        n = len(rows)
        aggregate[sample_size] = {
            "papers_evaluated": n,
            "per_paper": {r["paper_id"]: {k: v for k, v in r.items() if k != "paper_id"} for r in rows},
            "w": {
                "mean_overlap_at_k":                    mean_metric(rows, "overlap_at_k"),
                "mean_recall_at_k":                     mean_metric(rows, "recall_at_k"),
                "mean_hit_at_k":                        mean_metric(rows, "hit_at_k"),
                "mean_mrr_human_rank1_in_model_top_k":  mean_metric(rows, "mrr_human_rank1_in_model_top_k"),
                "std_overlap_at_k":                     std_metric(rows, "overlap_at_k"),
                "std_recall_at_k":                      std_metric(rows, "recall_at_k"),
                "std_hit_at_k":                         std_metric(rows, "hit_at_k"),
                "std_mrr_human_rank1_in_model_top_k":   std_metric(rows, "mrr_human_rank1_in_model_top_k"),
                "sem_overlap_at_k":                     sem_metric(rows, "overlap_at_k"),
                "sem_recall_at_k":                      sem_metric(rows, "recall_at_k"),
                "sem_hit_at_k":                         sem_metric(rows, "hit_at_k"),
                "sem_mrr_human_rank1_in_model_top_k":   sem_metric(rows, "mrr_human_rank1_in_model_top_k"),
            },
        }
        w = aggregate[sample_size]["w"]
        print(
            f"  [{model_name}] n={sample_size} ({tag})  papers={n}"
            f"  overlap={w['mean_overlap_at_k']:.3f}±{w['sem_overlap_at_k']:.3f}"
            f"  recall={w['mean_recall_at_k']:.3f}±{w['sem_recall_at_k']:.3f}"
            f"  mrr={w['mean_mrr_human_rank1_in_model_top_k']:.3f}±{w['sem_mrr_human_rank1_in_model_top_k']:.3f}"
        )

    return {"aggregate_by_sample_size": {str(k): v for k, v in aggregate.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-file", default="human_expert_annotations.json")
    parser.add_argument("--results-root",     default="paper_results")
    parser.add_argument("--papers-dir",        default="papers")
    parser.add_argument(
        "--output-json",
        default="results/sample_prefix_citation_metrics_all_models.json",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    papers_dir   = Path(args.papers_dir)
    annotations  = load_json(Path(args.annotations_file))

    resolver = CitationResolver()
    resolver.parse_all(results_root, papers_dir)

    summary = {
        "analyze_by": "samples",
        "citation_top_k": args.top_k,
        "models": {},
    }

    for model_name in MODEL_TAGS:
        print(f"\n=== {model_name} ===")
        summary["models"][model_name] = analyze_model(
            model_name, annotations, results_root, resolver, args.top_k
        )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
