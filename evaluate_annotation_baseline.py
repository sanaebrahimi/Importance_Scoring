import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from evaluate_human_section_scores import (
    aggregate_citation_report,
    aggregate_metric_report,
    align_section_score_maps,
    citation_match_score,
    human_citation_aliases,
    load_json,
    metric_bundle,
    normalize_scores,
    reciprocal_rank,
    uniform_vector,
)


def resolve_reference_top_k_citations(
    reference_items: List[dict],
    candidate_items: List[dict],
    k: int,
) -> List[Optional[str]]:
    ranked_reference = sorted(reference_items, key=lambda item: item.get("rank", 10**9))[:k]
    ranked_candidates = sorted(candidate_items, key=lambda item: item.get("rank", 10**9))[:k]

    candidate_keys = [str(item.get("citation", "")).strip() for item in ranked_candidates]
    candidate_aliases = {
        key: human_citation_aliases(key)
        for key in candidate_keys
        if key
    }
    used_candidate_keys = set()
    resolved: List[Optional[str]] = []

    for item in ranked_reference:
        citation_text = str(item.get("citation", "")).strip()
        reference_aliases = human_citation_aliases(citation_text)
        best_key: Optional[str] = None
        best_score = 0

        for candidate_key in candidate_keys:
            if not candidate_key or candidate_key in used_candidate_keys:
                continue
            score = citation_match_score(reference_aliases, candidate_aliases[candidate_key])
            if score > best_score:
                best_score = score
                best_key = candidate_key
                if score == 3:
                    break

        if best_key is not None and best_score > 0:
            used_candidate_keys.add(best_key)
        else:
            best_key = None
        resolved.append(best_key)

    return resolved


def annotation_top_k_report(
    paper_id: str,
    reference_items: List[dict],
    candidate_items: List[dict],
    k: int,
) -> dict:
    ranked_candidates = sorted(candidate_items, key=lambda item: item.get("rank", 10**9))[:k]
    candidate_top_k_keys = [str(item.get("citation", "")).strip() for item in ranked_candidates]
    ranked_reference = sorted(reference_items, key=lambda item: item.get("rank", 10**9))[:k]
    resolved_reference_keys = resolve_reference_top_k_citations(reference_items, candidate_items, k)

    resolved_reference_set = {key for key in resolved_reference_keys if key is not None}
    candidate_top_k_set = set(candidate_top_k_keys)
    overlap_count = len(resolved_reference_set.intersection(candidate_top_k_set))
    denom = max(1, len(ranked_reference))
    precision_at_k = overlap_count / max(1, k)
    recall_at_k = overlap_count / denom
    hit_at_k = 1.0 if overlap_count > 0 else 0.0
    mrr_rank1 = reciprocal_rank(
        resolved_reference_keys[0] if resolved_reference_keys else None,
        candidate_top_k_keys,
    )

    return {
        "paper_id": paper_id,
        "k": k,
        "human_top_k": ranked_reference,
        "human_top_k_resolved_baseline_keys": resolved_reference_keys,
        "baseline_top_k": ranked_candidates,
        "metrics": {
            "overlap_at_k": overlap_count,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "hit_at_k": hit_at_k,
            "mrr_human_rank1_in_model_top_k": mrr_rank1,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one annotation set directly against another (for example, ChatGPT baseline vs human annotations)."
    )
    parser.add_argument(
        "--reference-annotations",
        default="human_expert_annotations.json",
        help="Reference annotation file, typically the human annotations.",
    )
    parser.add_argument(
        "--candidate-annotations",
        default="chatgpt_baseline_annotations.json",
        help="Candidate annotation file to compare against the reference annotations.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Number of bootstrap resamples over papers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--citation-top-k",
        type=int,
        default=4,
        help="Top-k citations to compare between the two annotation files.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save the full comparison report as JSON.",
    )
    args = parser.parse_args()

    reference_annotations = load_json(Path(args.reference_annotations))
    candidate_annotations = load_json(Path(args.candidate_annotations))

    reference_papers: Dict[str, dict] = reference_annotations["papers"]
    candidate_papers: Dict[str, dict] = candidate_annotations["papers"]

    per_paper_reports = []
    per_paper_citation_reports = []
    skipped = []

    for paper_id, reference_payload in reference_papers.items():
        candidate_payload = candidate_papers.get(paper_id)
        if candidate_payload is None:
            skipped.append({"paper_id": paper_id, "reason": "missing_candidate_annotation"})
            continue

        reference_scores = reference_payload["top_level_scores_for_evaluation"]
        candidate_scores = candidate_payload["top_level_scores_for_evaluation"]

        overlap, reference_overlap_raw, candidate_overlap_raw, candidate_extras = align_section_score_maps(
            reference_scores,
            candidate_scores,
        )
        if len(overlap) < 2:
            skipped.append(
                {
                    "paper_id": paper_id,
                    "reason": "insufficient_overlap",
                    "reference_sections": sorted(reference_scores.keys()),
                    "candidate_sections": sorted(candidate_scores.keys()),
                }
            )
            continue

        reference_overlap = normalize_scores(reference_overlap_raw)
        candidate_overlap = normalize_scores(candidate_overlap_raw)

        ordered_reference = [reference_overlap[section] for section in overlap]
        ordered_candidate = [candidate_overlap[section] for section in overlap]
        ordered_uniform = uniform_vector(len(overlap))

        report = {
            "paper_id": paper_id,
            "aligned_sections": overlap,
            "reference_normalized": reference_overlap,
            "candidate_normalized": candidate_overlap,
            "candidate_extra_top_level_sections": candidate_extras,
            "metrics": {
                "model": metric_bundle(ordered_reference, ordered_candidate),
                "uniform": metric_bundle(ordered_reference, ordered_uniform),
            },
        }
        per_paper_reports.append(report)

        if reference_payload.get("important_papers") and candidate_payload.get("important_papers"):
            per_paper_citation_reports.append(
                annotation_top_k_report(
                    paper_id=paper_id,
                    reference_items=reference_payload["important_papers"],
                    candidate_items=candidate_payload["important_papers"],
                    k=max(1, args.citation_top_k),
                )
            )

    aggregate = aggregate_metric_report(
        [paper["metrics"] for paper in per_paper_reports],
        n_bootstrap=max(100, args.bootstrap_samples),
        seed=args.seed,
    )
    citation_aggregate = None
    if per_paper_citation_reports:
        citation_aggregate = aggregate_citation_report(
            per_paper_citation_reports,
            n_bootstrap=max(100, args.bootstrap_samples),
            seed=args.seed + 100,
        )

    summary = {
        "reference_annotations": args.reference_annotations,
        "candidate_annotations": args.candidate_annotations,
        "papers_evaluated": len(per_paper_reports),
        "paper_ids": [paper["paper_id"] for paper in per_paper_reports],
        "aggregate": aggregate,
        "per_paper": per_paper_reports,
        "citation_top_k": {
            "enabled": True,
            "k": args.citation_top_k,
            "papers_evaluated": len(per_paper_citation_reports),
            "paper_ids": [paper["paper_id"] for paper in per_paper_citation_reports],
            "aggregate": citation_aggregate,
            "per_paper": per_paper_citation_reports,
        },
        "skipped": skipped,
        "notes": [
            "Reference and candidate section scores are aligned on the overlapping top-level sections only.",
            "Both vectors are renormalized over that overlap before metric computation.",
            "Candidate citations are compared directly against the human top-k annotation using the same alias matching logic used in model evaluation.",
            "Uniform baseline rank correlations are typically undefined because the baseline is a constant vector; they are reported as null when appropriate.",
        ],
    }

    print(f"Evaluated papers: {summary['papers_evaluated']}")
    print(f"Paper IDs: {', '.join(summary['paper_ids'])}")
    print()
    print("Candidate annotation metrics")
    print(f"  Mean Spearman: {aggregate['model']['mean_spearman']}")
    print(f"  Mean Kendall tau-b: {aggregate['model']['mean_kendall_tau_b']}")
    print(f"  Mean L1: {aggregate['model']['mean_l1']}")
    print(f"  Bootstrap CI Spearman: {aggregate['model']['bootstrap_ci_spearman']}")
    print(f"  Bootstrap CI Kendall tau-b: {aggregate['model']['bootstrap_ci_kendall_tau_b']}")
    print(f"  Bootstrap CI L1: {aggregate['model']['bootstrap_ci_l1']}")
    print()
    print("Uniform baseline")
    print(f"  Mean Spearman: {aggregate['uniform']['mean_spearman']}")
    print(f"  Mean Kendall tau-b: {aggregate['uniform']['mean_kendall_tau_b']}")
    print(f"  Mean L1: {aggregate['uniform']['mean_l1']}")
    print(f"  Bootstrap CI L1: {aggregate['uniform']['bootstrap_ci_l1']}")

    if citation_aggregate is not None:
        print()
        print(f"Citation Top-{args.citation_top_k} metrics")
        print(f"  Papers evaluated: {len(per_paper_citation_reports)}")
        print(f"  Mean Overlap@{args.citation_top_k}: {citation_aggregate['mean_overlap_at_k']}")
        print(f"  Mean Precision@{args.citation_top_k}: {citation_aggregate['mean_precision_at_k']}")
        print(f"  Mean Recall@{args.citation_top_k}: {citation_aggregate['mean_recall_at_k']}")
        print(f"  Mean Hit@{args.citation_top_k}: {citation_aggregate['mean_hit_at_k']}")
        print(
            "  Mean MRR (human rank-1 in candidate top-k): "
            f"{citation_aggregate['mean_mrr_human_rank1_in_model_top_k']}"
        )

    if skipped:
        print()
        print("Skipped papers")
        for item in skipped:
            print(f"  {item['paper_id']}: {item['reason']}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
