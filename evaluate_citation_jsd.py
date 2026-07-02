from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from citation_resolver import CitationResolver


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_text_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def load_citation_score_map(path: Path) -> Dict[str, float]:
    raw = load_json(path)
    score_map: Dict[str, float] = {}
    for citation, payload in raw.items():
        if isinstance(payload, dict):
            score = float(payload.get("citation_score", 0.0))
        else:
            score = float(payload)
        score_map[str(citation)] = max(0.0, score)
    return score_map


def citation_target_id(
    paper_id: str,
    citation_key: str,
    resolver: Optional[CitationResolver],
) -> str:
    if resolver is not None:
        target_id = resolver.target_id_for(citation_key, paper_id)
        if target_id:
            return target_id
        entry = resolver.resolve(citation_key, paper_id=paper_id)
        if entry is not None:
            if entry.stable_external_id:
                return entry.stable_external_id
            if entry.canonical_id:
                return entry.canonical_id
            title_key = normalize_text_key(entry.title)
            if title_key:
                return title_key
    return normalize_text_key(citation_key) or citation_key.strip()


def aggregate_scores_by_target(
    paper_id: str,
    score_map: Dict[str, float],
    resolver: Optional[CitationResolver],
) -> Dict[str, dict]:
    grouped: Dict[str, dict] = {}
    for citation_key, score in score_map.items():
        target_id = citation_target_id(paper_id, citation_key, resolver)
        bucket = grouped.setdefault(
            target_id,
            {
                "score": 0.0,
                "display_key": citation_key,
                "best_single_score": float("-inf"),
                "raw_keys": [],
            },
        )
        bucket["score"] = float(bucket["score"]) + score
        bucket["raw_keys"].append(citation_key)
        if score > float(bucket["best_single_score"]):
            bucket["best_single_score"] = score
            bucket["display_key"] = citation_key
    return grouped


def normalize_citation_only_vector(
    assigned_scores: Sequence[float],
    tolerance: float = 1e-9,
) -> Tuple[List[float], float]:
    cleaned = [max(0.0, float(value)) for value in assigned_scores]
    assigned_mass = sum(cleaned)
    if assigned_mass <= tolerance:
        raise ValueError("Cannot normalize an all-zero citation vector.")

    vector = [value / assigned_mass for value in cleaned]
    total = sum(vector)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=max(tolerance, 1e-12)):
        vector[0] += 1.0 - total
    return vector, assigned_mass


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    total = 0.0
    for p_i, q_i in zip(p, q):
        if p_i <= 0.0:
            continue
        if q_i <= 0.0:
            raise ValueError("KL divergence received a zero-probability support mismatch.")
        total += p_i * math.log2(p_i / q_i)
    return total


def jensen_shannon_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    if len(p) != len(q):
        raise ValueError("JSD vectors must have the same length.")
    midpoint = [(p_i + q_i) / 2.0 for p_i, q_i in zip(p, q)]
    return 0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint)


def align_score_vectors(
    left_scores: Dict[str, dict],
    right_scores: Dict[str, dict],
) -> Tuple[List[str], List[str], List[float], List[float]]:
    all_target_ids = sorted(set(left_scores) | set(right_scores))
    labels: List[str] = []
    left_values: List[float] = []
    right_values: List[float] = []

    for target_id in all_target_ids:
        left_bucket = left_scores.get(target_id)
        right_bucket = right_scores.get(target_id)
        label = target_id
        if left_bucket and str(left_bucket["display_key"]).strip():
            label = str(left_bucket["display_key"])
        elif right_bucket and str(right_bucket["display_key"]).strip():
            label = str(right_bucket["display_key"])
        labels.append(label)
        left_values.append(float(left_bucket["score"]) if left_bucket else 0.0)
        right_values.append(float(right_bucket["score"]) if right_bucket else 0.0)

    return all_target_ids, labels, left_values, right_values


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
                "left_score": left_value,
                "right_score": right_value,
                "absolute_difference": abs(left_value - right_value),
            }
        )
    diffs.sort(key=lambda item: (-float(item["absolute_difference"]), str(item["citation"])))
    return diffs[: max(0, limit)]


def prepare_resolver(
    results_root: Path,
    papers_dir: Path,
    paper_ids: Sequence[str],
    left_tag: str,
    right_tag: str,
) -> CitationResolver:
    resolver = CitationResolver()
    resolver.register_corpus_papers(results_root, papers_dir)
    for paper_id in paper_ids:
        left_path = results_root / paper_id / f"{paper_id}_{left_tag}_citation_scores.json"
        right_path = results_root / paper_id / f"{paper_id}_{right_tag}_citation_scores.json"
        keys: List[str] = []
        seen = set()
        for path in (left_path, right_path):
            score_map = load_citation_score_map(path)
            for citation_key in score_map:
                normalized = re.sub(r"\s+", " ", citation_key).strip()
                if normalized in seen:
                    continue
                seen.add(normalized)
                keys.append(citation_key)
        resolver.parse_paper(paper_id, papers_dir / f"{paper_id}.pdf", keys)
    return resolver


def evaluate_paper(
    paper_id: str,
    results_root: Path,
    left_tag: str,
    right_tag: str,
    resolver: Optional[CitationResolver],
) -> dict:
    left_path = results_root / paper_id / f"{paper_id}_{left_tag}_citation_scores.json"
    right_path = results_root / paper_id / f"{paper_id}_{right_tag}_citation_scores.json"

    left_raw = load_citation_score_map(left_path)
    right_raw = load_citation_score_map(right_path)
    left_grouped = aggregate_scores_by_target(paper_id, left_raw, resolver)
    right_grouped = aggregate_scores_by_target(paper_id, right_raw, resolver)
    target_ids, labels, left_values, right_values = align_score_vectors(left_grouped, right_grouped)

    left_vector, left_mass = normalize_citation_only_vector(left_values)
    right_vector, right_mass = normalize_citation_only_vector(right_values)
    jsd = jensen_shannon_divergence(left_vector, right_vector)

    return {
        "paper_id": paper_id,
        "left_tag": left_tag,
        "right_tag": right_tag,
        "left_file": str(left_path),
        "right_file": str(right_path),
        "citation_components": len(target_ids),
        "left_raw_score_sum": left_mass,
        "right_raw_score_sum": right_mass,
        "jsd_base2": jsd,
        "top_absolute_differences": top_absolute_differences(
            target_ids=target_ids,
            labels=labels,
            left_values=left_values,
            right_values=right_values,
            limit=10,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute citation-distribution Jensen-Shannon divergence between two model outputs."
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
        help="Paper ids to evaluate.",
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
        "--output-json",
        default="",
        help="Optional path to save the JSD report as JSON.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    papers_dir = Path(args.papers_dir)
    paper_ids = list(args.paper_ids)

    resolver = prepare_resolver(
        results_root=results_root,
        papers_dir=papers_dir,
        paper_ids=paper_ids,
        left_tag=args.left_tag,
        right_tag=args.right_tag,
    )
    per_paper = [
        evaluate_paper(
            paper_id=paper_id,
            results_root=results_root,
            left_tag=args.left_tag,
            right_tag=args.right_tag,
            resolver=resolver,
        )
        for paper_id in paper_ids
    ]

    report = {
        "left_tag": args.left_tag,
        "right_tag": args.right_tag,
        "papers": per_paper,
        "mean_jsd_base2": sum(item["jsd_base2"] for item in per_paper) / len(per_paper) if per_paper else None,
    }

    for item in per_paper:
        print(
            f"{item['paper_id']}: JSD={item['jsd_base2']:.6f} "
            f"(left raw sum={item['left_raw_score_sum']:.6f}, right raw sum={item['right_raw_score_sum']:.6f})"
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
