import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from citation_resolver import CitationResolver


DIVERGENCE_EPSILON = 1e-12


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_scores(score_map: Dict[str, float]) -> Dict[str, float]:
    cleaned = {key: max(0.0, float(value)) for key, value in score_map.items()}
    total = sum(cleaned.values())
    if total <= 0.0:
        raise ValueError("Cannot normalize an all-zero score map.")
    normalized = {key: value / total for key, value in cleaned.items()}
    residual = 1.0 - sum(normalized.values())
    best_key = max(normalized, key=normalized.get)
    normalized[best_key] += residual
    return normalized


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]
    denom_x = math.sqrt(sum(x * x for x in centered_x))
    denom_y = math.sqrt(sum(y * y for y in centered_y))
    if denom_x <= 0.0 or denom_y <= 0.0:
        return None
    numer = sum(x * y for x, y in zip(centered_x, centered_y))
    return numer / (denom_x * denom_y)


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    return pearson_corr(average_ranks(xs), average_ranks(ys))


def kendall_tau_b(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n != len(ys) or n < 2:
        return None

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0

    for i in range(n):
        for j in range(i + 1, n):
            dx = 0
            if xs[i] < xs[j]:
                dx = -1
            elif xs[i] > xs[j]:
                dx = 1

            dy = 0
            if ys[i] < ys[j]:
                dy = -1
            elif ys[i] > ys[j]:
                dy = 1

            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
                continue
            if dy == 0:
                ties_y += 1
                continue
            if dx == dy:
                concordant += 1
            else:
                discordant += 1

    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom <= 0.0:
        return None
    return (concordant - discordant) / denom


def l1_distance(xs: Sequence[float], ys: Sequence[float]) -> float:
    return sum(abs(x - y) for x, y in zip(xs, ys))


def smooth_distribution(values: Sequence[float], epsilon: float = DIVERGENCE_EPSILON) -> Optional[List[float]]:
    if not values:
        return None
    cleaned = [max(0.0, float(value)) for value in values]
    if sum(cleaned) <= 0.0:
        return None
    if epsilon > 0.0:
        cleaned = [value + epsilon for value in cleaned]
    total = sum(cleaned)
    if total <= 0.0:
        return None
    return [value / total for value in cleaned]


def kl_divergence(
    p: Sequence[float],
    q: Sequence[float],
    epsilon: float = DIVERGENCE_EPSILON,
) -> Optional[float]:
    if len(p) != len(q) or not p:
        return None
    p_dist = smooth_distribution(p, epsilon=epsilon)
    q_dist = smooth_distribution(q, epsilon=epsilon)
    if p_dist is None or q_dist is None:
        return None
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p_dist, q_dist))


def jensen_shannon_divergence(
    p: Sequence[float],
    q: Sequence[float],
    epsilon: float = DIVERGENCE_EPSILON,
) -> Optional[float]:
    if len(p) != len(q) or not p:
        return None
    p_dist = smooth_distribution(p, epsilon=epsilon)
    q_dist = smooth_distribution(q, epsilon=epsilon)
    if p_dist is None or q_dist is None:
        return None
    midpoint = [(pi + qi) / 2.0 for pi, qi in zip(p_dist, q_dist)]
    return 0.5 * (
        kl_divergence(p_dist, midpoint, epsilon=0.0) +
        kl_divergence(q_dist, midpoint, epsilon=0.0)
    )


def mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def bootstrap_ci(
    values: Sequence[Optional[float]],
    n_bootstrap: int,
    seed: int,
) -> Optional[Tuple[float, float]]:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) == 1:
        return (present[0], present[0])

    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(present) for _ in range(len(present))]
        means.append(sum(sample) / len(sample))
    means.sort()

    lo_idx = max(0, int(0.025 * n_bootstrap))
    hi_idx = min(n_bootstrap - 1, int(0.975 * n_bootstrap))
    return (means[lo_idx], means[hi_idx])


def extract_top_level_model_scores(section_json: dict) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for section_name, payload in section_json.items():
        if isinstance(payload, dict):
            result[section_name] = float(payload.get("total_score", 0.0))
        else:
            result[section_name] = float(payload)
    return result


def metric_bundle(human: Sequence[float], model: Sequence[float]) -> Dict[str, Optional[float]]:
    return {
        "kl_divergence": kl_divergence(human, model),
        "jensen_shannon_divergence": jensen_shannon_divergence(human, model),
        "spearman": spearman_rho(human, model),
        "kendall_tau_b": kendall_tau_b(human, model),
        "l1": l1_distance(human, model),
    }


def uniform_vector(n: int) -> List[float]:
    if n <= 0:
        return []
    return [1.0 / n] * n


def normalize_text_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def align_section_score_maps(
    human_scores: Dict[str, float],
    model_scores: Dict[str, float],
) -> Tuple[List[str], Dict[str, float], Dict[str, float], List[str]]:
    """
    Align human and model top-level section scores using a case-insensitive,
    punctuation-insensitive key match while preserving human-readable labels.
    """
    model_by_norm = {
        normalize_text_key(section): section
        for section in model_scores
        if normalize_text_key(section)
    }

    overlap_labels: List[str] = []
    aligned_human: Dict[str, float] = {}
    aligned_model: Dict[str, float] = {}
    matched_model_sections = set()

    for human_section, human_value in human_scores.items():
        norm = normalize_text_key(human_section)
        model_section = model_by_norm.get(norm)
        if not model_section:
            continue
        overlap_labels.append(human_section)
        aligned_human[human_section] = human_value
        aligned_model[human_section] = model_scores[model_section]
        matched_model_sections.add(model_section)

    model_extras = sorted(section for section in model_scores if section not in matched_model_sections)
    return overlap_labels, aligned_human, aligned_model, model_extras


def extract_local_citation_numbers(text: str) -> List[int]:
    value = (text or "").strip()
    if not value:
        return []

    if re.fullmatch(r"\d+", value):
        number = int(value)
        return [number] if number > 0 else []

    numbers: List[int] = []
    for match in re.finditer(r"[\[(]\s*(\d+)(?:\s*[-–]\s*(\d+))?\s*[\])]", value):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if start <= 0 or end <= 0:
            continue
        if start > end:
            start, end = end, start
        numbers.extend(range(start, end + 1))

    return numbers


def get_top_k_model_citations(
    citation_json: dict,
    k: int,
    paper_id: Optional[str] = None,
    resolver: Optional[CitationResolver] = None,
) -> List[Tuple[str, float]]:
    if resolver is None or paper_id is None:
        ranked: List[Tuple[str, float]] = []
        for citation, payload in citation_json.items():
            if isinstance(payload, dict):
                score = float(payload.get("citation_score", 0.0))
            else:
                score = float(payload)
            ranked.append((citation, score))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[: max(0, k)]

    grouped: Dict[str, Dict[str, object]] = {}
    for citation, payload in citation_json.items():
        if isinstance(payload, dict):
            score = float(payload.get("citation_score", 0.0))
        else:
            score = float(payload)

        entry = resolver.resolve(citation, paper_id=paper_id)
        target_id = resolver.target_id_for(citation, paper_id) if paper_id is not None else None
        if target_id:
            group_key = target_id
        elif entry is not None:
            group_key = entry.stable_external_id or entry.canonical_id or normalize_text_key(entry.title) or citation
        else:
            group_key = normalize_text_key(citation) or citation

        bucket = grouped.setdefault(
            group_key,
            {
                "score": 0.0,
                "display_key": citation,
                "best_single_score": float("-inf"),
            },
        )
        bucket["score"] = float(bucket["score"]) + score
        if score > float(bucket["best_single_score"]):
            bucket["best_single_score"] = score
            bucket["display_key"] = citation

    ranked = [
        (str(payload["display_key"]), float(payload["score"]))
        for payload in grouped.values()
    ]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[: max(0, k)]


def model_citation_aliases(
    paper_id: str,
    citation_key: str,
    resolver: Optional[CitationResolver],
) -> Dict[str, List[str]]:
    aliases = {
        "numeric": [],
        "text": [],
    }

    raw_numeric = extract_local_citation_numbers(citation_key)
    if raw_numeric:
        aliases["numeric"].extend(str(num) for num in raw_numeric)

    raw_norm = normalize_text_key(citation_key)
    if raw_norm:
        aliases["text"].append(raw_norm)

    if resolver is None:
        return aliases

    entry = resolver.resolve(citation_key, paper_id=paper_id)
    if entry is None:
        return aliases

    if entry.numeric_key:
        aliases["numeric"].extend(str(num) for num in extract_local_citation_numbers(entry.numeric_key))
        num_norm = normalize_text_key(entry.numeric_key)
        if num_norm:
            aliases["text"].append(num_norm)

    title_norm = normalize_text_key(entry.title)
    if title_norm:
        aliases["text"].append(title_norm)

    if entry.first_author_last and entry.year:
        aliases["text"].append(normalize_text_key(f"{entry.first_author_last} {entry.year}"))
        aliases["text"].append(normalize_text_key(f"{entry.first_author_last}, {entry.year}"))

    raw_text_norm = normalize_text_key(entry.raw_text)
    if raw_text_norm:
        aliases["text"].append(raw_text_norm)

    return aliases


def human_citation_aliases(citation_text: str) -> Dict[str, List[str]]:
    aliases = {
        "numeric": [],
        "text": [],
    }
    aliases["numeric"].extend(str(num) for num in extract_local_citation_numbers(citation_text))
    norm = normalize_text_key(citation_text)
    if norm:
        aliases["text"].append(norm)
    return aliases


def citation_match_score(
    human_aliases: Dict[str, List[str]],
    model_aliases: Dict[str, List[str]],
) -> int:
    human_numeric = set(human_aliases["numeric"])
    model_numeric = set(model_aliases["numeric"])
    if human_numeric and model_numeric and human_numeric.intersection(model_numeric):
        return 3

    human_text = [text for text in human_aliases["text"] if text]
    model_text = [text for text in model_aliases["text"] if text]
    if not human_text or not model_text:
        return 0

    for h in human_text:
        for m in model_text:
            if h == m:
                return 3
            if len(h) >= 12 and h in m:
                return 2
            if len(m) >= 12 and m in h:
                return 2
    return 0


def resolve_human_top_k_citations(
    paper_id: str,
    human_items: List[dict],
    model_ranked_full: List[Tuple[str, float]],
    resolver: Optional[CitationResolver],
) -> List[Optional[str]]:
    resolved: List[Optional[str]] = []
    used_model_keys = set()

    for item in sorted(human_items, key=lambda entry: entry.get("rank", 10**9)):
        citation_text = str(item.get("citation", "")).strip()
        human_alias = human_citation_aliases(citation_text)
        best_key: Optional[str] = None
        best_score = 0

        for model_key, _ in model_ranked_full:
            if model_key in used_model_keys:
                continue
            model_alias = model_citation_aliases(paper_id, model_key, resolver)
            score = citation_match_score(human_alias, model_alias)
            if score > best_score:
                best_score = score
                best_key = model_key
                if score == 3:
                    break

        if best_key is not None and best_score > 0:
            used_model_keys.add(best_key)
        else:
            best_key = None
        resolved.append(best_key)

    return resolved


def reciprocal_rank(target: Optional[str], ranked_keys: Sequence[str]) -> float:
    if target is None:
        return 0.0
    for idx, key in enumerate(ranked_keys, start=1):
        if key == target:
            return 1.0 / idx
    return 0.0


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    total = 0.0
    for idx, rel in enumerate(relevances[: max(0, k)], start=1):
        total += (2.0**rel - 1.0) / math.log2(idx + 1.0)
    return total


def ndcg_at_k(relevance_by_key: Dict[str, float], ranked_keys: Sequence[str], k: int) -> float:
    observed = [float(relevance_by_key.get(key, 0.0)) for key in ranked_keys[: max(0, k)]]
    ideal = sorted((float(value) for value in relevance_by_key.values()), reverse=True)[: max(0, k)]
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg <= 0.0:
        return 0.0
    return dcg_at_k(observed, k) / ideal_dcg


def select_reference_important_papers(
    payload: dict,
    k: int,
    reference_field: str = "auto",
) -> List[dict]:
    if reference_field != "auto":
        return list(payload.get(reference_field, []) or [])

    if k > 4:
        return list(payload.get("important_papers_top_8", []) or [])
    return list(payload.get("important_papers", []) or [])


def citation_top_k_report(
    paper_id: str,
    important_papers: List[dict],
    citation_json: dict,
    resolver: Optional[CitationResolver],
    k: int,
) -> dict:
    ranked_full = get_top_k_model_citations(
        citation_json,
        max(k, len(citation_json)),
        paper_id=paper_id,
        resolver=resolver,
    )
    ranked_top_k = ranked_full[:k]
    ranked_top_k_keys = [key for key, _ in ranked_top_k]

    human_ranked = sorted(important_papers, key=lambda item: item.get("rank", 10**9))[:k]
    resolved_human_keys = resolve_human_top_k_citations(
        paper_id=paper_id,
        human_items=human_ranked,
        model_ranked_full=ranked_full,
        resolver=resolver,
    )
    resolved_human_set = {key for key in resolved_human_keys if key is not None}
    top_k_set = set(ranked_top_k_keys)
    overlap_count = len(resolved_human_set.intersection(top_k_set))
    denom = max(1, len(human_ranked))
    precision_at_k = overlap_count / max(1, k)
    recall_at_k = overlap_count / denom
    hit_at_k = 1.0 if overlap_count > 0 else 0.0
    mrr_rank1 = reciprocal_rank(resolved_human_keys[0] if resolved_human_keys else None, ranked_top_k_keys)

    ranked_full_keys = [key for key, _ in ranked_full]
    ranked_position = {key: idx for idx, key in enumerate(ranked_full_keys, start=1)}
    model_ranks_for_reference = [
        float(ranked_position.get(key, len(ranked_full_keys) + 1)) if key is not None else float(len(ranked_full_keys) + 1)
        for key in resolved_human_keys
    ]
    reference_ranks = [float(idx) for idx in range(1, len(human_ranked) + 1)]
    citation_spearman = spearman_rho(reference_ranks, model_ranks_for_reference)

    relevance_by_key = {
        key: float(len(human_ranked) - idx)
        for idx, key in enumerate(resolved_human_keys)
        if key is not None
    }
    ndcg = ndcg_at_k(relevance_by_key, ranked_top_k_keys, k)

    model_picked_citations = []
    for key, score in ranked_top_k:
        entry = resolver.resolve(key, paper_id=paper_id) if resolver is not None else None
        model_picked_citations.append(
            {
                "title": entry.title if entry is not None and entry.title else key,
                "citation_score": score,
            }
        )

    return {
        "k": k,
        "human_top_k": human_ranked,
        "human_top_k_resolved_model_keys": resolved_human_keys,
        "model_picked_citations": model_picked_citations,
        "model_top_k": [
            {
                "citation": key,
                "citation_score": score,
            }
            for key, score in ranked_top_k
        ],
        "metrics": {
            "overlap_at_k": overlap_count,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "hit_at_k": hit_at_k,
            "mrr_human_rank1_in_model_top_k": mrr_rank1,
            "ndcg_at_k": ndcg,
            "spearman_reference_top_k_vs_model_rank": citation_spearman,
        },
        "reference_ranks": reference_ranks,
        "model_ranks_for_reference": model_ranks_for_reference,
    }


def citation_reference_in_model_top_k_report(
    paper_id: str,
    human_items: List[dict],
    citation_json: dict,
    resolver: Optional[CitationResolver],
    reference_k: int,
    model_k: int,
) -> dict:
    ranked_full = get_top_k_model_citations(
        citation_json,
        max(model_k, len(citation_json)),
        paper_id=paper_id,
        resolver=resolver,
    )
    model_top_k = ranked_full[: max(0, model_k)]
    model_top_k_keys = [key for key, _ in model_top_k]

    human_ranked = sorted(human_items, key=lambda item: item.get("rank", 10**9))[: max(0, reference_k)]
    resolved_human_keys = resolve_human_top_k_citations(
        paper_id=paper_id,
        human_items=human_ranked,
        model_ranked_full=ranked_full,
        resolver=resolver,
    )
    resolved_human_set = {key for key in resolved_human_keys if key is not None}
    top_k_set = set(model_top_k_keys)
    overlap_count = len(resolved_human_set.intersection(top_k_set))
    denom = max(1, len(human_ranked))

    model_picked_citations = []
    for key, score in model_top_k:
        entry = resolver.resolve(key, paper_id=paper_id) if resolver is not None else None
        model_picked_citations.append(
            {
                "title": entry.title if entry is not None and entry.title else key,
                "citation_score": score,
            }
        )

    return {
        "reference_k": reference_k,
        "model_k": model_k,
        "human_top_k": human_ranked,
        "human_top_k_resolved_model_keys": resolved_human_keys,
        "model_top_k": [
            {
                "citation": key,
                "citation_score": score,
            }
            for key, score in model_top_k
        ],
        "model_picked_citations": model_picked_citations,
        "metrics": {
            "overlap_count": overlap_count,
            "recall": overlap_count / denom,
            "hit_any": 1.0 if overlap_count > 0 else 0.0,
            "hit_all": 1.0 if overlap_count >= len(human_ranked) and human_ranked else 0.0,
            "mrr_human_rank1_in_model_top_k": reciprocal_rank(
                resolved_human_keys[0] if resolved_human_keys else None,
                model_top_k_keys,
            ),
        },
    }


def aggregate_metric_report(per_paper: List[Dict[str, Dict[str, Optional[float]]]], n_bootstrap: int, seed: int) -> dict:
    model_kl = [paper["model"]["kl_divergence"] for paper in per_paper]
    model_jsd = [paper["model"]["jensen_shannon_divergence"] for paper in per_paper]
    model_spearman = [paper["model"]["spearman"] for paper in per_paper]
    model_kendall = [paper["model"]["kendall_tau_b"] for paper in per_paper]
    model_l1 = [paper["model"]["l1"] for paper in per_paper]

    uniform_kl = [paper["citation_frequency"]["kl_divergence"] for paper in per_paper]
    uniform_jsd = [paper["citation_frequency"]["jensen_shannon_divergence"] for paper in per_paper]
    uniform_spearman = [paper["citation_frequency"]["spearman"] for paper in per_paper]
    uniform_kendall = [paper["citation_frequency"]["kendall_tau_b"] for paper in per_paper]
    uniform_l1 = [paper["citation_frequency"]["l1"] for paper in per_paper]

    length_entries = [paper["length_weighted_frequency"] for paper in per_paper if paper.get("length_weighted_frequency") is not None]
    length_kl = [e["kl_divergence"] for e in length_entries]
    length_jsd = [e["jensen_shannon_divergence"] for e in length_entries]
    length_spearman = [e["spearman"] for e in length_entries]
    length_kendall = [e["kendall_tau_b"] for e in length_entries]
    length_l1 = [e["l1"] for e in length_entries]

    return {
        "model": {
            "mean_kl_divergence": mean_or_none(model_kl),
            "mean_jensen_shannon_divergence": mean_or_none(model_jsd),
            "mean_spearman": mean_or_none(model_spearman),
            "mean_kendall_tau_b": mean_or_none(model_kendall),
            "mean_l1": mean_or_none(model_l1),
            "bootstrap_ci_kl_divergence": bootstrap_ci(model_kl, n_bootstrap, seed),
            "bootstrap_ci_jensen_shannon_divergence": bootstrap_ci(model_jsd, n_bootstrap, seed + 1),
            "bootstrap_ci_spearman": bootstrap_ci(model_spearman, n_bootstrap, seed),
            "bootstrap_ci_kendall_tau_b": bootstrap_ci(model_kendall, n_bootstrap, seed + 2),
            "bootstrap_ci_l1": bootstrap_ci(model_l1, n_bootstrap, seed + 3),
        },
        "citation_frequency": {
            "mean_kl_divergence": mean_or_none(uniform_kl),
            "mean_jensen_shannon_divergence": mean_or_none(uniform_jsd),
            "mean_spearman": mean_or_none(uniform_spearman),
            "mean_kendall_tau_b": mean_or_none(uniform_kendall),
            "mean_l1": mean_or_none(uniform_l1),
            "bootstrap_ci_kl_divergence": bootstrap_ci(uniform_kl, n_bootstrap, seed + 4),
            "bootstrap_ci_jensen_shannon_divergence": bootstrap_ci(uniform_jsd, n_bootstrap, seed + 5),
            "bootstrap_ci_spearman": bootstrap_ci(uniform_spearman, n_bootstrap, seed + 6),
            "bootstrap_ci_kendall_tau_b": bootstrap_ci(uniform_kendall, n_bootstrap, seed + 7),
            "bootstrap_ci_l1": bootstrap_ci(uniform_l1, n_bootstrap, seed + 8),
        },
        "length_weighted_frequency": {
            "mean_kl_divergence": mean_or_none(length_kl),
            "mean_jensen_shannon_divergence": mean_or_none(length_jsd),
            "mean_spearman": mean_or_none(length_spearman),
            "mean_kendall_tau_b": mean_or_none(length_kendall),
            "mean_l1": mean_or_none(length_l1),
            "bootstrap_ci_kl_divergence": bootstrap_ci(length_kl, n_bootstrap, seed + 9),
            "bootstrap_ci_jensen_shannon_divergence": bootstrap_ci(length_jsd, n_bootstrap, seed + 10),
            "bootstrap_ci_spearman": bootstrap_ci(length_spearman, n_bootstrap, seed + 11),
            "bootstrap_ci_kendall_tau_b": bootstrap_ci(length_kendall, n_bootstrap, seed + 12),
            "bootstrap_ci_l1": bootstrap_ci(length_l1, n_bootstrap, seed + 13),
        },
    }


def aggregate_citation_report(per_paper: List[dict], n_bootstrap: int, seed: int) -> dict:
    overlap_values = [paper["metrics"]["overlap_at_k"] for paper in per_paper]
    precision_values = [paper["metrics"]["precision_at_k"] for paper in per_paper]
    recall_values = [paper["metrics"]["recall_at_k"] for paper in per_paper]
    hit_values = [paper["metrics"]["hit_at_k"] for paper in per_paper]
    mrr_values = [paper["metrics"]["mrr_human_rank1_in_model_top_k"] for paper in per_paper]
    ndcg_values = [paper["metrics"]["ndcg_at_k"] for paper in per_paper]
    spearman_values = [paper["metrics"]["spearman_reference_top_k_vs_model_rank"] for paper in per_paper]

    return {
        "mean_overlap_at_k": mean_or_none(overlap_values),
        "mean_precision_at_k": mean_or_none(precision_values),
        "mean_recall_at_k": mean_or_none(recall_values),
        "mean_hit_at_k": mean_or_none(hit_values),
        "mean_mrr_human_rank1_in_model_top_k": mean_or_none(mrr_values),
        "mean_ndcg_at_k": mean_or_none(ndcg_values),
        "mean_spearman_reference_top_k_vs_model_rank": mean_or_none(spearman_values),
        "bootstrap_ci_overlap_at_k": bootstrap_ci(overlap_values, n_bootstrap, seed),
        "bootstrap_ci_precision_at_k": bootstrap_ci(precision_values, n_bootstrap, seed + 1),
        "bootstrap_ci_recall_at_k": bootstrap_ci(recall_values, n_bootstrap, seed + 2),
        "bootstrap_ci_hit_at_k": bootstrap_ci(hit_values, n_bootstrap, seed + 3),
        "bootstrap_ci_mrr_human_rank1_in_model_top_k": bootstrap_ci(mrr_values, n_bootstrap, seed + 4),
        "bootstrap_ci_ndcg_at_k": bootstrap_ci(ndcg_values, n_bootstrap, seed + 5),
        "bootstrap_ci_spearman_reference_top_k_vs_model_rank": bootstrap_ci(spearman_values, n_bootstrap, seed + 6),
    }


def aggregate_reference_in_model_top_k_report(per_paper: List[dict], n_bootstrap: int, seed: int) -> dict:
    overlap_values = [paper["metrics"]["overlap_count"] for paper in per_paper]
    recall_values = [paper["metrics"]["recall"] for paper in per_paper]
    hit_any_values = [paper["metrics"]["hit_any"] for paper in per_paper]
    hit_all_values = [paper["metrics"]["hit_all"] for paper in per_paper]
    mrr_values = [paper["metrics"]["mrr_human_rank1_in_model_top_k"] for paper in per_paper]

    reference_k = per_paper[0]["reference_k"] if per_paper else 0
    model_k = per_paper[0]["model_k"] if per_paper else 0
    return {
        "reference_k": reference_k,
        "model_k": model_k,
        "mean_overlap_count": mean_or_none(overlap_values),
        "mean_recall": mean_or_none(recall_values),
        "mean_hit_any": mean_or_none(hit_any_values),
        "mean_hit_all": mean_or_none(hit_all_values),
        "mean_mrr_human_rank1_in_model_top_k": mean_or_none(mrr_values),
        "bootstrap_ci_overlap_count": bootstrap_ci(overlap_values, n_bootstrap, seed),
        "bootstrap_ci_recall": bootstrap_ci(recall_values, n_bootstrap, seed + 1),
        "bootstrap_ci_hit_any": bootstrap_ci(hit_any_values, n_bootstrap, seed + 2),
        "bootstrap_ci_hit_all": bootstrap_ci(hit_all_values, n_bootstrap, seed + 3),
        "bootstrap_ci_mrr_human_rank1_in_model_top_k": bootstrap_ci(mrr_values, n_bootstrap, seed + 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model section scores against human expert annotations.")
    parser.add_argument(
        "--annotations-file",
        default="human_expert_annotations.json",
        help="Path to the human annotation JSON file.",
    )
    parser.add_argument(
        "--results-root",
        default="paper_results",
        help="Root directory containing per-paper result folders.",
    )
    parser.add_argument(
        "--model-tag",
        default="",
        help="Model tag suffix used in saved section score filenames, e.g. qwen3_8b.",
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
        "--output-json",
        default="",
        help="Optional path to save the full evaluation report as JSON.",
    )
    parser.add_argument(
        "--papers-dir",
        default="papers",
        help="Directory containing source PDFs, used for citation resolution.",
    )
    parser.add_argument(
        "--load-mappings",
        default="",
        help="Optional citation_mappings.json to load and reuse during citation evaluation.",
    )
    parser.add_argument(
        "--save-mappings",
        default="",
        help="Optional path to save citation mappings after rebuilding them.",
    )
    parser.add_argument(
        "--citation-top-k",
        type=int,
        default=4,
        help="Top-k citations to compare against the human top-k citation annotation.",
    )
    parser.add_argument(
        "--enable-citation-eval",
        action="store_true",
        help="Also evaluate top-k citation agreement using the human important_papers lists.",
    )
    parser.add_argument(
        "--citation-reference-field",
        default="auto",
        help="Annotation field to use for citation evaluation. Default 'auto' uses important_papers_top_8 when k>4 and available, otherwise important_papers.",
    )
    parser.add_argument(
        "--chatgpt-annotations-file",
        default="chatgpt_baseline_annotations.json",
        help="Path to the ChatGPT annotation JSON file used as a second ground truth.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    citation_resolver: Optional[CitationResolver] = None
    if args.enable_citation_eval:
        if args.load_mappings and Path(args.load_mappings).exists():
            citation_resolver = CitationResolver.load(args.load_mappings)
        else:
            citation_resolver = CitationResolver()
            citation_resolver.parse_all(results_root, Path(args.papers_dir))
            if args.save_mappings:
                citation_resolver.save(args.save_mappings)

    def _run_evaluation(annotations: dict, n_bootstrap: int, seed: int) -> Tuple[dict, Dict[str, Optional[dict]], List[dict], Dict[str, List[dict]], List[dict]]:
        per_paper_reports: List[dict] = []
        per_paper_citation_reports: Dict[str, List[dict]] = {"model": [], "citation_frequency": [], "length_weighted_frequency": []}
        per_paper_human4_in_model10_reports: Dict[str, List[dict]] = {"model": [], "citation_frequency": [], "length_weighted_frequency": []}
        skipped: List[dict] = []
        suffix = f"_{args.model_tag}" if args.model_tag else ""

        for paper_id, payload in annotations["papers"].items():
            section_path = results_root / paper_id / f"{paper_id}{suffix}_section_scores.json"
            if not section_path.exists():
                skipped.append({"paper_id": paper_id, "reason": "missing_model_section_file", "path": str(section_path)})
                continue

            model_scores = extract_top_level_model_scores(load_json(section_path))
            ref_scores = payload["top_level_scores_for_evaluation"]

            overlap, ref_overlap_raw, model_overlap_raw, model_extras = align_section_score_maps(ref_scores, model_scores)
            if len(overlap) < 2:
                skipped.append({"paper_id": paper_id, "reason": "insufficient_overlap",
                                 "ref_sections": sorted(ref_scores.keys()), "model_sections": sorted(model_scores.keys())})
                continue

            try:
                ref_overlap = normalize_scores(ref_overlap_raw)
            except ValueError:
                skipped.append({
                    "paper_id": paper_id,
                    "reason": "all_zero_reference_overlap",
                    "aligned_sections": overlap,
                })
                continue

            try:
                model_overlap = normalize_scores(model_overlap_raw)
            except ValueError:
                skipped.append({
                    "paper_id": paper_id,
                    "reason": "all_zero_model_overlap",
                    "aligned_sections": overlap,
                    "section_file": str(section_path),
                })
                continue

            ordered_ref = [ref_overlap[s] for s in overlap]
            ordered_model = [model_overlap[s] for s in overlap]
            ordered_uniform = uniform_vector(len(overlap))

            length_section_metrics: Optional[Dict[str, Optional[float]]] = None
            length_section_path = results_root / paper_id / f"{paper_id}_length_weighted_frequency_section_scores.json"
            if length_section_path.exists():
                length_scores_raw = extract_top_level_model_scores(load_json(length_section_path))
                _, _, length_overlap_raw, _ = align_section_score_maps(ref_scores, length_scores_raw)
                if len(length_overlap_raw) >= 2:
                    try:
                        length_overlap = normalize_scores(length_overlap_raw)
                    except ValueError:
                        length_overlap = None
                    if length_overlap is not None:
                        ordered_length = [length_overlap.get(s, 0.0) for s in overlap]
                        length_section_metrics = metric_bundle(ordered_ref, ordered_length)

            per_paper_reports.append({
                "paper_id": paper_id,
                "section_file": str(section_path),
                "aligned_sections": overlap,
                "ref_normalized": ref_overlap,
                "model_normalized": model_overlap,
                "model_extra_top_level_sections": model_extras,
                "metrics": {
                    "model": metric_bundle(ordered_ref, ordered_model),
                    "citation_frequency": metric_bundle(ordered_ref, ordered_uniform),
                    "length_weighted_frequency": length_section_metrics,
                },
            })

            if args.enable_citation_eval:
                ref_important = select_reference_important_papers(payload, k=max(1, args.citation_top_k),
                                                                   reference_field=args.citation_reference_field)
                ref_human_top_4 = list(payload.get("important_papers", []) or [])[:4]
                if not ref_human_top_4:
                    ref_human_top_4 = select_reference_important_papers(payload, k=4, reference_field="auto")
                if not ref_important:
                    continue
                k_val = max(1, args.citation_top_k)
                for tag, filename in [
                    ("model", f"{paper_id}{suffix}_citation_scores.json"),
                    ("citation_frequency", f"{paper_id}_citation_frequency_citation_scores.json"),
                    ("length_weighted_frequency", f"{paper_id}_length_weighted_frequency_citation_scores.json"),
                ]:
                    cpath = results_root / paper_id / filename
                    if cpath.exists():
                        citation_json = load_json(cpath)
                        per_paper_citation_reports[tag].append({
                            "paper_id": paper_id,
                            "citation_file": str(cpath),
                            **citation_top_k_report(paper_id=paper_id, important_papers=ref_important,
                                                    citation_json=citation_json, resolver=citation_resolver, k=k_val),
                        })
                        if ref_human_top_4:
                            per_paper_human4_in_model10_reports[tag].append({
                                "paper_id": paper_id,
                                "citation_file": str(cpath),
                                **citation_reference_in_model_top_k_report(
                                    paper_id=paper_id,
                                    human_items=ref_human_top_4,
                                    citation_json=citation_json,
                                    resolver=citation_resolver,
                                    reference_k=4,
                                    model_k=10,
                                ),
                            })

        aggregate = aggregate_metric_report([p["metrics"] for p in per_paper_reports], n_bootstrap=n_bootstrap, seed=seed)
        citation_aggregates: Dict[str, Optional[dict]] = {"model": None, "citation_frequency": None, "length_weighted_frequency": None}
        human4_in_model10_aggregates: Dict[str, Optional[dict]] = {"model": None, "citation_frequency": None, "length_weighted_frequency": None}
        for source in citation_aggregates:
            if per_paper_citation_reports[source]:
                citation_aggregates[source] = aggregate_citation_report(
                    per_paper_citation_reports[source], n_bootstrap=n_bootstrap, seed=seed + 100)
            if per_paper_human4_in_model10_reports[source]:
                human4_in_model10_aggregates[source] = aggregate_reference_in_model_top_k_report(
                    per_paper_human4_in_model10_reports[source], n_bootstrap=n_bootstrap, seed=seed + 200)
        return (
            aggregate,
            citation_aggregates,
            human4_in_model10_aggregates,
            per_paper_reports,
            per_paper_citation_reports,
            per_paper_human4_in_model10_reports,
            skipped,
        )

    k = args.citation_top_k
    n_bootstrap = max(100, args.bootstrap_samples)

    def _print_section_metrics(label: str, agg: dict) -> None:
        print(label)
        print(f"  Mean KL divergence: {agg['mean_kl_divergence']}")
        print(f"  Mean Jensen-Shannon divergence: {agg['mean_jensen_shannon_divergence']}")
        print(f"  Mean Spearman: {agg['mean_spearman']}")
        print(f"  Mean Kendall tau-b: {agg['mean_kendall_tau_b']}")
        print(f"  Mean L1: {agg['mean_l1']}")
        print(f"  Bootstrap CI KL divergence: {agg['bootstrap_ci_kl_divergence']}")
        print(f"  Bootstrap CI Jensen-Shannon divergence: {agg['bootstrap_ci_jensen_shannon_divergence']}")
        print(f"  Bootstrap CI Spearman: {agg['bootstrap_ci_spearman']}")
        print(f"  Bootstrap CI Kendall tau-b: {agg['bootstrap_ci_kendall_tau_b']}")
        print(f"  Bootstrap CI L1: {agg['bootstrap_ci_l1']}")

    def _print_citation_metrics(label: str, agg: dict, n_papers: int) -> None:
        print(label)
        print(f"  Papers evaluated: {n_papers}")
        print(f"  Mean Precision@{k}: {agg['mean_precision_at_k']}")
        print(f"  Mean Recall@{k}: {agg['mean_recall_at_k']}")
        print(f"  Mean Hit@{k}: {agg['mean_hit_at_k']}")
        print(f"  Mean MRR (human rank-1 in model top-k): {agg['mean_mrr_human_rank1_in_model_top_k']}")
        print(f"  Mean nDCG@{k}: {agg['mean_ndcg_at_k']}")

    def _print_human4_in_model10_metrics(label: str, agg: dict, n_papers: int) -> None:
        print(label)
        print(f"  Papers evaluated: {n_papers}")
        print(f"  Mean overlap count (human top-{agg['reference_k']} in model top-{agg['model_k']}): {agg['mean_overlap_count']}")
        print(f"  Mean recall (human top-{agg['reference_k']} recovered in model top-{agg['model_k']}): {agg['mean_recall']}")
        print(f"  Mean hit-any: {agg['mean_hit_any']}")
        print(f"  Mean hit-all: {agg['mean_hit_all']}")
        print(f"  Mean MRR (human rank-1 in model top-{agg['model_k']}): {agg['mean_mrr_human_rank1_in_model_top_k']}")

    def _print_block(header: str, aggregate: dict, citation_aggregates: Dict[str, Optional[dict]],
                     human4_in_model10_aggregates: Dict[str, Optional[dict]],
                     per_paper_reports: List[dict], per_paper_citation_reports: Dict[str, List[dict]],
                     per_paper_human4_in_model10_reports: Dict[str, List[dict]],
                     skipped: List[dict]) -> None:
        print(f"{'=' * 60}")
        print(header)
        print(f"{'=' * 60}")
        print(f"Evaluated papers: {len(per_paper_reports)}")
        print(f"Paper IDs: {', '.join(p['paper_id'] for p in per_paper_reports)}")
        print()
        _print_section_metrics("Model section metrics", aggregate["model"])
        print()
        _print_section_metrics("Citation frequency baseline (section)", aggregate["citation_frequency"])
        print()
        _print_section_metrics("Length-weighted frequency baseline (section)", aggregate["length_weighted_frequency"])
        if args.enable_citation_eval:
            print()
            print(f"--- Citation Top-{k} metrics ---")
            for source, label in [
                ("model", f"Model citation @{k}"),
                ("citation_frequency", f"Citation frequency baseline citation @{k}"),
                ("length_weighted_frequency", f"Length-weighted frequency baseline citation @{k}"),
            ]:
                agg = citation_aggregates[source]
                n = len(per_paper_citation_reports[source])
                if agg is not None:
                    print()
                    _print_citation_metrics(label, agg, n)
            print()
            print("--- Human top-4 inside model top-10 ---")
            for source, label in [
                ("model", "Model retrieval of human top-4 in model top-10"),
                ("citation_frequency", "Citation frequency baseline retrieval of human top-4 in model top-10"),
                ("length_weighted_frequency", "Length-weighted frequency baseline retrieval of human top-4 in model top-10"),
            ]:
                agg = human4_in_model10_aggregates[source]
                n = len(per_paper_human4_in_model10_reports[source])
                if agg is not None:
                    print()
                    _print_human4_in_model10_metrics(label, agg, n)
        if skipped:
            print()
            print("Skipped papers")
            for item in skipped:
                print(f"  {item['paper_id']}: {item['reason']}")

    # --- Human annotations ---
    human_annotations = load_json(Path(args.annotations_file))
    h_aggregate, h_citation_agg, h_human4_model10_agg, h_reports, h_cit_reports, h_human4_model10_reports, h_skipped = _run_evaluation(
        human_annotations, n_bootstrap=n_bootstrap, seed=args.seed)
    _print_block(
        "Ground truth: Human expert annotations",
        h_aggregate,
        h_citation_agg,
        h_human4_model10_agg,
        h_reports,
        h_cit_reports,
        h_human4_model10_reports,
        h_skipped,
    )

    # --- ChatGPT annotations ---
    chatgpt_path = Path(args.chatgpt_annotations_file)
    if chatgpt_path.exists():
        print()
        chatgpt_annotations = load_json(chatgpt_path)
        g_aggregate, g_citation_agg, g_human4_model10_agg, g_reports, g_cit_reports, g_human4_model10_reports, g_skipped = _run_evaluation(
            chatgpt_annotations, n_bootstrap=n_bootstrap, seed=args.seed + 200)
        _print_block(
            "Ground truth: ChatGPT annotations",
            g_aggregate,
            g_citation_agg,
            g_human4_model10_agg,
            g_reports,
            g_cit_reports,
            g_human4_model10_reports,
            g_skipped,
        )

    summary = {
        "model_tag": args.model_tag,
        "metric_notes": [
            f"KL and Jensen-Shannon divergences are computed on normalized top-level section score vectors over the aligned human-model overlap using epsilon smoothing with epsilon={DIVERGENCE_EPSILON}.",
            "KL and Jensen-Shannon are reported in nats because natural logarithms are used.",
            "Top-k citation agreement uses the human important_papers lists and citation alias resolution before overlap is computed.",
        ],
        "human": {
            "papers_evaluated": len(h_reports),
            "aggregate": h_aggregate,
            "citation_top_k": {
                "enabled": args.enable_citation_eval,
                "k": k,
                "model": {"papers_evaluated": len(h_cit_reports["model"]), "aggregate": h_citation_agg["model"]},
                "citation_frequency": {"papers_evaluated": len(h_cit_reports["citation_frequency"]), "aggregate": h_citation_agg["citation_frequency"]},
                "length_weighted_frequency": {"papers_evaluated": len(h_cit_reports["length_weighted_frequency"]), "aggregate": h_citation_agg["length_weighted_frequency"]},
            },
            "human_top_4_in_model_top_10": {
                "enabled": args.enable_citation_eval,
                "model": {"papers_evaluated": len(h_human4_model10_reports["model"]), "aggregate": h_human4_model10_agg["model"]},
                "citation_frequency": {"papers_evaluated": len(h_human4_model10_reports["citation_frequency"]), "aggregate": h_human4_model10_agg["citation_frequency"]},
                "length_weighted_frequency": {"papers_evaluated": len(h_human4_model10_reports["length_weighted_frequency"]), "aggregate": h_human4_model10_agg["length_weighted_frequency"]},
            },
            "skipped": h_skipped,
        },
    }
    if chatgpt_path.exists():
        summary["chatgpt"] = {
            "papers_evaluated": len(g_reports),
            "aggregate": g_aggregate,
            "citation_top_k": {
                "enabled": args.enable_citation_eval,
                "k": k,
                "model": {"papers_evaluated": len(g_cit_reports["model"]), "aggregate": g_citation_agg["model"]},
                "citation_frequency": {"papers_evaluated": len(g_cit_reports["citation_frequency"]), "aggregate": g_citation_agg["citation_frequency"]},
                "length_weighted_frequency": {"papers_evaluated": len(g_cit_reports["length_weighted_frequency"]), "aggregate": g_citation_agg["length_weighted_frequency"]},
            },
            "human_top_4_in_model_top_10": {
                "enabled": args.enable_citation_eval,
                "model": {"papers_evaluated": len(g_human4_model10_reports["model"]), "aggregate": g_human4_model10_agg["model"]},
                "citation_frequency": {"papers_evaluated": len(g_human4_model10_reports["citation_frequency"]), "aggregate": g_human4_model10_agg["citation_frequency"]},
                "length_weighted_frequency": {"papers_evaluated": len(g_human4_model10_reports["length_weighted_frequency"]), "aggregate": g_human4_model10_agg["length_weighted_frequency"]},
            },
            "skipped": g_skipped,
        }

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
