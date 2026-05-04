import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


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
        "spearman": spearman_rho(human, model),
        "kendall_tau_b": kendall_tau_b(human, model),
        "l1": l1_distance(human, model),
    }


def uniform_vector(n: int) -> List[float]:
    if n <= 0:
        return []
    return [1.0 / n] * n


def aggregate_metric_report(per_paper: List[Dict[str, Dict[str, Optional[float]]]], n_bootstrap: int, seed: int) -> dict:
    model_spearman = [paper["model"]["spearman"] for paper in per_paper]
    model_kendall = [paper["model"]["kendall_tau_b"] for paper in per_paper]
    model_l1 = [paper["model"]["l1"] for paper in per_paper]

    uniform_spearman = [paper["uniform"]["spearman"] for paper in per_paper]
    uniform_kendall = [paper["uniform"]["kendall_tau_b"] for paper in per_paper]
    uniform_l1 = [paper["uniform"]["l1"] for paper in per_paper]

    return {
        "model": {
            "mean_spearman": mean_or_none(model_spearman),
            "mean_kendall_tau_b": mean_or_none(model_kendall),
            "mean_l1": mean_or_none(model_l1),
            "bootstrap_ci_spearman": bootstrap_ci(model_spearman, n_bootstrap, seed),
            "bootstrap_ci_kendall_tau_b": bootstrap_ci(model_kendall, n_bootstrap, seed + 1),
            "bootstrap_ci_l1": bootstrap_ci(model_l1, n_bootstrap, seed + 2),
        },
        "uniform": {
            "mean_spearman": mean_or_none(uniform_spearman),
            "mean_kendall_tau_b": mean_or_none(uniform_kendall),
            "mean_l1": mean_or_none(uniform_l1),
            "bootstrap_ci_spearman": bootstrap_ci(uniform_spearman, n_bootstrap, seed + 3),
            "bootstrap_ci_kendall_tau_b": bootstrap_ci(uniform_kendall, n_bootstrap, seed + 4),
            "bootstrap_ci_l1": bootstrap_ci(uniform_l1, n_bootstrap, seed + 5),
        },
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
    args = parser.parse_args()

    annotations = load_json(Path(args.annotations_file))
    results_root = Path(args.results_root)

    per_paper_reports = []
    skipped = []

    for paper_id, payload in annotations["papers"].items():
        suffix = f"_{args.model_tag}" if args.model_tag else ""
        section_path = results_root / paper_id / f"{paper_id}{suffix}_section_scores.json"
        if not section_path.exists():
            skipped.append({
                "paper_id": paper_id,
                "reason": "missing_model_section_file",
                "path": str(section_path),
            })
            continue

        model_section_json = load_json(section_path)
        model_scores = extract_top_level_model_scores(model_section_json)
        human_scores = payload["top_level_scores_for_evaluation"]

        overlap = [section for section in human_scores if section in model_scores]
        if len(overlap) < 2:
            skipped.append({
                "paper_id": paper_id,
                "reason": "insufficient_overlap",
                "human_sections": sorted(human_scores.keys()),
                "model_sections": sorted(model_scores.keys()),
            })
            continue

        human_overlap = normalize_scores({section: human_scores[section] for section in overlap})
        model_overlap = normalize_scores({section: model_scores[section] for section in overlap})

        ordered_human = [human_overlap[section] for section in overlap]
        ordered_model = [model_overlap[section] for section in overlap]
        ordered_uniform = uniform_vector(len(overlap))

        report = {
            "paper_id": paper_id,
            "section_file": str(section_path),
            "aligned_sections": overlap,
            "human_normalized": human_overlap,
            "model_normalized": model_overlap,
            "model_extra_top_level_sections": sorted(section for section in model_scores if section not in overlap),
            "metrics": {
                "model": metric_bundle(ordered_human, ordered_model),
                "uniform": metric_bundle(ordered_human, ordered_uniform),
            },
        }
        per_paper_reports.append(report)

    aggregate = aggregate_metric_report(
        [paper["metrics"] for paper in per_paper_reports],
        n_bootstrap=max(100, args.bootstrap_samples),
        seed=args.seed,
    )

    summary = {
        "model_tag": args.model_tag,
        "papers_evaluated": len(per_paper_reports),
        "paper_ids": [paper["paper_id"] for paper in per_paper_reports],
        "aggregate": aggregate,
        "per_paper": per_paper_reports,
        "skipped": skipped,
        "notes": [
            "Human and model scores are aligned on the annotated top-level section overlap only.",
            "Both human and model vectors are renormalized over that overlap before metric computation.",
            "Uniform baseline rank correlations are typically undefined because the baseline is a constant vector; they are reported as null when appropriate."
        ],
    }

    print(f"Evaluated papers: {summary['papers_evaluated']}")
    print(f"Paper IDs: {', '.join(summary['paper_ids'])}")
    print()
    print("Model metrics")
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
