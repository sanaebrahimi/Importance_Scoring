import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_citation(
    paper_id: str,
    raw_citation: str,
    raw_map: Dict[str, str],
    entries: Dict[str, Dict[str, Any]],
    corpus_titles: Dict[str, str],
) -> Tuple[str, Optional[str]]:
    combined_key = f"{paper_id}|||{raw_citation}"
    canonical_id = raw_map.get(combined_key)
    if canonical_id is None:
        return raw_citation, None

    if canonical_id in entries:
        return canonical_id, entries[canonical_id].get("title")

    if canonical_id in corpus_titles:
        return canonical_id, corpus_titles[canonical_id]

    return canonical_id, None


def build_top4_export(
    mappings: Dict[str, Any],
    top_k: int = 4,
    model_filter: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    entries = mappings.get("entries", {})
    raw_map = mappings.get("raw_map", {})
    corpus_titles = mappings.get("corpus_titles", {})
    paper_model_scores = mappings.get("paper_model_citation_scores", {})
    allowed_models = set(model_filter or [])

    export: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for paper_id, model_scores in sorted(paper_model_scores.items()):
        export[paper_id] = {}
        for model_tag, raw_scores in sorted(model_scores.items()):
            if allowed_models and model_tag not in allowed_models:
                continue
            if not isinstance(raw_scores, dict) or not raw_scores:
                continue

            aggregated_scores: Dict[str, float] = defaultdict(float)
            title_by_canonical: Dict[str, Optional[str]] = {}
            raw_keys_by_canonical: Dict[str, List[str]] = defaultdict(list)

            for raw_citation, score in raw_scores.items():
                try:
                    numeric_score = float(score)
                except (TypeError, ValueError):
                    continue
                canonical_id, title = resolve_citation(
                    paper_id=paper_id,
                    raw_citation=raw_citation,
                    raw_map=raw_map,
                    entries=entries,
                    corpus_titles=corpus_titles,
                )
                aggregated_scores[canonical_id] += numeric_score
                if canonical_id not in title_by_canonical or not title_by_canonical[canonical_id]:
                    title_by_canonical[canonical_id] = title
                raw_keys_by_canonical[canonical_id].append(raw_citation)

            ranked = sorted(
                aggregated_scores.items(),
                key=lambda item: (-item[1], item[0].lower()),
            )[:top_k]

            export[paper_id][model_tag] = [
                {
                    "rank": idx + 1,
                    "canonical_id": canonical_id,
                    "title": title_by_canonical.get(canonical_id),
                    "score": score,
                    "raw_citations": sorted(set(raw_keys_by_canonical.get(canonical_id, []))),
                }
                for idx, (canonical_id, score) in enumerate(ranked)
            ]

    return export


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export the top-k ranked citations for each paper and model into a nested JSON file. "
            "Outer keys are paper ids; inner keys are model tags."
        )
    )
    parser.add_argument(
        "--mappings",
        default="citation_mappings.json",
        help="Path to citation_mappings.json containing resolver data and per-model citation scores.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of highest-ranked citations to keep per paper/model.",
    )
    parser.add_argument(
        "--output",
        default="results/top4_citations_by_model.json",
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of model tags to keep. Example: --models llama3_2_3 qwen3_4b",
    )
    args = parser.parse_args()

    mappings = load_json(args.mappings)
    export = build_top4_export(
        mappings,
        top_k=max(1, args.top_k),
        model_filter=args.models,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)

    print(f"Saved top-{max(1, args.top_k)} citation export to {output_path}")


if __name__ == "__main__":
    main()
