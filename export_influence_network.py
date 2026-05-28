"""
Export a corpus-only influence network as JSON and CSV.

The export is designed for figure-making: it includes a node table with
technical and propagated scores, plus an edge table with direct citation
weights and direct propagated contributions under the Step 6 recurrences.

Usage:
    python3 export_influence_network.py \
      --results "papers/Case 1/results" \
      --papers "papers/Case 1" \
      --load-mappings "papers/Case 1/citation_mappings.json" \
      --model-tag "llama3_2_3b_3" \
      --exclude-papers Fast-RCNN LSTM RNN-Encode-Decode layer-norm weight-decay \
      --output-prefix "papers/Case 1/results/case1_influence_network_llama3_2_3b_3"
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from citation_graph_framework import KnowledgeDiscoveryFramework
from citation_resolver import CitationResolver


def _edge_rows(
    fw: KnowledgeDiscoveryFramework,
    source_mass: Dict[str, float],
    source_weighted_scores: Dict[str, float],
) -> List[Dict[str, object]]:
    graph = fw.graph
    audit = graph.internal_citation_audit()
    diag_by_edge: Dict[Tuple[str, str], List] = defaultdict(list)
    for item in audit:
        diag_by_edge[(item.source_paper, item.target_paper)].append(item)

    rows: List[Dict[str, object]] = []
    for (source, target), weight in sorted(graph.edges().items()):
        if source not in graph.papers or target not in graph.papers:
            continue
        diags = diag_by_edge.get((source, target), [])
        used_diags = [
            item for item in diags if item.status in {"paper_score", "paragraph_fallback"}
        ]
        sigma_target = graph.papers[target].originality_score()
        source_step6_mass = source_mass.get(source, 1.0)
        direct_mass_contribution = weight * source_step6_mass
        direct_pi_contribution = sigma_target * direct_mass_contribution
        direct_source_weighted_contribution = weight * source_weighted_scores.get(source, 0.0)

        rows.append(
            {
                "source": source,
                "target": target,
                "edge_weight": weight,
                "source_sigma_tech": graph.papers[source].originality_score(),
                "target_sigma_tech": sigma_target,
                "source_step6_mass": source_step6_mass,
                "direct_step6_mass_contribution": direct_mass_contribution,
                "direct_pi_contribution": direct_pi_contribution,
                "direct_source_weighted_contribution": direct_source_weighted_contribution,
                "n_internal_citation_keys": len(diags),
                "used_citation_keys": [item.citation_key for item in used_diags],
                "all_candidate_citation_keys": [item.citation_key for item in diags],
                "edge_statuses": [item.status for item in diags],
            }
        )

    rows.sort(key=lambda row: (-float(row["edge_weight"]), str(row["source"]), str(row["target"])))
    return rows


def _node_rows(
    fw: KnowledgeDiscoveryFramework,
    step6_scores: Dict[str, float],
    step6_mass: Dict[str, float],
    propagated_mass: Dict[str, float],
    normalized_influence: Dict[str, float],
    step6b_scores: Dict[str, float],
    step6b_mass: Dict[str, float],
    step6b_norm: Dict[str, float],
    seminal_scores: Dict[str, float],
    reference_scores: Dict[str, float],
) -> List[Dict[str, object]]:
    graph = fw.graph
    rows: List[Dict[str, object]] = []
    for paper_id in sorted(graph.papers):
        rows.append(
            {
                "paper_id": paper_id,
                "sigma_tech": graph.papers[paper_id].originality_score(),
                "sigma_P": step6_scores.get(paper_id, 0.0),
                "step6_mass": step6_mass.get(paper_id, 1.0),
                "pi_P": propagated_mass.get(paper_id, 0.0),
                "I_P": normalized_influence.get(paper_id, 0.0),
                "sigma_P_src": step6b_scores.get(paper_id, 0.0),
                "pi_P_src": step6b_mass.get(paper_id, 0.0),
                "I_P_src": step6b_norm.get(paper_id, 0.0),
                "seminal_score": seminal_scores.get(paper_id, 0.0),
                "reference_score": reference_scores.get(paper_id, 0.0),
                "incoming_edge_weight": graph.in_weight(paper_id),
                "outgoing_edge_weight": graph.out_weight(paper_id),
            }
        )

    rows.sort(key=lambda row: (-float(row["I_P"]), str(row["paper_id"])))
    return rows


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export corpus-only influence network.")
    parser.add_argument("--results", required=True, help="Results directory")
    parser.add_argument("--papers", required=True, help="PDF directory")
    parser.add_argument("--model-tag", default="", help="Model tag to load")
    parser.add_argument("--load-mappings", default="", help="Optional citation_mappings.json to load first")
    parser.add_argument("--save-mappings", default="", help="Optional path to save refreshed mappings")
    parser.add_argument(
        "--exclude-papers",
        nargs="+",
        default=[],
        metavar="PAPER_ID",
        help="Optional paper ids to exclude from resolution and graph construction.",
    )
    parser.add_argument("--influence-depth", type=int, default=3)
    parser.add_argument("--influence-decay", type=float, default=0.5)
    parser.add_argument("--output-prefix", required=True, help="Output prefix without extension")
    args = parser.parse_args()

    excluded = {paper.strip() for paper in args.exclude_papers if paper.strip()}

    if args.load_mappings and Path(args.load_mappings).exists():
        resolver = CitationResolver.load(args.load_mappings)
        resolver.parse_all(args.results, args.papers, exclude_papers=excluded)
    else:
        resolver = CitationResolver()
        resolver.parse_all(args.results, args.papers, exclude_papers=excluded)
    resolver.register_corpus_papers(args.results, args.papers, exclude_papers=excluded)
    if args.save_mappings:
        resolver.save(args.save_mappings)

    fw = KnowledgeDiscoveryFramework.from_results_dir(
        args.results,
        citation_mappings=resolver.build_citation_mappings(),
        model_tag=args.model_tag,
        exclude_papers=excluded,
    )

    step6_scores, step6_mass = fw.corpus_contribution._compute_with_mass()
    propagated_mass = fw.corpus_contribution.propagated_influence_mass()
    normalized_influence = fw.corpus_contribution.normalized_influence()

    step6b_scores, step6b_mass = fw.corpus_contribution._compute_source_weighted_with_mass()
    step6b_norm = fw.corpus_contribution.normalized_source_weighted_influence()

    seminal_scores = fw.influence.seminal_scores(
        restrict_to_corpus=True,
        max_depth=args.influence_depth,
        decay=args.influence_decay,
    )
    reference_scores = fw.pagerank.compute()

    nodes = _node_rows(
        fw,
        step6_scores=step6_scores,
        step6_mass=step6_mass,
        propagated_mass=propagated_mass,
        normalized_influence=normalized_influence,
        step6b_scores=step6b_scores,
        step6b_mass=step6b_mass,
        step6b_norm=step6b_norm,
        seminal_scores=seminal_scores,
        reference_scores=reference_scores,
    )
    edges = _edge_rows(
        fw,
        source_mass=step6_mass,
        source_weighted_scores=step6b_scores,
    )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    nodes_csv = output_prefix.with_name(output_prefix.name + "_nodes.csv")
    edges_csv = output_prefix.with_name(output_prefix.name + "_edges.csv")
    bundle_json = output_prefix.with_name(output_prefix.name + ".json")

    _write_csv(nodes_csv, nodes)
    _write_csv(edges_csv, edges)

    bundle = {
        "metadata": {
            "results_dir": args.results,
            "papers_dir": args.papers,
            "model_tag": args.model_tag,
            "exclude_papers": sorted(excluded),
            "influence_depth": args.influence_depth,
            "influence_decay": args.influence_decay,
            "corpus_size": len(fw.graph.papers),
            "internal_edge_count": len(edges),
        },
        "nodes": nodes,
        "edges": edges,
    }
    with bundle_json.open("w") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)

    print(f"Wrote {nodes_csv}")
    print(f"Wrote {edges_csv}")
    print(f"Wrote {bundle_json}")
    print(f"Corpus papers: {len(fw.graph.papers)}")
    print(f"Internal weighted edges: {len(edges)}")


if __name__ == "__main__":
    main()
