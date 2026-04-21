"""
Knowledge graph runner — resolves citations, builds the graph, and prints
all analysis results to the terminal.

Usage:
    python3 run_knowledge_graph.py
    python3 run_knowledge_graph.py --results paper_results/ --papers papers/
    python3 run_knowledge_graph.py --save-mappings citation_mappings.json
"""

import argparse
from pathlib import Path

from citation_resolver import CitationResolver
from citation_graph_framework import KnowledgeDiscoveryFramework


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)

def subheader(title: str) -> None:
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run knowledge-graph analyses.")
    parser.add_argument("--results",  default="paper_results/", help="Results directory")
    parser.add_argument("--papers",   default="papers/",        help="PDF directory")
    parser.add_argument("--save-mappings", default="",
                        help="Save resolved citation mappings to this JSON file")
    parser.add_argument("--load-mappings", default="",
                        help="Load pre-saved mappings instead of re-parsing PDFs")
    parser.add_argument("--top-k",    type=int, default=10,     help="Top-K for ranked lists")
    parser.add_argument("--pagerank-damping", type=float, default=0.85,
                        help="Deprecated — no longer used")
    parser.add_argument("--influence-depth",  type=int,   default=3)
    parser.add_argument("--influence-decay",  type=float, default=0.5)
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Step 1 – Citation resolver                                          #
    # ------------------------------------------------------------------ #
    header("STEP 1 — Citation Resolver")

    if args.load_mappings and Path(args.load_mappings).exists():
        print(f"Loading pre-saved mappings from {args.load_mappings}")
        resolver = CitationResolver.load(args.load_mappings)
        resolver.parse_all(args.results, args.papers)
    else:
        resolver = CitationResolver()
        resolver.parse_all(args.results, args.papers)

    resolver.register_corpus_papers(args.results, args.papers)

    if args.save_mappings:
        resolver.save(args.save_mappings)

    print(f"\n{resolver}")
    mappings = resolver.build_citation_mappings()

    # show a sample of resolved entries
    subheader("Sample resolved citations")
    for (pid, cit), entry in list(resolver.all_resolved().items())[:8]:
        print(f"  [{pid}] {cit!r:30s} → [{entry.canonical_id}]  {entry.title[:45]}")

    # ------------------------------------------------------------------ #
    # Step 2 – Build framework                                            #
    # ------------------------------------------------------------------ #
    header("STEP 2 — Build Citation Graph")

    fw = KnowledgeDiscoveryFramework.from_results_dir(
        args.results,
        citation_mappings=mappings,
    )
    print(f"\n{fw.graph}")

    # ------------------------------------------------------------------ #
    # Step 3 – Summary                                                    #
    # ------------------------------------------------------------------ #
    header("STEP 3 — Corpus Summary")

    summary = fw.summary()
    print(f"\n  Corpus papers   : {summary['corpus_papers']}")
    print(f"  Total nodes     : {summary['total_nodes']}")
    print(f"  Total edges     : {summary['total_edges']}")
    print(f"  Total weight    : {summary['total_weight']:.4f}")
    print(f"  Avg originality : {summary['avg_originality']:.4f}")

    subheader("Per-paper breakdown")
    print(f"  {'Paper':<35} {'Originality':>11}  {'Cit. Import.':>12}  {'# Citations':>11}")
    print(f"  {'-'*35} {'-'*11}  {'-'*12}  {'-'*11}")
    for pid, d in summary["papers"].items():
        print(
            f"  {pid:<35} {d['originality']:>11.4f}  "
            f"{d['citation_importance']:>12.4f}  {d['unique_citations']:>11}"
        )

    # ------------------------------------------------------------------ #
    # Step 4 – Originality ranking                                        #
    # ------------------------------------------------------------------ #
    header("STEP 4 — Originality Ranking  (Orig(p) = Σ S_tech)")

    print(f"\n  Papers ranked by original technical content (higher = more original):\n")
    for rank, (pid, score) in enumerate(fw.originality.rank_by_originality(), 1):
        bar = "█" * int(score * 30)
        print(f"  {rank}. {pid:<35} {score:.4f}  {bar}")

    # ------------------------------------------------------------------ #
    # Step 5 – Importance-Weighted PageRank                               #
    # ------------------------------------------------------------------ #
    header("STEP 5 — Importance-Weighted PageRank")

    print(f"\n  Top-{args.top_k} cited works by Normalized Reference Score:\n")
    ipr = fw.pagerank.compute()
    top_pr = sorted(ipr.items(), key=lambda x: x[1], reverse=True)[: args.top_k]
    for rank, (node, score) in enumerate(top_pr, 1):
        entry = resolver.resolve(node)
        label = (
            f"{entry.first_author_last} ({entry.year})  {entry.title[:40]}"
            if entry else node
        )
        print(f"  {rank:>2}. {score:.6f}  {label}")

    # ------------------------------------------------------------------ #
    # Step 6 – Seminal works (influence propagation)                      #
    # ------------------------------------------------------------------ #
    header("STEP 6 — Seminal Works  (Influence Propagation)")

    print(
        f"\n  Top cited works by total downstream influence "
        f"(depth={args.influence_depth}, decay={args.influence_decay}):\n"
    )
    seminal = fw.influence.seminal_works(
        top_k=args.top_k,
        restrict_to_corpus=False,
        max_depth=args.influence_depth,
        decay=args.influence_decay,
    )
    for rank, (node, score) in enumerate(seminal, 1):
        entry = resolver.resolve(node)
        label = (
            f"{entry.first_author_last} ({entry.year})  {entry.title[:40]}"
            if entry else node
        )
        print(f"  {rank:>2}. {score:.6f}  {label}")

    # ------------------------------------------------------------------ #
    # Step 7 – Section-level citation roles                               #
    # ------------------------------------------------------------------ #
    header("STEP 7 — Citation Roles  (Technical vs Rhetorical)")

    print(f"\n  For each corpus paper: top citations and their role.\n")
    for pid, paper in fw.graph.papers.items():
        top_cits = sorted(paper.citation_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  {pid}")
        for cit, total_w in top_cits:
            rho, label = fw.section_citations.classify_citation(pid, cit)
            entry = resolver.resolve(cit)
            author_year = (
                f"{entry.first_author_last} ({entry.year})"
                if entry else cit[:30]
            )
            rho_str = f"{rho:.2f}" if rho != float("inf") else "∞"
            print(f"    {author_year:<28}  w={total_w:.4f}  ρ={rho_str:<6}  [{label}]")
        print()

    # ------------------------------------------------------------------ #
    # Step 8 – Foundational vs Peripheral                                 #
    # ------------------------------------------------------------------ #
    header("STEP 8 — Foundational vs Peripheral Works")

    print(f"\n  Top-{args.top_k} works by foundational score "
          f"(cited in methods/experiments):\n")
    found_data = fw.foundational.compute_all()
    top_found = sorted(
        found_data.items(), key=lambda x: x[1]["foundational_score"], reverse=True
    )[: args.top_k]
    print(f"  {'Work':<32} {'Found':>7}  {'Periph':>7}  {'Class':<14}")
    print(f"  {'-'*32} {'-'*7}  {'-'*7}  {'-'*14}")
    for cit, d in top_found:
        entry = resolver.resolve(cit)
        label = (
            f"{entry.first_author_last}_{entry.year}"
            if entry else cit[:28]
        )
        print(
            f"  {label:<32} {d['foundational_score']:>7.4f}  "
            f"{d['peripheral_score']:>7.4f}  {d['classification']}"
        )

    # ------------------------------------------------------------------ #
    # Step 9 – Community detection                                        #
    # ------------------------------------------------------------------ #
    header("STEP 9 — Community Detection")

    partition = fw.communities.detect(restrict_to_corpus=True)
    q_score   = fw.communities.modularity(partition)
    communities: dict = {}
    for pid, cid in partition.items():
        communities.setdefault(cid, []).append(pid)

    print(f"\n  Weighted modularity Q = {q_score:.4f}")
    print(f"  {len(communities)} communities detected:\n")
    for cid, members in sorted(communities.items()):
        print(f"  Community {cid}: {', '.join(members)}")

    # ------------------------------------------------------------------ #
    # Step 10 – Research gap                                              #
    # ------------------------------------------------------------------ #
    header("STEP 10 — Research Gap Score  (Gap(C, t))")

    all_papers = list(fw.graph.papers.keys())
    gap_full   = fw.research_gaps.gap_score(all_papers)
    print(f"\n  Full-corpus gap score: {gap_full:.4f}")
    print(
        "  (positive = technically original work not yet adopted "
        "as a dependency by other papers in the corpus)"
    )

    print(f"\n  Per-paper gap score (cluster = single paper):\n")
    print(f"  {'Paper':<35} {'Gap score':>10}  {'Originality':>11}")
    print(f"  {'-'*35} {'-'*10}  {'-'*11}")
    for pid in sorted(fw.graph.papers):
        g  = fw.research_gaps.gap_score([pid])
        o  = fw.originality.originality(pid)
        print(f"  {pid:<35} {g:>10.4f}  {o:>11.4f}")

    header("DONE")
    print(f"\n  All analyses complete.\n")


if __name__ == "__main__":
    main()
