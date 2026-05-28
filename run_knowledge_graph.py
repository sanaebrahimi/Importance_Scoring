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
    parser.add_argument(
        "--model-tag",
        default="",
        help="Optional model tag for selecting tagged score files during graph analysis (e.g. llama3_2).",
    )
    parser.add_argument(
        "--exclude-papers",
        nargs="+",
        default=[],
        metavar="PAPER_ID",
        help="Optional paper ids to exclude from citation resolution and graph construction.",
    )
    parser.add_argument("--save-mappings", default="",
                        help="Save resolved citation mappings to this JSON file")
    parser.add_argument("--load-mappings", default="",
                        help="Load pre-saved mappings instead of re-parsing PDFs")
    parser.add_argument(
        "--llm-repair-model",
        default="",
        help="Optional Ollama model used to repair low-confidence reference titles.",
    )
    parser.add_argument(
        "--llm-repair-host",
        default="http://localhost:11434",
        help="Ollama host for low-confidence reference repair.",
    )
    parser.add_argument(
        "--llm-repair-min-confidence",
        type=float,
        default=0.60,
        help="Minimum LLM confidence required to accept repaired reference metadata.",
    )
    parser.add_argument(
        "--llm-repair-max-calls",
        type=int,
        default=200,
        help="Maximum number of LLM repair calls during one resolver run.",
    )
    parser.add_argument("--top-k",    type=int, default=10,     help="Top-K for ranked lists")
    parser.add_argument("--pagerank-damping", type=float, default=0.85,
                        help="Deprecated — no longer used")
    parser.add_argument("--influence-depth",  type=int,   default=3)
    parser.add_argument("--influence-decay",  type=float, default=0.5)
    args = parser.parse_args()
    excluded = {paper.strip() for paper in args.exclude_papers if paper.strip()}

    # ------------------------------------------------------------------ #
    # Step 1 – Citation resolver                                          #
    # ------------------------------------------------------------------ #
    header("STEP 1 — Citation Resolver")

    resolver_kwargs = dict(
        llm_repair_model=args.llm_repair_model,
        llm_repair_host=args.llm_repair_host,
        llm_repair_min_confidence=args.llm_repair_min_confidence,
        llm_repair_max_calls=args.llm_repair_max_calls,
    )

    if args.load_mappings and Path(args.load_mappings).exists():
        print(f"Loading pre-saved mappings from {args.load_mappings}")
        resolver = CitationResolver.load(args.load_mappings, **resolver_kwargs)
        resolver.parse_all(args.results, args.papers, exclude_papers=excluded)
    else:
        resolver = CitationResolver(**resolver_kwargs)
        resolver.parse_all(args.results, args.papers, exclude_papers=excluded)

    resolver.register_corpus_papers(args.results, args.papers, exclude_papers=excluded)

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
        model_tag=args.model_tag,
        exclude_papers=excluded,
    )
    print(f"\n{fw.graph}")

    audit = fw.graph.internal_citation_audit()
    if audit:
        subheader("Internal citation audit")
        paper_scored = sum(1 for item in audit if item.status == "paper_score")
        paragraph_fallback = sum(1 for item in audit if item.status == "paragraph_fallback")
        missing = sum(1 for item in audit if item.status == "missing")
        ignored_dupes = sum(1 for item in audit if item.status == "ignored_duplicate_missing")
        print(f"  Internal citation candidates : {len(audit)}")
        print(f"  Using paper-level scores      : {paper_scored}")
        print(f"  Using paragraph fallback      : {paragraph_fallback}")
        print(f"  Missing after fallback        : {missing}")
        print(f"  Ignored duplicate missings    : {ignored_dupes}")

        if paragraph_fallback:
            print("\n  Paragraph-fallback edges:")
            for item in sorted(
                (entry for entry in audit if entry.status == "paragraph_fallback"),
                key=lambda entry: (entry.source_paper, entry.target_paper, entry.citation_key),
            ):
                print(
                    f"    {item.source_paper} :: {item.citation_key} -> {item.target_paper} "
                    f"(fallback={item.fallback_score:.9f})"
                )

        if missing:
            print("\n  Still-missing internal citations:")
            for item in sorted(
                (entry for entry in audit if entry.status == "missing"),
                key=lambda entry: (entry.source_paper, entry.target_paper, entry.citation_key),
            ):
                print(f"    {item.source_paper} :: {item.citation_key} -> {item.target_paper}")

    # ------------------------------------------------------------------ #
    # Step 3 – Summary                                                    #
    # ------------------------------------------------------------------ #
    header("STEP 3 — Corpus Summary")

    summary = fw.summary()
    print(f"\n  Corpus papers   : {summary['corpus_papers']}")
    print(f"  Total nodes     : {summary['total_nodes']}")
    print(f"  Total edges     : {summary['total_edges']}")
    print(f"  Total weight    : {summary['total_weight']:.9f}")
    print(f"  Avg originality : {summary['avg_originality']:.9f}")

    subheader("Per-paper breakdown")
    print(f"  {'Paper':<35} {'Originality':>15}  {'Cit. Import.':>15}  {'# Citations':>11}")
    print(f"  {'-'*35} {'-'*15}  {'-'*15}  {'-'*11}")
    for pid, d in summary["papers"].items():
        print(
            f"  {pid:<35} {d['originality']:>15.9f}  "
            f"{d['citation_importance']:>15.9f}  {d['unique_citations']:>11}"
        )

    # ------------------------------------------------------------------ #
    # Step 4 – Originality ranking                                        #
    # ------------------------------------------------------------------ #
    header("STEP 4 — Originality Ranking  (Orig(p) = Σ S_tech)")

    print(f"\n  Papers ranked by original technical content (higher = more original):\n")
    for rank, (pid, score) in enumerate(fw.originality.rank_by_originality(), 1):
        bar = "█" * int(score * 30)
        print(f"  {rank}. {pid:<35} {score:.9f}  {bar}")

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
        print(f"  {rank:>2}. {score:.9f}  {label}")

    # ------------------------------------------------------------------ #
    # Step 6 – Contribution Analyzer                                      #
    # ------------------------------------------------------------------ #
    header("STEP 6 — Contribution Analyzer")

    print(
        "\n  Corpus-level total score:\n"
        "  σ_P(p) = σ_tech(p) · (1 + Σ_{paths a→p} Π_{edges} W(u,v))\n"
    )
    contrib_scores = fw.corpus_contribution.compute()
    top_contrib = sorted(contrib_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"  {'Paper':<35} {'σ_P':>15}  {'σ_tech':>15}")
    print(f"  {'-'*35} {'-'*15}  {'-'*15}")
    for pid, score in top_contrib:
        orig = fw.graph.papers[pid].originality_score()
        print(f"  {pid:<35} {score:>15.9f}  {orig:>15.9f}")

    subheader("Normalized Influence Score  I_P(p)")
    print(
        "\n  Propagated cross-paper mass:\n"
        "  π_P(p) = σ_tech(p) · Σ_{paths a→p} Π_{edges} W(u,v)\n"
        "  I_P(p) = π_P(p) / Σ_{p'} π_P(p')  (share of propagated cross-paper influence)\n"
    )
    norm_influence = fw.corpus_contribution.normalized_influence()
    propagated_mass = fw.corpus_contribution.propagated_influence_mass()
    top_influence  = sorted(norm_influence.items(), key=lambda x: x[1], reverse=True)
    print(f"  {'Paper':<35} {'π_P(p)':>15}  {'I_P(p)':>15}")
    print(f"  {'-'*35} {'-'*15}  {'-'*15}")
    for pid, inf_score in top_influence:
        propagated = propagated_mass[pid]
        print(f"  {pid:<35} {propagated:>15.9f}  {inf_score:>15.9f}")

    # ------------------------------------------------------------------ #
    # Step 6B – Source-weighted contribution analyzer                     #
    # ------------------------------------------------------------------ #
    header("STEP 6B — Source-Weighted Contribution Analyzer")

    print(
        "\n  Corpus-level total score:\n"
        "  σ_P(p) = σ_tech(p) + Σ_{paths a→p} σ_tech(a) · Π_{edges} W(u,v)\n"
    )
    src_contrib_scores = fw.corpus_contribution.compute_source_weighted()
    top_src_contrib = sorted(src_contrib_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"  {'Paper':<35} {'σ_P':>15}  {'σ_tech':>15}")
    print(f"  {'-'*35} {'-'*15}  {'-'*15}")
    for pid, score in top_src_contrib:
        orig = fw.graph.papers[pid].originality_score()
        print(f"  {pid:<35} {score:>15.9f}  {orig:>15.9f}")

    subheader("Normalized Source-Weighted Influence Score  I_P^src(p)")
    print(
        "\n  Propagated cross-paper mass:\n"
        "  π_P^src(p) = Σ_{paths a→p} σ_tech(a) · Π_{edges} W(u,v)\n"
        "  I_P^src(p) = π_P^src(p) / Σ_{p'} π_P^src(p')\n"
    )
    src_norm_influence = fw.corpus_contribution.normalized_source_weighted_influence()
    src_propagated_mass = fw.corpus_contribution.source_weighted_propagated_mass()
    top_src_influence = sorted(src_norm_influence.items(), key=lambda x: x[1], reverse=True)
    print(f"  {'Paper':<35} {'π_P^src(p)':>18}  {'I_P^src(p)':>18}")
    print(f"  {'-'*35} {'-'*18}  {'-'*18}")
    for pid, inf_score in top_src_influence:
        propagated = src_propagated_mass[pid]
        print(f"  {pid:<35} {propagated:>18.9e}  {inf_score:>18.9e}")

    # ------------------------------------------------------------------ #
    # Step 7 – Seminal works (influence propagation)                      #
    # ------------------------------------------------------------------ #
    header("STEP 7 — Seminal Works  (Influence Propagation)")

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
        print(f"  {rank:>2}. {score:.9f}  {label}")

    # ------------------------------------------------------------------ #
    # Step 8 – Section-level citation roles                               #
    # ------------------------------------------------------------------ #
    header("STEP 8 — Citation Roles  (Technical vs Rhetorical)")

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
            print(f"    {author_year:<28}  w={total_w:.9f}  ρ={rho_str:<6}  [{label}]")
        print()

    # ------------------------------------------------------------------ #
    # Step 9 – Foundational vs Peripheral                                 #
    # ------------------------------------------------------------------ #
    header("STEP 9 — Foundational vs Peripheral Works")

    print(f"\n  Top-{args.top_k} works by foundational score "
          f"(cited in methods/experiments):\n")
    found_data = fw.foundational.compute_all()
    top_found = sorted(
        found_data.items(), key=lambda x: x[1]["foundational_score"], reverse=True
    )[: args.top_k]
    print(f"  {'Work':<32} {'Found':>13}  {'Periph':>13}  {'Class':<14}")
    print(f"  {'-'*32} {'-'*13}  {'-'*13}  {'-'*14}")
    for cit, d in top_found:
        entry = resolver.resolve(cit)
        label = (
            f"{entry.first_author_last}_{entry.year}"
            if entry else cit[:28]
        )
        print(
            f"  {label:<32} {d['foundational_score']:>13.9f}  "
            f"{d['peripheral_score']:>13.9f}  {d['classification']}"
        )

    # ------------------------------------------------------------------ #
    # Step 10 – Community detection                                       #
    # ------------------------------------------------------------------ #
    header("STEP 10 — Community Detection")

    partition_dir  = fw.communities.detect(restrict_to_corpus=True, directed=True)
    partition_undir = fw.communities.detect(restrict_to_corpus=True, directed=False)

    q_directed = fw.communities.modularity(partition_dir,   directed=True)
    q_undir    = fw.communities.modularity(partition_undir, directed=False)

    communities_dir: dict = {}
    for pid, cid in partition_dir.items():
        communities_dir.setdefault(cid, []).append(pid)

    communities_undir: dict = {}
    for pid, cid in partition_undir.items():
        communities_undir.setdefault(cid, []).append(pid)

    print(f"\n  Weighted modularity Q (directed)   = {q_directed:.9f}")
    print(f"  {len(communities_dir)} communities detected (directed):\n")
    for cid, members in sorted(communities_dir.items()):
        print(f"  Community {cid}: {', '.join(members)}")

    print(f"\n  Weighted modularity Q (undirected) = {q_undir:.9f}")
    print(f"  {len(communities_undir)} communities detected (undirected):\n")
    for cid, members in sorted(communities_undir.items()):
        print(f"  Community {cid}: {', '.join(members)}")

    subheader("Community-detection baselines")

    baseline_specs = [
        ("binary", "Binary citation graph"),
        ("mention_count", "Raw citation-mention count graph"),
    ]
    for weighting, label in baseline_specs:
        partition = fw.communities.detect_baseline(weighting, restrict_to_corpus=True)
        q_baseline = fw.communities.baseline_modularity(
            partition,
            weighting,
            restrict_to_corpus=True,
        )
        communities: dict = {}
        for pid, cid in partition.items():
            communities.setdefault(cid, []).append(pid)

        print(f"\n  {label}:")
        print(f"    Weighted modularity Q = {q_baseline:.9f}")
        print(f"    {len(communities)} communities detected:\n")
        for cid, members in sorted(communities.items()):
            print(f"    Community {cid}: {', '.join(members)}")

    # ------------------------------------------------------------------ #
    # Step 11 – Research gap                                              #
    # ------------------------------------------------------------------ #
    header("STEP 11 — Research Gap Score  (Gap(C, t))")

    all_papers = list(fw.graph.papers.keys())
    gap_full   = fw.research_gaps.gap_score(all_papers)
    print(f"\n  Full-corpus gap score: {gap_full:.9f}")
    print(
        "  (positive = technically original work not yet adopted "
        "as a dependency by other papers in the corpus)"
    )

    print(f"\n  Per-paper gap score (cluster = single paper):\n")
    print(f"  {'Paper':<35} {'Gap score':>15}  {'Originality':>15}")
    print(f"  {'-'*35} {'-'*15}  {'-'*15}")
    for pid in sorted(fw.graph.papers):
        g  = fw.research_gaps.gap_score([pid])
        o  = fw.originality.originality(pid)
        print(f"  {pid:<35} {g:>15.9f}  {o:>15.9f}")

    header("DONE")
    print(f"\n  All analyses complete.\n")


if __name__ == "__main__":
    main()
