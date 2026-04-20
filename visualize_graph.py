"""
visualize_graph.py — Interactive HTML citation graph.

Node types:
  • Corpus papers  (blue, community-tinted) — every paper we have full data for
  • External cited works  (amber) — top-K resolved works by PageRank

Node size   → PageRank score
Node color  → community (corpus) or amber (external)
Edge width  → importance weight W(p, q)
Hover       → full metadata + scores

Usage:
    python3 visualize_graph.py
    python3 visualize_graph.py --top-k 30 --min-weight 0.002
    python3 visualize_graph.py --load-mappings citation_mappings.json --output graph.html

Requirements:
    pip install pyvis
"""

import argparse
import re
from pathlib import Path

try:
    from pyvis.network import Network
except ImportError:
    raise SystemExit("Install pyvis:  pip install pyvis")

from citation_resolver import CitationResolver
from citation_graph_framework import KnowledgeDiscoveryFramework


# ── colour palette ────────────────────────────────────────────────────────────
# One colour per community for corpus nodes
_COMMUNITY_COLOURS = [
    "#4e9af1",  # blue
    "#4ef17a",  # green
    "#f1a14e",  # orange
    "#a14ef1",  # purple
    "#f14e4e",  # red
    "#f1e84e",  # yellow
    "#4ef1e8",  # cyan
    "#f14ea1",  # pink
]
_EXTERNAL_COLOUR = "#f0a500"   # amber for external cited works
_EDGE_COLOUR     = "#666666"


def _scale(val: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi <= lo:
        return (out_lo + out_hi) / 2
    return out_lo + (val - lo) / (hi - lo) * (out_hi - out_lo)


# ── build ─────────────────────────────────────────────────────────────────────

def build(
    fw: KnowledgeDiscoveryFramework,
    resolver: CitationResolver,
    top_k: int,
    min_weight: float,
) -> Network:

    net = Network(
        height="920px",
        width="100%",
        bgcolor="#0f0f1a",
        font_color="#e8e8e8",
        directed=True,
        notebook=False,
    )

    # ── analytics ─────────────────────────────────────────────────────────
    pagerank   = fw.pagerank.compute(damping=0.85)
    corpus_ids = set(fw.graph.papers.keys())
    partition  = fw.communities.detect(restrict_to_corpus=True)   # {pid: community_id}

    # Pick top-K external nodes by PageRank
    external_pr  = {n: s for n, s in pagerank.items() if n not in corpus_ids}
    top_external = set(
        sorted(external_pr, key=lambda x: external_pr[x], reverse=True)[:top_k]
    )
    included = corpus_ids | top_external

    # Scale node sizes
    pr_vals      = [pagerank.get(n, 0.0) for n in included]
    pr_min, pr_max = min(pr_vals, default=0.0), max(pr_vals, default=1.0)

    # ── nodes ─────────────────────────────────────────────────────────────
    for node_id in included:
        pr   = pagerank.get(node_id, 0.0)
        size = _scale(pr, pr_min, pr_max, 14, 60)

        if node_id in corpus_ids:
            paper     = fw.graph.papers[node_id]
            orig      = paper.originality_score()
            community = partition.get(node_id, 0)
            colour    = _COMMUNITY_COLOURS[community % len(_COMMUNITY_COLOURS)]
            label     = node_id
            tooltip   = "\n".join([
                node_id,
                f"Year: —",
                "corpus paper",
            ])
        else:
            entry  = resolver.resolve(node_id)
            colour = _EXTERNAL_COLOUR
            label  = node_id
            if entry:
                authors = ", ".join(entry.authors[:3])
                if len(entry.authors) > 3:
                    authors += " et al."
                title = re.sub(r"<[^>]+>", "", entry.title)
                tooltip = "\n".join([
                    title,
                    f"Year: {entry.year}",
                    authors,
                ])
            else:
                tooltip = node_id

        net.add_node(
            node_id,
            label=label,
            title=tooltip,
            color={"background": colour, "border": "#ffffff",
                   "highlight": {"background": "#ffffff", "border": colour}},
            size=size,
            borderWidth=1.5,
            font={"size": 10, "face": "monospace"},
        )

    # ── edges ─────────────────────────────────────────────────────────────
    all_edges  = fw.graph.edges()                            # {(src,dst): weight}
    valid_w    = [w for (s, d), w in all_edges.items()
                  if s in included and d in included and w >= min_weight]
    w_max      = max(valid_w, default=1.0)

    for (src, dst), weight in all_edges.items():
        if src not in included or dst not in included:
            continue
        if weight < min_weight:
            continue

        thickness = max(0.5, weight / w_max * 8)
        entry     = resolver.resolve(dst)
        dst_label = f"{entry.first_author_last} ({entry.year})" if entry else dst

        net.add_edge(
            src, dst,
            value=thickness,
            title=f"<b>{src}</b> → {dst_label}<br>weight: {weight:.4f}",
            color={"color": _EDGE_COLOUR, "opacity": 0.55,
                   "highlight": "#ffffff"},
            arrows="to",
        )

    # ── physics + interaction options ─────────────────────────────────────
    net.set_options("""
    {
      "physics": {
        "solver": "barnesHut",
        "barnesHut": {
          "gravitationalConstant": -9000,
          "centralGravity": 0.25,
          "springLength": 220,
          "springConstant": 0.04,
          "damping": 0.12
        },
        "maxVelocity": 60,
        "minVelocity": 0.5,
        "stabilization": { "iterations": 250 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 80,
        "navigationButtons": true,
        "keyboard": { "enabled": true }
      },
      "edges": {
        "smooth": { "type": "curvedCW", "roundness": 0.15 },
        "scaling": { "min": 0.5, "max": 8 }
      },
      "nodes": {
        "scaling": { "min": 14, "max": 60 },
        "shadow": { "enabled": true, "size": 6 }
      }
    }
    """)

    return net


# ── legend injected into the HTML ────────────────────────────────────────────

def _inject_legend(html_path: Path, corpus_ids: set, partition: dict) -> None:
    """Append a small legend div to the generated HTML."""
    used_communities = sorted({partition.get(p, 0) for p in corpus_ids})
    items = []
    for c in used_communities:
        col = _COMMUNITY_COLOURS[c % len(_COMMUNITY_COLOURS)]
        items.append(
            f'<span style="display:inline-flex;align-items:center;margin-right:12px">'
            f'<span style="width:12px;height:12px;border-radius:50%;background:{col};'
            f'display:inline-block;margin-right:4px"></span>Community {c}</span>'
        )
    items.append(
        f'<span style="display:inline-flex;align-items:center">'
        f'<span style="width:12px;height:12px;border-radius:50%;background:{_EXTERNAL_COLOUR};'
        f'display:inline-block;margin-right:4px"></span>External cited work</span>'
    )
    legend_html = (
        '<div style="position:fixed;bottom:16px;left:16px;background:rgba(0,0,0,0.7);'
        'color:#e8e8e8;padding:10px 14px;border-radius:8px;font-family:monospace;'
        'font-size:12px;z-index:9999">'
        + "".join(items)
        + "<br><span style='opacity:0.6;font-size:11px'>"
        "Node size = Normalized Reference Score &nbsp;|&nbsp; Edge width = citation importance weight"
        "</span></div>"
    )
    text = html_path.read_text(encoding="utf-8")
    text = text.replace("</body>", legend_html + "\n</body>")
    html_path.write_text(text, encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML citation graph."
    )
    parser.add_argument("--results",       default="paper_results/",
                        help="Results directory")
    parser.add_argument("--papers",        default="papers/",
                        help="PDF directory (for citation resolver)")
    parser.add_argument("--load-mappings", default="",
                        help="Load pre-saved citation_mappings.json")
    parser.add_argument("--save-mappings", default="",
                        help="Save resolved mappings to this file")
    parser.add_argument("--top-k",         type=int,   default=20,
                        help="Number of top external cited works to include (default 20)")
    parser.add_argument("--min-weight",    type=float, default=0.001,
                        help="Minimum edge weight to display (default 0.001)")
    parser.add_argument("--output",        default="graph.html",
                        help="Output HTML file (default graph.html)")
    args = parser.parse_args()

    # ── resolver ──────────────────────────────────────────────────────────
    if args.load_mappings and Path(args.load_mappings).exists():
        print(f"Loading mappings from {args.load_mappings} ...")
        resolver = CitationResolver.load(args.load_mappings)
    else:
        print("Parsing PDFs to resolve citations ...")
        resolver = CitationResolver()
        resolver.parse_all(args.results, args.papers)

    resolver.register_corpus_papers(args.results, args.papers)

    if args.save_mappings:
        resolver.save(args.save_mappings)
        print(f"Mappings saved to {args.save_mappings}")

    mappings = resolver.build_citation_mappings()

    # ── framework ─────────────────────────────────────────────────────────
    fw = KnowledgeDiscoveryFramework.from_results_dir(
        args.results, citation_mappings=mappings
    )

    corpus_ids = set(fw.graph.papers.keys())
    partition  = fw.communities.detect(restrict_to_corpus=True)
    print(f"Corpus: {len(corpus_ids)} papers  |  "
          f"Total graph nodes: {len(fw.graph.all_nodes())}")
    print(f"Including top-{args.top_k} external works by PageRank, "
          f"min edge weight {args.min_weight}")

    # ── build + save ──────────────────────────────────────────────────────
    net = build(fw, resolver, top_k=args.top_k, min_weight=args.min_weight)
    net.show(args.output, notebook=False)

    _inject_legend(Path(args.output), corpus_ids, partition)
    print(f"\nGraph saved → {args.output}")
    print("Open it in any browser.")


if __name__ == "__main__":
    main()
