"""
visualize_graph.py - Generate an interactive HTML citation graph.

The generated page preloads the full graph and lets the viewer change
browser-side filters such as:
  - top-k external cited works by PageRank
  - minimum edge weight

Usage:
    python3 visualize_graph.py
    python3 visualize_graph.py --top-k 30 --min-weight 0.002
    python3 visualize_graph.py --model-tag llama3_2 --top-k 40 --min-weight 0.005
    python3 visualize_graph.py --load-mappings citation_mappings.json --output graph.html
"""

import argparse
import colorsys
import json
import re
from pathlib import Path

from citation_resolver import CitationResolver
from citation_graph_framework import KnowledgeDiscoveryFramework


_COMMUNITY_COLOURS = [
    "#2E86DE",
    "#E74C3C",
    "#16A085",
    "#8E44AD",
    "#F39C12",
    "#00B8D4",
    "#D81B60",
    "#7CB342",
    "#6D4C41",
    "#3949AB",
    "#FF7043",
    "#00897B",
    "#C2185B",
    "#9C27B0",
    "#43A047",
    "#5C6BC0",
    "#EF5350",
    "#26A69A",
    "#FFB300",
    "#1E88E5",
    "#8D6E63",
    "#EC407A",
    "#66BB6A",
    "#AB47BC",
]
_EXTERNAL_COLOUR = "#f0a500"
_EDGE_COLOUR = "#666666"


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#{:02X}{:02X}{:02X}".format(
        max(0, min(255, round(r * 255))),
        max(0, min(255, round(g * 255))),
        max(0, min(255, round(b * 255))),
    )


def _community_colour(community: int) -> str:
    if community < len(_COMMUNITY_COLOURS):
        return _COMMUNITY_COLOURS[community]

    # Golden-angle hue stepping keeps later colors well separated even when the
    # number of detected communities exceeds the curated palette.
    hue = ((community - len(_COMMUNITY_COLOURS)) * 0.61803398875) % 1.0
    saturation = 0.72
    value = 0.92 if community % 2 == 0 else 0.78
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return _rgb_to_hex(r, g, b)


def _scale(val: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi <= lo:
        return (out_lo + out_hi) / 2
    return out_lo + (val - lo) / (hi - lo) * (out_hi - out_lo)


def _corpus_tooltip(node_id: str) -> str:
    return "\n".join([node_id, "Year: -", "corpus paper"])


def _external_label_and_tooltip(resolver: CitationResolver, node_id: str) -> tuple[str, str]:
    entry = resolver.resolve(node_id)
    if not entry:
        return node_id, node_id

    authors = ", ".join(entry.authors[:3])
    if len(entry.authors) > 3:
        authors += " et al."
    title = re.sub(r"<[^>]+>", "", entry.title)

    label = node_id
    if entry.first_author_last and entry.year:
        label = f"{entry.first_author_last} ({entry.year})"

    tooltip = "\n".join([title, f"Year: {entry.year}", authors])
    return label, tooltip


def build_graph_payload(
    fw: KnowledgeDiscoveryFramework,
    resolver: CitationResolver,
    model_tag: str = "",
) -> dict:
    pagerank = fw.pagerank.compute()
    corpus_ids = sorted(fw.graph.papers.keys())
    corpus_set = set(corpus_ids)
    partition = fw.communities.detect(restrict_to_corpus=True)
    all_node_ids = sorted(fw.graph.all_nodes())

    external_ranked = sorted(
        (node_id for node_id in all_node_ids if node_id not in corpus_set),
        key=lambda node_id: (-pagerank.get(node_id, 0.0), node_id),
    )

    pr_vals = [pagerank.get(node_id, 0.0) for node_id in all_node_ids]
    pr_min = min(pr_vals, default=0.0)
    pr_max = max(pr_vals, default=1.0)

    nodes = []
    for node_id in all_node_ids:
        pr = pagerank.get(node_id, 0.0)
        size = _scale(pr, pr_min, pr_max, 14, 60)

        if node_id in corpus_set:
            community = partition.get(node_id, 0)
            colour = _community_colour(community)
            label = node_id
            tooltip = _corpus_tooltip(node_id)
        else:
            community = None
            colour = _EXTERNAL_COLOUR
            label, tooltip = _external_label_and_tooltip(resolver, node_id)

        nodes.append(
            {
                "id": node_id,
                "label": label,
                "title": tooltip,
                "size": size,
                "borderWidth": 1.5,
                "font": {"size": 10, "face": "monospace", "color": "#e8e8e8"},
                "color": {
                    "background": colour,
                    "border": "#ffffff",
                    "highlight": {"background": "#ffffff", "border": colour},
                },
                "is_corpus": node_id in corpus_set,
                "community": community,
                "pagerank": pr,
            }
        )

    edges = []
    for (src, dst), weight in sorted(
        fw.graph.edges().items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    ):
        entry = resolver.resolve(dst)
        dst_label = dst
        if entry and entry.first_author_last and entry.year:
            dst_label = f"{entry.first_author_last} ({entry.year})"

        edges.append(
            {
                "from": src,
                "to": dst,
                "raw_weight": weight,
                "title": f"<b>{src}</b> -> {dst_label}<br>weight: {weight:.4f}",
                "color": {
                    "color": _EDGE_COLOUR,
                    "opacity": 0.55,
                    "highlight": "#ffffff",
                },
                "arrows": "to",
            }
        )

    used_communities = sorted({partition.get(paper_id, 0) for paper_id in corpus_ids})
    community_legend = [
        {
            "community": community,
            "color": _community_colour(community),
        }
        for community in used_communities
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "corpus_ids": corpus_ids,
        "external_order": external_ranked,
        "community_legend": community_legend,
        "external_color": _EXTERNAL_COLOUR,
        "graph_stats": {
            "total_nodes": len(all_node_ids),
            "total_edges": len(edges),
            "corpus_nodes": len(corpus_ids),
            "external_nodes": len(external_ranked),
        },
        "model_tag": model_tag,
    }


def render_html(payload: dict, default_top_k: int, default_min_weight: float) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False)
    default_state_json = json.dumps(
        {"top_k": default_top_k, "min_weight": default_min_weight},
        ensure_ascii=False,
    )
    title = "Importance-Weighted Citation Graph"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <link rel="stylesheet" href="lib/vis-9.1.2/vis-network.css">
  <script src="lib/vis-9.1.2/vis-network.min.js"></script>
  <style>
    :root {{
      --bg: #07111f;
      --panel: rgba(8, 20, 38, 0.88);
      --panel-strong: rgba(10, 26, 48, 0.96);
      --text: #e9f0f8;
      --muted: #9fb2c8;
      --line: rgba(170, 194, 219, 0.18);
      --accent: #f0a500;
      --accent-2: #4e9af1;
      --field: rgba(255, 255, 255, 0.05);
      --shadow: 0 18px 42px rgba(0, 0, 0, 0.34);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(78, 154, 241, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(240, 165, 0, 0.12), transparent 24%),
        linear-gradient(180deg, #08111d 0%, #0d1726 45%, #0a121d 100%);
    }}

    .page {{
      width: min(1400px, calc(100vw - 28px));
      margin: 18px auto 24px;
    }}

    .hero {{
      display: grid;
      grid-template-columns: 1.35fr 0.9fr;
      gap: 16px;
      align-items: stretch;
      margin-bottom: 16px;
    }}

    .hero-card,
    .panel,
    .legend {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }}

    .hero-card {{
      padding: 22px 24px;
    }}

    .eyebrow {{
      display: inline-block;
      margin-bottom: 10px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(240, 165, 0, 0.12);
      color: #ffd58d;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}

    h1 {{
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 42px);
      line-height: 1.02;
      letter-spacing: -0.03em;
    }}

    .intro {{
      margin: 0;
      max-width: 70ch;
      color: var(--muted);
      line-height: 1.55;
      font-size: 15px;
    }}

    .panel {{
      padding: 18px;
    }}

    .controls-title,
    .meta-title {{
      margin: 0 0 10px;
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #d2dfec;
    }}

    .controls-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}

    label {{
      display: block;
      margin-bottom: 6px;
      font-size: 13px;
      color: var(--muted);
    }}

    input {{
      width: 100%;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px;
      background: var(--field);
      color: var(--text);
      padding: 11px 12px;
      font-size: 15px;
      outline: none;
    }}

    input:focus {{
      border-color: rgba(78, 154, 241, 0.9);
      box-shadow: 0 0 0 3px rgba(78, 154, 241, 0.18);
    }}

    input[readonly] {{
      opacity: 0.92;
      cursor: default;
    }}

    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }}

    button {{
      appearance: none;
      border: 0;
      border-radius: 999px;
      padding: 11px 16px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 600;
      transition: transform 120ms ease, opacity 120ms ease, background 120ms ease;
    }}

    button:hover {{
      transform: translateY(-1px);
    }}

    .primary {{
      background: linear-gradient(135deg, #ffbf40 0%, #f0a500 100%);
      color: #11161f;
    }}

    .secondary {{
      background: rgba(255, 255, 255, 0.08);
      color: var(--text);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }}

    .hint,
    .status-line,
    .meta-copy {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}

    .status-line {{
      margin-top: 12px;
    }}

    .meta-stack {{
      display: grid;
      gap: 12px;
    }}

    .meta-chip {{
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.06);
    }}

    .meta-value {{
      display: block;
      margin-top: 4px;
      font-size: 22px;
      font-weight: 700;
      color: var(--text);
    }}

    .graph-shell {{
      position: relative;
      overflow: hidden;
      border-radius: 22px;
      border: 1px solid var(--line);
      background: rgba(5, 11, 22, 0.74);
      box-shadow: var(--shadow);
    }}

    #mynetwork {{
      width: 100%;
      height: 920px;
      background:
        radial-gradient(circle at 20% 10%, rgba(78, 154, 241, 0.07), transparent 16%),
        radial-gradient(circle at 80% 0%, rgba(240, 165, 0, 0.06), transparent 18%),
        #0f0f1a;
    }}

    .overlay {{
      position: absolute;
      left: 18px;
      right: 18px;
      top: 18px;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 14px;
      pointer-events: none;
    }}

    .legend,
    .command-box {{
      pointer-events: auto;
      padding: 12px 14px;
      max-width: min(460px, 100%);
    }}

    .legend-items {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      margin-top: 8px;
    }}

    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: var(--text);
    }}

    .legend-dot {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
      flex: 0 0 12px;
      border: 1px solid rgba(255, 255, 255, 0.28);
    }}

    .command-box {{
      background: rgba(7, 15, 28, 0.9);
      border: 1px solid rgba(255, 255, 255, 0.1);
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 12px;
      color: #dce7f2;
    }}

    code {{
      color: #ffd58d;
      word-break: break-word;
    }}

    @media (max-width: 980px) {{
      .page {{
        width: min(100vw - 18px, 1400px);
      }}

      .hero {{
        grid-template-columns: 1fr;
      }}

      .controls-grid {{
        grid-template-columns: 1fr;
      }}

      #mynetwork {{
        height: 74vh;
        min-height: 560px;
      }}

      .overlay {{
        position: static;
        display: grid;
        padding: 14px;
        gap: 12px;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="hero-card">
        <span class="eyebrow">Importance Scoring</span>
        <h1>{title}</h1>
        <p class="intro">
          This page preloads the citation graph and lets you change the same
          filtering knobs you would normally pass on the command line. Update
          <code>top-k</code> and <code>min-weight</code>, then redraw the graph
          without regenerating the site.
        </p>
      </div>
      <div class="panel">
        <h2 class="controls-title">Graph Controls</h2>
        <div class="controls-grid">
          <div>
            <label for="model-tag-input">Scoring model</label>
            <input id="model-tag-input" type="text" readonly>
          </div>
          <div>
            <label for="top-k-input">Top external works by PageRank</label>
            <input id="top-k-input" type="number" min="0" step="1">
          </div>
          <div>
            <label for="min-weight-input">Minimum edge weight</label>
            <input id="min-weight-input" type="number" min="0" step="0.0001">
          </div>
        </div>
        <div class="actions">
          <button class="primary" id="apply-btn" type="button">Update Graph</button>
          <button class="secondary" id="reset-btn" type="button">Reset Defaults</button>
        </div>
        <p class="status-line" id="stats-line"></p>
        <p class="hint">
          The graph keeps all corpus papers visible and adds the highest-ranked
          external works that satisfy the current filters.
        </p>
      </div>
    </section>

    <section class="hero" style="grid-template-columns: 0.9fr 1.35fr;">
      <div class="panel">
        <h2 class="meta-title">Current View</h2>
        <div class="meta-stack">
          <div class="meta-chip">
            Visible nodes
            <span class="meta-value" id="visible-nodes">-</span>
          </div>
          <div class="meta-chip">
            Visible edges
            <span class="meta-value" id="visible-edges">-</span>
          </div>
          <div class="meta-chip">
            Max external candidates
            <span class="meta-value" id="max-external">-</span>
          </div>
        </div>
      </div>
      <div class="panel">
        <h2 class="meta-title">Equivalent Local Command</h2>
        <p class="meta-copy">
          The website filters preloaded data in the browser, but these values
          correspond to the same parameters in the Python generator.
        </p>
        <div class="command-box">
          <code id="command-preview"></code>
        </div>
      </div>
    </section>

    <section class="graph-shell">
      <div class="overlay">
        <div class="legend">
          <div class="meta-title" style="margin-bottom:0">Legend</div>
          <div class="legend-items" id="legend-items"></div>
          <div class="hint" style="margin-top:8px">
            Node size = normalized reference score. Edge width = citation importance weight.
          </div>
        </div>
      </div>
      <div id="mynetwork"></div>
    </section>
  </div>

  <script>
    const GRAPH_PAYLOAD = {payload_json};
    const DEFAULT_STATE = {default_state_json};
    const NETWORK_OPTIONS = {{
      physics: {{
        solver: "barnesHut",
        barnesHut: {{
          gravitationalConstant: -9000,
          centralGravity: 0.25,
          springLength: 220,
          springConstant: 0.04,
          damping: 0.12
        }},
        maxVelocity: 60,
        minVelocity: 0.5,
        stabilization: {{ iterations: 250 }}
      }},
      interaction: {{
        hover: true,
        tooltipDelay: 80,
        navigationButtons: true,
        keyboard: {{ enabled: true }}
      }},
      edges: {{
        smooth: {{ type: "curvedCW", roundness: 0.15 }},
        scaling: {{ min: 0.5, max: 8 }}
      }},
      nodes: {{
        scaling: {{ min: 14, max: 60 }},
        shadow: {{ enabled: true, size: 6 }}
      }}
    }};

    const maxExternal = GRAPH_PAYLOAD.external_order.length;
    const nodesDataset = new vis.DataSet([]);
    const edgesDataset = new vis.DataSet([]);
    const network = new vis.Network(
      document.getElementById("mynetwork"),
      {{ nodes: nodesDataset, edges: edgesDataset }},
      NETWORK_OPTIONS
    );

    function formatNumber(value) {{
      if (value === 0) {{
        return "0";
      }}
      if (Math.abs(value) < 0.01) {{
        return value.toFixed(4);
      }}
      return value.toFixed(3);
    }}

    function clampTopK(value) {{
      if (!Number.isFinite(value)) {{
        return DEFAULT_STATE.top_k;
      }}
      return Math.max(0, Math.min(maxExternal, Math.round(value)));
    }}

    function clampMinWeight(value) {{
      if (!Number.isFinite(value) || value < 0) {{
        return DEFAULT_STATE.min_weight;
      }}
      return value;
    }}

    function parseQueryState() {{
      const params = new URLSearchParams(window.location.search);
      const topK = clampTopK(Number(params.get("top_k")));
      const minWeight = clampMinWeight(Number(params.get("min_weight")));
      return {{
        top_k: params.has("top_k") ? topK : DEFAULT_STATE.top_k,
        min_weight: params.has("min_weight") ? minWeight : DEFAULT_STATE.min_weight
      }};
    }}

    function writeInputs(state) {{
      const modelTag = GRAPH_PAYLOAD.model_tag || "default";
      document.getElementById("model-tag-input").value = modelTag;
      document.getElementById("top-k-input").value = String(state.top_k);
      document.getElementById("min-weight-input").value = String(state.min_weight);
    }}

    function readInputs() {{
      return {{
        top_k: clampTopK(Number(document.getElementById("top-k-input").value)),
        min_weight: clampMinWeight(Number(document.getElementById("min-weight-input").value))
      }};
    }}

    function buildLegend() {{
      const legendItems = document.getElementById("legend-items");
      legendItems.innerHTML = "";

      for (const item of GRAPH_PAYLOAD.community_legend) {{
        const row = document.createElement("span");
        row.className = "legend-item";
        row.innerHTML =
          `<span class="legend-dot" style="background:${{item.color}}"></span>` +
          `Community ${{item.community}}`;
        legendItems.appendChild(row);
      }}

      const external = document.createElement("span");
      external.className = "legend-item";
      external.innerHTML =
        `<span class="legend-dot" style="background:${{GRAPH_PAYLOAD.external_color}}"></span>` +
        `External cited work`;
      legendItems.appendChild(external);
    }}

    function filteredGraph(state) {{
      const externalIncluded = new Set(
        GRAPH_PAYLOAD.external_order.slice(0, state.top_k)
      );
      const allowedNodes = new Set(GRAPH_PAYLOAD.corpus_ids);
      for (const nodeId of externalIncluded) {{
        allowedNodes.add(nodeId);
      }}

      const visibleEdges = GRAPH_PAYLOAD.edges.filter((edge) =>
        allowedNodes.has(edge.from) &&
        allowedNodes.has(edge.to) &&
        edge.raw_weight >= state.min_weight
      );

      const connectedNodes = new Set(GRAPH_PAYLOAD.corpus_ids);
      for (const edge of visibleEdges) {{
        connectedNodes.add(edge.from);
        connectedNodes.add(edge.to);
      }}

      const visibleNodes = GRAPH_PAYLOAD.nodes.filter((node) => {{
        if (node.is_corpus) {{
          return true;
        }}
        return externalIncluded.has(node.id) && connectedNodes.has(node.id);
      }});

      const visibleNodeIds = new Set(visibleNodes.map((node) => node.id));
      const trimmedEdges = visibleEdges.filter((edge) =>
        visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to)
      );

      const maxWeight = trimmedEdges.reduce(
        (current, edge) => Math.max(current, edge.raw_weight),
        1.0
      );

      const scaledEdges = trimmedEdges.map((edge) => {{
        return {{
          ...edge,
          value: Math.max(0.5, (edge.raw_weight / maxWeight) * 8)
        }};
      }});

      const corpusVisible = visibleNodes.filter((node) => node.is_corpus).length;
      const externalVisible = visibleNodes.length - corpusVisible;

      return {{
        nodes: visibleNodes,
        edges: scaledEdges,
        corpusVisible,
        externalVisible
      }};
    }}

    function updateSummary(state, filtered) {{
      document.getElementById("visible-nodes").textContent = String(filtered.nodes.length);
      document.getElementById("visible-edges").textContent = String(filtered.edges.length);
      document.getElementById("max-external").textContent = String(maxExternal);
      document.getElementById("stats-line").textContent =
        `Showing ${{filtered.corpusVisible}} corpus papers, ` +
        `${{filtered.externalVisible}} external works, and ` +
        `${{filtered.edges.length}} edges.`;

      document.getElementById("command-preview").textContent =
        `python3 visualize_graph.py ` +
        `${{GRAPH_PAYLOAD.model_tag ? `--model-tag ${{GRAPH_PAYLOAD.model_tag}} ` : ``}}` +
        `--load-mappings citation_mappings.json ` +
        `--top-k ${{state.top_k}} --min-weight ${{formatNumber(state.min_weight)}} ` +
        `--output docs/index.html`;
    }}

    function updateUrl(state) {{
      const params = new URLSearchParams();
      params.set("top_k", String(state.top_k));
      params.set("min_weight", String(state.min_weight));
      const nextUrl = `${{window.location.pathname}}?${{params.toString()}}`;
      window.history.replaceState(null, "", nextUrl);
    }}

    function render(state) {{
      writeInputs(state);
      const filtered = filteredGraph(state);
      nodesDataset.clear();
      edgesDataset.clear();
      nodesDataset.add(filtered.nodes);
      edgesDataset.add(filtered.edges);
      updateSummary(state, filtered);
      updateUrl(state);
      network.fit({{ animation: {{ duration: 400, easingFunction: "easeInOutQuad" }} }});
    }}

    function applyFromInputs() {{
      render(readInputs());
    }}

    document.getElementById("apply-btn").addEventListener("click", applyFromInputs);
    document.getElementById("reset-btn").addEventListener("click", () => render(DEFAULT_STATE));
    document.getElementById("top-k-input").addEventListener("keydown", (event) => {{
      if (event.key === "Enter") {{
        applyFromInputs();
      }}
    }});
    document.getElementById("min-weight-input").addEventListener("keydown", (event) => {{
      if (event.key === "Enter") {{
        applyFromInputs();
      }}
    }});

    buildLegend();
    render(parseQueryState());
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML citation graph."
    )
    parser.add_argument("--results", default="paper_results/", help="Results directory")
    parser.add_argument("--papers", default="papers/", help="PDF directory")
    parser.add_argument(
        "--model-tag",
        default="",
        help="Optional model tag for selecting tagged score files (e.g. llama3_2).",
    )
    parser.add_argument(
        "--load-mappings",
        default="",
        help="Load pre-saved citation_mappings.json",
    )
    parser.add_argument(
        "--save-mappings",
        default="",
        help="Save resolved mappings to this file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Initial number of top external cited works to show (default 20)",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.001,
        help="Initial minimum edge weight to display (default 0.001)",
    )
    parser.add_argument(
        "--output",
        default="graph.html",
        help="Output HTML file (default graph.html)",
    )
    args = parser.parse_args()

    if args.load_mappings and Path(args.load_mappings).exists():
        print(f"Loading mappings from {args.load_mappings} ...")
        resolver = CitationResolver.load(args.load_mappings)
        resolver.parse_all(args.results, args.papers)
    else:
        print("Parsing PDFs to resolve citations ...")
        resolver = CitationResolver()
        resolver.parse_all(args.results, args.papers)

    resolver.register_corpus_papers(args.results, args.papers)

    if args.save_mappings:
        resolver.save(args.save_mappings)
        print(f"Mappings saved to {args.save_mappings}")

    mappings = resolver.build_citation_mappings()
    fw = KnowledgeDiscoveryFramework.from_results_dir(
        args.results,
        citation_mappings=mappings,
        model_tag=args.model_tag,
    )

    payload = build_graph_payload(fw, resolver)
    output_path = Path(args.output)
    output_path.write_text(
        render_html(payload, default_top_k=args.top_k, default_min_weight=args.min_weight),
        encoding="utf-8",
    )

    print(
        f"Corpus: {payload['graph_stats']['corpus_nodes']} papers  |  "
        f"Total graph nodes: {payload['graph_stats']['total_nodes']}"
    )
    print(
        f"Website defaults: top-k={args.top_k}, min-weight={args.min_weight}  |  "
        f"max external candidates={payload['graph_stats']['external_nodes']}"
    )
    print(f"\nGraph saved -> {output_path}")
    print("Open it in any browser.")


if __name__ == "__main__":
    main()
