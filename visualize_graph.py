"""
visualize_graph.py - Generate an interactive HTML citation graph.

The generated page preloads the full graph and lets the viewer change
browser-side filters such as:
  - top-k external cited works by PageRank
  - minimum edge weight

Usage:
    python3 visualize_graph.py
    python3 visualize_graph.py --top-k 40 --min-weight 0.005
    python3 visualize_graph.py --model-tag llama3_2 --top-k 40 --min-weight 0.005
    python3 visualize_graph.py --load-mappings citation_mappings.json --output graph.html
"""

import argparse
import colorsys
import json
import os
import re
from pathlib import Path
from typing import Optional, Union

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
_DEFAULT_MODEL_STORAGE_KEY = "__default__"


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


def _result_model_tag_from_filename(paper_id: str, filename: str) -> Optional[str]:
    suffixes = [
        "_paragraph_citation_scores.json",
        "_paragraph_scores.json",
        "_section_scores.json",
        "_citation_scores.json",
    ]
    for suffix in suffixes:
        if filename == f"{paper_id}{suffix}":
            return ""
        prefix = f"{paper_id}_"
        if filename.startswith(prefix) and filename.endswith(suffix):
            return filename[len(prefix) : -len(suffix)]
    return None


def discover_available_model_tags(results_dir: Union[str, Path]) -> list[str]:
    tags = set()
    for paper_dir in sorted(Path(results_dir).iterdir()):
        if not paper_dir.is_dir() or paper_dir.name.startswith("."):
            continue
        paper_id = paper_dir.name
        for child in paper_dir.iterdir():
            if not child.is_file() or not child.name.endswith(".json"):
                continue
            model_tag = _result_model_tag_from_filename(paper_id, child.name)
            if model_tag is None:
                continue
            tags.add(model_tag)
    return sorted(tags, key=lambda tag: (tag != "", tag))


def _model_storage_key(model_tag: str) -> str:
    return model_tag or _DEFAULT_MODEL_STORAGE_KEY


def _model_display_name(model_tag: str) -> str:
    return model_tag or "default"


def _scale(val: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi <= lo:
        return (out_lo + out_hi) / 2
    return out_lo + (val - lo) / (hi - lo) * (out_hi - out_lo)


def _corpus_tooltip(node_id: str) -> str:
    return "\n".join([node_id, "Year: -", "corpus paper"])


def _corpus_label(node_id: str) -> str:
    label = node_id.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", label).strip()


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
    seminal = fw.influence.seminal_scores(restrict_to_corpus=False)
    norm_influence = fw.corpus_contribution.normalized_influence()
    corpus_ids = sorted(fw.graph.papers.keys())
    corpus_set = set(corpus_ids)
    partition = fw.communities.detect(restrict_to_corpus=True)
    all_node_ids = sorted(fw.graph.all_nodes())

    external_ranked = sorted(
        (node_id for node_id in all_node_ids if node_id not in corpus_set),
        key=lambda node_id: (-pagerank.get(node_id, 0.0), node_id),
    )

    ip_vals = list(norm_influence.values())
    ip_min = min(ip_vals, default=0.0)
    ip_max = max(ip_vals, default=1.0)

    seminal_vals = [seminal.get(node_id, 0.0) for node_id in all_node_ids if node_id not in corpus_set]
    seminal_min = min(seminal_vals, default=0.0)
    seminal_max = max(seminal_vals, default=1.0)

    nodes = []
    for node_id in all_node_ids:
        pr = pagerank.get(node_id, 0.0)
        seminal_score = seminal.get(node_id, 0.0)

        if node_id in corpus_set:
            ip_score = norm_influence.get(node_id, 0.0)
            size = _scale(ip_score, ip_min, ip_max, 20, 72)
            community = partition.get(node_id, 0)
            colour = _community_colour(community)
            label = _corpus_label(node_id)
            tooltip = "\n".join(
                [
                    _corpus_tooltip(node_id),
                    f"Normalized influence I_P: {ip_score:.6f}",
                    f"Seminal score: {seminal_score:.6f}",
                    f"Reference score: {pr:.6f}",
                ]
            )
            border_width = 3.0
            font = {
                "size": 16,
                "face": "IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
                "color": "#f5f7fb",
                "strokeWidth": 4,
                "strokeColor": "rgba(5, 11, 19, 0.92)",
                "bold": True,
            }
        else:
            ip_score = 0.0
            size = _scale(seminal_score, seminal_min, seminal_max, 14, 55)
            community = None
            colour = _EXTERNAL_COLOUR
            label, tooltip_base = _external_label_and_tooltip(resolver, node_id)
            tooltip = "\n".join(
                [
                    tooltip_base,
                    f"Seminal score: {seminal_score:.6f}",
                    f"Reference score: {pr:.6f}",
                ]
            )
            border_width = 2.0
            font = {
                "size": 13,
                "face": "IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
                "color": "#fff6de",
                "strokeWidth": 4,
                "strokeColor": "rgba(5, 11, 19, 0.92)",
            }

        nodes.append(
            {
                "id": node_id,
                "label": label,
                "title": tooltip,
                "size": size,
                "borderWidth": border_width,
                "font": font,
                "color": {
                    "background": colour,
                    "border": "#ffffff",
                    "highlight": {"background": "#ffffff", "border": colour},
                },
                "is_corpus": node_id in corpus_set,
                "community": community,
                "pagerank": pr,
                "seminal_score": seminal_score,
                "norm_influence": ip_score,
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
                    "opacity": 0.72,
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
        "model_key": _model_storage_key(model_tag),
        "model_label": _model_display_name(model_tag),
        "model_tag": model_tag,
    }


def render_html(
    payloads_by_model: dict,
    default_model_key: str,
    default_top_k: int,
    default_min_weight: float,
    vis_css_text: str,
    vis_js_text: str,
) -> str:
    payloads_json = json.dumps(payloads_by_model, ensure_ascii=False)
    default_state_json = json.dumps(
        {
            "model": default_model_key,
            "top_k": default_top_k,
            "min_weight": default_min_weight,
        },
        ensure_ascii=False,
    )
    title = "Importance-Weighted Citation Graph"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
{vis_css_text}
  </style>
  <script>
{vis_js_text}
  </script>
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

    input,
    select {{
      width: 100%;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px;
      background: var(--field);
      color: var(--text);
      padding: 11px 12px;
      font-size: 15px;
      outline: none;
    }}

    input:focus,
    select:focus {{
      border-color: rgba(78, 154, 241, 0.9);
      box-shadow: 0 0 0 3px rgba(78, 154, 241, 0.18);
    }}

    input[readonly],
    select:disabled {{
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
            <select id="model-tag-input"></select>
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
          external works that satisfy the current filters. Set
          <code>min-weight</code> to <code>0</code> to disable edge-weight pruning.
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
            Corpus node size = Normalized Influence I_P(p). External node size = seminal score. Edge width = citation importance weight.
          </div>
        </div>
      </div>
      <div id="mynetwork"></div>
    </section>
  </div>

  <script>
    const GRAPH_PAYLOADS = {payloads_json};
    const DEFAULT_STATE = {default_state_json};
    const NETWORK_OPTIONS = {{
      physics: {{
        solver: "barnesHut",
        barnesHut: {{
          gravitationalConstant: -7200,
          centralGravity: 0.18,
          springLength: 250,
          springConstant: 0.045,
          damping: 0.16
        }},
        enabled: true,
        maxVelocity: 60,
        minVelocity: 0.5,
        stabilization: {{ iterations: 350, fit: true }}
      }},
      layout: {{
        improvedLayout: true,
        randomSeed: 7
      }},
      interaction: {{
        hover: true,
        tooltipDelay: 80,
        navigationButtons: true,
        keyboard: {{ enabled: true }},
        hideEdgesOnDrag: true
      }},
      edges: {{
        smooth: false,
        scaling: {{ min: 0.5, max: 8 }}
      }},
      nodes: {{
        scaling: {{ min: 18, max: 72 }},
        shadow: {{ enabled: true, size: 8 }},
        shape: "dot"
      }}
    }};

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

    function clampTopK(value, modelKey = null) {{
      const payload = modelKey ? (GRAPH_PAYLOADS[modelKey] || currentPayload()) : currentPayload();
      const maxExternal = payload.external_order.length;
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

    function availableModelKeys() {{
      return Object.keys(GRAPH_PAYLOADS);
    }}

    function currentPayload(state = null) {{
      const modelKey = state?.model ?? DEFAULT_STATE.model;
      return GRAPH_PAYLOADS[modelKey] || GRAPH_PAYLOADS[DEFAULT_STATE.model];
    }}

    function parseQueryState() {{
      const params = new URLSearchParams(window.location.search);
      const requestedModel = params.get("model");
      const validModel = availableModelKeys().includes(requestedModel || "")
        ? (requestedModel || DEFAULT_STATE.model)
        : DEFAULT_STATE.model;
      const topK = clampTopK(Number(params.get("top_k")), validModel);
      const minWeight = clampMinWeight(Number(params.get("min_weight")));
      return {{
        model: validModel,
        top_k: params.has("top_k") ? topK : DEFAULT_STATE.top_k,
        min_weight: params.has("min_weight") ? minWeight : DEFAULT_STATE.min_weight
      }};
    }}

    function writeInputs(state) {{
      document.getElementById("model-tag-input").value = state.model;
      document.getElementById("top-k-input").value = String(state.top_k);
      document.getElementById("min-weight-input").value = String(state.min_weight);
    }}

    function readInputs() {{
      return {{
        model: document.getElementById("model-tag-input").value || DEFAULT_STATE.model,
        top_k: clampTopK(
          Number(document.getElementById("top-k-input").value),
          document.getElementById("model-tag-input").value || DEFAULT_STATE.model,
        ),
        min_weight: clampMinWeight(Number(document.getElementById("min-weight-input").value))
      }};
    }}

    function buildModelOptions() {{
      const modelSelect = document.getElementById("model-tag-input");
      modelSelect.innerHTML = "";
      for (const modelKey of availableModelKeys()) {{
        const payload = GRAPH_PAYLOADS[modelKey];
        const option = document.createElement("option");
        option.value = modelKey;
        option.textContent = payload.model_label;
        modelSelect.appendChild(option);
      }}
      modelSelect.disabled = availableModelKeys().length <= 1;
    }}

    function buildLegend(payload) {{
      const legendItems = document.getElementById("legend-items");
      legendItems.innerHTML = "";

      for (const item of payload.community_legend) {{
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
        `<span class="legend-dot" style="background:${{payload.external_color}}"></span>` +
        `External cited work`;
      legendItems.appendChild(external);
    }}

    function filteredGraph(state) {{
      const payload = currentPayload(state);
      const externalIncluded = new Set(
        payload.external_order.slice(0, state.top_k)
      );
      const allowedNodes = new Set(payload.corpus_ids);
      for (const nodeId of externalIncluded) {{
        allowedNodes.add(nodeId);
      }}

      const visibleEdges = payload.edges.filter((edge) =>
        allowedNodes.has(edge.from) &&
        allowedNodes.has(edge.to) &&
        edge.raw_weight >= state.min_weight
      );

      const connectedNodes = new Set(payload.corpus_ids);
      for (const edge of visibleEdges) {{
        connectedNodes.add(edge.from);
        connectedNodes.add(edge.to);
      }}

      const visibleNodes = payload.nodes.filter((node) => {{
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
        payload,
        nodes: visibleNodes,
        edges: scaledEdges,
        corpusVisible,
        externalVisible
      }};
    }}

    function updateSummary(state, filtered) {{
      document.getElementById("visible-nodes").textContent = String(filtered.nodes.length);
      document.getElementById("visible-edges").textContent = String(filtered.edges.length);
      document.getElementById("max-external").textContent = String(filtered.payload.external_order.length);
      document.getElementById("stats-line").textContent =
        `Model ${{filtered.payload.model_label}}: showing ${{filtered.corpusVisible}} corpus papers, ` +
        `${{filtered.externalVisible}} external works, and ` +
        `${{filtered.edges.length}} edges.`;

      document.getElementById("command-preview").textContent =
        `python3 visualize_graph.py ` +
        `${{filtered.payload.model_tag ? `--model-tag ${{filtered.payload.model_tag}} ` : ``}}` +
        `--load-mappings citation_mappings.json ` +
        `--top-k ${{state.top_k}} ` +
        `${{state.min_weight > 0 ? `--min-weight ${{formatNumber(state.min_weight)}} ` : ``}}` +
        `--output docs/index.html`;
    }}

    function updateUrl(state) {{
      const params = new URLSearchParams();
      if (state.model && state.model !== DEFAULT_STATE.model) {{
        params.set("model", String(state.model));
      }}
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
      buildLegend(filtered.payload);
      updateSummary(state, filtered);
      updateUrl(state);
      network.setOptions({{ physics: {{ enabled: true }} }});
      network.once("stabilized", () => {{
        network.fit({{ animation: {{ duration: 500, easingFunction: "easeInOutQuad" }} }});
        network.setOptions({{ physics: {{ enabled: false }} }});
      }});
      network.stabilize(350);
    }}

    function applyFromInputs() {{
      render(readInputs());
    }}

    document.getElementById("apply-btn").addEventListener("click", applyFromInputs);
    document.getElementById("reset-btn").addEventListener("click", () => render(DEFAULT_STATE));
    document.getElementById("model-tag-input").addEventListener("change", () => render(readInputs()));
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

    buildModelOptions();
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
        "--exclude-papers",
        nargs="+",
        default=[],
        metavar="PAPER_ID",
        help="Optional paper ids to exclude from citation resolution and graph construction.",
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
        default=18,
        help="Initial number of top external cited works to show (default 18)",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.0,
        help="Initial minimum edge weight to display (default 0.0; disables weight pruning)",
    )
    parser.add_argument(
        "--output",
        default="graph.html",
        help="Output HTML file (default graph.html)",
    )
    args = parser.parse_args()
    excluded = {paper.strip() for paper in args.exclude_papers if paper.strip()}

    if args.load_mappings and Path(args.load_mappings).exists():
        print(f"Loading mappings from {args.load_mappings} ...")
        resolver = CitationResolver.load(args.load_mappings)
        resolver.parse_all(args.results, args.papers, exclude_papers=excluded)
    else:
        print("Parsing PDFs to resolve citations ...")
        resolver = CitationResolver()
        resolver.parse_all(args.results, args.papers, exclude_papers=excluded)

    resolver.register_corpus_papers(args.results, args.papers, exclude_papers=excluded)

    if args.save_mappings:
        resolver.save(args.save_mappings)
        print(f"Mappings saved to {args.save_mappings}")

    mappings = resolver.build_citation_mappings()
    if args.model_tag:
        model_tags = [args.model_tag.strip()]
    else:
        model_tags = discover_available_model_tags(args.results)
        if not model_tags:
            model_tags = [""]

    payloads_by_model = {}
    for model_tag in model_tags:
        fw = KnowledgeDiscoveryFramework.from_results_dir(
            args.results,
            citation_mappings=mappings,
            model_tag=model_tag,
            exclude_papers=excluded,
        )
        payload = build_graph_payload(fw, resolver, model_tag=model_tag)
        payloads_by_model[_model_storage_key(model_tag)] = payload

    default_model_key = _model_storage_key(model_tags[0])
    if not args.model_tag and _model_storage_key("llama3_2") in payloads_by_model:
        default_model_key = _model_storage_key("llama3_2")

    default_payload = payloads_by_model[default_model_key]
    output_path = Path(args.output)
    vis_asset_root = Path("lib") / "vis-9.1.2"
    vis_css_path = vis_asset_root / "vis-network.css"
    vis_js_path = vis_asset_root / "vis-network.min.js"
    vis_css_text = vis_css_path.read_text(encoding="utf-8")
    vis_js_text = vis_js_path.read_text(encoding="utf-8")
    output_path.write_text(
        render_html(
            payloads_by_model,
            default_model_key=default_model_key,
            default_top_k=args.top_k,
            default_min_weight=args.min_weight,
            vis_css_text=vis_css_text,
            vis_js_text=vis_js_text,
        ),
        encoding="utf-8",
    )

    print(
        f"Corpus: {default_payload['graph_stats']['corpus_nodes']} papers  |  "
        f"Total graph nodes: {default_payload['graph_stats']['total_nodes']}"
    )
    print(
        f"Website defaults: top-k={args.top_k}, min-weight={args.min_weight}  |  "
        f"max external candidates={default_payload['graph_stats']['external_nodes']}"
    )
    print(
        "Models embedded: "
        + ", ".join(payload["model_label"] for payload in payloads_by_model.values())
    )
    print(f"\nGraph saved -> {output_path}")
    print("Open it in any browser.")


if __name__ == "__main__":
    main()
