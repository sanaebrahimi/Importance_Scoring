import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

NODES_FILE = Path('papers/Case 1/results/case1_influence_network_qwen3_1_7b_nodes.csv')
EDGES_FILE = Path('papers/Case 1/results/case1_influence_network_qwen3_1_7b_edges.csv')
OUTPUT_DIR = Path('outputs')
FINAL_PDF = Path('results/final_sonnet46_model_comparison/citation_network_IP.pdf')
FINAL_PNG = Path('results/final_sonnet46_model_comparison/citation_network_IP.png')


def load_influence(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Nodes file not found: {path}')
    influence = {}
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            influence[row['paper_id']] = float(row['I_P'])
    return influence


def load_edges(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Edges file not found: {path}')
    edges = []
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append((row['target'], row['source'], float(row['edge_weight'])))
    return edges


influence = load_influence(NODES_FILE)
edges_raw = load_edges(EDGES_FILE)

G = nx.DiGraph()
for target_paper, source_paper, weight in edges_raw:
    # Edge source -> target with weight equal to the citation contribution
    # assigned by the source paper to the cited target paper.
    G.add_edge(source_paper, target_paper, weight=weight)
for node in influence:
    if node not in G.nodes():
        G.add_node(node)

np.random.seed(17)
pos = nx.spring_layout(G, k=3.4, iterations=300, weight='weight', seed=17)
# Compress and shift the network left to reserve a clean annotation column.
pos = {node: (coords[0] * 0.82 - 0.18, coords[1] * 0.92) for node, coords in pos.items()}

nodes_list = list(G.nodes())
inf_vals = np.array([influence.get(node, 0.0) for node in nodes_list])
inf_log = np.log1p(inf_vals * 1000)
inf_norm = (inf_log - inf_log.min()) / (inf_log.max() - inf_log.min() + 1e-12)
node_sizes = 200 + inf_norm ** 0.55 * 4500
node_size_map = {node: size for node, size in zip(nodes_list, node_sizes)}
node_marker_r = {node: np.sqrt(size / np.pi) for node, size in node_size_map.items()}

fig, ax = plt.subplots(figsize=(19, 13.5))
BG_COLOR = '#FFFFFF'
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_axis_off()

NODE_CMAP = LinearSegmentedColormap.from_list(
    'reds', ['#FFCCCC', '#CC1111', '#6B0000'], N=256
)
EDGE_CMAP = LinearSegmentedColormap.from_list(
    'blues', ['#C8DEFF', '#5599DD', '#1A3A7A'], N=256
)

weights = np.array([data['weight'] for _, _, data in G.edges(data=True)])
w_log = np.log1p(weights)
w_norm = (w_log - w_log.min()) / (w_log.max() - w_log.min() + 1e-12)


def node_radius_data(node):
    r_pts = node_marker_r.get(node, 10)
    fig_w, fig_h = fig.get_size_inches()
    ax_pos = ax.get_position()
    xl, yl = ax.get_xlim(), ax.get_ylim()
    r_data_x = (r_pts / 72.0) * (xl[1] - xl[0]) / (ax_pos.width * fig_w)
    r_data_y = (r_pts / 72.0) * (yl[1] - yl[0]) / (ax_pos.height * fig_h)
    return (r_data_x + r_data_y) / 2.0


for (src, tgt, data), t in zip(G.edges(data=True), w_norm):
    color = EDGE_CMAP(0.15 + t * 0.85)
    lw = 1.2 + t * 3.5
    alpha = 0.50 + t * 0.45

    x0, y0 = pos[src]
    x1, y1 = pos[tgt]
    dx, dy = x1 - x0, y1 - y0
    dist = np.sqrt(dx ** 2 + dy ** 2) or 1e-9
    ux, uy = dx / dist, dy / dist

    margin = 1.6
    end_x = x1 - ux * node_radius_data(tgt) * margin
    end_y = y1 - uy * node_radius_data(tgt) * margin
    start_x = x0 + ux * node_radius_data(src) * margin
    start_y = y0 + uy * node_radius_data(src) * margin
    mutation_scale = 10 + node_marker_r.get(tgt, 10) * 0.8

    ax.annotate(
        '',
        xy=(end_x, end_y),
        xytext=(start_x, start_y),
        arrowprops=dict(
            arrowstyle='->',
            color=color,
            lw=lw,
            alpha=alpha,
            mutation_scale=mutation_scale,
            connectionstyle='arc3,rad=0.07',
        ),
        zorder=2,
    )

xs = [pos[node][0] for node in nodes_list]
ys = [pos[node][1] for node in nodes_list]
node_colors = [NODE_CMAP(inf_norm[i]) for i in range(len(nodes_list))]

for glow_scale, glow_alpha in [(5.5, 0.04), (3.8, 0.07), (2.4, 0.10), (1.6, 0.14)]:
    ax.scatter(
        xs,
        ys,
        s=node_sizes * glow_scale,
        c=node_colors,
        alpha=glow_alpha,
        zorder=3,
        linewidths=0,
    )

ax.scatter(
    xs,
    ys,
    s=node_sizes,
    c=node_colors,
    zorder=5,
    edgecolors='none',
    linewidths=0,
    alpha=1.0,
)

for i, node in enumerate(nodes_list):
    x, y = pos[node]
    t = inf_norm[i]
    fontsize = 11.0 + t * 6.0
    label_color = '#111111'

    parts = node.split('-')
    if len(node) > 18 and len(parts) > 1:
        mid = max(1, len(parts) // 2)
        label = '-'.join(parts[:mid]) + '\n' + '-'.join(parts[mid:])
    else:
        label = node.replace('-', '‑')

    fig_w, fig_h = fig.get_size_inches()
    ax_pos = ax.get_position()
    xl, yl = ax.get_xlim(), ax.get_ylim()
    r_pts = node_marker_r[node]
    r_data_x = (r_pts / 72.0) * (xl[1] - xl[0]) / (ax_pos.width * fig_w)
    r_data_y = (r_pts / 72.0) * (yl[1] - yl[0]) / (ax_pos.height * fig_h)

    txt = ax.text(
        x + r_data_x * 1.3,
        y + r_data_y * 0.3,
        label,
        ha='left',
        va='center',
        fontsize=fontsize,
        color=label_color,
        fontweight='bold',
        linespacing=1.2,
        zorder=7,
    )
    txt.set_path_effects([pe.withStroke(linewidth=3.4, foreground='white')])

sm = plt.cm.ScalarMappable(
    cmap=NODE_CMAP,
    norm=plt.Normalize(vmin=0, vmax=max(influence.values())),
)
sm.set_array([])
fig.subplots_adjust(left=0.03, right=0.72, top=0.98, bottom=0.05)
cbar_ax = fig.add_axes([0.825, 0.21, 0.022, 0.24])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Influence Score  $I_p$', color='#111111', fontsize=17, labelpad=14, fontweight='bold')
cbar.ax.yaxis.set_tick_params(color='#111111', labelcolor='#111111', labelsize=15)
cbar.outline.set_edgecolor('#AAAAAA')

edge_legend_items = [
    mpatches.Patch(facecolor=EDGE_CMAP(0.15), label='Low weight'),
    mpatches.Patch(facecolor=EDGE_CMAP(0.50), label='Medium weight'),
    mpatches.Patch(facecolor=EDGE_CMAP(1.00), label='High weight'),
]
leg = fig.legend(
    handles=edge_legend_items,
    loc='upper left',
    bbox_to_anchor=(0.825, 0.76),
    ncol=1,
    prop={'size': 15.5, 'weight': 'bold'},
    facecolor='#F5F8FF',
    edgecolor='#5599DD',
    labelcolor='#111111',
    framealpha=0.98,
    title='Edge Weight',
    title_fontsize=17,
    borderpad=1.2,
    labelspacing=0.70,
    handlelength=1.8,
    handleheight=1.2,
)
leg.get_title().set_color('#1A3A7A')
leg.get_title().set_fontweight('bold')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_PDF.parent.mkdir(parents=True, exist_ok=True)
pdf_path = OUTPUT_DIR / 'citation_network_IP.pdf'
png_path = OUTPUT_DIR / 'citation_network_IP.png'
plt.savefig(pdf_path, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
plt.savefig(png_path, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
plt.savefig(FINAL_PDF, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
plt.savefig(FINAL_PNG, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
print('Done')
