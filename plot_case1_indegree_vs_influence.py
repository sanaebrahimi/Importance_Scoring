import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})

DATA = [
    ("N-Machine-Translation", 3, 4.75e-1, 8.34e-1),
    ("AttentionAllYouNeed", 6, 1.22e-1, 6.54e-1),
    ("shift-reduce-const-pars", 1, 9.16e-2, 7.28e-1),
    ("LSM-Memory", 1, 8.76e-2, 7.61e-1),
    ("struc-attention-net", 1, 5.80e-2, 7.35e-1),
    ("output-embed-llm", 1, 5.67e-2, 7.57e-1),
    ("yolov10", 1, 3.99e-2, 7.37e-1),
    ("gemma2", 2, 2.31e-2, 8.41e-1),
    ("Gen-Sequences-RNN", 2, 1.86e-2, 7.15e-1),
    ("Deep-Residual-Learning", 2, 1.15e-2, 6.31e-1),
    ("llama3", 1, 9.25e-3, 8.42e-1),
    ("ExploringLimitLLMs", 1, 6.96e-3, 7.55e-1),
    ("Smarter-Better-Faster-Longer", 0, 0.0, 6.82e-1),
    ("kimiK2", 0, 0.0, 8.54e-1),
    ("YOLO-World-Real-Time", 0, 0.0, 8.04e-1),
    ("d-fine-redefine", 0, 0.0, 7.86e-1),
    ("simPo", 0, 0.0, 8.33e-1),
]

DISPLAY = {
    "N-Machine-Translation": "N-Machine-\nTranslation",
    "AttentionAllYouNeed": "Attention Is\nAll You Need",
    "shift-reduce-const-pars": "Shift-Reduce\nConst Pars",
    "LSM-Memory": "LSM-Memory",
    "struc-attention-net": "Struct-Attention\nNet",
    "output-embed-llm": "Output-Embed\nLLM",
    "yolov10": "YOLOv10",
    "Gen-Sequences-RNN": "Gen-Sequences-\nRNN",
    "Deep-Residual-Learning": "Deep Residual\nLearning",
    "ExploringLimitLLMs": "Exploring\nLimit LLMs",
    "Smarter-Better-Faster-Longer": "Smarter-Better-\nFaster-Longer",
    "YOLO-World-Real-Time": "YOLO-World-\nReal-Time",
    "d-fine-redefine": "d-fine-\nredefine",
}


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    pos = 1
    j = 0
    while j < len(order):
        k = j
        value = values[order[j]]
        while k < len(order) and values[order[k]] == value:
            k += 1
        avg = (pos + pos + (k - j) - 1) / 2
        for idx in range(j, k):
            ranks[order[idx]] = avg
        pos += (k - j)
        j = k
    return ranks


def spearman(xs, ys):
    rx = average_ranks(xs)
    ry = average_ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = sum((a - mx) ** 2 for a in rx) ** 0.5
    deny = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (denx * deny)


nonzero = [(paper, indeg, influence, sigma) for paper, indeg, influence, sigma in DATA if influence > 0]
xs = [indeg for _, indeg, _, _ in nonzero]
ys = [influence for _, _, influence, _ in nonzero]
rho = spearman(xs, ys)

jitter_map = {
    1: [-0.22, -0.13, -0.05, 0.03, 0.11, 0.19, 0.27],
    2: [-0.12, 0.00, 0.12],
    3: [0.0],
    6: [0.0],
}
used = {1: 0, 2: 0, 3: 0, 6: 0}
plot_x = []
for indeg in xs:
    jitter = jitter_map[indeg][used[indeg]]
    used[indeg] += 1
    plot_x.append(indeg + jitter)

sigmas = [sigma for _, _, _, sigma in nonzero]
fig, ax = plt.subplots(figsize=(7.4, 5.4))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_linewidth(0.8)
    ax.spines[spine].set_color("#9aa0a6")

ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
ax.set_axisbelow(True)

sc = ax.scatter(
    plot_x,
    ys,
    c=sigmas,
    cmap="Reds",
    s=130,
    edgecolors="white",
    linewidths=1.2,
    zorder=3,
)

coef = np.polyfit(xs, ys, 1)
line_x = np.linspace(-0.1, 6.25, 100)
line_y = coef[0] * line_x + coef[1]
ax.plot(line_x, line_y, color="#4a6fa5", linewidth=1.5, alpha=0.8, zorder=2)

label_offsets = {
    "N-Machine-Translation": (0.07, 0.012),
    "AttentionAllYouNeed": (0.08, 0.008),
    "shift-reduce-const-pars": (-0.55, 0.012),
    "LSM-Memory": (-0.20, 0.010),
    "struc-attention-net": (-0.52, 0.000),
    "output-embed-llm": (-0.42, -0.010),
    "yolov10": (0.10, -0.010),
    "gemma2": (0.09, 0.002),
    "Gen-Sequences-RNN": (0.09, -0.006),
    "Deep-Residual-Learning": (0.09, -0.012),
    "llama3": (-0.15, -0.010),
    "ExploringLimitLLMs": (0.08, -0.002),
}

for (paper, indeg, influence, _), x in zip(nonzero, plot_x):
    dx, dy = label_offsets.get(paper, (0.05, 0.01))
    txt = ax.text(
        x + dx,
        influence + dy,
        DISPLAY.get(paper, paper),
        fontsize=10.5,
        fontweight="bold",
        color="#222222",
        ha="left",
        va="center",
        linespacing=0.95,
        zorder=4,
    )
    txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

ax.set_xlim(-0.35, 6.55)
ax.set_ylim(0.0, 0.52)
ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.tick_params(axis="both", labelsize=12, width=0, length=0)

ax.set_xlabel("In-degree within the induced Case 1 corpus", fontsize=15, fontweight="bold", labelpad=10)
ax.set_ylabel(r"Normalized influence $I_{\mathcal{P}}(d)$", fontsize=15, fontweight="bold", labelpad=10)

ax.text(
    0.02,
    0.98,
    rf"Nonzero-influence papers only ($n=12$)\nSpearman $\rho = {rho:.3f}$",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=12.5,
    fontweight="bold",
    color="#1f2937",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.96),
)

cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.86)
cbar.set_label(r"$\sigma_{\mathcal{P}}(d)$", fontsize=13, fontweight="bold", labelpad=10)
cbar.ax.tick_params(labelsize=11)
cbar.outline.set_edgecolor("#cbd5e1")

plt.tight_layout()

output_dir = Path("results/final_sonnet46_model_comparison")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "case1_indegree_vs_influence_scatter.pdf", dpi=220, bbox_inches="tight")
plt.savefig(output_dir / "case1_indegree_vs_influence_scatter.png", dpi=220, bbox_inches="tight")
print(f"Spearman rho (nonzero influence): {rho:.6f}")
print(output_dir / "case1_indegree_vs_influence_scatter.pdf")
