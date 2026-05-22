"""
Dot-and-whisker plot for section-level validation results.
Produces: section_validation_dotwhisker.pdf  +  .png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────
# Each entry: (point_estimate, ci_low, ci_high)  |  None = undefined

HUMAN = {
    "llama3.2:1b":      dict(spearman=(0.149,0.046,0.251), kendall=(0.125,0.042,0.211), l1=(0.716,0.651,0.777)),
    "llama3.2:3b":      dict(spearman=(0.565,0.445,0.678), kendall=(0.469,0.369,0.566), l1=(0.691,0.572,0.824)),
    "gemma2:2b":        dict(spearman=(0.549,0.449,0.643), kendall=(0.463,0.378,0.545), l1=(0.591,0.533,0.650)),
    "gemma3:4b":        dict(spearman=(0.324,0.172,0.472), kendall=(0.262,0.142,0.376), l1=(0.640,0.565,0.718)),
    "qwen3:1.7b":       dict(spearman=(0.636,0.533,0.729), kendall=(0.511,0.421,0.594), l1=(0.548,0.488,0.609)),
    "phi3:medium":      dict(spearman=(0.448,0.270,0.607), kendall=(0.362,0.221,0.499), l1=(0.596,0.522,0.678)),
    "qwen2.5:3b":       dict(spearman=(0.548,0.418,0.666), kendall=(0.453,0.346,0.558), l1=(0.610,0.535,0.686)),
    "Citation freq.":   dict(spearman=None,                 kendall=None,                l1=(0.687,0.633,0.742)),
    "Length-wt. freq.": dict(spearman=(0.624,0.520,0.720), kendall=(0.528,0.440,0.617), l1=(0.526,0.467,0.584)),
}

CHATGPT = {
    "llama3.2:1b":      dict(spearman=(0.220,0.109,0.335), kendall=(0.172,0.084,0.264), l1=(0.628,0.565,0.692)),
    "llama3.2:3b":      dict(spearman=(0.585,0.462,0.698), kendall=(0.489,0.376,0.594), l1=(0.579,0.456,0.722)),
    "gemma2:2b":        dict(spearman=(0.567,0.461,0.660), kendall=(0.476,0.397,0.556), l1=(0.471,0.418,0.527)),
    "gemma3:4b":        dict(spearman=(0.339,0.193,0.477), kendall=(0.286,0.170,0.394), l1=(0.536,0.466,0.609)),
    "qwen3:1.7b":       dict(spearman=(0.659,0.562,0.746), kendall=(0.532,0.439,0.617), l1=(0.411,0.369,0.459)),
    "phi3:medium":      dict(spearman=(0.474,0.317,0.622), kendall=(0.377,0.242,0.501), l1=(0.500,0.433,0.574)),
    "qwen2.5:3b":       dict(spearman=(0.550,0.418,0.672), kendall=(0.466,0.359,0.569), l1=(0.484,0.415,0.560)),
    "Citation freq.":   dict(spearman=None,                 kendall=None,                l1=(0.573,0.521,0.624)),
    "Length-wt. freq.": dict(spearman=(0.673,0.563,0.770), kendall=(0.564,0.465,0.650), l1=(0.429,0.375,0.486)),
}

MODELS_LLM = [
    "llama3.2:1b",
    "llama3.2:3b",
    "gemma2:2b",
    "gemma3:4b",
    "qwen3:1.7b",
    "phi3:medium",
    "qwen2.5:3b",
]
MODELS_BASELINE = [
    "Citation freq.",
    "Length-wt. freq.",
]

METRICS = [
    ("spearman", "Spearman  ρ  ↑"),
    ("kendall",  "Kendall  τ$_b$  ↑"),
    ("l1",       "$L_1$  ↓"),
]

# ── Layout ────────────────────────────────────────────────────────────────
# Y positions: LLMs top-to-bottom (y = n_llm … 1), gap, baselines below
N_LLM = len(MODELS_LLM)
Y_LLM      = {m: N_LLM - i for i, m in enumerate(MODELS_LLM)}      # 7 … 1
Y_BASELINE = {m: -0.6 - i * 0.9 for i, m in enumerate(MODELS_BASELINE)}  # -0.6, -1.5
SEP_Y      = 0.3          # separator line position
OFFSET     = 0.17         # vertical jitter between human / chatgpt dots

C_HUMAN  = "#1f77b4"   # blue
C_GPT    = "#d62728"   # red

# ── Figure ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 6.2), sharey=True)
fig.subplots_adjust(wspace=0.06, left=0.16, right=0.97, top=0.90, bottom=0.10)

all_models = MODELS_LLM + MODELS_BASELINE
ytick_pos    = [Y_LLM[m]      for m in MODELS_LLM] + \
               [Y_BASELINE[m] for m in MODELS_BASELINE]
ytick_labels = MODELS_LLM + MODELS_BASELINE

for ax, (mkey, mlabel) in zip(axes, METRICS):

    for model in all_models:
        y_center = Y_LLM.get(model) or Y_BASELINE.get(model)

        for src, data, col, mk, yo in [
            ("Human",   HUMAN,  C_HUMAN, "o",  +OFFSET),
            ("ChatGPT", CHATGPT, C_GPT,  "s",  -OFFSET),
        ]:
            entry = data[model][mkey]
            if entry is None:
                continue
            val, lo, hi = entry
            ax.errorbar(
                val, y_center + yo,
                xerr=[[val - lo], [hi - val]],
                fmt=mk,
                color=col,
                markersize=5.5,
                markeredgewidth=0.6,
                markeredgecolor="white",
                capsize=3.0,
                capthick=1.3,
                elinewidth=1.3,
                zorder=3,
            )

    # Separator between LLMs and baselines
    ax.axhline(SEP_Y, color="#888888", lw=0.8, ls="--", zorder=1)

    # Vertical reference line at 0 for correlation metrics
    if mkey in ("spearman", "kendall"):
        ax.axvline(0, color="#cccccc", lw=0.7, ls=":", zorder=0)

    ax.set_xlabel(mlabel, fontsize=11.5)
    ax.grid(axis="x", color="#dddddd", lw=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

# Y-axis ticks (only on left panel because sharey=True)
axes[0].set_yticks(ytick_pos)
axes[0].set_yticklabels(ytick_labels, fontsize=9.5, fontfamily="monospace")
axes[0].set_ylim(min(ytick_pos) - 0.55, max(ytick_pos) + 0.55)
axes[0].tick_params(axis="y", length=0)
for ax in axes[1:]:
    ax.tick_params(axis="y", length=0)

# X-axis limits
axes[0].set_xlim(-0.05, 0.85)
axes[1].set_xlim(-0.05, 0.75)
axes[2].set_xlim(0.28,  0.95)

# Annotation: LLM models vs baselines
axes[0].text(-0.38, (Y_LLM[MODELS_LLM[0]] + Y_LLM[MODELS_LLM[-1]]) / 2,
             "LLM\nmodels", va="center", ha="center", fontsize=8.5,
             color="#444444", rotation=90,
             transform=axes[0].get_yaxis_transform(), clip_on=False)
axes[0].text(-0.38, (Y_BASELINE[MODELS_BASELINE[0]] + Y_BASELINE[MODELS_BASELINE[-1]]) / 2,
             "Baselines", va="center", ha="center", fontsize=8.5,
             color="#444444", rotation=90,
             transform=axes[0].get_yaxis_transform(), clip_on=False)

# Legend
legend_elems = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor=C_HUMAN,
           markersize=7, markeredgecolor="white", label="Human annotations"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor=C_GPT,
           markersize=7, markeredgecolor="white", label="ChatGPT annotations"),
]
axes[1].legend(handles=legend_elems, loc="lower right", fontsize=9.5,
               framealpha=0.9, edgecolor="#cccccc")

fig.suptitle("Section-level validation  —  95% bootstrap CI  (n = 28 papers)",
             fontsize=12, y=0.97)

out_base = "section_validation_dotwhisker"
plt.savefig(f"{out_base}.pdf", bbox_inches="tight")
plt.savefig(f"{out_base}.png", bbox_inches="tight", dpi=200)
print(f"Saved {out_base}.pdf  and  {out_base}.png")
plt.show()
