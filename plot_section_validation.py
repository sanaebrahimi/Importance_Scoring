from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from evaluate_human_section_scores import (
    align_section_score_maps,
    bootstrap_ci,
    extract_top_level_model_scores,
    load_json,
    mean_or_none,
    metric_bundle,
    normalize_scores,
)


ROOT = Path(__file__).resolve().parent
PLOTS_DIR = ROOT / "results" / "plots"
METRICS_JSON = ROOT / "results" / "section_validation_three_refs_metrics.json"

plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 14,
        "legend.fontsize": 12.5,
        "figure.titlesize": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "x",
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "figure.titleweight": "bold",
    }
)

BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 7

BASELINES = [
    "Citation freq.",
    "Length-wtd. freq.",
]
LLM_MODELS = [
    "llama3.2:1b",
    "llama3.2:3b",
    "gemma2:2b",
    "gemma3:4b",
    "phi3:medium",
    "qwen3:1.7b",
    "qwen3:4b",
    "qwen2.5:3b",
]
ALL_MODELS = BASELINES + LLM_MODELS
N_BASE = len(BASELINES)
N = len(ALL_MODELS)
Y = np.arange(N)

MODEL_COLORS = {
    "gemma2:2b": "#e15759",
    "gemma3:4b": "#76b7b2",
    "llama3.2:1b": "#4e79a7",
    "llama3.2:3b": "#f28e2b",
    "qwen2.5:3b": "#b07aa1",
    "qwen3:1.7b": "#59a14f",
    "qwen3:4b": "#9c755f",
    "phi3:medium": "#edc948",
    "Citation freq.": "#aaaaaa",
    "Length-wtd. freq.": "#666666",
}

MODEL_TAGS = {
    "Citation freq.": ("citation_frequency",),
    "Length-wtd. freq.": ("length_weighted_frequency",),
    "llama3.2:1b": ("llama3_2_1b_promptv2", "llama3_2_1b_promptv2_retry8"),
    "llama3.2:3b": ("llama3_2_promptv2", "llama3_2_3b_promptv2", "llama3_2_3b_promptv2_retry8"),
    "gemma2:2b": ("gemma2_2b_promptv2", "gemma2_2b_promptv2_retry8"),
    "gemma3:4b": ("gemma3_4b_promptv2", "gemma3_4b_promptv2_retry8"),
    "qwen3:1.7b": ("qwen3_1_7_promptv2", "qwen3_1_7b_promptv2_retry8"),
    "qwen3:4b": ("qwen3_4b_promptv2", "qwen3_4b_promptv2_retry8"),
    "phi3:medium": ("phi3_medium_promptv2", "phi3_med_promptv2", "phi3_medium_promptv2_retry8"),
    "qwen2.5:3b": ("qwen2_5_3b_promptv2", "qwen2_5_3b_promptv2_retry8"),
}
REFERENCE_TAGS = {
    "human": (),
    "openai": ("openai_full_paper_gpt_oss_120b",),
    "anthropic": ("anthropic_full_paper_claude_sonnet_4_6_direct",),
}

REFERENCES = [
    {
        "key": "human",
        "label": "Human annotations",
        "marker": "o",
        "offset": 0.24,
    },
    {
        "key": "openai",
        "label": "gpt-oss-120b reference",
        "marker": "s",
        "offset": 0.00,
    },
    {
        "key": "anthropic",
        "label": "Sonnet-4.6 reference",
        "marker": "^",
        "offset": -0.24,
    },
]

DATA = {
    "spearman": {
        "xlabel": r"Spearman $\rho$ $\uparrow$",
        "xlim": (-0.26, 0.90),
    },
    "kendall": {
        "xlabel": r"Kendall $\tau_b$ $\uparrow$",
        "xlim": (-0.26, 0.84),
    },
    "l1": {
        "xlabel": r"$L_1$ distance $\downarrow$",
        "xlim": (0.18, 0.88),
    },
}

SEP_Y = N_BASE - 0.5

def rgba(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
    return (*mcolors.to_rgb(hex_color), alpha)


def load_human_reference_maps() -> dict[str, dict[str, float]]:
    annotations = load_json(ROOT / "human_expert_annotations.json")
    return {
        paper_id: payload["top_level_scores_for_evaluation"]
        for paper_id, payload in annotations["papers"].items()
    }


def resolve_existing_tag(paper_id: str, tags: tuple[str, ...]) -> str | None:
    for tag in tags:
        path = ROOT / "paper_results" / paper_id / f"{paper_id}_{tag}_section_scores.json"
        if path.exists():
            return tag
    return None


def load_section_score_maps(tags: tuple[str, ...]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    paper_root = ROOT / "paper_results"
    for paper_dir in sorted(path for path in paper_root.iterdir() if path.is_dir()):
        paper_id = paper_dir.name
        resolved_tag = resolve_existing_tag(paper_id, tags)
        if resolved_tag is None:
            continue
        path = paper_dir / f"{paper_id}_{resolved_tag}_section_scores.json"
        result[paper_id] = extract_top_level_model_scores(load_json(path))
    return result


def summarize_metric(values: list[float | None], seed_offset: int) -> list[float | None]:
    mean = mean_or_none(values)
    ci = bootstrap_ci(values, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED + seed_offset)
    if mean is None or ci is None:
        return [None, None, None]
    return [mean, ci[0], ci[1]]


def evaluate_against_reference(
    candidate_maps: dict[str, dict[str, float]],
    reference_maps: dict[str, dict[str, float]],
    paper_ids: list[str],
) -> dict[str, list[float | None]]:
    per_metric = {
        "spearman": [],
        "kendall": [],
        "l1": [],
    }

    for paper_id in paper_ids:
        overlap, ref_raw, cand_raw, _ = align_section_score_maps(
            reference_maps[paper_id],
            candidate_maps[paper_id],
        )
        if len(overlap) < 2:
            continue

        ref_norm = normalize_scores(ref_raw)
        cand_norm = normalize_scores(cand_raw)
        ordered_ref = [ref_norm[section] for section in overlap]
        ordered_cand = [cand_norm[section] for section in overlap]
        bundle = metric_bundle(ordered_ref, ordered_cand)

        per_metric["spearman"].append(bundle["spearman"])
        per_metric["kendall"].append(bundle["kendall_tau_b"])
        per_metric["l1"].append(bundle["l1"])

    return {
        "papers_evaluated": len(per_metric["l1"]),
        "spearman": summarize_metric(per_metric["spearman"], 0),
        "kendall": summarize_metric(per_metric["kendall"], 1),
        "l1": summarize_metric(per_metric["l1"], 2),
    }


def build_summary() -> dict:
    human_maps = load_human_reference_maps()
    openai_maps = load_section_score_maps(REFERENCE_TAGS["openai"])
    anthropic_maps = load_section_score_maps(REFERENCE_TAGS["anthropic"])

    reference_maps = {
        "human": human_maps,
        "openai": openai_maps,
        "anthropic": anthropic_maps,
    }

    summary = {
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "references": {
            "human": "human_expert_annotations.json",
            "openai": list(REFERENCE_TAGS["openai"]),
            "anthropic": list(REFERENCE_TAGS["anthropic"]),
        },
        "models": {},
    }

    for label in ALL_MODELS:
        model_tags = MODEL_TAGS[label]
        candidate_maps = load_section_score_maps(model_tags)
        common_papers = sorted(
            set(candidate_maps)
            & set(human_maps)
            & set(openai_maps)
            & set(anthropic_maps)
        )

        model_summary = {
            "model_tag": model_tags[0],
            "model_tags": list(model_tags),
            "common_papers": common_papers,
            "paper_count": len(common_papers),
            "references": {},
        }

        for ref_key, ref_maps in reference_maps.items():
            model_summary["references"][ref_key] = evaluate_against_reference(
                candidate_maps=candidate_maps,
                reference_maps=ref_maps,
                paper_ids=common_papers,
            )

        summary["models"][label] = model_summary

    return summary


def model_metric_triplet(summary: dict, label: str, ref_key: str, metric_key: str) -> list[float | None]:
    return summary["models"][label]["references"][ref_key][metric_key]


def main() -> None:
    summary = build_summary()
    METRICS_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for metric_key in DATA:
        for ref in REFERENCES:
            DATA[metric_key][ref["key"]] = [
                model_metric_triplet(summary, label, ref["key"], metric_key)
                for label in ALL_MODELS
            ]

    fig, axes = plt.subplots(1, 3, figsize=(17.8, 7.1), sharey=True)
    fig.subplots_adjust(wspace=0.07, left=0.18, right=0.98, top=0.84, bottom=0.11)

    for ax, (metric_key, cfg) in zip(axes, DATA.items()):
        for i in range(N):
            if i < N_BASE:
                face = "#f0f0ee"
            elif i % 2 == 0:
                face = "#f7f7f5"
            else:
                face = "white"
            ax.axhspan(i - 0.5, i + 0.5, color=face, zorder=0)

        ax.axhline(SEP_Y, color="#999999", lw=1.1, ls=(0, (5, 4)), zorder=5)

        if metric_key in {"spearman", "kendall"}:
            ax.axvline(0, color="#c8c8c8", lw=1.0, zorder=1)

        for i, label in enumerate(ALL_MODELS):
            color = MODEL_COLORS[label]
            for ref in REFERENCES:
                point, lo, hi = cfg[ref["key"]][i]
                y = Y[i] + ref["offset"]
                marker = ref["marker"]

                if point is None:
                    ax.text(
                        cfg["xlim"][0] + 0.01,
                        y,
                        "—",
                        color="#aaaaaa",
                        va="center",
                        ha="left",
                        fontsize=11,
                        zorder=4,
                    )
                    continue

                ax.barh(
                    y,
                    hi - lo,
                    left=lo,
                    height=0.22,
                    color=rgba(color, 0.20),
                    linewidth=0,
                    zorder=2,
                )
                ax.plot([lo, hi], [y, y], color=color, lw=1.8, zorder=3, solid_capstyle="butt")
                for xc in (lo, hi):
                    ax.plot([xc, xc], [y - 0.095, y + 0.095], color=color, lw=1.8, zorder=3)
                ax.plot(
                    point,
                    y,
                    marker,
                    color=color,
                    ms=7,
                    mec="white",
                    mew=1.4,
                    zorder=4,
                )

        ax.set_xlim(cfg["xlim"])
        ax.set_ylim(-0.65, N - 0.35)
        ax.set_xlabel(cfg["xlabel"], labelpad=7)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(6, prune="both"))
        ax.tick_params(axis="x", which="both", bottom=True, top=False)
        ax.tick_params(axis="y", left=False)
        ax.spines["bottom"].set_color("#bbbbbb")
        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")

    axes[0].set_yticks(Y)
    axes[0].set_yticklabels(ALL_MODELS, fontweight="bold")

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=ref["marker"],
            color="none",
            markerfacecolor="#666666",
            markeredgecolor="#666666",
            markeredgewidth=1.0,
            markersize=8,
            label=ref["label"],
        )
        for ref in REFERENCES
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.55, 1.01),
        handlelength=1.8,
        handletextpad=0.6,
        columnspacing=1.2,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_base = PLOTS_DIR / "section_validation_three_refs"
    plt.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.savefig(f"{out_base}.png", bbox_inches="tight", dpi=200)
    print(f"Saved {out_base}.pdf")
    print(f"Saved {out_base}.png")
    print(f"Saved {METRICS_JSON}")


if __name__ == "__main__":
    main()
