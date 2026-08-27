import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from evaluate_human_section_scores import (
    align_section_score_maps,
    bootstrap_ci,
    jensen_shannon_divergence,
    kendall_tau_b,
    l1_distance,
    load_json,
    normalize_scores,
    spearman_rho,
)


ROOT = Path(__file__).resolve().parent
ROUND1_PATH = ROOT / "human_expert_annotations.json"
ROUND2_PATH = ROOT / "human_expert_annotations_round2.json"
OUT_DIR = ROOT / "results" / "human_annotation_noise"

BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 7

plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "figure.titlesize": 16,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "figure.titleweight": "bold",
    }
)


def mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def top_k_labels(score_map: Dict[str, float], k: int) -> List[str]:
    return [
        label
        for label, _ in sorted(
            score_map.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )[:k]
    ]


def top_1_agreement(score_map_a: Dict[str, float], score_map_b: Dict[str, float]) -> float:
    if not score_map_a or not score_map_b:
        return 0.0
    max_a = max(score_map_a.values())
    max_b = max(score_map_b.values())
    top_a = {label for label, value in score_map_a.items() if value == max_a}
    top_b = {label for label, value in score_map_b.items() if value == max_b}
    return 1.0 if top_a & top_b else 0.0


def top_k_overlap_fraction(score_map_a: Dict[str, float], score_map_b: Dict[str, float], k: int) -> float:
    top_a = set(top_k_labels(score_map_a, k))
    top_b = set(top_k_labels(score_map_b, k))
    if not top_a or not top_b:
        return 0.0
    denom = float(k)
    return len(top_a & top_b) / denom


def format_ci(mean_value: float, ci: Tuple[float, float]) -> str:
    return f"{mean_value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def load_round(path: Path) -> Dict[str, Dict[str, float]]:
    payload = load_json(path)
    return {
        paper_id: paper_payload["top_level_scores_for_evaluation"]
        for paper_id, paper_payload in payload["papers"].items()
    }


def evaluate_rounds(
    round1_maps: Dict[str, Dict[str, float]],
    round2_maps: Dict[str, Dict[str, float]],
) -> Tuple[List[dict], dict]:
    common_papers = sorted(set(round1_maps) & set(round2_maps))
    per_paper: List[dict] = []

    metrics = {
        "spearman": [],
        "kendall_tau_b": [],
        "l1": [],
        "jsd": [],
        "top_1_agreement": [],
        "top_2_overlap": [],
    }

    for paper_id in common_papers:
        overlap, round1_raw, round2_raw, extras = align_section_score_maps(
            round1_maps[paper_id],
            round2_maps[paper_id],
        )
        if len(overlap) < 2:
            continue

        round1_norm = normalize_scores(round1_raw)
        round2_norm = normalize_scores(round2_raw)
        ordered_1 = [round1_norm[label] for label in overlap]
        ordered_2 = [round2_norm[label] for label in overlap]

        paper_metrics = {
            "paper_id": paper_id,
            "n_shared_sections": len(overlap),
            "shared_sections": overlap,
            "round2_extra_sections": extras,
            "spearman": spearman_rho(ordered_1, ordered_2),
            "kendall_tau_b": kendall_tau_b(ordered_1, ordered_2),
            "l1": l1_distance(ordered_1, ordered_2),
            "jsd": jensen_shannon_divergence(ordered_1, ordered_2),
            "top_1_agreement": top_1_agreement(round1_norm, round2_norm),
            "top_2_overlap": top_k_overlap_fraction(round1_norm, round2_norm, 2),
        }
        per_paper.append(paper_metrics)

        for key in metrics:
            metrics[key].append(paper_metrics[key])

    summary = {
        "papers_evaluated": len(per_paper),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "metrics": {},
    }
    seed_offsets = {
        "spearman": 1,
        "kendall_tau_b": 2,
        "l1": 3,
        "jsd": 4,
        "top_1_agreement": 5,
        "top_2_overlap": 6,
    }
    for key, values in metrics.items():
        summary["metrics"][key] = {
            "mean": mean(values),
            "bootstrap_ci": bootstrap_ci(values, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED + seed_offsets[key]),
        }

    return per_paper, summary


def write_summary_table(summary: dict) -> None:
    metric_rows = [
        ("Mean Spearman $\\rho$ $\\uparrow$", "spearman"),
        ("Mean Kendall $\\tau_b$ $\\uparrow$", "kendall_tau_b"),
        ("Mean $L_1$ $\\downarrow$", "l1"),
        ("Mean JSD $\\downarrow$", "jsd"),
        ("Top-1 agreement $\\uparrow$", "top_1_agreement"),
        ("Top-2 overlap $\\uparrow$", "top_2_overlap"),
    ]

    tsv_lines = ["metric\tmean\tci_low\tci_high"]
    tex_lines = [
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Metric & Human Round 1 vs. Round 2 \\\\",
        "\\midrule",
    ]

    for label, key in metric_rows:
        mean_value = summary["metrics"][key]["mean"]
        ci = summary["metrics"][key]["bootstrap_ci"]
        tsv_lines.append(f"{key}\t{mean_value:.6f}\t{ci[0]:.6f}\t{ci[1]:.6f}")
        tex_lines.append(
            f"{label} & {mean_value:.3f} {{\\scriptsize[$ {ci[0]:.3f},\\,{ci[1]:.3f} $]}} \\\\"
        )

    tex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )

    (OUT_DIR / "human_annotation_rounds_summary.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
    (OUT_DIR / "human_annotation_rounds_summary.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def write_per_paper_tsv(per_paper: Sequence[dict]) -> None:
    lines = [
        "paper_id\tn_shared_sections\tspearman\tkendall_tau_b\tl1\tjsd\ttop_1_agreement\ttop_2_overlap"
    ]
    for row in per_paper:
        lines.append(
            "\t".join(
                [
                    row["paper_id"],
                    str(row["n_shared_sections"]),
                    f"{row['spearman']:.6f}",
                    f"{row['kendall_tau_b']:.6f}",
                    f"{row['l1']:.6f}",
                    f"{row['jsd']:.6f}",
                    f"{row['top_1_agreement']:.6f}",
                    f"{row['top_2_overlap']:.6f}",
                ]
            )
        )
    (OUT_DIR / "human_annotation_rounds_per_paper.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_dot_whisker(summary: dict, per_paper: Sequence[dict]) -> None:
    metric_specs = [
        ("spearman", r"Spearman $\rho$ $\uparrow$", "#4e79a7"),
        ("kendall_tau_b", r"Kendall $\tau_b$ $\uparrow$", "#f28e2b"),
        ("l1", r"$L_1$ $\downarrow$", "#e15759"),
        ("jsd", "JSD ↓", "#76b7b2"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.8))
    fig.patch.set_facecolor("white")
    rng = np.random.default_rng(7)

    for ax, (metric_key, label, color) in zip(axes.flatten(), metric_specs):
        ax.set_facecolor("white")
        values = [row[metric_key] for row in per_paper]
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            values,
            np.zeros(len(values)) + jitter,
            s=20,
            color=mcolors.to_rgba(color, 0.22),
            edgecolors="none",
            zorder=1,
        )

        mean_value = summary["metrics"][metric_key]["mean"]
        ci_low, ci_high = summary["metrics"][metric_key]["bootstrap_ci"]
        ax.hlines(0.0, ci_low, ci_high, color=color, linewidth=3.0, zorder=3)
        ax.scatter(
            [mean_value],
            [0.0],
            s=70,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=4,
        )

        margin = max((max(values) - min(values)) * 0.12, 0.02)
        x_min = min(min(values), ci_low) - margin
        x_max = max(max(values), ci_high) + margin
        if metric_key in {"top_1_agreement", "top_2_overlap"}:
            x_min = max(-0.02, x_min)
            x_max = min(1.02, x_max)
        if metric_key == "jsd":
            x_min = max(-0.001, x_min)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.22, 0.22)
        ax.set_yticks([])
        ax.grid(axis="x", color="#d9d9d9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", labelsize=11)
        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")
        ax.set_title(label, pad=8)
        ax.set_xlabel("Per-paper values", fontweight="bold", fontsize=12)

    fig.suptitle("Human Annotation Noise at the Section Level", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "human_annotation_rounds_dot_whisker.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(OUT_DIR / "human_annotation_rounds_dot_whisker.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_noise_heatmap(per_paper: Sequence[dict]) -> None:
    ordered = sorted(per_paper, key=lambda row: (row["jsd"], row["l1"]), reverse=True)
    papers = [row["paper_id"] for row in ordered]
    values = np.array([[row["l1"], row["jsd"]] for row in ordered], dtype=float)

    fig_height = max(6.0, 0.22 * len(papers) + 1.6)
    fig, ax = plt.subplots(figsize=(4.8, fig_height))
    im = ax.imshow(values, aspect="auto", cmap="OrRd")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["L1", "JSD"], fontweight="bold")
    ax.set_yticks(np.arange(len(papers)))
    ax.set_yticklabels(papers, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=12)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#2f2f2f",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.ax.tick_params(labelsize=11)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    cbar.set_label("Disagreement", fontweight="bold")

    ax.set_title("Per-paper Human Annotation Noise", pad=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "human_annotation_rounds_noise_heatmap.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(OUT_DIR / "human_annotation_rounds_noise_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary_json(per_paper: Sequence[dict], summary: dict) -> None:
    payload = {
        "round_1": str(ROUND1_PATH.name),
        "round_2": str(ROUND2_PATH.name),
        "summary": summary,
        "per_paper": per_paper,
    }
    (OUT_DIR / "human_annotation_rounds_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    round1_maps = load_round(ROUND1_PATH)
    round2_maps = load_round(ROUND2_PATH)
    per_paper, summary = evaluate_rounds(round1_maps, round2_maps)

    write_summary_json(per_paper, summary)
    write_summary_table(summary)
    write_per_paper_tsv(per_paper)
    plot_dot_whisker(summary, per_paper)
    plot_noise_heatmap(per_paper)

    print(f"Evaluated {summary['papers_evaluated']} papers")
    print(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
