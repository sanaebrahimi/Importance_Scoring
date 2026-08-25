from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
    }
)


MODEL_ORDER = [
    "gemma2:2b",
    "gemma3:4b",
    "llama3.2:1b",
    "llama3.2:3b",
    "qwen2.5:3b",
    "qwen3:1.7b",
    "qwen3:4b",
    "phi3:medium",
    "Citation freq.",
    "Length-wtd. freq.",
]

BASELINE_MODELS = [
    "Citation freq.",
    "Length-wtd. freq.",
]

MODEL_COLORS: Dict[str, str] = {
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

ANCHOR_MODEL = "qwen3:1.7b"
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 17
PRIMARY_REFERENCE_LABEL = "Sonnet-4.6"
SECONDARY_REFERENCE_LABEL = "gpt-oss-120b"

METRICS = [
    {
        "key": "kl_divergence",
        "label": "KL",
        "better": "lower",
        "difference_label": "Model - qwen3:1.7b",
        "summary_label": "Mean KL",
    },
    {
        "key": "jensen_shannon_divergence",
        "label": "JSD",
        "better": "lower",
        "difference_label": "Model - qwen3:1.7b",
        "summary_label": "Mean JSD",
    },
    {
        "key": "spearman",
        "label": r"Spearman $\rho$",
        "better": "higher",
        "difference_label": "qwen3:1.7b - Model",
        "summary_label": r"Mean Spearman $\rho$",
    },
    {
        "key": "top_k_overlap_fraction",
        "label": "Top-4 Overlap",
        "better": "higher",
        "difference_label": "qwen3:1.7b - Model",
        "summary_label": "Mean Top-4 overlap",
    },
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_mean_ci(
    values: Sequence[float],
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Optional[Tuple[float, float]]:
    if not values:
        return None
    if len(values) == 1:
        only = float(values[0])
        return (only, only)
    rng = random.Random(seed)
    n = len(values)
    means: List[float] = []
    for _ in range(n_bootstrap):
        sample = [float(values[rng.randrange(n)]) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_bootstrap)]
    hi = means[min(n_bootstrap - 1, int(0.975 * n_bootstrap))]
    return (lo, hi)


def metric_value(entry: dict, metric_key: str) -> float:
    if metric_key == "top_k_overlap_fraction":
        return float(entry["top_k_overlap"]["overlap_fraction"])
    return float(entry["metrics"][metric_key])


def better_value(left: float, right: float, better: str) -> int:
    if math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
        return 0
    if better == "lower":
        return 1 if left < right else -1
    return 1 if left > right else -1


def resolved_model_order(model_names: Sequence[str]) -> List[str]:
    present = set(model_names)
    ordered = [model for model in MODEL_ORDER if model in present]
    ordered.extend(sorted(model for model in present if model not in ordered))
    return ordered


def paired_advantage_values(
    anchor_entries: Sequence[dict],
    model_entries: Sequence[dict],
    metric_key: str,
    better: str,
) -> List[float]:
    anchor_by_paper = {entry["paper_id"]: entry for entry in anchor_entries}
    model_by_paper = {entry["paper_id"]: entry for entry in model_entries}
    shared = sorted(set(anchor_by_paper) & set(model_by_paper))
    values = []
    for paper_id in shared:
        anchor_value = metric_value(anchor_by_paper[paper_id], metric_key)
        model_value = metric_value(model_by_paper[paper_id], metric_key)
        if better == "lower":
            values.append(model_value - anchor_value)
        else:
            values.append(anchor_value - model_value)
    return values


def pairwise_better_differences(
    left_entries: Sequence[dict],
    right_entries: Sequence[dict],
    metric_key: str,
    better: str,
) -> List[float]:
    left_by_paper = {entry["paper_id"]: entry for entry in left_entries}
    right_by_paper = {entry["paper_id"]: entry for entry in right_entries}
    shared = sorted(set(left_by_paper) & set(right_by_paper))
    values = []
    for paper_id in shared:
        left_value = metric_value(left_by_paper[paper_id], metric_key)
        right_value = metric_value(right_by_paper[paper_id], metric_key)
        if better == "lower":
            values.append(right_value - left_value)
        else:
            values.append(left_value - right_value)
    return values


def build_difference_summary(report: dict) -> Dict[str, dict]:
    per_model = report["per_model_results"]
    anchor_entries = per_model[ANCHOR_MODEL]
    model_order = resolved_model_order(per_model.keys())
    summary: Dict[str, dict] = {}
    for metric in METRICS:
        metric_key = metric["key"]
        metric_payload = {}
        for model in model_order:
            if model == ANCHOR_MODEL:
                continue
            values = paired_advantage_values(
                anchor_entries=anchor_entries,
                model_entries=per_model[model],
                metric_key=metric_key,
                better=metric["better"],
            )
            ci = bootstrap_mean_ci(values, seed=BOOTSTRAP_SEED + len(metric_key) + len(model))
            mean_value = float(np.mean(values)) if values else None
            significant = bool(ci and (ci[0] > 0.0 or ci[1] < 0.0))
            metric_payload[model] = {
                "values": values,
                "mean": mean_value,
                "ci": ci,
                "significant": significant,
            }
        summary[metric_key] = metric_payload
    return summary


def summary_keys_for_metric(metric_key: str) -> Tuple[str, str]:
    if metric_key == "top_k_overlap_fraction":
        return ("mean_top_4_overlap_fraction", "mean_top_4_overlap_fraction_ci")
    if metric_key == "spearman":
        return ("mean_spearman", "mean_spearman_ci")
    if metric_key == "kl_divergence":
        return ("mean_kl_divergence", "mean_kl_divergence_ci")
    return ("mean_jensen_shannon_divergence", "mean_jensen_shannon_divergence_ci")


def plot_difference_from_best(
    output_dir: Path,
    difference_summary: Dict[str, dict],
) -> None:
    models = [model for model in resolved_model_order(difference_summary[METRICS[0]["key"]].keys()) if model != ANCHOR_MODEL]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(18.0, 7.6), constrained_layout=False)
    rng = random.Random(11)

    for ax, metric in zip(axes, METRICS):
        metric_key = metric["key"]
        payload = difference_summary[metric_key]

        all_values = [value for model in models for value in payload[model]["values"]]
        max_abs = max((abs(v) for v in all_values), default=0.05)
        x_pad = max(0.03, max_abs * 0.18)
        ax.axvline(0.0, color="#777777", linewidth=1.1, linestyle="--", zorder=1)

        y_positions = list(range(len(models), 0, -1))
        for y, model in zip(y_positions, models):
            values = payload[model]["values"]
            mean_value = payload[model]["mean"]
            ci = payload[model]["ci"]
            color = MODEL_COLORS[model]

            jittered_y = [y + rng.uniform(-0.12, 0.12) for _ in values]
            ax.scatter(
                values,
                jittered_y,
                s=16,
                alpha=0.28,
                color=color,
                edgecolors="none",
                zorder=2,
            )

            if ci is not None and mean_value is not None:
                ax.errorbar(
                    mean_value,
                    y,
                    xerr=[[mean_value - ci[0]], [ci[1] - mean_value]],
                    fmt="o",
                    color=color,
                    ecolor=color,
                    elinewidth=2.0,
                    capsize=4,
                    markersize=7,
                    zorder=4,
                )
                if payload[model]["significant"]:
                    x_star = ci[1] + x_pad * 0.28 if mean_value >= 0 else ci[0] - x_pad * 0.28
                    ax.text(
                        x_star,
                        y,
                        "*",
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                        color="#222222",
                        zorder=5,
                    )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(models, fontsize=10)
        ax.set_xlabel(metric["difference_label"], fontsize=11)
        ax.set_title(metric["label"], fontsize=13, fontweight="semibold")
        ax.grid(True, axis="x", color="#d9dde2", linewidth=0.8, alpha=0.95)
        ax.grid(False, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.set_xlim(-max_abs - x_pad, max_abs + x_pad)

    fig.suptitle(
        "Difference from qwen3:1.7b across 50 papers\n(* = paired bootstrap 95% CI excludes 0)",
        fontsize=15,
        fontweight="semibold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.12, top=0.84, wspace=0.26)
    for suffix in (".pdf", ".png"):
        fig.savefig(output_dir / f"difference_from_best_dotwhisker{suffix}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_sorted_mean_ci(
    output_dir: Path,
    primary_report: dict,
    secondary_report: dict,
    basename: str = "sorted_mean_ci_cleveland",
) -> None:
    primary_summaries = primary_report["model_summaries"]
    secondary_summaries = secondary_report["model_summaries"]
    model_order = resolved_model_order(
        set(primary_summaries.keys()).intersection(secondary_summaries.keys())
    )
    fig, axes = plt.subplots(1, len(METRICS), figsize=(18.4, 7.6), constrained_layout=False)
    legend_handles = None

    for panel_idx, (ax, metric) in enumerate(zip(axes, METRICS)):
        metric_key = metric["key"]
        better = metric["better"]
        summary_key, ci_key = summary_keys_for_metric(metric_key)

        llm_rows = []
        baseline_rows = []
        for model in model_order:
            primary_mean = float(primary_summaries[model][summary_key])
            primary_ci = primary_summaries[model][ci_key]
            secondary_mean = float(secondary_summaries[model][summary_key])
            secondary_ci = secondary_summaries[model][ci_key]
            row = (model, primary_mean, primary_ci, secondary_mean, secondary_ci)
            if model in BASELINE_MODELS:
                baseline_rows.append(row)
            else:
                llm_rows.append(row)

        reverse = better == "higher"
        llm_rows.sort(key=lambda item: item[1], reverse=reverse)
        baseline_rows.sort(key=lambda item: BASELINE_MODELS.index(item[0]))
        rows = llm_rows + baseline_rows
        y_positions = list(range(len(rows), 0, -1))
        n_rows = len(rows)
        n_baselines = len(baseline_rows)
        n_llms = len(llm_rows)

        values = [row[1] for row in rows] + [row[3] for row in rows]
        lows = [row[2][0] for row in rows] + [row[4][0] for row in rows]
        highs = [row[2][1] for row in rows] + [row[4][1] for row in rows]
        x_min = min(lows)
        x_max = max(highs)
        x_pad = max(0.01, (x_max - x_min) * 0.10)

        for idx, y in enumerate(y_positions):
            if idx < n_llms:
                face = "#f7f7f5" if idx % 2 == 0 else "white"
            else:
                face = "#f0f0ee"
            ax.axhspan(y - 0.5, y + 0.5, color=face, zorder=0)

        if n_baselines:
            sep_y = n_baselines + 0.5
            ax.axhline(sep_y, color="#999999", lw=1.1, ls=(0, (5, 4)), zorder=2)

        primary_handle = None
        secondary_handle = None
        for y, (model, primary_mean, primary_ci, secondary_mean, secondary_ci) in zip(y_positions, rows):
            color = MODEL_COLORS[model]
            ax.barh(
                y + 0.13,
                primary_ci[1] - primary_ci[0],
                left=primary_ci[0],
                height=0.18,
                color=color,
                alpha=0.20,
                linewidth=0,
                zorder=2.4,
            )
            primary_container = ax.errorbar(
                primary_mean,
                y + 0.13,
                xerr=[[primary_mean - primary_ci[0]], [primary_ci[1] - primary_mean]],
                fmt="^",
                color=color,
                ecolor=color,
                elinewidth=2.0,
                capsize=4,
                markersize=8.5,
                zorder=3,
            )
            ax.barh(
                y - 0.13,
                secondary_ci[1] - secondary_ci[0],
                left=secondary_ci[0],
                height=0.18,
                color=color,
                alpha=0.20,
                linewidth=0,
                zorder=2.4,
            )
            secondary_container = ax.errorbar(
                secondary_mean,
                y - 0.13,
                xerr=[[secondary_mean - secondary_ci[0]], [secondary_ci[1] - secondary_mean]],
                fmt="s",
                color=color,
                ecolor=color,
                elinewidth=2.0,
                capsize=4,
                markersize=7.7,
                markerfacecolor="white",
                markeredgewidth=1.8,
                zorder=3,
            )
            if primary_handle is None:
                primary_handle = primary_container.lines[0]
            if secondary_handle is None:
                secondary_handle = secondary_container.lines[0]

        ax.set_yticks(y_positions)
        if panel_idx == 0:
            ax.set_yticklabels([row[0] for row in rows], fontsize=12, fontweight="bold")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(metric["summary_label"], fontsize=13, fontweight="bold")
        ax.grid(True, axis="x", color="#d9dde2", linewidth=0.8, alpha=0.95)
        ax.grid(False, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#bbbbbb")
        ax.set_axisbelow(True)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", left=False)
        for label in ax.get_xticklabels():
            label.set_fontweight("bold")
        legend_handles = (primary_handle, secondary_handle)

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            [PRIMARY_REFERENCE_LABEL, SECONDARY_REFERENCE_LABEL],
            loc="upper center",
            ncol=2,
            frameon=False,
            fontsize=12,
            bbox_to_anchor=(0.5, 0.985),
            handletextpad=0.6,
            columnspacing=1.5,
        )
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.14, top=0.84, wspace=0.08)
    for suffix in (".pdf", ".png"):
        fig.savefig(output_dir / f"{basename}{suffix}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_pairwise_winrate_tables(report: dict) -> Dict[str, np.ndarray]:
    per_model = report["per_model_results"]
    model_order = resolved_model_order(per_model.keys())
    tables: Dict[str, np.ndarray] = {}
    for metric in METRICS:
        metric_key = metric["key"]
        matrix = np.zeros((len(model_order), len(model_order)), dtype=int)
        for i, left_model in enumerate(model_order):
            for j, right_model in enumerate(model_order):
                if i == j:
                    matrix[i, j] = 0
                    continue
                left_by_paper = {entry["paper_id"]: entry for entry in per_model[left_model]}
                right_by_paper = {entry["paper_id"]: entry for entry in per_model[right_model]}
                shared = sorted(set(left_by_paper) & set(right_by_paper))
                wins = 0
                for paper_id in shared:
                    left_value = metric_value(left_by_paper[paper_id], metric_key)
                    right_value = metric_value(right_by_paper[paper_id], metric_key)
                    if better_value(left_value, right_value, metric["better"]) > 0:
                        wins += 1
                matrix[i, j] = wins
        tables[metric_key] = matrix
    return tables


def plot_winrate_heatmaps(output_dir: Path, tables: Dict[str, np.ndarray], n_papers: int) -> None:
    model_order = resolved_model_order(MODEL_ORDER[: tables[METRICS[0]["key"]].shape[0]])
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 13.2), constrained_layout=False)
    axes = axes.flatten()
    cmap = plt.get_cmap("YlGnBu")

    for ax, metric in zip(axes, METRICS):
        matrix = tables[metric["key"]]
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=n_papers)
        ax.set_title(metric["label"], fontsize=13, fontweight="semibold")
        ax.set_xticks(range(len(model_order)))
        ax.set_yticks(range(len(model_order)))
        ax.set_xticklabels(model_order, rotation=34, ha="right", rotation_mode="anchor", fontsize=11)
        ax.set_yticklabels(model_order, fontsize=11)
        for label in ax.get_xticklabels():
            label.set_fontweight("bold")
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")
        for i in range(len(model_order)):
            for j in range(len(model_order)):
                text = "—" if i == j else str(int(matrix[i, j]))
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=10.2,
                    fontweight="bold",
                    color="#102235",
                )

    fig.subplots_adjust(left=0.12, right=0.90, bottom=0.16, top=0.90, wspace=0.25, hspace=0.28)
    cax = fig.add_axes([0.915, 0.21, 0.015, 0.56])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Papers where row model beats column model", fontsize=9.5, labelpad=10)
    cbar.ax.tick_params(labelsize=8.5)
    fig.suptitle("Pairwise win-rate heatmaps across 50 papers", fontsize=15, fontweight="semibold", y=0.98)
    for suffix in (".pdf", ".png"):
        fig.savefig(output_dir / f"pairwise_winrate_heatmaps{suffix}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_pairwise_significance_tables(report: dict) -> Dict[str, dict]:
    per_model = report["per_model_results"]
    model_order = resolved_model_order(per_model.keys())
    results: Dict[str, dict] = {}

    for metric in METRICS:
        metric_key = metric["key"]
        status = np.zeros((len(model_order), len(model_order)), dtype=int)
        mean_diff = np.zeros((len(model_order), len(model_order)), dtype=float)
        ci_low = np.zeros((len(model_order), len(model_order)), dtype=float)
        ci_high = np.zeros((len(model_order), len(model_order)), dtype=float)

        for i, left_model in enumerate(model_order):
            for j, right_model in enumerate(model_order):
                if i == j:
                    continue
                deltas = pairwise_better_differences(
                    left_entries=per_model[left_model],
                    right_entries=per_model[right_model],
                    metric_key=metric_key,
                    better=metric["better"],
                )
                mean_value = float(np.mean(deltas)) if deltas else 0.0
                ci = bootstrap_mean_ci(deltas, seed=BOOTSTRAP_SEED + i * 31 + j * 17 + len(metric_key))
                mean_diff[i, j] = mean_value
                ci_low[i, j] = ci[0] if ci else 0.0
                ci_high[i, j] = ci[1] if ci else 0.0
                if ci is not None:
                    if ci[0] > 0.0:
                        status[i, j] = 1
                    elif ci[1] < 0.0:
                        status[i, j] = -1

        results[metric_key] = {
            "status": status,
            "mean_diff": mean_diff,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    return results


def plot_significance_heatmaps(output_dir: Path, tables: Dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 13.2), constrained_layout=False)
    axes = axes.flatten()
    cmap = plt.get_cmap("RdYlGn")

    for ax, metric in zip(axes, METRICS):
        status = tables[metric["key"]]["status"]
        im = ax.imshow(status, cmap=cmap, vmin=-1, vmax=1)
        ax.set_title(metric["label"], fontsize=13, fontweight="semibold")
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_yticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels(MODEL_ORDER, rotation=40, ha="right", fontsize=9)
        ax.set_yticklabels(MODEL_ORDER, fontsize=9)
        for i in range(len(MODEL_ORDER)):
            for j in range(len(MODEL_ORDER)):
                if i == j:
                    label = "—"
                else:
                    code = int(status[i, j])
                    label = "+" if code > 0 else "-" if code < 0 else "ns"
                ax.text(j, i, label, ha="center", va="center", fontsize=8.4, color="#102235")

    fig.subplots_adjust(left=0.09, right=0.90, bottom=0.12, top=0.90, wspace=0.25, hspace=0.28)
    cax = fig.add_axes([0.915, 0.21, 0.015, 0.56])
    cbar = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(["row worse", "not signif.", "row better"])
    cbar.ax.tick_params(labelsize=8.5)
    fig.suptitle("Pairwise bootstrap significance heatmaps (95% CI on paired mean difference)", fontsize=15, fontweight="semibold", y=0.98)
    for suffix in (".pdf", ".png"):
        fig.savefig(output_dir / f"pairwise_significance_heatmaps{suffix}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_difference_tsv(output_dir: Path, difference_summary: Dict[str, dict]) -> None:
    path = output_dir / "difference_from_best_summary.tsv"
    model_order = resolved_model_order(difference_summary[METRICS[0]["key"]].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "metric",
                "model",
                "mean_difference",
                "ci_low",
                "ci_high",
                "significant_vs_anchor",
            ]
        )
        for metric in METRICS:
            metric_key = metric["key"]
            for model in model_order:
                if model == ANCHOR_MODEL:
                    continue
                payload = difference_summary[metric_key][model]
                ci = payload["ci"] or ("", "")
                writer.writerow(
                    [
                        metric_key,
                        model,
                        f"{payload['mean']:.6f}" if payload["mean"] is not None else "",
                        f"{ci[0]:.6f}" if ci[0] != "" else "",
                        f"{ci[1]:.6f}" if ci[1] != "" else "",
                        "yes" if payload["significant"] else "no",
                    ]
                )


def write_pairwise_tables(
    output_dir: Path,
    winrate_tables: Dict[str, np.ndarray],
    significance_tables: Dict[str, dict],
) -> None:
    model_order = resolved_model_order(MODEL_ORDER[: winrate_tables[METRICS[0]["key"]].shape[0]])
    for metric in METRICS:
        metric_key = metric["key"]

        winrate_path = output_dir / f"pairwise_winrates_{metric_key}.tsv"
        with open(winrate_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["model", *model_order])
            matrix = winrate_tables[metric_key]
            for i, model in enumerate(model_order):
                writer.writerow([model, *[int(value) for value in matrix[i]]])

        sig_path = output_dir / f"pairwise_significance_{metric_key}.tsv"
        with open(sig_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["row_model", "col_model", "status", "mean_difference", "ci_low", "ci_high"])
            payload = significance_tables[metric_key]
            for i, row_model in enumerate(model_order):
                for j, col_model in enumerate(model_order):
                    if i == j:
                        continue
                    writer.writerow(
                        [
                            row_model,
                            col_model,
                            int(payload["status"][i, j]),
                            f"{payload['mean_diff'][i, j]:.6f}",
                            f"{payload['ci_low'][i, j]:.6f}",
                            f"{payload['ci_high'][i, j]:.6f}",
                        ]
                    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate final publication figures for the Sonnet-4.6 vs local-model citation comparison."
    )
    parser.add_argument(
        "--comparison-json",
        default="results/anthropic_promptv2_citation_model_comparison.json",
    )
    parser.add_argument(
        "--secondary-comparison-json",
        default="results/openai_gptoss_promptv2_citation_model_comparison.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/final_sonnet46_model_comparison",
    )
    parser.add_argument(
        "--sorted-mean-basename",
        default="sorted_mean_ci_cleveland",
    )
    parser.add_argument(
        "--only-sorted-mean",
        action="store_true",
        help="Generate only the sorted mean + 95% CI Cleveland plot.",
    )
    args = parser.parse_args()

    report = load_json(Path(args.comparison_json))
    secondary_report = load_json(Path(args.secondary_comparison_json))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.only_sorted_mean:
        plot_sorted_mean_ci(
            output_dir,
            report,
            secondary_report,
            basename=args.sorted_mean_basename,
        )
    else:
        difference_summary = build_difference_summary(report)
        winrate_tables = build_pairwise_winrate_tables(report)
        plot_difference_from_best(output_dir, difference_summary)
        plot_sorted_mean_ci(
            output_dir,
            report,
            secondary_report,
            basename=args.sorted_mean_basename,
        )
        plot_winrate_heatmaps(output_dir, winrate_tables, n_papers=int(report["shared_paper_count"]))

    manifest = {
        "comparison_json": args.comparison_json,
        "secondary_comparison_json": args.secondary_comparison_json,
        "output_dir": str(output_dir),
        "shared_paper_count": int(report["shared_paper_count"]),
        "anchor_model": ANCHOR_MODEL,
        "model_order": MODEL_ORDER,
        "metrics": [metric["key"] for metric in METRICS],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved final comparison artifacts to {output_dir}")


if __name__ == "__main__":
    main()
