from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

from plot_final_sonnet46_model_comparison import (
    BASELINE_MODELS,
    METRICS,
    MODEL_ORDER,
    PRIMARY_REFERENCE_LABEL,
    SECONDARY_REFERENCE_LABEL,
    build_difference_summary,
    build_pairwise_winrate_tables,
    load_json,
    metric_value,
    resolved_model_order,
    summary_keys_for_metric,
)
from plot_section_validation import ALL_MODELS as SECTION_MODEL_ORDER


ROOT = Path(__file__).resolve().parent
FINAL_DIR = ROOT / "results" / "final_sonnet46_model_comparison"

PRIMARY_COMPARISON_JSON = ROOT / "results" / "anthropic_promptv2_plus_baselines_citation_model_comparison.json"
SECONDARY_COMPARISON_JSON = ROOT / "results" / "openai_gptoss_plus_baselines_citation_model_comparison.json"
SECTION_VALIDATION_JSON = ROOT / "results" / "section_validation_three_refs_metrics.json"

TOP4_BAR_EXPORTS = {
    "anthropic_promptv2_top4_bars.tsv": (
        ROOT / "results" / "anthropic_promptv2_top4_bar_metrics.json",
        "Sonnet-4.6",
    ),
    "openai_promptv2_top4_bars.tsv": (
        ROOT / "results" / "openai_promptv2_top4_bar_metrics.json",
        "gpt-oss-120b",
    ),
    "human_promptv2_top4_bars.tsv": (
        FINAL_DIR / "human_promptv2_top4_bar_metrics.json",
        "Human",
    ),
}

VIOLIN_EXPORTS = {
    "anthropic_promptv2_kl_violin.tsv": (
        PRIMARY_COMPARISON_JSON,
        "kl_divergence",
        "KL",
        "Sonnet-4.6",
    ),
    "anthropic_promptv2_jsd_violin.tsv": (
        PRIMARY_COMPARISON_JSON,
        "jensen_shannon_divergence",
        "JSD",
        "Sonnet-4.6",
    ),
    "openai_gptoss_promptv2_kl_violin.tsv": (
        SECONDARY_COMPARISON_JSON,
        "kl_divergence",
        "KL",
        "gpt-oss-120b",
    ),
    "openai_gptoss_promptv2_jsd_violin.tsv": (
        SECONDARY_COMPARISON_JSON,
        "jensen_shannon_divergence",
        "JSD",
        "gpt-oss-120b",
    ),
}


def write_tsv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def fmt_float(value: object) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.6f}"


def export_difference_tsvs(primary_report: dict) -> None:
    difference_summary = build_difference_summary(primary_report)
    model_order = resolved_model_order(primary_report["per_model_results"].keys())
    anchor_entries = {
        entry["paper_id"]: entry
        for entry in primary_report["per_model_results"]["qwen3:1.7b"]
    }

    summary_rows = []
    per_paper_rows = []
    for metric in METRICS:
        metric_key = metric["key"]
        for model in model_order:
            if model == "qwen3:1.7b":
                continue
            payload = difference_summary[metric_key][model]
            ci = payload["ci"] or ("", "")
            summary_rows.append(
                [
                    metric_key,
                    metric["label"],
                    model,
                    fmt_float(payload["mean"]),
                    fmt_float(ci[0]),
                    fmt_float(ci[1]),
                    "yes" if payload["significant"] else "no",
                ]
            )

            for entry in primary_report["per_model_results"][model]:
                paper_id = entry["paper_id"]
                anchor_entry = anchor_entries[paper_id]
                anchor_value = metric_value(anchor_entry, metric_key)
                model_value = metric_value(entry, metric_key)
                plotted_difference = model_value - anchor_value
                if metric["better"] == "higher":
                    plotted_difference = anchor_value - model_value
                per_paper_rows.append(
                    [
                        metric_key,
                        metric["label"],
                        model,
                        paper_id,
                        fmt_float(anchor_value),
                        fmt_float(model_value),
                        fmt_float(plotted_difference),
                    ]
                )

    write_tsv(
        FINAL_DIR / "difference_from_best_summary.tsv",
        [
            "metric_key",
            "metric_label",
            "model",
            "mean_difference",
            "ci_low",
            "ci_high",
            "significant_vs_anchor",
        ],
        summary_rows,
    )
    write_tsv(
        FINAL_DIR / "difference_from_best_per_paper.tsv",
        [
            "metric_key",
            "metric_label",
            "model",
            "paper_id",
            "anchor_value",
            "model_value",
            "plotted_difference",
        ],
        per_paper_rows,
    )


def export_sorted_mean_tsvs(primary_report: dict, secondary_report: dict) -> None:
    primary_summaries = primary_report["model_summaries"]
    secondary_summaries = secondary_report["model_summaries"]
    model_order = resolved_model_order(
        set(primary_summaries.keys()).intersection(secondary_summaries.keys())
    )

    def build_rows(include_baselines: bool) -> list[list[object]]:
        rows = []
        for metric in METRICS:
            metric_key = metric["key"]
            summary_key, ci_key = summary_keys_for_metric(metric_key)
            filtered_models = [
                model
                for model in model_order
                if include_baselines or model not in BASELINE_MODELS
            ]

            sortable = []
            for model in filtered_models:
                sortable.append((model, float(primary_summaries[model][summary_key])))
            sortable.sort(key=lambda item: item[1], reverse=metric["better"] == "higher")

            for sort_rank, (model, _) in enumerate(sortable, start=1):
                primary_ci = primary_summaries[model][ci_key]
                secondary_ci = secondary_summaries[model][ci_key]
                rows.append(
                    [
                        metric_key,
                        metric["label"],
                        model,
                        sort_rank,
                        "yes" if model in BASELINE_MODELS else "no",
                        PRIMARY_REFERENCE_LABEL,
                        fmt_float(primary_summaries[model][summary_key]),
                        fmt_float(primary_ci[0]),
                        fmt_float(primary_ci[1]),
                        SECONDARY_REFERENCE_LABEL,
                        fmt_float(secondary_summaries[model][summary_key]),
                        fmt_float(secondary_ci[0]),
                        fmt_float(secondary_ci[1]),
                        int(primary_summaries[model]["papers_evaluated"]),
                    ]
                )
        return rows

    header = [
        "metric_key",
        "metric_label",
        "model",
        "sort_rank",
        "is_baseline",
        "primary_reference",
        "primary_mean",
        "primary_ci_low",
        "primary_ci_high",
        "secondary_reference",
        "secondary_mean",
        "secondary_ci_low",
        "secondary_ci_high",
        "papers_evaluated",
    ]
    write_tsv(FINAL_DIR / "sorted_mean_ci_cleveland.tsv", header, build_rows(False))
    write_tsv(
        FINAL_DIR / "sorted_mean_ci_cleveland_with_baselines.tsv",
        header,
        build_rows(True),
    )


def export_pairwise_winrate_tsvs(primary_report: dict) -> None:
    tables = build_pairwise_winrate_tables(primary_report)
    model_order = resolved_model_order(primary_report["per_model_results"].keys())
    for metric in METRICS:
        metric_key = metric["key"]
        matrix = tables[metric_key]
        rows = []
        for i, row_model in enumerate(model_order):
            rows.append([row_model, *[int(value) for value in matrix[i]]])
        write_tsv(
            FINAL_DIR / f"pairwise_winrates_{metric_key}.tsv",
            ["model", *model_order],
            rows,
        )


def export_violin_tsvs() -> None:
    for filename, (json_path, metric_key, metric_label, reference_label) in VIOLIN_EXPORTS.items():
        report = load_json(json_path)
        model_order = resolved_model_order(report["per_model_results"].keys())
        rows = []
        for model in model_order:
            for entry in report["per_model_results"][model]:
                rows.append(
                    [
                        reference_label,
                        metric_key,
                        metric_label,
                        model,
                        entry["paper_id"],
                        fmt_float(metric_value(entry, metric_key)),
                    ]
                )
        write_tsv(
            FINAL_DIR / filename,
            ["reference", "metric_key", "metric_label", "model", "paper_id", "value"],
            rows,
        )


def export_top4_bar_tsvs() -> None:
    for filename, (json_path, reference_label) in TOP4_BAR_EXPORTS.items():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        rows = []
        for model_payload in payload["models"]:
            for metric_name, metric_value_raw in model_payload["metrics"].items():
                rows.append(
                    [
                        reference_label,
                        payload.get("reference_tag", payload.get("reference_source", "")),
                        payload["papers_evaluated"],
                        model_payload["display_name"],
                        metric_name,
                        fmt_float(metric_value_raw),
                    ]
                )
        write_tsv(
            FINAL_DIR / filename,
            ["reference", "reference_tag", "papers_evaluated", "model", "metric", "value"],
            rows,
        )


def export_section_validation_tsv() -> None:
    payload = json.loads(SECTION_VALIDATION_JSON.read_text(encoding="utf-8"))
    rows = []
    for model in [name for name in SECTION_MODEL_ORDER if name in payload["models"]]:
        model_payload = payload["models"][model]
        for reference_key in ("human", "openai", "anthropic"):
            reference_payload = model_payload["references"][reference_key]
            rows.append(
                [
                    model,
                    reference_key,
                    reference_payload["papers_evaluated"],
                    fmt_float(reference_payload["spearman"][0]),
                    fmt_float(reference_payload["spearman"][1]),
                    fmt_float(reference_payload["spearman"][2]),
                    fmt_float(reference_payload["kendall"][0]),
                    fmt_float(reference_payload["kendall"][1]),
                    fmt_float(reference_payload["kendall"][2]),
                    fmt_float(reference_payload["l1"][0]),
                    fmt_float(reference_payload["l1"][1]),
                    fmt_float(reference_payload["l1"][2]),
                    "|".join(model_payload.get("model_tags", [])),
                ]
            )
    write_tsv(
        FINAL_DIR / "section_validation_three_refs.tsv",
        [
            "model",
            "reference",
            "papers_evaluated",
            "spearman_mean",
            "spearman_ci_low",
            "spearman_ci_high",
            "kendall_mean",
            "kendall_ci_low",
            "kendall_ci_high",
            "l1_mean",
            "l1_ci_low",
            "l1_ci_high",
            "model_tags",
        ],
        rows,
    )


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    primary_report = load_json(PRIMARY_COMPARISON_JSON)
    secondary_report = load_json(SECONDARY_COMPARISON_JSON)

    export_difference_tsvs(primary_report)
    export_sorted_mean_tsvs(primary_report, secondary_report)
    export_pairwise_winrate_tsvs(primary_report)
    export_violin_tsvs()
    export_top4_bar_tsvs()
    export_section_validation_tsv()
    print(f"Saved TSV exports to {FINAL_DIR}")


if __name__ == "__main__":
    main()
