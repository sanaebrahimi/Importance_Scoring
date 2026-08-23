import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_PAPERS_DIR = Path("papers")
DEFAULT_SECTIONS_FILE = Path("papers_section_titles.txt")
DEFAULT_RESULTS_ROOT = Path("paper_results")


def sanitize_tag_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return cleaned


def sections_var_name(pdf_path: Path) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", pdf_path.stem).strip("_").upper()
    return f"{base}_SECTIONS"


def load_assignment_names(assignments_path: Path) -> Dict[str, dict]:
    source = assignments_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(assignments_path))
    assignments: Dict[str, dict] = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = ast.literal_eval(node.value)
        if isinstance(value, dict):
            assignments[target.id] = value

    return assignments


def output_files_exist(prefix: Path, model_tag: str, citation_only: bool = False) -> bool:
    tagged_prefix = Path(f"{prefix}_{model_tag}")
    required = [Path(f"{tagged_prefix}_citation_scores.json")]
    if not citation_only:
        required.extend(
            [
                Path(f"{tagged_prefix}_section_scores.json"),
                Path(f"{tagged_prefix}_paragraph_scores.json"),
                Path(f"{tagged_prefix}_paragraph_citation_scores.json"),
            ]
        )
    return all(path.exists() for path in required)


def default_model_tag(baseline_name: str, api_provider: str = "") -> str:
    mapping = {
        "citation_frequency": "citation_frequency",
        "length_weighted_frequency": "length_weighted_frequency",
        "technical_section_prior": "technical_section_prior",
        "single_pass_llm": "single_pass_llm",
        "openai_full_paper": "openai_full_paper",
        "anthropic_full_paper": "anthropic_full_paper",
        "single_shot_citation_api": f"single_shot_citation_{api_provider}" if api_provider else "single_shot_citation_api",
    }
    return mapping[baseline_name]


def compose_model_tag(baseline_name: str, explicit_tag: str, run_suffix: str, api_provider: str = "") -> str:
    base_tag = explicit_tag.strip() or default_model_tag(baseline_name, api_provider=api_provider)
    suffix = sanitize_tag_component(run_suffix)
    return f"{base_tag}_{suffix}" if suffix else base_tag


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one baseline model for every paper/section mapping pair.")
    parser.add_argument(
        "--baseline",
        required=True,
        choices=[
            "citation_frequency",
            "length_weighted_frequency",
            "single_pass_llm",
            "technical_section_prior",
            "openai_full_paper",
            "anthropic_full_paper",
            "single_shot_citation_api",
        ],
        help="Baseline model to run across the full corpus.",
    )
    parser.add_argument("--papers-dir", default=str(DEFAULT_PAPERS_DIR), help="Directory containing PDF papers.")
    parser.add_argument(
        "--sections-file",
        default=str(DEFAULT_SECTIONS_FILE),
        help="Text file containing section-tree assignments.",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory where per-paper result folders will be created.",
    )
    parser.add_argument(
        "--model-tag",
        default="",
        help="Optional suffix for output filenames. Defaults to the baseline name.",
    )
    parser.add_argument(
        "--run-suffix",
        default="",
        help="Optional extra suffix appended to the effective model tag.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Explicit model name used by the selected baseline. Required for single_pass_llm, openai_full_paper, anthropic_full_paper, and single_shot_citation_api.",
    )
    parser.add_argument(
        "--api-provider",
        choices=["openai", "anthropic"],
        default="",
        help="API provider used by single_shot_citation_api. Required for that baseline.",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama host used only by the single_pass_llm baseline.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature used only by the single_pass_llm baseline.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum parse retries used only by the single_pass_llm baseline.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional API key for the selected full-paper API baseline. Prefer using --api-key-env instead of passing secrets on the command line.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable that stores the API key. For anthropic_full_paper, the code automatically falls back to ANTHROPIC_API_KEY; single_shot_citation_api with --api-provider anthropic still falls back to AWS_BEARER_TOKEN_BEDROCK.",
    )
    parser.add_argument(
        "--api-endpoint",
        default="",
        help="Optional full API endpoint URL for the selected full-paper baseline. If omitted, the baseline-specific default endpoint is used.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=600,
        help="HTTP timeout in seconds for full-paper API requests.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=12000,
        help="Maximum output tokens requested from the full-paper API baseline.",
    )
    parser.add_argument(
        "--api-response-format",
        choices=["json_object", "none"],
        default="none",
        help="Response-format hint for OpenAI-compatible baselines. Anthropic ignores this flag.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use when invoking baselines/run_baseline.py.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a paper if its expected baseline output JSON files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining papers if one paper fails.",
    )
    parser.add_argument(
        "--papers",
        nargs="+",
        default=[],
        metavar="STEM",
        help="Run only these paper stems (filename without .pdf). If omitted, all papers are run.",
    )
    args = parser.parse_args()

    papers_dir = Path(args.papers_dir)
    sections_file = Path(args.sections_file)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    assignments = load_assignment_names(sections_file)
    model_tag = compose_model_tag(args.baseline, args.model_tag, args.run_suffix, api_provider=args.api_provider)
    citation_only = args.baseline == "single_shot_citation_api"
    paper_filter = set(args.papers)
    manifest = []

    for pdf_path in sorted(papers_dir.glob("*.pdf")):
        if paper_filter and pdf_path.stem not in paper_filter:
            continue
        section_var = sections_var_name(pdf_path)
        if section_var not in assignments:
            raise SystemExit(
                f"Missing sections mapping for {pdf_path.name}. Expected variable '{section_var}' in {sections_file}."
            )

        paper_dir = results_root / pdf_path.stem
        paper_dir.mkdir(parents=True, exist_ok=True)
        prefix = paper_dir / pdf_path.stem
        debug_log = paper_dir / f"debug_{model_tag}.log"

        command = [
            args.python,
            "baselines/run_baseline.py",
            "--baseline",
            args.baseline,
            "--pdf",
            str(pdf_path),
            "--paper-id",
            pdf_path.stem,
            "--sections-file",
            str(sections_file),
            "--sections-var",
            section_var,
            "--output1",
            str(prefix),
            "--output2",
            str(prefix),
            "--output3",
            str(prefix),
            "--model-tag",
            model_tag,
            "--model",
            args.model,
            "--host",
            args.host,
            "--temperature",
            str(max(0.0, args.temperature)),
            "--max-retries",
            str(max(1, args.max_retries)),
            "--debug-log",
            str(debug_log),
            "--api-key",
            args.api_key,
            "--api-key-env",
            args.api_key_env,
            "--api-endpoint",
            args.api_endpoint,
            "--request-timeout",
            str(max(30, args.request_timeout)),
            "--max-output-tokens",
            str(max(512, args.max_output_tokens)),
            "--api-response-format",
            args.api_response_format,
        ]
        if args.api_provider:
            command.extend(["--api-provider", args.api_provider])

        record = {
            "paper": pdf_path.name,
            "paper_dir": str(paper_dir),
            "baseline": args.baseline,
            "model_tag": model_tag,
            "sections_var": section_var,
            "command": command,
            "skipped": False,
            "status": "pending",
        }

        if args.skip_existing and output_files_exist(prefix, model_tag, citation_only=citation_only):
            record["skipped"] = True
            record["status"] = "skipped"
            manifest.append(record)
            print(f"[skip] {pdf_path.name}: outputs already exist in {paper_dir}")
            continue

        print(f"[run] {pdf_path.name} -> {paper_dir}")
        if args.dry_run:
            record["status"] = "dry_run"
            print(" ".join(command))
        else:
            try:
                subprocess.run(command, check=True, env=os.environ.copy())
                record["status"] = "completed"
            except subprocess.CalledProcessError as exc:
                record["status"] = "failed"
                record["returncode"] = exc.returncode
                print(f"[fail] {pdf_path.name}: command exited with status {exc.returncode}")
                manifest.append(record)
                if args.continue_on_error:
                    continue
                raise

        manifest.append(record)

    manifest_path = results_root / f"manifest_{model_tag}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
