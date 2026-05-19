import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict


DEFAULT_PAPERS_DIR = Path("papers")
DEFAULT_SECTIONS_FILE = Path("papers_section_titles.txt")
DEFAULT_RESULTS_ROOT = Path("paper_results")
DEFAULT_VENDOR_DIR = Path(".vendor")


def sanitize_model_tag(model: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").lower()
    return cleaned or "model"


def sanitize_tag_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return cleaned


def compose_model_tag(model: str, explicit_tag: str, run_suffix: str) -> str:
    base_tag = explicit_tag.strip() or sanitize_model_tag(model)
    suffix = sanitize_tag_component(run_suffix)
    return f"{base_tag}_{suffix}" if suffix else base_tag


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


def sections_var_name(pdf_path: Path) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", pdf_path.stem).strip("_").upper()
    return f"{base}_SECTIONS"


def output_files_exist(prefix: Path, model_tag: str = "") -> bool:
    tagged_prefix = Path(f"{prefix}_{model_tag}") if model_tag else prefix
    return all(
        path.exists()
        for path in (
            Path(f"{tagged_prefix}_citation_scores.json"),
            Path(f"{tagged_prefix}_section_scores.json"),
            Path(f"{tagged_prefix}_paragraph_scores.json"),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run importance_score.py for every paper/section mapping pair.")
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
    parser.add_argument("--model", default="llama3.2", help="Ollama model name to pass through.")
    parser.add_argument(
        "--model-tag",
        default="",
        help="Optional suffix for output/debug/prompt filenames. Defaults to a sanitized version of --model.",
    )
    parser.add_argument(
        "--run-suffix",
        default="",
        help=(
            "Optional extra suffix appended to the effective model tag so reruns write to new files "
            "(for example: --model-tag gemma2_2b --run-suffix run2 -> gemma2_2b_run2)."
        ),
    )
    parser.add_argument("--host", default="localhost:11434", help="Ollama host to pass through.")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of LLM samples to average.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Base sampling temperature.")
    parser.add_argument(
        "--sample-temperature-jitter",
        type=float,
        default=0.03,
        help="Maximum per-sample temperature increase added on top of --temperature.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible per-sample temperature jitter.",
    )
    parser.add_argument("--max-retries", type=int, default=5, help="Retries per sample for parsing.")
    parser.add_argument(
        "--paragraph-direct-max-tokens",
        type=int,
        default=0,
        help="Token threshold for optional paragraph compression.",
    )
    parser.add_argument(
        "--paragraph-compressed-snippet-limit",
        type=int,
        default=180,
        help="Per-paragraph snippet length used during compressed scoring.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use when invoking importance_score.py.",
    )
    parser.add_argument(
        "--vendor-dir",
        default=str(DEFAULT_VENDOR_DIR),
        help="Optional local site-packages directory to prepend to PYTHONPATH if it exists.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a paper if its three output JSON files already exist.",
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
    vendor_dir = Path(args.vendor_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    model_tag = compose_model_tag(args.model, args.model_tag, args.run_suffix)
    paper_filter = set(args.papers)

    assignments = load_assignment_names(sections_file)
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
        debug_log = paper_dir / "debug.log"
        prompts_output = paper_dir / "prompts.json"

        outputs_present = output_files_exist(prefix, model_tag=model_tag)
        command = [
            args.python,
            "importance_score.py",
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
            "--model",
            args.model,
            "--model-tag",
            model_tag,
            "--host",
            args.host,
            "--n-samples",
            str(max(1, args.n_samples)),
            "--temperature",
            str(max(0.0, args.temperature)),
            "--sample-temperature-jitter",
            str(max(0.0, args.sample_temperature_jitter)),
            "--max-retries",
            str(max(1, args.max_retries)),
            "--paragraph-direct-max-tokens",
            str(max(0, args.paragraph_direct_max_tokens)),
            "--paragraph-compressed-snippet-limit",
            str(max(60, args.paragraph_compressed_snippet_limit)),
            "--debug-log",
            str(debug_log),
            "--prompts-output",
            str(prompts_output),
        ]
        if args.seed is not None:
            command.extend(["--seed", str(args.seed)])

        record = {
            "paper": pdf_path.name,
            "paper_dir": str(paper_dir),
            "model_tag": model_tag,
            "sections_var": section_var,
            "command": command,
            "skipped": False,
            "status": "pending",
        }

        if args.skip_existing and outputs_present:
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
            env = os.environ.copy()
            if vendor_dir.exists():
                current_pythonpath = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = (
                    f"{vendor_dir}{os.pathsep}{current_pythonpath}" if current_pythonpath else str(vendor_dir)
                )
            try:
                subprocess.run(command, check=True, env=env)
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

    manifest_name = f"manifest_{model_tag}.json" if model_tag else "manifest.json"
    manifest_path = results_root / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
