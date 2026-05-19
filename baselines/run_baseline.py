import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.models import run_baseline_from_args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one of the baseline models and write outputs in the standard JSON format."
    )
    parser.add_argument(
        "--baseline",
        required=True,
        choices=["citation_frequency", "length_weighted_frequency", "single_pass_llm", "technical_section_prior"],
        help="Baseline model to run.",
    )
    parser.add_argument("--pdf", required=True, help="Path to the source PDF.")
    parser.add_argument("--paper-id", required=True, help="Paper identifier used in output files.")
    parser.add_argument("--sections-file", required=True, help="Path to papers_section_titles.txt.")
    parser.add_argument("--sections-var", required=True, help="Variable name for this paper inside the sections file.")
    parser.add_argument("--output1", required=True, help="Prefix for citation output file.")
    parser.add_argument("--output2", required=True, help="Prefix for section output file.")
    parser.add_argument(
        "--output3",
        default="",
        help="Optional prefix for paragraph outputs. Defaults to --output2.",
    )
    parser.add_argument(
        "--model-tag",
        default="",
        help="Optional suffix to append to output files. Defaults to the baseline name.",
    )
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="Ollama model name used by the single_pass_llm baseline.",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama host used by the single_pass_llm baseline.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature used by the single_pass_llm baseline.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum parse retries for the single_pass_llm baseline.",
    )
    parser.add_argument(
        "--debug-log",
        default="",
        help="Optional debug log path. For single_pass_llm, raw prompts/responses are appended here.",
    )
    args = parser.parse_args()

    citation_path, section_path, paragraph_path, paragraph_citation_path = run_baseline_from_args(args)
    print("Saved baseline outputs:")
    print(citation_path)
    print(section_path)
    print(paragraph_path)
    print(paragraph_citation_path)


if __name__ == "__main__":
    main()
