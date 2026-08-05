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
        choices=[
            "citation_frequency",
            "length_weighted_frequency",
            "single_pass_llm",
            "technical_section_prior",
            "openai_full_paper",
            "anthropic_full_paper",
            "single_shot_citation_api",
        ],
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
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional API key for the selected full-paper API baseline. Prefer using --api-key-env instead of passing secrets on the command line.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable that stores the API key. For anthropic_full_paper, the code automatically falls back to AWS_BEARER_TOKEN_BEDROCK when this is left at the OpenAI default.",
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
    args = parser.parse_args()

    citation_path, section_path, paragraph_path, paragraph_citation_path = run_baseline_from_args(args)
    print("Saved baseline outputs:")
    for path in (citation_path, section_path, paragraph_path, paragraph_citation_path):
        if path:
            print(path)


if __name__ == "__main__":
    main()
