"""Batch section extractor that rewrites papers_section_titles.txt.

This script reuses the single-PDF extractor in extract_sections.py so that
batch and one-off extraction follow the same heading-detection logic.
"""

from pathlib import Path

from extract_sections import extract_sections, render_block, _var_name


PAPERS_DIR = Path("papers")
OUTPUT_PATH = Path("papers_section_titles.txt")


def main() -> None:
    output_blocks = []

    for pdf_path in sorted(PAPERS_DIR.glob("*.pdf")):
        entries = extract_sections(pdf_path)
        block = render_block(pdf_path, entries, _var_name(pdf_path))
        output_blocks.append(block)

    OUTPUT_PATH.write_text("\n\n\n".join(output_blocks) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
