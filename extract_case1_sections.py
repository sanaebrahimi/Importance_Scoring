"""
Extract section headings from every PDF in papers/Case 1/ and append
the results to case1_section_titles.txt in the same folder.

Uses the same extraction logic as extract_paper_sections.py
(bookmarks → font analysis → numbered-pattern fallback).

Usage:
    python3 extract_case1_sections.py
"""

from pathlib import Path

from extract_sections import extract_sections, render_block, _var_name


CASE1_DIR = Path("papers") / "Case 1"
OUTPUT_FILE = CASE1_DIR / "case1_section_titles.txt"


def main() -> None:
    pdfs = sorted(CASE1_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {CASE1_DIR}")
        return

    blocks = []
    for pdf in pdfs:
        print(f"Processing {pdf.name} ...", end=" ", flush=True)
        entries = extract_sections(pdf)
        block = render_block(pdf, entries, _var_name(pdf))
        blocks.append(block)
        count = sum(1 for line in block.splitlines() if '": ' in line)
        print(f"{count} headings found")

    OUTPUT_FILE.write_text("\n\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"\nWrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
