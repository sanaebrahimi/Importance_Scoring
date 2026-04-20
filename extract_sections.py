"""
extract_sections.py — Extract section/subsection structure from a single PDF
without using any language model.

Detection strategy (in order of preference):
  1. PDF embedded outline / bookmarks  (most reliable when present)
  2. Font-size + bold analysis via pymupdf  (accurate for most papers)
  3. Numbered-heading text patterns  (1. / 1.1 / A. fallback)

Handles papers where the section number ("5.1") and title
("Calculating the agent contributions") appear on separate consecutive lines.

Usage:
    python3 extract_sections.py paper.pdf
    python3 extract_sections.py paper.pdf --var MY_VAR_NAME
    python3 extract_sections.py paper.pdf --append papers_section_titles.txt

Requirements:
    pip install pymupdf        (recommended — font-aware, most accurate)
    pip install pypdf          (fallback if pymupdf is not available)
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# ── backend ───────────────────────────────────────────────────────────────────
try:
    import fitz as _fitz                           # type: ignore[import-untyped]
    _BACKEND = "fitz"
except ImportError:
    _fitz = None
    try:
        from pypdf import PdfReader as _PdfReader  # type: ignore[import-untyped]
        _BACKEND = "pypdf"
    except ImportError:
        raise SystemExit(
            "No PDF backend found.\n"
            "  pip install pymupdf    (recommended)\n"
            "  pip install pypdf      (fallback)"
        )

# ── constants ──────────────────────────────────────────────────────────────────
_STOP = frozenset({"acknowledgments", "acknowledgements"})
# "references"/"bibliography" are skipped (not collected) but scanning continues
# so appendix sections appearing after the reference list are still detected.
_SKIP = frozenset({"abstract", "references", "bibliography", "appendix"})
_CONNECTORS = frozenset({"and", "or", "of", "the", "on", "in", "with", "vs",
                          "for", "to", "a", "an", "at", "by", "from", "is"})
_JUNK_CHARS  = re.compile(r"[=+*<>|\\@#$%^&]")
_FLOAT_ONLY  = re.compile(r"^\d+(?:\.\d+)*$")   # e.g. "5", "5.1", "6.3.1"
_ALPHA_ONLY  = re.compile(r"^[A-Z]$")           # appendix letters: "A", "B"
_NUMBERED    = re.compile(r"^(\d+(?:\.\d+){0,3})\.?\s+(.+)$")
_APPENDIX    = re.compile(r"^(Appendix\s+)?([A-Z])[\.\s]\s+(.+)$")


# ── shared helpers ─────────────────────────────────────────────────────────────

def _clean(title: str) -> str:
    """Strip leading section numbers and normalise whitespace."""
    title = " ".join(title.replace("\n", " ").split())
    title = re.sub(r"^(?:\d+(?:\.\d+)*)\.?\s+", "", title)
    title = re.sub(r"^Appendix\s+[A-Z][\.\s]\s*", "", title)
    return title.strip()


def _var_name(path: Path) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", path.stem).strip("_").upper()
    return f"{base}_SECTIONS"


def _build_tree(flat: list[tuple[int, str]]) -> list:
    """flat: [(level, title), …]  Returns nested [(title, children_or_None)]."""
    root: list = []
    stack: list[tuple[int, list]] = [(0, root)]
    for level, title in flat:
        node = [title, []]
        while len(stack) > 1 and stack[-1][0] >= level:
            stack.pop()
        stack[-1][1].append(node)
        stack.append((level, node[1]))
    return _collapse(root)


def _collapse(nodes: list) -> list:
    return [(t, _collapse(c) if c else None) for t, c in nodes]


def _level_from_number(num: str) -> int:
    """'5.1' → 2,  '6.3.1' → 3,  '1' → 1."""
    return num.count(".") + 1


# ── Strategy 1 — PDF outline / bookmarks ──────────────────────────────────────

def _outline_fitz(doc) -> list:
    toc = doc.get_toc(simple=True)   # [[level, title, page], …]
    if not toc:
        return []
    flat: list[tuple[int, str]] = []
    for level, raw, _ in toc:
        title = _clean(raw)
        lower = title.lower()
        if lower in _SKIP:
            continue
        if lower in _STOP:
            break
        flat.append((level, title))
    return _build_tree(flat) if flat else []


def _outline_pypdf(reader) -> list:
    def _walk(items: list, level: int = 1) -> list[tuple[int, str]]:
        flat: list[tuple[int, str]] = []
        i = 0
        while i < len(items):
            item = items[i]
            if isinstance(item, list):
                i += 1
                continue
            title = _clean(getattr(item, "title", "") or "")
            lower = title.lower()
            if lower in _SKIP:
                i += 1
                continue
            if lower in _STOP:
                return flat
            flat.append((level, title))
            if i + 1 < len(items) and isinstance(items[i + 1], list):
                flat.extend(_walk(items[i + 1], level + 1))
                i += 2
            else:
                i += 1
        return flat

    outline = getattr(reader, "outline", []) or []
    flat = _walk(outline)
    return _build_tree(flat) if flat else []


# ── Strategy 2 — font-size analysis (pymupdf only) ────────────────────────────

def _font_analysis(doc) -> list:
    """
    Identify headings by font size and bold flags.

    Body text = most common size for lines > 40 chars.
    Heading candidates:
      • Lines whose dominant span is notably larger than body size.
      • Bold lines close to body size (sub-headings), excluding body text.

    Two special cases handled:
      • Skip title/author block: only start collecting after the "Abstract"
        heading has been seen (anything before Abstract is metadata).
      • Split section numbers: some papers put "5.1" on one line and the title
        on the next bold line of the same size; we join them.
    """
    records: list[tuple[str, float, bool]] = []   # (line_text, size, bold)

    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                line_text = " ".join(s["text"] for s in spans).strip()
                if not line_text or len(line_text) < 2:
                    continue
                dom = max(spans, key=lambda s: len(s["text"]))
                size = round(dom["size"], 1)
                bold = bool(dom["flags"] & 16)
                records.append((line_text, size, bold))

    if not records:
        return []

    # Body size = mode of sizes for long lines (≥ 40 chars → body paragraphs)
    long_sizes = [r[1] for r in records if len(r[0]) >= 40]
    body_size  = Counter(long_sizes or [r[1] for r in records]).most_common(1)[0][0]

    size_threshold = body_size * 1.04   # at least 4% larger than body → heading

    # ── Pass 1: collect heading-candidate records ──────────────────────────
    # started = True after we have seen the Abstract heading (skips title/authors)
    started       = False
    pending_num   = ""   # holds a standalone "5.1" line until next title line
    heading_sizes: set[float] = set()
    flat: list[tuple[int, str, float]] = []
    stop = False

    n = len(records)
    i = 0
    while i < n:
        text, size, bold = records[i]

        if stop:
            break

        # Detect the Abstract marker → start collecting after it
        if _clean(text).lower() == "abstract":
            started = True
            i += 1
            continue

        if not started:
            i += 1
            continue

        is_body = abs(size - body_size) < 0.6

        # ── Standalone section-number lines (e.g. "5", "5.1", "A") ────
        if bold and (_FLOAT_ONLY.match(text.strip()) or _ALPHA_ONLY.match(text.strip())):
            pending_num = text.strip()
            i += 1
            continue

        # When a section number was just seen, accept the next bold line as
        # heading regardless of its size.
        if pending_num and bold and 3 <= len(text.strip()) <= 120:
            pass   # fall through to title collection
        else:
            is_large      = size >= size_threshold
            is_bold_short = (bold
                             and size >= body_size - 0.5
                             and not is_body
                             and len(text) <= 90)

            if not (is_large or is_bold_short):
                pending_num = ""
                i += 1
                continue

        title = _clean(text)
        lower = title.lower()

        # ── Sanity filters ─────────────────────────────────────────────
        if not title or len(title) > 120:
            pending_num = ""; i += 1; continue
        if not re.search(r"[A-Za-z]", title):   # numbers-only → page number or label
            pending_num = ""; i += 1; continue
        if _JUNK_CHARS.search(title):
            pending_num = ""; i += 1; continue
        if re.match(r"^(Figure|Fig\.|Table|Algorithm|Listing|Theorem|Lemma|Proof|Eq\.)\s", title):
            pending_num = ""; i += 1; continue
        if re.search(r"[.!?]\s*\S", title):   # mid-sentence punctuation
            pending_num = ""; i += 1; continue
        if title.endswith((".", "!", "?")) and not title.endswith("..."):
            pending_num = ""; i += 1; continue
        if re.search(r"@|\bemail\b", title, re.I):
            pending_num = ""; i += 1; continue
        if re.search(r"\b(ISSN|ISBN|DOI|arXiv)\b|\d{4}-\d{4,}", title):
            pending_num = ""; i += 1; continue

        if lower in _SKIP:
            pending_num = ""; i += 1; continue
        if lower in _STOP:
            stop = True
            break

        i += 1

        # ── Merge wrapped continuation lines ───────────────────────────
        # Some papers typeset long headings across two consecutive bold lines
        # of the same size with no body text in between, e.g.:
        #   "Model Capacity Matters But Only With"   ← wrapped
        #   "Coordination"
        while i < n:
            nt, ns, nb = records[i]
            nt_clean = _clean(nt)
            # Accept continuation only if immediately next (no body text gap),
            # same/similar size, bold, and short (< 25 chars)
            if (nb
                    and abs(ns - size) <= 0.3
                    and 1 <= len(nt_clean) <= 40
                    and not _FLOAT_ONLY.match(nt.strip())
                    and not _ALPHA_ONLY.match(nt.strip())
                    and not nt_clean.endswith((".", "!", "?"))
                    and nt_clean.lower() not in _STOP
                    and nt_clean.lower() not in _SKIP):
                title = title + " " + nt_clean
                i += 1
            else:
                break

        # Determine level
        if pending_num:
            explicit_level = _level_from_number(pending_num)
            heading_sizes.add(size)
            flat.append((explicit_level, title, size))
            pending_num = ""
        else:
            heading_sizes.add(size)
            flat.append((0, title, size))
            pending_num = ""

    if not flat:
        return []

    # ── Pass 2: assign levels to entries without explicit numbering ────────
    # Cluster sizes within 5% of each other into the same level so that minor
    # font-size variations (e.g. 11.8 vs 12.0) don't create spurious sub-levels.
    # Largest cluster → level 1, next → level 2, etc.
    def _cluster_sizes(sizes: set[float], tol: float = 0.06) -> dict[float, float]:
        """Return {size: representative} where nearby sizes share a rep."""
        reps: dict[float, float] = {}
        for sz in sorted(sizes, reverse=True):
            found = next(
                (r for r in reps if abs(sz - r) / max(sz, r) <= tol), None
            )
            reps[sz] = found if found is not None else sz
        return reps

    all_sizes  = {sz for _, _, sz in flat}
    size_rep   = _cluster_sizes(all_sizes)   # size → cluster representative

    # Build explicit rep→level from numbered entries
    rep_level: dict[float, int] = {}
    for lv, _, sz in flat:
        if lv != 0:
            rep = size_rep[sz]
            rep_level[rep] = min(rep_level.get(rep, lv), lv)

    # Assign levels to clusters not yet covered (unnumbered headings)
    used_levels = set(rep_level.values())
    next_level  = 1
    for rep in sorted({size_rep[sz] for lv, _, sz in flat if lv == 0
                        and size_rep[sz] not in rep_level}, reverse=True):
        while next_level in used_levels:
            next_level += 1
        rep_level[rep] = next_level
        used_levels.add(next_level)
        next_level += 1

    flat_levelled = [
        (lv if lv != 0 else rep_level.get(size_rep.get(sz, sz), 1), title)
        for lv, title, sz in flat
    ]

    return _build_tree(flat_levelled)


# ── Strategy 3 — numbered heading text patterns ────────────────────────────────

def _preprocess_split_numbers(lines: list[str]) -> list[str]:
    """
    Merge lines where a section number stands alone on one line and the title
    follows immediately on the next, e.g.:
        "5.1"
        "Calculating the agent contributions"
    → "5.1 Calculating the agent contributions"
    """
    result: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if _FLOAT_ONLY.match(stripped) or _ALPHA_ONLY.match(stripped):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                nxt = lines[j].strip()
                if nxt and not _FLOAT_ONLY.match(nxt) and not _ALPHA_ONLY.match(nxt) and not _NUMBERED.match(nxt):
                    result.append(f"{stripped} {nxt}")
                    i = j + 1
                    continue
        result.append(lines[i])
        i += 1
    return result


def _text_patterns(text: str) -> list:
    """
    Detect headings by numbered prefixes: 1. / 1.1 / 1.1.1 / A.
    Handles papers where the section number is on its own line.
    """
    lines = _preprocess_split_numbers(text.splitlines())
    flat: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if stripped.lower() in _STOP or stripped == "References":
            break

        m = _NUMBERED.match(stripped) or _APPENDIX.match(stripped)
        if m:
            if _APPENDIX.match(stripped):
                num_part   = m.group(2)
                title_part = (m.group(3) or "").strip()
                level      = 1
            else:
                num_part   = m.group(1)
                title_part = m.group(2).strip()
                level      = _level_from_number(num_part)

            if not re.search(r"[A-Za-z]", title_part):
                i += 1
                continue
            if _JUNK_CHARS.search(title_part):
                i += 1
                continue
            if len(title_part) > 120:
                i += 1
                continue

            # Multi-line continuation
            j = i + 1
            while j < len(lines):
                nxt = lines[j].strip()
                if not nxt or _NUMBERED.match(nxt) or nxt.lower() in _STOP:
                    break
                if _looks_like_continuation(nxt):
                    title_part = f"{title_part} {nxt}"
                    j += 1
                else:
                    break

            flat.append((level, _clean(f"{num_part} {title_part}")))
            i = j
            continue

        i += 1

    return _build_tree(flat) if flat else []


def _looks_like_continuation(line: str) -> bool:
    if not line or len(line) > 40 or line.endswith((".", ":", ";")):
        return False
    if _NUMBERED.match(line):
        return False
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9.-]*", line)
    if not words or max(len(w) for w in words) > 22:
        return False
    good = sum(1 for w in words if w.lower() in _CONNECTORS or w[:1].isupper() or w.isupper())
    return good >= max(1, len(words) - 1)


# ── top-level entry point ──────────────────────────────────────────────────────

def extract_sections(pdf_path: Path) -> list:
    """
    Return a nested list of (title, children_or_None) for one PDF.
    Tries three strategies in order; returns the first non-empty result.
    """
    if _BACKEND == "fitz":
        doc = _fitz.open(str(pdf_path))

        result = _outline_fitz(doc)
        if result:
            return result

        result = _font_analysis(doc)
        if result:
            return result

        text = "\n".join(page.get_text() for page in doc)
        return _text_patterns(text)

    else:
        reader = _PdfReader(str(pdf_path))

        result = _outline_pypdf(reader)
        if result:
            return result

        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return _text_patterns(text)


# ── output formatting ──────────────────────────────────────────────────────────

def _format(entries: list, indent: int = 1) -> list[str]:
    pad = "    " * indent
    lines: list[str] = []
    for title, children in entries:
        if children:
            lines.append(f'{pad}"{title}": {{')
            lines.extend(_format(children, indent + 1))
            lines.append(f"{pad}}},")
        else:
            lines.append(f'{pad}"{title}": None,')
    return lines


def render_block(pdf_path: Path, entries: list, var_name: str) -> str:
    return "\n".join(
        [f"# {pdf_path.name}", f"{var_name} = {{"]
        + _format(entries) + ["}"]
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract section structure from a PDF (no LLM)."
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument("--var", default="",
                        help="Variable name (default: derived from filename)")
    parser.add_argument("--append", type=Path, default=None,
                        help="Append result to this .txt file")
    args = parser.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"File not found: {args.pdf}")

    var_name = args.var.strip() or _var_name(args.pdf)
    entries  = extract_sections(args.pdf)

    if not entries:
        print(f"[WARNING] No sections detected in {args.pdf.name}", file=sys.stderr)

    block = render_block(args.pdf, entries, var_name)
    print(block)

    if args.append:
        existing = args.append.read_text(encoding="utf-8") if args.append.exists() else ""
        sep = "\n\n\n" if existing.strip() else ""
        args.append.write_text(
            existing.rstrip() + sep + block + "\n", encoding="utf-8"
        )
        print(f"[Appended to {args.append}]", file=sys.stderr)


if __name__ == "__main__":
    main()
