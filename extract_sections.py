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
_FLOAT_ONLY  = re.compile(r"^\d+(?:\.\d+)*\.?$")   # e.g. "5", "5.", "5.1", "6.3.1"
_ALPHA_ONLY  = re.compile(r"^[A-Z]\.?$")           # appendix letters: "A", "A."
_ROMAN_ONLY  = re.compile(r"^[IVXLCM]+\.?$")
_NUMBERED    = re.compile(r"^(\d+(?:\.\d+){0,3})\.?\s+(.+)$")
_ROMAN_HEADING = re.compile(r"^([IVXLCM]+)\.?\s+(.+)$")
_ALPHA_HEADING = re.compile(r"^([A-Z])[.)]\s+(.+)$")
_APPENDIX    = re.compile(r"^Appendix\s+([A-Z])(?:[.)])?\s+(.+)$", re.I)
_CANONICAL_HEADING_RE = re.compile(
    r"\b("
    r"introduction|preliminar(?:y|ies)?|background|problem definition|problem|"
    r"method|methodology|solution|technical|architecture(?: overview)?|"
    r"experiment(?:s|al| setup)?|dataset(?:s)?|baseline models?|"
    r"related work|discussion|benefit(?:s)?|conclusion|limitations?|"
    r"final remarks?|proof of concept|performance evaluation|"
    r"case study|user study"
    r")\b",
    re.I,
)


# ── shared helpers ─────────────────────────────────────────────────────────────

def _clean(title: str) -> str:
    """Strip leading section numbers and normalise whitespace."""
    title = " ".join(title.replace("\n", " ").split())
    title = re.sub(r"^(?:\d+(?:\.\d+)*)\.?\s+", "", title)
    title = re.sub(r"^[IVXLCM]+\.?\s+", "", title)
    title = re.sub(r"^[A-Z](?:[.)])\s+", "", title)
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


def _is_plausible_numbered_heading(raw_line: str, num_part: str, title_part: str) -> bool:
    """Reject axis labels, caption fragments, and OCR'd numeric junk."""
    title = title_part.strip()
    if not title or not re.search(r"[A-Za-z]", title):
        return False
    if num_part.startswith("0."):
        return False
    if title[:1].islower():
        return False
    if title[:1] in "([{-−" or title.startswith(("Fig.", "Figure", "Table", "Algorithm")):
        return False
    if re.match(r"^[nmrkdxty]\b", title, re.I):
        return False
    if re.match(r"^Regret-ratio\b", title, re.I):
        return False
    if re.match(r"^(The|This|These|Those|We|Our|For|At)\b", title):
        return False
    if "…" in title or "..." in title:
        return False
    if re.search(r"[≤≥=<>∀∈]", title):
        return False
    if len(re.findall(r"\b\d+\.", title)) >= 2:
        return False
    if title.count(",") >= 2 and re.search(r"\b\d+\b", title):
        return False
    if re.search(r"\b[A-Z][a-z]+,\s*\d+\b", title):
        return False
    if re.search(
        r"\b(Time\s*\(Sec\)|Time\(sec\)|Duration|Disparity|Skyline size|Percentage of people)\b",
        title,
        re.I,
    ):
        return False
    first_part = num_part.rstrip(".").split(".", 1)[0]
    try:
        first_value = int(first_part)
        if first_value <= 0 or first_value > 20:
            return False
    except ValueError:
        return False
    # For top-level Arabic headings, allow either "2." style or a strong heading
    # like "4 EXPERIMENTS"; reject plain numeric/table lines.
    if "." not in num_part and not re.match(r"^\d+\.\s+", raw_line):
        if not (title.isupper() or _looks_like_heading_shape(title)):
            return False
    return True


def _trim_heading_title(title: str, level: int) -> str:
    """Trim trailing body text from OCR lines like '4.2 Proof of Concept. We ...'."""
    title = title.strip()
    if not title:
        return title

    # Prefer the short heading before the first sentence break.
    parts = re.split(r"(?<=[.?!])\s+", title, maxsplit=1)
    if len(parts) == 2:
        first = parts[0].strip()
        remainder = parts[1].strip()
        if (
            first
            and remainder
            and len(first.split()) <= 10
            and not first.endswith(("et al.", "etc."))
        ):
            return first.rstrip(".")

    # If a colon introduces body text, keep the lead phrase.
    colon_parts = title.split(":", 1)
    if len(colon_parts) == 2:
        first = colon_parts[0].strip()
        remainder = colon_parts[1].strip()
        if first and remainder and len(first.split()) <= 10:
            return first

    return title


def _is_plausible_roman_marker(marker: str) -> bool:
    marker = marker.rstrip(".")
    if not marker or not _ROMAN_ONLY.match(marker):
        return False
    if len(marker) == 1 and marker not in {"I", "V", "X"}:
        return False
    return True


def _explicit_heading_counts(text: str) -> tuple[int, int]:
    """Return (roman_top_count, numbered_top_count) for a document."""
    lines = _preprocess_split_numbers(text.splitlines())
    roman_top_count = 0
    numbered_top_count = 0

    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        roman_match = _ROMAN_HEADING.match(stripped)
        if roman_match and _is_plausible_roman_marker(roman_match.group(1)):
            roman_top_count += 1
            continue

        numbered_match = _NUMBERED.match(stripped)
        if numbered_match and _is_plausible_numbered_heading(
            stripped, numbered_match.group(1), numbered_match.group(2)
        ):
            if _level_from_number(numbered_match.group(1)) == 1:
                numbered_top_count += 1

    return roman_top_count, numbered_top_count


def _has_strong_explicit_heading_scheme(text: str) -> bool:
    roman_top_count, numbered_top_count = _explicit_heading_counts(text)
    return roman_top_count >= 3 or numbered_top_count >= 3


def _is_suspicious_title(title: str) -> bool:
    if not title:
        return True
    stripped = title.strip()
    lower = stripped.lower()
    canonical_heading = _CANONICAL_HEADING_RE.search(stripped)

    if stripped[:1].islower() or stripped[:1] in ",.)-−•":
        return True
    if len(stripped) > 120:
        return True
    if re.search(r"[≤≥∀∈µ⃗]", stripped):
        return True
    if re.match(r"^(figure|fig\.|table|algorithm)\b", stripped, re.I):
        return True
    if re.search(r"\b(Time\s*\(Sec\)|Duration|Disparity|Skyline size|Regret-ratio \(|Accuracy CrS)\b", stripped, re.I):
        return True
    if re.search(r"^(Reliab(?:i|l)il?ity|Frequency)\b", stripped, re.I):
        return True
    if "error (log scale)" in lower:
        return True
    if re.match(r"^(Please|At each|The results|A detailed|Across all|We use)\b", stripped, re.I):
        return True
    if re.match(r"^(HR|Candidate)\s+(Attribute|Ranking)\s+(Values|Weights)\b", stripped, re.I):
        return True
    if re.match(r"^O\d+(?:\s+O\d+)+\s+Centroid$", stripped):
        return True
    if not canonical_heading and re.match(r"^[A-Z][a-z]+$", stripped):
        return True
    if not canonical_heading and re.match(r"^[A-Z][a-z]+(?:\s+[A-Z](?:\.)?)?,\s*\d+", stripped):
        return True
    if not canonical_heading and re.match(r"^[A-Z][a-z]+-dataset$", stripped):
        return True
    if stripped in {"Shapley Values", "Number dimensions", "Candidate Attribute Values", "HR Attribute Values", "HR Ranking Weights", "Candidate Ranking Weights"}:
        return True
    if stripped in {
        "PHP", "MySQL", "HTML", "CSS", "JavaScript", "AJAX",
        "Bootstrap", "MongoDB", "Node.js", "Reactjs", "JS", "R",
        "Performance_PG", "Performance_12", "Performance_UG", "Grad Year",
        "Deep Learning", "Degree Master of Science", "Stream Computer Science",
        "Current City Bangalore",
    }:
        return True
    if re.match(r"^(Zipf|Uniform)\b", stripped):
        return True
    if "_" in stripped:
        return True
    if "powered by tcpdf" in lower or "http://" in lower or "https://" in lower:
        return True
    if re.match(r"^\[\d+\]", stripped):
        return True
    if stripped.count("-") >= 3:
        return True
    if stripped.count(",") >= 3 and len(stripped.split()) >= 8:
        return True
    if "which contain" in lower or "issue of under-representation" in lower:
        return True
    if re.match(r"^\(?[a-z]\)\s+[A-Z][A-Za-z-]+(?:,\s*.+)?$", stripped):
        return True

    words = re.findall(r"[A-Za-z][A-Za-z'’.-]*", stripped)
    if len(words) >= 6:
        shaped = sum(1 for w in words if w[0].isupper() or w.isupper())
        lower_nonconnector = sum(1 for w in words if w.islower() and w not in _CONNECTORS)
        if lower_nonconnector > shaped + 1:
            return True

    return False


def _prune_tree(nodes: list) -> list:
    """Drop obvious OCR/caption junk; promote good children of bad parents."""
    pruned: list = []
    for title, children in nodes:
        child_nodes = _prune_tree(children) if children else None
        if _is_suspicious_title(title):
            if child_nodes:
                pruned.extend(child_nodes)
            continue
        pruned.append((title, child_nodes if child_nodes else None))
    return pruned


def _flatten_titles(nodes: list) -> list[str]:
    titles: list[str] = []
    for title, children in nodes:
        titles.append(title)
        if children:
            titles.extend(_flatten_titles(children))
    return titles


def _tree_canonical_count(nodes: list) -> int:
    return sum(
        1
        for title in _flatten_titles(nodes)
        if _CANONICAL_HEADING_RE.search(title)
    )


def _top_level_canonical_count(nodes: list) -> int:
    return sum(
        1
        for title, _ in nodes
        if _CANONICAL_HEADING_RE.search(title)
    )


def _looks_like_heading_shape(title: str) -> bool:
    if _is_suspicious_title(title):
        return False
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.-]*", title)
    if not words or len(words) > 16:
        return False
    if title.isupper():
        return True
    shaped = sum(
        1
        for word in words
        if word[0].isupper() or word.isupper() or word[0].isdigit()
    )
    return shaped >= max(1, len(words) - 2)


def _tree_quality_score(nodes: list) -> int:
    titles = _flatten_titles(nodes)
    if not titles:
        return -10_000

    top_level = len(nodes)
    total = len(titles)
    heading_like = sum(1 for title in titles if _looks_like_heading_shape(title))
    suspicious = sum(1 for title in titles if _is_suspicious_title(title))
    duplicates = sum(count - 1 for count in Counter(titles).values() if count > 1)
    canonical = _tree_canonical_count(nodes)

    score = heading_like * 4 + canonical * 5 + top_level * 3 + total
    score -= suspicious * 6
    score -= duplicates * 3
    if top_level > 12:
        score -= (top_level - 12) * 5
    if total > 30:
        score -= (total - 30) * 2
    return score


def _tree_max_depth(nodes: list, depth: int = 1) -> int:
    if not nodes:
        return max(0, depth - 1)
    best = depth
    for _, children in nodes:
        if children:
            best = max(best, _tree_max_depth(children, depth + 1))
    return best


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
        if bold and (
            _FLOAT_ONLY.match(text.strip())
            or _ALPHA_ONLY.match(text.strip())
            or _ROMAN_ONLY.match(text.strip())
        ):
            pending_num = text.strip()
            i += 1
            continue

        # When a section number was just seen, accept the next bold line as
        # heading regardless of its size.
        explicit_marker = (
            _NUMBERED.match(text.strip())
            or _ROMAN_HEADING.match(text.strip())
            or _ALPHA_HEADING.match(text.strip())
        )

        if pending_num and bold and 3 <= len(text.strip()) <= 120:
            pass   # fall through to title collection
        elif explicit_marker:
            pending_num = pending_num or explicit_marker.group(1)
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
        if title[:1].islower():
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
            if _ROMAN_ONLY.match(pending_num):
                explicit_level = 1
            elif _ALPHA_ONLY.match(pending_num):
                explicit_level = 2
            else:
                explicit_level = _level_from_number(pending_num.rstrip("."))
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
        if _FLOAT_ONLY.match(stripped) or _ALPHA_ONLY.match(stripped) or _ROMAN_ONLY.match(stripped):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                nxt = lines[j].strip()
                if (
                    nxt
                    and not _FLOAT_ONLY.match(nxt)
                    and not _ALPHA_ONLY.match(nxt)
                    and not _ROMAN_ONLY.match(nxt)
                    and not _NUMBERED.match(nxt)
                    and not _ROMAN_HEADING.match(nxt)
                    and not _ALPHA_HEADING.match(nxt)
                ):
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

    roman_top_count = 0
    numbered_top_count = 0
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        roman_match = _ROMAN_HEADING.match(stripped)
        if roman_match and _is_plausible_roman_marker(roman_match.group(1)):
            roman_top_count += 1
            continue

        numbered_match = _NUMBERED.match(stripped)
        if numbered_match and _is_plausible_numbered_heading(
            stripped, numbered_match.group(1), numbered_match.group(2)
        ):
            if _level_from_number(numbered_match.group(1)) == 1:
                numbered_top_count += 1

    alpha_subsections_enabled = roman_top_count >= 3 and roman_top_count >= numbered_top_count
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if stripped.lower() in _STOP or stripped == "References":
            break

        appendix_match = _APPENDIX.match(stripped)
        roman_match = _ROMAN_HEADING.match(stripped)
        if roman_match and not _is_plausible_roman_marker(roman_match.group(1)):
            roman_match = None
        numbered_match = _NUMBERED.match(stripped)
        if appendix_match or roman_match or numbered_match:
            if appendix_match:
                num_part   = appendix_match.group(1)
                title_part = (appendix_match.group(2) or "").strip()
                level      = 1
            elif roman_match:
                num_part   = roman_match.group(1)
                title_part = roman_match.group(2).strip()
                if not title_part or not re.match(r"^[A-Z]", title_part):
                    i += 1
                    continue
                level      = 1
            else:
                assert numbered_match is not None
                num_part   = numbered_match.group(1)
                title_part = numbered_match.group(2).strip()
                lower_title = _clean(f"{num_part} {title_part}").lower()
                if lower_title in {"references", "bibliography", "acknowledgments", "acknowledgements"}:
                    break
                if not _is_plausible_numbered_heading(stripped, num_part, title_part):
                    i += 1
                    continue
                level      = _level_from_number(num_part)

            title_part = _trim_heading_title(title_part, level)

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
                if (
                    not nxt
                    or _NUMBERED.match(nxt)
                    or _ROMAN_HEADING.match(nxt)
                    or _ALPHA_HEADING.match(nxt)
                    or _APPENDIX.match(nxt)
                    or nxt.lower() in _STOP
                ):
                    break
                if _looks_like_continuation(nxt):
                    title_part = f"{title_part} {nxt}"
                    j += 1
                else:
                    break

            cleaned = _clean(f"{num_part} {title_part}")
            lower_clean = cleaned.lower()
            if lower_clean in _SKIP or lower_clean in _STOP:
                if lower_clean in {"references", "bibliography", "acknowledgments", "acknowledgements"}:
                    break
                i = j
                continue

            flat.append((level, cleaned))
            i = j
            continue

        am = _ALPHA_HEADING.match(stripped)
        if am and alpha_subsections_enabled:
            title_part = am.group(2).strip()
            if (
                title_part
                and re.search(r"[A-Za-z]", title_part)
                and not _JUNK_CHARS.search(title_part)
                and len(title_part) <= 120
                and not title_part[:1].islower()
            ):
                j = i + 1
                while j < len(lines):
                    nxt = lines[j].strip()
                    if (
                        not nxt
                        or _NUMBERED.match(nxt)
                        or _ROMAN_HEADING.match(nxt)
                        or _ALPHA_HEADING.match(nxt)
                        or nxt.lower() in _STOP
                    ):
                        break
                    if _looks_like_continuation(nxt):
                        title_part = f"{title_part} {nxt}"
                        j += 1
                    else:
                        break

                cleaned = _clean(title_part)
                lower_clean = cleaned.lower()
                if lower_clean not in _SKIP and lower_clean not in _STOP:
                    flat.append((2, cleaned))
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
    if len(words) == 1 and words[0].lower() in _CONNECTORS:
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

        text = "\n".join(page.get_text() for page in doc)
        prefer_explicit = _has_strong_explicit_heading_scheme(text)
        text_result = _prune_tree(_text_patterns(text))
        font_result = _prune_tree(_font_analysis(doc))

        candidates = []
        if text_result:
            text_score = _tree_quality_score(text_result)
            if prefer_explicit:
                text_score += 10
            candidates.append((text_score, text_result))
        if font_result:
            font_score = _tree_quality_score(font_result)
            if not prefer_explicit:
                font_score += 10
            candidates.append((font_score, font_result))

        if candidates:
            if text_result and font_result:
                text_total = len(_flatten_titles(text_result))
                font_total = len(_flatten_titles(font_result))
                font_canonical = _tree_canonical_count(font_result)
                text_top_canonical = _top_level_canonical_count(text_result)
                font_top_canonical = _top_level_canonical_count(font_result)
                text_depth = _tree_max_depth(text_result)
                font_depth = _tree_max_depth(font_result)
                if prefer_explicit and text_depth > font_depth:
                    return text_result
                if font_top_canonical >= 5 and text_top_canonical <= 3:
                    return font_result
                if font_canonical >= 6 and text_total > max(18, int(font_total * 1.8)):
                    return font_result
            candidates.sort(key=lambda item: item[0], reverse=True)
            return candidates[0][1]

        return []

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
