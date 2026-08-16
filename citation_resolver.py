"""
Citation Resolver — maps raw citation strings to structured reference metadata
and produces cross-paper canonical identifiers.

Reads the reference section of each paper's PDF and links every citation key
found in that paper's *_citation_scores.json to its full bibliographic entry
(authors, title, year).

Two citation styles are supported:
  - Numeric : [1], [2], ...  (e.g. Fair_Epsilon_Net, FCM_SIGMOD_CRV)
  - Author-year: (Hong et al., 2023)  (e.g. adv_res_paper)

Cross-paper unification
-----------------------
Both `[7]` in FCM_SIGMOD_CRV and `(Asudeh et al., 2023)` in adv_res_paper
can refer to the same paper. The resolver assigns a canonical_id
(e.g. "asudeh_2023") that is the same for both, so that
CitationGraph.add_citation_mappings() can connect them to a single node.

Quick start
-----------
    from citation_resolver import CitationResolver
    from citation_graph_framework import KnowledgeDiscoveryFramework

    resolver = CitationResolver()
    resolver.parse_all("paper_results/", "papers/")

    print(resolver.resolve("(Hong et al., 2023)"))
    print(resolver.as_dict()["[7]"])

    fw = KnowledgeDiscoveryFramework.from_results_dir(
        "paper_results/",
        citation_mappings=resolver.build_citation_mappings(),
    )
"""

from __future__ import annotations

import difflib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import PyPDF2  # type: ignore[import-untyped]

try:
    from ollama import Client
except ImportError:  # pragma: no cover - optional dependency
    Client = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ReferenceEntry:
    """Parsed bibliographic entry for a single cited work."""

    raw_text: str               # cleaned full text of the reference entry
    numeric_key: Optional[str]  # "[1]" for numeric style, None for author-year
    authors: List[str]          # author strings as they appear in the reference
    first_author_last: str      # first author's last name, lowercase (for canonical_id)
    year: Optional[int]
    title: str
    canonical_id: str           # e.g. "hong_2023" — stable across citation styles
    stable_external_id: str = ""
    title_source: str = "parsed"
    confidence: Optional[float] = None

    def __str__(self) -> str:
        authors_str = "; ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        return f"[{self.canonical_id}] {authors_str} ({self.year}). {self.title[:80]}..."


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def _pdf_full_text(pdf_path: str | Path) -> str:
    text = ""
    with open(pdf_path, "rb") as f:
        for page in PyPDF2.PdfReader(f).pages:
            text += (page.extract_text() or "") + "\n"
    return text


def _extract_references_block(text: str) -> str:
    lines = (text or "").splitlines()
    start_idx: Optional[int] = None
    for idx, line in enumerate(lines):
        stripped = re.sub(r"\s+", " ", line).strip().lower()
        stripped = re.sub(r"^(?:\d+(?:\.\d+)*|[ivxlcdm]+)[\.\)]\s+", "", stripped)
        if stripped in {"references", "bibliography"}:
            start_idx = idx + 1
            break

    if start_idx is None:
        tail = "\n".join(lines[-400:]) if lines else text
        return tail.strip()

    refs_lines = lines[start_idx:]
    while refs_lines and not refs_lines[0].strip():
        refs_lines.pop(0)
    return "\n".join(refs_lines).strip()


def _normalize_title_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _titles_match(t1: str, t2: str, threshold: float = 0.85) -> bool:
    """True if two title strings refer to the same work despite minor OCR/formatting variation.

    Operates on pre-normalized (alphanumeric-only, lowercase) forms so that
    punctuation, spacing, and ligature differences are invisible to the comparison.
    A 40-char prefix equality check catches truncated reference titles cheaply;
    difflib ratio catches the remaining cases (transpositions, missing articles, etc.).
    """
    n1 = _normalize_title_key(t1)
    n2 = _normalize_title_key(t2)
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True
    if len(n1) >= 40 and len(n2) >= 40 and n1[:40] == n2[:40]:
        return True
    return difflib.SequenceMatcher(None, n1, n2).ratio() >= threshold


def _looks_like_low_quality_title(title: str) -> bool:
    cleaned = _normalize_ws(title).strip(" .,:;\"'`()[]{}")
    normalized = _normalize_title_key(cleaned)
    if not normalized:
        return True

    lower = cleaned.lower()
    words = re.findall(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'’.\-]*", cleaned)
    if re.fullmatch(r"(1[89]\d{2}|20\d{2})", cleaned):
        return True
    if len(words) <= 1:
        return True
    if len(normalized) < 12:
        return True
    if sum(ch.isdigit() for ch in cleaned) > max(4, len(cleaned) // 3):
        return True
    suspicious_phrases = (
        "attention visualizations",
        "input-input",
        "layer5",
        "pages ",
        "volume ",
        "proceedings of",
    )
    if any(phrase in lower for phrase in suspicious_phrases):
        return True
    return False


def _stable_external_id_from_title(title: str, year: Optional[int]) -> str:
    normalized = _normalize_title_key(title)
    if not normalized or year is None:
        return ""
    return f"ref_{normalized[:96]}_{year}"


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    stripped = text.strip()
    candidates = [stripped]
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidates.insert(0, fence_match.group(1))
    brace_match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _model_tag_from_citation_filename(paper_id: str, path: Path) -> Optional[str]:
    filename = path.name
    suffix = "_citation_scores.json"
    prefix = f"{paper_id}_"
    if filename == f"{paper_id}{suffix}":
        return ""
    if filename.startswith(prefix) and filename.endswith(suffix):
        middle = filename[len(prefix) : -len(suffix)]
        if middle == "paragraph" or middle.endswith("_paragraph"):
            return None
        return middle
    return None


def _discover_citation_score_files(paper_dir: Path, paper_id: str) -> List[Tuple[str, Path]]:
    discovered: List[Tuple[str, Path]] = []
    for candidate in sorted(paper_dir.iterdir()):
        if not candidate.is_file():
            continue
        model_tag = _model_tag_from_citation_filename(paper_id, candidate)
        if model_tag is None:
            continue
        discovered.append((model_tag, candidate))
    discovered.sort(key=lambda item: (item[0] != "", item[0]))
    return discovered


def _extract_pdf_title(pdf_path: str | Path) -> Optional[str]:
    reader = PyPDF2.PdfReader(str(pdf_path))
    metadata = reader.metadata or {}
    meta_title = getattr(metadata, "title", None) if hasattr(metadata, "title") else metadata.get("/Title")
    meta_title = _normalize_ws(str(meta_title or ""))
    if meta_title and meta_title.lower() not in {"untitled", "unknown"}:
        return meta_title

    if not reader.pages:
        return None

    first_page_text = reader.pages[0].extract_text() or ""
    raw_lines = [_normalize_ws(line) for line in first_page_text.splitlines()]
    lines = [line for line in raw_lines if line]
    if not lines:
        return None

    def clean_front_matter_line(line: str) -> str:
        lower = line.lower()
        if lower.startswith("provided proper attribution is provided"):
            return ""
        if lower.startswith("reproduce the tables and figures in this paper solely for use"):
            return ""
        if lower.startswith("scholarly works"):
            return ""
        if lower.startswith("published as a conference paper at "):
            return ""
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", line):
            return ""
        line = re.sub(
            r"^arxiv:\S+\s+\[[^\]]+\]\s+\d{1,2}\s+\w+\s+\d{4}\s*",
            "",
            line,
            flags=re.IGNORECASE,
        )
        return _normalize_ws(line)

    def is_front_matter_noise(line: str) -> bool:
        lower = line.lower()
        if any(token in lower for token in ("@", "university", "department", "school", "institute", "laboratory")):
            return False
        if "pages" in lower or "association for computational linguistics" in lower:
            return True
        if "proceedings" in lower or "findings of the association for computational linguistics" in lower:
            return True
        if "copyright" in lower or "©" in line:
            return True
        if re.search(r"\b(?:19|20)\d{2}\b", line) and re.search(
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|june)\b", lower
        ):
            return True
        return False

    def is_continuation_title_line(line: str) -> bool:
        lower = line.lower()
        if any(token in lower for token in ("@", "university", "department", "school", "institute", "laboratory")):
            return False
        normalized = re.sub(r"([a-z])and([A-Z])", r"\1 and \2", line)
        normalized = re.sub(r"\band(?=[A-Z])", "and ", normalized)

        def looks_like_author_segment(segment: str) -> bool:
            words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’.\-]*", segment)
            if not 1 <= len(words) <= 4:
                return False
            forbidden = {
                "of", "for", "with", "from", "into", "under", "using", "towards",
                "toward", "through", "via", "machine", "reading", "attention",
                "networks", "network", "translation", "sequence", "model", "models",
            }
            if any(word.lower() in forbidden for word in words):
                return False
            return all(word[:1].isupper() for word in words)

        author_segments = [
            segment.strip()
            for segment in re.split(r",|\band\b", normalized)
            if segment.strip()
        ]
        if len(author_segments) >= 2:
            author_like_count = sum(1 for segment in author_segments if looks_like_author_segment(segment))
            if author_like_count >= 2 and author_like_count == len(author_segments):
                return False
        if lower.startswith("abstract"):
            return False
        words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’.\-]*", line)
        if len(words) < 2:
            return False
        has_title_stopword = any(
            word in {"of", "the", "to", "at", "for", "in", "on", "with", "by", "from"}
            for word in (w.lower() for w in words)
        )
        titlecase_ratio = (
            sum(1 for word in words if word[:1].isupper()) / len(words)
            if words else 0.0
        )
        letters = [ch for ch in line if ch.isalpha()]
        upper_ratio = (
            sum(1 for ch in letters if ch.isupper()) / len(letters)
            if letters else 0.0
        )
        return has_title_stopword or upper_ratio >= 0.65 or (len(words) >= 3 and titlecase_ratio >= 0.80)

    pre_abstract: List[str] = []
    for raw_line in lines:
        line = clean_front_matter_line(raw_line)
        if not line or is_front_matter_noise(line):
            continue
        if line.strip().lower() == "abstract":
            break
        pre_abstract.append(line)

    if not pre_abstract:
        return None

    candidate_lines = [pre_abstract[0]]
    for line in pre_abstract[1:]:
        if len(candidate_lines) >= 3:
            break
        if is_continuation_title_line(line):
            candidate_lines.append(line)
            continue
        break

    title = _normalize_ws(" ".join(candidate_lines).rstrip("*"))
    return title or None


def _extract_refs_text(pdf_path: str | Path) -> str:
    """Return text from the first 'References' heading to end of document."""
    text = _pdf_full_text(pdf_path)
    return _extract_references_block(text)


# ---------------------------------------------------------------------------
# Style detection
# ---------------------------------------------------------------------------

def _detect_style(citation_keys: List[str]) -> str:
    """Return 'numeric', 'author_year', or 'mixed'."""
    numeric = sum(1 for k in citation_keys if re.match(r"^\s*(?:\[\d|\(\d)", k))
    author  = sum(1 for k in citation_keys if re.match(r"^\s*\([A-Z]", k))
    if numeric > author:
        return "numeric"
    if author > numeric:
        return "author_year"
    return "mixed"


# ---------------------------------------------------------------------------
# Entry extraction helpers
# ---------------------------------------------------------------------------

def _normalize_ws(text: str) -> str:
    """Collapse all whitespace (including newlines) to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def _find_numeric_entry(refs_text: str, key: str) -> Optional[str]:
    """Return the raw text for numeric reference [N] or (N)."""
    m = re.match(r"(?:\[\s*(\d+)\s*\]|\(\s*(\d+)\s*\))", key.strip())
    if not m:
        return None
    n = int(m.group(1) or m.group(2))
    # Match from [n] to [n+1] (or end of text)
    pattern = rf"\[{n}\](.*?)(?=\[{n + 1}\]|\Z)"
    match = re.search(pattern, refs_text, re.DOTALL)
    if not match:
        return None
    return _normalize_ws(match.group(1))


def _find_authoryear_entry(refs_text: str, key: str) -> Optional[str]:
    """
    Return the raw text for an author-year citation like (Hong et al., 2023).

    Strategy:
    1. Find last_name + year within a 900-char window.
    2. Anchor on the year position rather than the name position so the
       entry boundary search is relative to a stable point.
    3. Walk forward from (year + 130 chars) to find the next entry start,
       giving the title and venue enough room to be included.
    """
    # Normalize the key to handle PDF artifacts like missing spaces
    norm_key = re.sub(r"([a-z])([A-Z])", r"\1 \2", key)   # "Xiet" → "Xi et"
    norm_key = re.sub(r"\s+", " ", norm_key).strip()

    km = re.match(r"\(\s*([A-Z][A-Za-zÀ-ÿ\-']+)", norm_key)
    # Year may have a letter suffix: 2023a, 2024b
    ym = re.search(r"(\d{4})([a-z]?)", norm_key)
    if not km or not ym:
        return None
    last_name = km.group(1)
    year      = ym.group(1)

    name_re = re.compile(rf"\b{re.escape(last_name)}\b", re.IGNORECASE)
    for name_match in name_re.finditer(refs_text):
        window = refs_text[name_match.start(): name_match.start() + 900]
        if year not in window:
            continue

        # Anchor on the year: find its absolute position.
        year_rel = window.find(year)
        year_abs = name_match.start() + year_rel

        # Entry start: last newline + capital-letter before the name match.
        text_before = refs_text[: name_match.start()]
        nl_matches  = list(re.finditer(r"\n(?=[A-Z])", text_before))
        entry_start = (nl_matches[-1].end() - 1) if nl_matches else 0

        # Entry end: search for next entry start starting at (year_abs + 130).
        # 130 chars after the year is enough to clear a short title line; we
        # then look for a `\n` followed by a capital letter, which marks either
        # the title-continuation (skip) or the next entry's author (stop).
        # To distinguish author lines from title lines we use two passes:
        #   pass 1: find the first `\n[A-Z]` after (year_abs + 130) as a candidate
        #   pass 2: if that candidate line looks like a title (no comma near start),
        #           advance one more `\n[A-Z]` further.
        search_base  = year_abs + 130
        candidate_re = re.compile(r"\n([A-Z][A-Za-zÀ-ÿ0-9\-:'\" ]{3,})")
        entry_end    = min(year_abs + 700, len(refs_text))

        for cand in candidate_re.finditer(refs_text[search_base:]):
            cand_line = cand.group(1)
            # Author lines contain a comma (separating names) or end with a year.
            # Title lines typically start with a capital word followed by colon or
            # lowercase continuation.  Use comma in first 40 chars as author signal.
            if "," in cand_line[:40] or re.search(r"\d{4}", cand_line[:20]):
                entry_end = search_base + cand.start()
                break
            # Otherwise treat as a title/venue continuation and keep searching,
            # but cap to avoid runaway.
            if search_base + cand.start() > year_abs + 600:
                entry_end = search_base + cand.start()
                break

        raw = _normalize_ws(refs_text[entry_start:entry_end])
        if year not in raw:
            continue
        # Verify last_name is the FIRST author, not a co-author buried mid-list.
        # In ACL format the entry starts "Firstname Lastname, ..." so we check
        # only the text before the first comma.
        first_author_text = raw.split(",")[0] if "," in raw else raw[:60]
        if last_name.lower() not in first_author_text.lower():
            continue
        # Verify the FIRST year in the raw entry matches the citation year.
        # This prevents two consecutive same-author entries from bleeding together
        # (e.g. "Hendrycks 2020" entry absorbing a trailing "2021" from the next).
        first_year_match = _YEAR_RE.search(raw)
        if first_year_match and first_year_match.group(1) != year:
            continue
        return raw

    return None


# ---------------------------------------------------------------------------
# Entry parsing (text → structured fields)
# ---------------------------------------------------------------------------

# Matches a year (optionally followed by a letter suffix like "2023a") that is
# not immediately followed by another digit (avoids matching "20231" etc.).
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})(?=[^0-9]|$)")
_NUMERIC_KEY_RE = re.compile(r"^\[(\d+)\]\s*")


def _find_pub_year_match(text: str) -> Optional[re.Match]:
    """
    Return the regex match for the publication year in a reference entry.

    Preference order:
    1. Year preceded by ', ' — ACM/IEEE end-of-entry style: '…, pp. X–Y, YEAR.'
    2. Year preceded by '. ' — NeurIPS/ACL style: 'Authors. YEAR. Title.'
    3. First year not inside parentheses (venue names like '(ICALP 2019)' skipped).
    4. Any year.
    """
    all_matches = list(_YEAR_RE.finditer(text))
    if not all_matches:
        return None

    def _in_parens(m: re.Match) -> bool:
        before = text[: m.start()]
        return before.count('(') > before.count(')')

    # 1. Year preceded by ", " and NOT inside parentheses (e.g. in-text cite "(Smith, 2023)")
    for m in reversed(all_matches):
        pre = text[max(0, m.start() - 3): m.start()]
        if re.search(r',\s*$', pre) and not _in_parens(m):
            return m
    # 2. Year preceded by ". " and not inside parens (NeurIPS/ACL: "Authors. YEAR. Title.")
    for m in all_matches:
        pre = text[max(0, m.start() - 3): m.start()]
        if re.search(r'\.\s+$', pre) and not _in_parens(m):
            return m
    # 3. First year not inside parentheses
    for m in all_matches:
        if not _in_parens(m):
            return m
    return all_matches[0]


def _split_authors_title_year_at_end(text: str, year_pos: int) -> tuple[str, str]:
    """
    Split a reference whose year appears at the end into (authors_raw, title).

    Handles three sub-formats in priority order:
    1. "et al." present  → author block ends right after "et al."
    2. Period-boundary   → first ". " not preceded by an uppercase initial
    3. IEEE quoted title → Authors, "Title," venue, Year.
                          All initials are uppercase so (2) finds nothing;
                          split on the opening quotation mark instead.
    """
    # 1. et al. boundary
    et_al_m = re.search(r"\bet\s+al\.\s*", text[:year_pos])
    if et_al_m:
        authors_raw = text[: et_al_m.end()].strip().rstrip(". ")
        rest = re.sub(r"-\s+", "", text[et_al_m.end():].strip())
        tm = re.match(r"([^.]+\.)", rest)
        title = tm.group(1).strip() if tm else rest[:150].strip()
        return authors_raw, title

    # 2. IEEE quoted-title format: Authors, "Title," venue, Year.
    #    Must be checked BEFORE period boundary because venue abbreviations like
    #    "no. " and "vol. " (preceded by lowercase) would falsely trigger the
    #    period boundary.  A quote before year_pos is a reliable IEEE signal.
    #    Only open/close DOUBLE quotes are used for IEEE titles; U+2019 (') is an
    #    apostrophe inside words like "It's" and must not be treated as a close-quote.
    quote_m = re.search(r'["\u201c\u201d]', text[:year_pos])
    if quote_m:
        authors_raw = text[: quote_m.start()].strip().rstrip(", ")
        after_q = re.sub(r"-\s+", "", text[quote_m.start() + 1:])
        close_m = re.search(r'["\u201d]', after_q)
        if close_m:
            title = after_q[: close_m.start()].strip().rstrip(",")
        else:
            tm = re.match(r"([^,]+),", after_q)
            title = tm.group(1).strip() if tm else after_q[:150].strip()
        return authors_raw, title

    # 3. Period boundary (not preceded by uppercase initial or "al")
    boundary = re.search(r"(?<![A-Z])(?<!al)\.\s+", text)
    if boundary and boundary.start() < year_pos:
        authors_raw = text[: boundary.start()].strip()
        rest = re.sub(r"-\s+", "", text[boundary.end():].strip())
        tm = re.match(r"([^.]+\.)", rest)
        title = tm.group(1).strip() if tm else rest[:150].strip()
        return authors_raw, title

    # Fallback: no split found
    rest = re.sub(r"-\s+", "", text)
    tm = re.match(r"([^.]+\.)", rest)
    return "", tm.group(1).strip() if tm else rest[:150].strip()


def _parse_entry(
    raw: str,
    numeric_key: Optional[str] = None,
    year_suffix: str = "",
) -> Optional[ReferenceEntry]:
    """
    Parse authors, year, and title from a raw reference string.

    Handles two patterns:
    - Numeric : "Authors. Year. Title. Venue."
    - Author-year: "Firstname Lastname, ... Year. Title. Venue."
    Both PDF styles put the year as a bare 4-digit number somewhere after
    the author block.

    year_suffix: letter from the original citation key (e.g. "a" from "2024a")
    that distinguishes between two papers by the same first-author in the same
    year; it is appended to the canonical_id ("zhang_2024a").
    """
    if not raw:
        return None

    # Strip any leading [N] prefix (numeric style already extracted but key kept)
    text = _NUMERIC_KEY_RE.sub("", raw).strip()

    year_match = _find_pub_year_match(text)
    if not year_match:
        return None

    year = int(year_match.group(1))
    year_pos = year_match.start()

    # Skip optional year-suffix letter immediately after the digits (e.g. "2023a" → skip "a")
    year_end = year_match.end()
    if year_end < len(text) and text[year_end].isalpha() and text[year_end].islower():
        year_end += 1

    # Detect format:
    #   "early year"  → Authors. Year. Title. Venue.   (NeurIPS/ACL style)
    #   "year at end" → Authors. Title. Venue, Year.   (ACM/IEEE style)
    pre_year = text[max(0, year_pos - 3): year_pos]
    # "year at end": explicitly preceded by ", " OR year sits in the last 20% of the text
    # (catches "Authors. Title. 2012." where nothing follows the year)
    # Also treat as year-at-end when there is a quoted title before the year:
    # IEEE entries like 'Authors, "Title," venue, DD Mon YEAR, pp. N.' have the year
    # embedded in a conference date (e.g. "Feb 2018"), not preceded by ", ", yet the
    # authors and title split must use the quote-based logic.
    # Finally, if there are clearly two sentence boundaries before the year, we are
    # almost certainly in an "Authors. Title. Venue ... Year." format, even when OCR
    # spillover after the reference makes the year appear far from the end.
    has_ieee_quote = bool(re.search(r'["\u201c\u201d]', text[:year_pos]))
    first_boundary = re.search(r"(?<![A-Z])(?<!al)\.\s+", text)
    has_second_boundary_before_year = False
    if first_boundary and first_boundary.end() < year_pos:
        second_boundary = re.search(r"\.\s+", text[first_boundary.end():year_pos])
        has_second_boundary_before_year = second_boundary is not None
    year_at_end = (
        bool(re.search(r",\s*$", pre_year))
        or year_pos > len(text) * 0.80
        or has_ieee_quote
        or has_second_boundary_before_year
    )

    if year_at_end:
        authors_raw, title = _split_authors_title_year_at_end(text, year_pos)
    else:
        # Everything before the year ≈ authors
        authors_raw = text[:year_pos].strip().rstrip(",. ")
        # Everything after "YYYY[a]." ≈ title (first complete sentence)
        after_year = text[year_end:].lstrip(". ").strip()
        after_year = re.sub(r"-\s+", "", after_year)
        title_match = re.match(r"([^.]+\.)", after_year)
        title = title_match.group(1).strip() if title_match else after_year[:150].strip()

    title = re.sub(r"\s+", " ", title).strip()

    authors   = _parse_authors(authors_raw)
    first_last = _first_author_last(authors_raw)
    suffix     = year_suffix.strip().lower()
    canonical_id = f"{first_last}_{year}{suffix}"

    return ReferenceEntry(
        raw_text=raw,
        numeric_key=numeric_key,
        authors=authors,
        first_author_last=first_last,
        year=year,
        title=title,
        canonical_id=canonical_id,
    )


def _parse_authors(authors_raw: str) -> List[str]:
    """
    Split an author string into a list of individual author names.
    Handles ", and " / " and " conjunctions and "et al." truncation.
    """
    # Normalise "et al." variants
    text = re.sub(r"\bet\s+al\.?", "et al.", authors_raw)
    # Split on ", " and " and "
    parts = re.split(r",\s*(?:and\s+)?|\s+and\s+", text)
    authors = [p.strip().rstrip(",. ") for p in parts if p.strip()]
    # Filter out empty strings and lone punctuation
    return [a for a in authors if len(a) > 1]


def _first_author_last(authors_raw: str) -> str:
    """
    Extract the last name of the first author as a lowercase ASCII string
    for use in canonical IDs.

    Handles both "Firstname Lastname, ..." and "Lastname, Firstname, ..." formats.
    Also handles single-word org names like "Mistral AI".
    """
    # Take the first comma-separated token
    first_token = authors_raw.split(",")[0].strip()
    # Remove et al.
    first_token = re.sub(r"\bet\s+al\.?", "", first_token).strip()
    # If token has spaces: last word is the last name (e.g. "Sirui Hong" → "hong")
    words = first_token.split()
    if words:
        last = words[-1].lower()
        # Strip non-alpha chars
        last = re.sub(r"[^a-z]", "", last)
        return last or "unknown"
    return "unknown"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CitationResolver:
    """
    Resolves citation strings to structured bibliographic entries and
    provides canonical cross-paper identifiers.
    """

    def __init__(
        self,
        llm_repair_model: str = "",
        llm_repair_host: str = "http://localhost:11434",
        llm_repair_min_confidence: float = 0.60,
        llm_repair_max_calls: int = 200,
    ) -> None:
        # (paper_id, citation_str) → ReferenceEntry  — scoped per paper so that
        # [2] in paper A and [2] in paper B are resolved independently.
        self._raw_map: Dict[Tuple[str, str], ReferenceEntry] = {}
        # canonical_id → ReferenceEntry (first entry that established this id)
        self._canonical_map: Dict[str, ReferenceEntry] = {}
        # stable external id (usually title+year based) → ReferenceEntry
        self._stable_id_map: Dict[str, ReferenceEntry] = {}
        # track disambiguation suffixes: canonical_id_base → count
        self._id_counts: Dict[str, int] = {}
        # corpus paper_id → extracted paper title
        self._corpus_titles: Dict[str, str] = {}
        # normalized extracted title → [paper_id, ...]
        self._title_to_paper_ids: Dict[str, List[str]] = defaultdict(list)
        # paper_id → {model_tag: {citation_str: score}}
        self._paper_model_citation_scores: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
        # paper_id → {model_tag: file_path}
        self._paper_model_citation_files: Dict[str, Dict[str, str]] = defaultdict(dict)
        # (paper_id, citation_str) → graph target id
        self._graph_target_map: Dict[Tuple[str, str], str] = {}
        # optional LLM repair configuration
        self._llm_repair_model = llm_repair_model.strip()
        self._llm_repair_host = llm_repair_host.strip() or "http://localhost:11434"
        self._llm_repair_min_confidence = max(0.0, min(1.0, llm_repair_min_confidence))
        self._llm_repair_max_calls = max(0, llm_repair_max_calls)
        self._llm_repair_calls = 0
        self._llm_client: Optional[Client] = None
        self._llm_repair_cache: Dict[str, Optional[ReferenceEntry]] = {}

    def _apply_exclusions(self, exclude_papers: set[str]) -> None:
        if not exclude_papers:
            return

        for paper_id in list(exclude_papers):
            title = self._corpus_titles.pop(paper_id, None)
            if title:
                normalized = _normalize_title_key(title)
                if normalized in self._title_to_paper_ids:
                    self._title_to_paper_ids[normalized] = [
                        existing_id
                        for existing_id in self._title_to_paper_ids[normalized]
                        if existing_id != paper_id
                    ]
                    if not self._title_to_paper_ids[normalized]:
                        del self._title_to_paper_ids[normalized]
            self._paper_model_citation_scores.pop(paper_id, None)
            self._paper_model_citation_files.pop(paper_id, None)
            for key in [raw_key for raw_key in self._raw_map if raw_key[0] == paper_id]:
                del self._raw_map[key]

    def register_corpus_papers(
        self,
        results_dir: str | Path,
        papers_dir: str | Path,
        pdf_ext: str = ".pdf",
        exclude_papers: Optional[set[str]] = None,
    ) -> "CitationResolver":
        """Index corpus paper titles so resolved citations can collapse onto corpus nodes."""
        results_path = Path(results_dir)
        papers_path = Path(papers_dir)
        excluded = {paper.strip() for paper in (exclude_papers or set()) if paper.strip()}
        self._apply_exclusions(excluded)
        self._corpus_titles.clear()
        self._title_to_paper_ids.clear()

        for paper_dir in sorted(results_path.iterdir()):
            if not paper_dir.is_dir() or paper_dir.name.startswith("."):
                continue
            paper_id = paper_dir.name
            if paper_id in excluded:
                continue
            pdf_path = papers_path / f"{paper_id}{pdf_ext}"
            if not pdf_path.exists():
                continue
            title = _extract_pdf_title(pdf_path)
            if not title:
                continue
            self._corpus_titles[paper_id] = title
            normalized = _normalize_title_key(title)
            if normalized and paper_id not in self._title_to_paper_ids[normalized]:
                self._title_to_paper_ids[normalized].append(paper_id)

        return self

    def _get_llm_client(self) -> Optional[Client]:
        if not self._llm_repair_model or self._llm_repair_max_calls <= 0 or Client is None:
            return None
        if self._llm_client is None:
            self._llm_client = Client(host=self._llm_repair_host)
        return self._llm_client

    def _reference_context(
        self,
        refs_text: str,
        citation_key: str,
        is_numeric: bool,
    ) -> tuple[Optional[str], Optional[str]]:
        if not is_numeric:
            return None, None
        match = re.match(r"(?:\[\s*(\d+)\s*\]|\(\s*(\d+)\s*\))", citation_key.strip())
        if not match:
            return None, None
        idx = int(match.group(1) or match.group(2))
        prev_ref = _find_numeric_entry(refs_text, f"[{idx - 1}]") if idx > 1 else None
        next_ref = _find_numeric_entry(refs_text, f"[{idx + 1}]")
        return prev_ref, next_ref

    def _assign_stable_external_id(self, entry: ReferenceEntry) -> ReferenceEntry:
        if entry.stable_external_id:
            return entry
        stable = _stable_external_id_from_title(entry.title, entry.year)
        entry.stable_external_id = stable or entry.canonical_id
        return entry

    def _should_attempt_llm_repair(self, entry: Optional[ReferenceEntry], raw: str) -> bool:
        if self._get_llm_client() is None:
            return False
        if entry is None:
            return True
        if entry.year is None:
            return True
        if not entry.authors or entry.first_author_last == "unknown":
            return True
        if _looks_like_low_quality_title(entry.title):
            return True
        normalized_title = _normalize_title_key(entry.title)
        raw_prefix_norm = _normalize_title_key(raw[:260])
        if normalized_title and normalized_title not in raw_prefix_norm and len(normalized_title) < 24:
            return True
        return False

    def _repair_entry_with_llm(
        self,
        raw: str,
        *,
        citation_key: str,
        year_suffix: str,
        previous_reference: Optional[str] = None,
        next_reference: Optional[str] = None,
    ) -> Optional[ReferenceEntry]:
        client = self._get_llm_client()
        if client is None or self._llm_repair_calls >= self._llm_repair_max_calls:
            return None

        cache_key = _normalize_ws(" || ".join(filter(None, [previous_reference or "", raw, next_reference or ""])))
        if cache_key in self._llm_repair_cache:
            cached = self._llm_repair_cache[cache_key]
            if cached is None:
                return None
            return ReferenceEntry(
                raw_text=raw,
                numeric_key=cached.numeric_key,
                authors=list(cached.authors),
                first_author_last=cached.first_author_last,
                year=cached.year,
                title=cached.title,
                canonical_id=cached.canonical_id,
                stable_external_id=cached.stable_external_id,
                title_source=cached.title_source,
                confidence=cached.confidence,
            )

        system_prompt = (
            "You extract clean bibliographic metadata from one noisy reference entry. "
            "Return strict JSON only with the keys: title, authors, year, confidence. "
            "authors must be an array of author strings. year must be an integer or null. "
            "confidence must be a number between 0 and 1. Do not include any extra text."
        )
        context_bits = [f"Citation key: {citation_key}", f"Reference entry: {raw}"]
        if previous_reference:
            context_bits.append(f"Previous reference: {previous_reference}")
        if next_reference:
            context_bits.append(f"Next reference: {next_reference}")
        user_prompt = "\n".join(context_bits)

        self._llm_repair_calls += 1
        try:
            response = client.chat(
                model=self._llm_repair_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0},
                think=False,
            )
        except Exception:
            self._llm_repair_cache[cache_key] = None
            return None
        response_text = response.message.content if getattr(response, "message", None) else response["message"]["content"]
        parsed = _extract_json_object(response_text)
        if not isinstance(parsed, dict):
            self._llm_repair_cache[cache_key] = None
            return None

        title = _normalize_ws(str(parsed.get("title", ""))).strip()
        if not title or _looks_like_low_quality_title(title):
            self._llm_repair_cache[cache_key] = None
            return None

        authors_raw = parsed.get("authors", [])
        authors = [str(author).strip() for author in authors_raw if str(author).strip()] if isinstance(authors_raw, list) else []
        year_val = parsed.get("year")
        try:
            year = int(year_val) if year_val is not None and str(year_val).strip() else None
        except (TypeError, ValueError):
            year = None
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < self._llm_repair_min_confidence:
            self._llm_repair_cache[cache_key] = None
            return None

        authors_raw_text = ", ".join(authors)
        first_last = _first_author_last(authors_raw_text)
        suffix = year_suffix.strip().lower()
        canonical_id = (
            f"{first_last}_{year}{suffix}"
            if year is not None
            else f"{first_last}_unknown"
        )
        repaired = ReferenceEntry(
            raw_text=raw,
            numeric_key=citation_key if re.match(r"^(?:\[\d|\(\d)", citation_key) else None,
            authors=authors,
            first_author_last=first_last,
            year=year,
            title=title,
            canonical_id=canonical_id,
            title_source="llm_repaired",
            confidence=confidence,
        )
        self._assign_stable_external_id(repaired)
        self._llm_repair_cache[cache_key] = repaired
        return repaired

    def _maybe_repair_entry_with_llm(
        self,
        entry: Optional[ReferenceEntry],
        raw: str,
        *,
        citation_key: str,
        year_suffix: str,
        previous_reference: Optional[str] = None,
        next_reference: Optional[str] = None,
    ) -> Optional[ReferenceEntry]:
        if not self._should_attempt_llm_repair(entry, raw):
            return entry
        repaired = self._repair_entry_with_llm(
            raw,
            citation_key=citation_key,
            year_suffix=year_suffix,
            previous_reference=previous_reference,
            next_reference=next_reference,
        )
        if repaired is not None:
            return repaired
        return entry

    # --- building ---

    def parse_paper(
        self,
        paper_id: str,
        pdf_path: str | Path,
        citation_keys: List[str],
    ) -> int:
        """
        Parse the reference section of a PDF and resolve the given citation keys.

        Args:
            paper_id      : identifier for this paper (used in log messages only).
            pdf_path      : path to the PDF file.
            citation_keys : list of citation strings from that paper's
                            *_citation_scores.json.

        Returns the number of keys successfully resolved.
        """
        refs_text = _extract_refs_text(pdf_path)
        style = _detect_style(citation_keys)
        resolved = 0

        for key in citation_keys:
            norm_key = re.sub(r"\s+", " ", key).strip()
            # Do NOT skip if norm_key was seen in another paper —
            # [2] in paper A and [2] in paper B are different works.
            if (paper_id, norm_key) in self._raw_map:
                resolved += 1
                continue

            raw: Optional[str] = None
            is_numeric = bool(re.match(r"^(?:\[\d|\(\d)", norm_key))

            if style in ("numeric", "mixed") and is_numeric:
                raw = _find_numeric_entry(refs_text, norm_key)
            if raw is None and not is_numeric and re.match(r"\([A-Z]", norm_key):
                raw = _find_authoryear_entry(refs_text, norm_key)

            if raw is None:
                continue

            # Pass year suffix (a/b) from the citation key for canonical_id
            year_suffix = ""
            if not is_numeric:
                ys = re.search(r"\d{4}([a-z])", norm_key)
                if ys:
                    year_suffix = ys.group(1)

            entry = _parse_entry(
                raw,
                numeric_key=norm_key if is_numeric else None,
                year_suffix=year_suffix,
            )
            prev_ref, next_ref = self._reference_context(refs_text, norm_key, is_numeric)
            entry = self._maybe_repair_entry_with_llm(
                entry,
                raw,
                citation_key=norm_key,
                year_suffix=year_suffix,
                previous_reference=prev_ref,
                next_reference=next_ref,
            )
            if entry is None:
                continue
            self._assign_stable_external_id(entry)

            if entry.stable_external_id in self._stable_id_map:
                existing = self._stable_id_map[entry.stable_external_id]
                if _titles_match(existing.title, entry.title):
                    self._raw_map[(paper_id, norm_key)] = existing
                    self._graph_target_map.pop((paper_id, norm_key), None)
                    resolved += 1
                    continue

            # Disambiguate canonical_id when author+year collide
            base_id = entry.canonical_id
            if base_id in self._canonical_map:
                existing = self._canonical_map[base_id]
                # Same paper referenced again under a different key — reuse entry
                if _titles_match(existing.title, entry.title):
                    self._raw_map[(paper_id, norm_key)] = existing
                    resolved += 1
                    continue
                # Different paper with same author+year — add suffix
                count = self._id_counts.get(base_id, 1) + 1
                self._id_counts[base_id] = count
                entry.canonical_id = f"{base_id}_{chr(96 + count)}"  # _b, _c, ...

            entry.canonical_id = entry.canonical_id  # (may have been updated above)
            self._canonical_map[entry.canonical_id] = entry
            if entry.stable_external_id:
                self._stable_id_map.setdefault(entry.stable_external_id, entry)
            self._raw_map[(paper_id, norm_key)] = entry
            self._graph_target_map.pop((paper_id, norm_key), None)
            resolved += 1

        print(f"[CitationResolver] {paper_id}: resolved {resolved}/{len(citation_keys)} citations")
        return resolved

    def parse_all(
        self,
        results_dir: str | Path,
        papers_dir: str | Path,
        pdf_ext: str = ".pdf",
        exclude_papers: Optional[set[str]] = None,
    ) -> "CitationResolver":
        """
        Auto-discover all papers: match each paper subdirectory in results_dir
        to a same-named PDF in papers_dir and parse it.

        Args:
            results_dir : directory containing one subdirectory per paper.
            papers_dir  : directory containing PDF files.
            pdf_ext     : PDF file extension (default '.pdf').
        """
        results_path = Path(results_dir)
        papers_path  = Path(papers_dir)
        excluded = {paper.strip() for paper in (exclude_papers or set()) if paper.strip()}
        self._apply_exclusions(excluded)
        self.register_corpus_papers(results_dir, papers_dir, pdf_ext=pdf_ext, exclude_papers=excluded)

        for paper_dir in sorted(results_path.iterdir()):
            if not paper_dir.is_dir() or paper_dir.name.startswith("."):
                continue
            paper_id = paper_dir.name
            if paper_id in excluded:
                continue
            pdf_path = papers_path / f"{paper_id}{pdf_ext}"
            if not pdf_path.exists():
                print(f"[CitationResolver] No PDF found for {paper_id}, skipping.")
                continue

            citation_files = _discover_citation_score_files(paper_dir, paper_id)
            if not citation_files:
                print(f"[CitationResolver] No citation scores for {paper_id}, skipping.")
                continue

            keys: List[str] = []
            seen_keys = set()
            model_tags: List[str] = []
            for model_tag, citation_file in citation_files:
                with open(citation_file) as f:
                    raw_scores = json.load(f)
                score_map = {
                    citation: float(payload.get("citation_score", 0.0))
                    for citation, payload in raw_scores.items()
                }
                self._paper_model_citation_scores[paper_id][model_tag] = score_map
                self._paper_model_citation_files[paper_id][model_tag] = str(citation_file)
                label = model_tag or "default"
                model_tags.append(label)
                for citation in score_map:
                    norm_key = re.sub(r"\s+", " ", citation).strip()
                    if norm_key in seen_keys:
                        continue
                    seen_keys.add(norm_key)
                    keys.append(citation)

            self.parse_paper(
                paper_id,
                pdf_path,
                keys,
            )
            print(
                f"[CitationResolver] {paper_id}: discovered {len(citation_files)} citation-score files "
                f"({', '.join(model_tags)})"
            )

        return self

    # --- querying ---

    def resolve(self, citation_str: str, paper_id: str = "") -> Optional[ReferenceEntry]:
        """
        Look up a citation string.

        If paper_id is given, the lookup is scoped to that paper (correct for
        numeric citations like [2] which differ across papers).

        If paper_id is omitted, citation_str is treated as a canonical_id
        (e.g. "hong_2023") and looked up in the canonical map — useful for
        display in analysis steps where only the canonical id is known.
        """
        if paper_id:
            norm = re.sub(r"\s+", " ", citation_str).strip()
            entry = self._raw_map.get((paper_id, norm))
            if entry:
                return entry
            flat = re.sub(r"\s+", "", citation_str)
            for (pid, k), v in self._raw_map.items():
                if pid == paper_id and re.sub(r"\s+", "", k) == flat:
                    return v
            return None

        # No paper_id: treat as canonical_id lookup (used by analysis display code)
        if citation_str in self._canonical_map:
            return self._canonical_map[citation_str]
        if citation_str in self._stable_id_map:
            return self._stable_id_map[citation_str]
        return None

    def _canonical_target(self, entry: ReferenceEntry) -> str:
        """Return the graph target id for a resolved citation."""
        self._assign_stable_external_id(entry)

        def prefix_match_with_authorish_suffix(entry_title: str, corpus_title: str) -> bool:
            entry_words = re.findall(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'’.\-]*", entry_title)
            corpus_words = re.findall(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'’.\-]*", corpus_title)
            entry_norm_words = [re.sub(r"[^a-z0-9]+", "", word.lower()) for word in entry_words]
            corpus_norm_words = [re.sub(r"[^a-z0-9]+", "", word.lower()) for word in corpus_words]
            entry_norm_words = [word for word in entry_norm_words if word]
            corpus_norm_words = [word for word in corpus_norm_words if word]
            if len(entry_norm_words) < 3 or len(corpus_norm_words) <= len(entry_norm_words):
                return False
            if corpus_norm_words[: len(entry_norm_words)] != entry_norm_words:
                return False

            suffix_words = corpus_words[len(entry_norm_words) :]
            suffix_lower = [word.lower() for word in suffix_words]
            if any(
                token in {"university", "department", "school", "institute", "laboratory", "google", "deepmind", "team"}
                for token in suffix_lower
            ):
                return True
            core_suffix = [word for word in suffix_words if word.lower() not in {"and", "of", "the"}]
            if 2 <= len(core_suffix) <= 12 and all(word[:1].isupper() for word in core_suffix):
                return True
            if any(char.isdigit() for char in " ".join(suffix_words)):
                return True
            return False

        def title_match(normalized_text: str) -> Optional[str]:
            if not normalized_text:
                return None

            matched_papers = self._title_to_paper_ids.get(normalized_text, [])
            if len(matched_papers) == 1:
                return matched_papers[0]

            author_suffix_hits: List[Tuple[float, str]] = []
            best_ratio, best_id = 0.0, None
            for paper_id, corpus_title in self._corpus_titles.items():
                corpus_norm = _normalize_title_key(corpus_title)
                if not corpus_norm:
                    continue
                if prefix_match_with_authorish_suffix(entry.title, corpus_title):
                    ratio = difflib.SequenceMatcher(None, normalized_text, corpus_norm).ratio()
                    author_suffix_hits.append((ratio, paper_id))
                ratio = difflib.SequenceMatcher(None, normalized_text, corpus_norm).ratio()
                if ratio > best_ratio:
                    best_ratio, best_id = ratio, paper_id

            if author_suffix_hits:
                author_suffix_hits.sort(reverse=True)
                if len(author_suffix_hits) == 1 or author_suffix_hits[0][0] > author_suffix_hits[1][0]:
                    return author_suffix_hits[0][1]
            if best_ratio >= 0.90 and best_id is not None:
                return best_id
            return None

        normalized_title = _normalize_title_key(entry.title)
        matched = title_match(normalized_title)
        if matched is not None:
            return matched

        # Only fall back to raw reference text when the parsed title is clearly
        # low quality; otherwise long parser spillover can create false corpus
        # matches from the next reference entry.
        if len(normalized_title) >= 20:
            return entry.stable_external_id or entry.canonical_id

        raw_prefix_norm = _normalize_title_key(entry.raw_text[:220])
        raw_hits = [
            paper_ids[0]
            for corpus_norm, paper_ids in self._title_to_paper_ids.items()
            if len(paper_ids) == 1 and len(corpus_norm) >= 18 and corpus_norm in raw_prefix_norm
        ]
        if len(raw_hits) == 1:
            return raw_hits[0]
        return entry.stable_external_id or entry.canonical_id

    def build_citation_mappings(self) -> Dict[str, Dict[str, str]]:
        """
        Return {paper_id: {citation_str: target_id}}.

        The CitationGraph uses this to resolve each paper's citations
        independently, so [2] in paper A and [2] in paper B map to
        their respective graph targets. For corpus papers the target is the
        corpus paper id; for external works it is a stable external id, usually
        derived from normalized title + year.
        """
        result: Dict[str, Dict[str, str]] = {}
        for (paper_id, cit_str), entry in self._raw_map.items():
            result.setdefault(paper_id, {})[cit_str] = self.target_id_for(cit_str, paper_id) or self._canonical_target(entry)
        return result

    def target_id_for(self, citation_str: str, paper_id: str) -> Optional[str]:
        """Return the graph target id for a paper-scoped citation key."""
        norm = re.sub(r"\s+", " ", citation_str).strip()
        cache_key = (paper_id, norm)
        if cache_key in self._graph_target_map:
            return self._graph_target_map[cache_key]

        entry = self.resolve(norm, paper_id=paper_id)
        if entry is None:
            flat = re.sub(r"\s+", "", citation_str)
            for (pid, key), target in self._graph_target_map.items():
                if pid == paper_id and re.sub(r"\s+", "", key) == flat:
                    return target
            return None

        target = self._canonical_target(entry)
        self._graph_target_map[cache_key] = target
        return target

    def all_resolved(self) -> Dict[Tuple[str, str], ReferenceEntry]:
        """Return all (paper_id, citation_str) → ReferenceEntry mappings."""
        return dict(self._raw_map)

    def as_dict(self) -> Dict:
        """Export all resolved entries as plain dicts (JSON-serialisable)."""
        entries: Dict[str, Dict] = {}
        for cid, entry in self._canonical_map.items():
            entries[cid] = {
                "title": entry.title,
                "authors": entry.authors,
                "year": entry.year,
                "canonical_id": entry.canonical_id,
                "stable_external_id": entry.stable_external_id,
                "numeric_key": entry.numeric_key,
                "raw_text": entry.raw_text,
                "title_source": entry.title_source,
                "confidence": entry.confidence,
            }
        raw: Dict[str, str] = {}
        graph_targets: Dict[str, str] = {}
        for (paper_id, cit_str), entry in self._raw_map.items():
            raw[f"{paper_id}|||{cit_str}"] = entry.canonical_id
            graph_targets[f"{paper_id}|||{cit_str}"] = self.target_id_for(cit_str, paper_id) or self._canonical_target(entry)
        return {
            "entries": entries,
            "raw_map": raw,
            "graph_targets": graph_targets,
            "corpus_titles": dict(self._corpus_titles),
            "paper_model_citation_scores": {
                paper_id: dict(model_scores)
                for paper_id, model_scores in self._paper_model_citation_scores.items()
            },
            "paper_model_citation_files": {
                paper_id: dict(model_files)
                for paper_id, model_files in self._paper_model_citation_files.items()
            },
        }

    def save(self, path: str | Path) -> None:
        """Save all resolved entries to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.as_dict(), f, indent=2, ensure_ascii=False)
        print(f"[CitationResolver] Saved {len(self._raw_map)} entries to {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        llm_repair_model: str = "",
        llm_repair_host: str = "http://localhost:11434",
        llm_repair_min_confidence: float = 0.60,
        llm_repair_max_calls: int = 200,
    ) -> "CitationResolver":
        """Restore a previously saved resolver (skips re-parsing PDFs)."""
        resolver = cls(
            llm_repair_model=llm_repair_model,
            llm_repair_host=llm_repair_host,
            llm_repair_min_confidence=llm_repair_min_confidence,
            llm_repair_max_calls=llm_repair_max_calls,
        )
        with open(path) as f:
            data = json.load(f)
        # Support both old flat format and new nested format
        if "entries" in data and "raw_map" in data:
            for cid, d in data["entries"].items():
                first_last = _first_author_last(d["authors"][0] if d["authors"] else "")
                entry = ReferenceEntry(
                    raw_text=d["raw_text"],
                    numeric_key=d.get("numeric_key"),
                    authors=d["authors"],
                    first_author_last=first_last,
                    year=d.get("year"),
                    title=d["title"],
                    canonical_id=d["canonical_id"],
                    stable_external_id=d.get("stable_external_id", ""),
                    title_source=d.get("title_source", "parsed"),
                    confidence=d.get("confidence"),
                )
                resolver._canonical_map[cid] = entry
                resolver._assign_stable_external_id(entry)
                resolver._stable_id_map.setdefault(entry.stable_external_id, entry)
            for key, cid in data["raw_map"].items():
                paper_id, cit_str = key.split("|||", 1)
                entry = resolver._canonical_map[cid]
                resolver._raw_map[(paper_id, cit_str)] = entry
            for key, target in data.get("graph_targets", {}).items():
                paper_id, cit_str = key.split("|||", 1)
                resolver._graph_target_map[(paper_id, cit_str)] = target
            for paper_id, title in data.get("corpus_titles", {}).items():
                resolver._corpus_titles[paper_id] = title
                normalized = _normalize_title_key(title)
                if normalized and paper_id not in resolver._title_to_paper_ids[normalized]:
                    resolver._title_to_paper_ids[normalized].append(paper_id)
            for paper_id, model_scores in data.get("paper_model_citation_scores", {}).items():
                resolver._paper_model_citation_scores[paper_id] = {
                    model_tag: {citation: float(score) for citation, score in score_map.items()}
                    for model_tag, score_map in model_scores.items()
                }
            for paper_id, model_files in data.get("paper_model_citation_files", {}).items():
                resolver._paper_model_citation_files[paper_id] = {
                    model_tag: str(file_path) for model_tag, file_path in model_files.items()
                }
        else:
            # Old flat format: {citation_str: entry_dict} — no paper scope
            for key, d in data.items():
                first_last = _first_author_last(d["authors"][0] if d["authors"] else "")
                entry = ReferenceEntry(
                    raw_text=d["raw_text"],
                    numeric_key=d.get("numeric_key"),
                    authors=d["authors"],
                    first_author_last=first_last,
                    year=d.get("year"),
                    title=d["title"],
                    canonical_id=d["canonical_id"],
                )
                resolver._canonical_map.setdefault(entry.canonical_id, entry)
                resolver._assign_stable_external_id(entry)
                resolver._stable_id_map.setdefault(entry.stable_external_id, entry)
        return resolver

    def __len__(self) -> int:
        return len(self._raw_map)

    def __repr__(self) -> str:
        return (
            f"CitationResolver(resolved={len(self._raw_map)}, "
            f"canonical_ids={len(self._canonical_map)})"
        )
