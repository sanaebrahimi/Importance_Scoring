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

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import PyPDF2  # type: ignore[import-untyped]


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


def _normalize_title_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


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

    pre_abstract: List[str] = []
    for line in lines:
        if line.strip().lower() == "abstract":
            break
        pre_abstract.append(line)

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

    def is_author_or_affiliation_line(line: str) -> bool:
        lower = line.lower()
        if any(token in lower for token in ("@", "university", "department", "school", "institute", "laboratory")):
            return True
        words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’.\-]*", line)
        if not 2 <= len(words) <= 5:
            return False
        if re.search(r"\b(?:19|20)\d{2}\b", line):
            return False
        if any(token in lower for token in ("abstract", "pages", "association", "proceedings", "findings")):
            return False
        return all(word[:1].isupper() for word in words)

    filtered = [line for line in pre_abstract if not is_front_matter_noise(line)]
    if not filtered:
        return None

    stop_idx = next((idx for idx, line in enumerate(filtered) if is_author_or_affiliation_line(line)), len(filtered))
    candidate_lines = filtered[:stop_idx] if stop_idx > 0 else []
    if not candidate_lines:
        candidate_lines = filtered[: min(3, len(filtered))]

    title = _normalize_ws(" ".join(candidate_lines).rstrip("*"))
    return title or None


def _extract_refs_text(pdf_path: str | Path) -> str:
    """Return text from the first 'References' heading to end of document."""
    text = _pdf_full_text(pdf_path)
    for marker in ("References\n", "REFERENCES\n", "Bibliography\n", "BIBLIOGRAPHY\n"):
        idx = text.rfind(marker)
        if idx != -1:
            return text[idx:]
    return text[-8000:]   # fallback: last 8 000 chars


# ---------------------------------------------------------------------------
# Style detection
# ---------------------------------------------------------------------------

def _detect_style(citation_keys: List[str]) -> str:
    """Return 'numeric', 'author_year', or 'mixed'."""
    numeric = sum(1 for k in citation_keys if re.match(r"^\s*\[\d", k))
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
    """Return the raw text for numeric reference [N]."""
    m = re.match(r"\[\s*(\d+)\s*\]", key.strip())
    if not m:
        return None
    n = int(m.group(1))
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
    year_at_end = (
        bool(re.search(r",\s*$", pre_year))
        or year_pos > len(text) * 0.80
    )

    if year_at_end:
        # Prefer "et al." as the explicit author-block end
        et_al_m = re.search(r"\bet\s+al\.\s*", text[:year_pos])
        if et_al_m:
            authors_raw = text[: et_al_m.end()].strip().rstrip(". ")
            rest = text[et_al_m.end():].strip()
        else:
            # First ". " not preceded by an uppercase initial ("J.") or "et al."
            boundary = re.search(r"(?<![A-Z])(?<!al)\.\s+", text)
            if boundary and boundary.start() < year_pos:
                authors_raw = text[: boundary.start()].strip()
                rest = text[boundary.end():].strip()
            else:
                authors_raw = ""
                rest = text
        rest = re.sub(r"-\s+", "", rest)
        title_match = re.match(r"([^.]+\.)", rest)
        title = title_match.group(1).strip() if title_match else rest[:150].strip()
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

    def __init__(self) -> None:
        # (paper_id, citation_str) → ReferenceEntry  — scoped per paper so that
        # [2] in paper A and [2] in paper B are resolved independently.
        self._raw_map: Dict[Tuple[str, str], ReferenceEntry] = {}
        # canonical_id → ReferenceEntry (first entry that established this id)
        self._canonical_map: Dict[str, ReferenceEntry] = {}
        # track disambiguation suffixes: canonical_id_base → count
        self._id_counts: Dict[str, int] = {}
        # corpus paper_id → extracted paper title
        self._corpus_titles: Dict[str, str] = {}
        # normalized extracted title → [paper_id, ...]
        self._title_to_paper_ids: Dict[str, List[str]] = defaultdict(list)

    def register_corpus_papers(
        self,
        results_dir: str | Path,
        papers_dir: str | Path,
        pdf_ext: str = ".pdf",
    ) -> "CitationResolver":
        """Index corpus paper titles so resolved citations can collapse onto corpus nodes."""
        results_path = Path(results_dir)
        papers_path = Path(papers_dir)

        for paper_dir in sorted(results_path.iterdir()):
            if not paper_dir.is_dir() or paper_dir.name.startswith("."):
                continue
            paper_id = paper_dir.name
            if paper_id in self._corpus_titles:
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
            is_numeric = bool(re.match(r"\[\d", norm_key))

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
            if entry is None:
                continue

            # Disambiguate canonical_id when author+year collide
            base_id = entry.canonical_id
            if base_id in self._canonical_map:
                existing = self._canonical_map[base_id]
                # Same paper referenced again under a different key — reuse entry
                if _normalize_ws(existing.title).lower()[:60] == _normalize_ws(entry.title).lower()[:60]:
                    self._raw_map[(paper_id, norm_key)] = existing
                    resolved += 1
                    continue
                # Different paper with same author+year — add suffix
                count = self._id_counts.get(base_id, 1) + 1
                self._id_counts[base_id] = count
                entry.canonical_id = f"{base_id}_{chr(96 + count)}"  # _b, _c, ...

            entry.canonical_id = entry.canonical_id  # (may have been updated above)
            self._canonical_map[entry.canonical_id] = entry
            self._raw_map[(paper_id, norm_key)] = entry
            resolved += 1

        print(f"[CitationResolver] {paper_id}: resolved {resolved}/{len(citation_keys)} citations")
        return resolved

    def parse_all(
        self,
        results_dir: str | Path,
        papers_dir: str | Path,
        pdf_ext: str = ".pdf",
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
        self.register_corpus_papers(results_dir, papers_dir, pdf_ext=pdf_ext)

        for paper_dir in sorted(results_path.iterdir()):
            if not paper_dir.is_dir() or paper_dir.name.startswith("."):
                continue
            paper_id = paper_dir.name
            pdf_path = papers_path / f"{paper_id}{pdf_ext}"
            if not pdf_path.exists():
                print(f"[CitationResolver] No PDF found for {paper_id}, skipping.")
                continue

            citation_file = paper_dir / f"{paper_id}_citation_scores.json"
            if not citation_file.exists():
                print(f"[CitationResolver] No citation scores for {paper_id}, skipping.")
                continue

            with open(citation_file) as f:
                keys = list(json.load(f).keys())

            self.parse_paper(paper_id, pdf_path, keys)

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
        return None

    def _canonical_target(self, entry: ReferenceEntry) -> str:
        """Return the graph target id for a resolved citation."""
        normalized_title = _normalize_title_key(entry.title)
        if normalized_title:
            matched_papers = self._title_to_paper_ids.get(normalized_title, [])
            if len(matched_papers) == 1:
                return matched_papers[0]
        return entry.canonical_id

    def build_citation_mappings(self) -> Dict[str, Dict[str, str]]:
        """
        Return {paper_id: {citation_str: canonical_id}}.

        The CitationGraph uses this to resolve each paper's citations
        independently, so [2] in paper A and [2] in paper B map to
        their respective canonical ids.
        """
        result: Dict[str, Dict[str, str]] = {}
        for (paper_id, cit_str), entry in self._raw_map.items():
            result.setdefault(paper_id, {})[cit_str] = self._canonical_target(entry)
        return result

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
                "numeric_key": entry.numeric_key,
                "raw_text": entry.raw_text,
            }
        raw: Dict[str, str] = {}
        for (paper_id, cit_str), entry in self._raw_map.items():
            raw[f"{paper_id}|||{cit_str}"] = entry.canonical_id
        return {"entries": entries, "raw_map": raw, "corpus_titles": dict(self._corpus_titles)}

    def save(self, path: str | Path) -> None:
        """Save all resolved entries to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.as_dict(), f, indent=2, ensure_ascii=False)
        print(f"[CitationResolver] Saved {len(self._raw_map)} entries to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "CitationResolver":
        """Restore a previously saved resolver (skips re-parsing PDFs)."""
        resolver = cls()
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
                )
                resolver._canonical_map[cid] = entry
            for key, cid in data["raw_map"].items():
                paper_id, cit_str = key.split("|||", 1)
                entry = resolver._canonical_map[cid]
                resolver._raw_map[(paper_id, cit_str)] = entry
            for paper_id, title in data.get("corpus_titles", {}).items():
                resolver._corpus_titles[paper_id] = title
                normalized = _normalize_title_key(title)
                if normalized and paper_id not in resolver._title_to_paper_ids[normalized]:
                    resolver._title_to_paper_ids[normalized].append(paper_id)
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
        return resolver

    def __len__(self) -> int:
        return len(self._raw_map)

    def __repr__(self) -> str:
        return (
            f"CitationResolver(resolved={len(self._raw_map)}, "
            f"canonical_ids={len(self._canonical_map)})"
        )
