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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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

_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_NUMERIC_KEY_RE = re.compile(r"^\[(\d+)\]\s*")


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

    year_match = _YEAR_RE.search(text)
    if not year_match:
        return None

    year = int(year_match.group(1))
    year_pos = year_match.start()

    # Everything before the year ≈ authors
    authors_raw = text[:year_pos].strip().rstrip(",. ")
    # Everything after "YYYY." ≈ title (first complete sentence)
    after_year = text[year_match.end():].lstrip(". ").strip()
    # Un-hyphenate line-break artefacts: "collabo-\nrative" → "collaborative"
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
        # raw citation key → ReferenceEntry (can be from any paper)
        self._raw_map: Dict[str, ReferenceEntry] = {}
        # canonical_id → ReferenceEntry (first entry that established this canonical id)
        self._canonical_map: Dict[str, ReferenceEntry] = {}
        # track disambiguation suffixes: canonical_id_base → count
        self._id_counts: Dict[str, int] = {}

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
            if norm_key in self._raw_map:
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
                    self._raw_map[norm_key] = existing
                    resolved += 1
                    continue
                # Different paper with same author+year — add suffix
                count = self._id_counts.get(base_id, 1) + 1
                self._id_counts[base_id] = count
                entry.canonical_id = f"{base_id}_{chr(96 + count)}"  # _b, _c, ...

            entry.canonical_id = entry.canonical_id  # (may have been updated above)
            self._canonical_map[entry.canonical_id] = entry
            self._raw_map[norm_key] = entry
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

    def resolve(self, citation_str: str) -> Optional[ReferenceEntry]:
        """Look up a citation string (tolerant of minor whitespace differences)."""
        norm = re.sub(r"\s+", " ", citation_str).strip()
        if norm in self._raw_map:
            return self._raw_map[norm]
        # Whitespace-collapsed fallback
        flat = re.sub(r"\s+", "", citation_str)
        for key, entry in self._raw_map.items():
            if re.sub(r"\s+", "", key) == flat:
                return entry
        return None

    def build_citation_mappings(self) -> Dict[str, str]:
        """
        Return {citation_str: canonical_id} ready for
        CitationGraph.add_citation_mappings().

        All citation strings that refer to the same paper share one
        canonical_id, enabling cross-paper edges in the citation graph.
        """
        return {k: v.canonical_id for k, v in self._raw_map.items()}

    def all_resolved(self) -> Dict[str, ReferenceEntry]:
        """Return all (citation_str → ReferenceEntry) mappings."""
        return dict(self._raw_map)

    def as_dict(self) -> Dict[str, Dict]:
        """Export all resolved entries as plain dicts (JSON-serialisable)."""
        return {
            k: {
                "title": v.title,
                "authors": v.authors,
                "year": v.year,
                "canonical_id": v.canonical_id,
                "numeric_key": v.numeric_key,
                "raw_text": v.raw_text,
            }
            for k, v in self._raw_map.items()
        }

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
            resolver._raw_map[key] = entry
            resolver._canonical_map.setdefault(entry.canonical_id, entry)
        return resolver

    def __len__(self) -> int:
        return len(self._raw_map)

    def __repr__(self) -> str:
        n_canonical = len(self._canonical_map)
        return (
            f"CitationResolver(resolved={len(self._raw_map)}, "
            f"canonical_ids={n_canonical})"
        )
