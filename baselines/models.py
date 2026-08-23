from __future__ import annotations

"""Baseline models for section and citation importance scoring.

The goal of this module is to provide lightweight alternatives to the full
hierarchical LLM pipeline while keeping the output format compatible with the
existing evaluation and graph tooling.
"""

import argparse
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from ollama import Client
except ImportError:  # pragma: no cover - optional dependency for non-LLM baselines
    Client = None  # type: ignore[assignment]

from importance_score import (
    CITATION_BLOCK_PATTERN,
    build_citation_focus_text,
    canonicalize_citation_key,
    classify_citation_block,
    detect_dominant_citation_style,
    extract_citations_by_section,
    flatten_content_to_text,
    load_sections_from_file,
    normalize_for_match,
    normalize_distribution,
    normalized_key,
    parse_score_map_from_response,
    read_pdf_text,
    safe_float,
    split_citation_block,
    split_text_into_paragraphs,
)


def append_debug_log(debug_log_path: str, entry: str) -> None:
    if not debug_log_path:
        return
    path = Path(debug_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry.rstrip() + "\n")


def token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", text or ""))


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def is_retryable_openai_network_error(exc: urllib.error.URLError) -> bool:
    reason = getattr(exc, "reason", None)
    if isinstance(reason, socket.gaierror):
        return True
    if isinstance(reason, TimeoutError):
        return True

    message = str(reason or exc).lower()
    retryable_fragments = (
        "nodename nor servname provided",
        "name or service not known",
        "temporary failure in name resolution",
        "timed out",
        "connection reset",
        "connection refused",
        "network is unreachable",
        "no route to host",
    )
    return any(fragment in message for fragment in retryable_fragments)


ROOT_DIR = Path(__file__).resolve().parents[1]
HUMAN_ANNOTATION_INSTRUCTIONS_PATH = ROOT_DIR / "human_section_annotation_instructions.txt"
NODE_SCORE_LINE_RE = re.compile(
    r"^(?:(?:node\s+title)\s*:\s*)?(?P<title>.+?)\s*(?:(?::|[–—-])\s*|\|\s*)"
    r"(?:(?:total(?:\s+score)?)\s*:\s*)?"
    r"(?P<total>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*\|\s*"
    r"(?:(?:citation(?:\s+score)?)\s*:\s*)?"
    r"(?P<citation>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
    flags=re.IGNORECASE,
)
SINGLE_VALUE_NODE_SCORE_LINE_RE = re.compile(
    r"^(?:(?:node\s+title)\s*:\s*)?(?P<title>.+?)\s*(?::|[–—-])\s*"
    r"(?P<score>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
    flags=re.IGNORECASE,
)
CITATION_SCORE_LINE_RE = re.compile(
    r"^(?P<citation>.+)\s*:\s*(?P<score>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)
CITATION_SCORE_TABLE_ROW_RE = re.compile(
    r"^\|\s*(?P<citation>.+?)\s*\|\s*(?P<score>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*\|?\s*$"
)
PARAGRAPH_TITLE_RE = re.compile(r"^Paragraph\s+(?P<index>\d+)\s*$", flags=re.IGNORECASE)


def parse_node_score_line(line: str) -> Optional[Tuple[str, float, float, bool]]:
    match = NODE_SCORE_LINE_RE.match(line)
    if match:
        title = str(match.group("title")).strip()
        if not title:
            return None
        total_score = max(0.0, safe_float(match.group("total"), 0.0))
        citation_score = max(0.0, safe_float(match.group("citation"), 0.0))
        return title, total_score, citation_score, True

    match = SINGLE_VALUE_NODE_SCORE_LINE_RE.match(line)
    if match:
        title = str(match.group("title")).strip()
        if not title:
            return None
        total_score = max(0.0, safe_float(match.group("score"), 0.0))
        return title, total_score, 0.0, False

    return None


@dataclass
class SectionRecord:
    name: str
    path: List[str]
    text: str
    citation_contexts: Dict[str, List[str]]
    children: List["SectionRecord"] = field(default_factory=list)
    token_count: int = 0
    citation_mentions: Counter = field(default_factory=Counter)
    total_score: float = 0.0
    citation_score: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def title_key(self) -> str:
        return " > ".join(self.path)


def flatten_citation_contexts(raw: Any) -> Dict[str, List[str]]:
    flattened: Dict[str, List[str]] = defaultdict(list)
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, dict):
                nested = flatten_citation_contexts(value)
                for nested_key, contexts in nested.items():
                    flattened[nested_key].extend(contexts)
            elif isinstance(value, list):
                flattened[str(key)].extend(str(item) for item in value if str(item).strip())
            elif value is not None:
                flattened[str(key)].append(str(value))
    return dict(flattened)


def load_human_annotation_instructions() -> str:
    return HUMAN_ANNOTATION_INSTRUCTIONS_PATH.read_text(encoding="utf-8").strip()


def load_human_section_only_instructions() -> str:
    full_text = load_human_annotation_instructions()
    marker = "==================================================\nPART 2: TOP 4 FOUNDATIONAL PAPERS"
    if marker in full_text:
        return full_text.split(marker, 1)[0].rstrip()
    return full_text


def extract_json_payload(text: str) -> Dict[str, Any]:
    candidate = (text or "").strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, flags=re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()

    try:
        loaded = json.loads(candidate)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", candidate):
        try:
            loaded, _ = decoder.raw_decode(candidate[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded

    raise ValueError("Model response did not contain a parseable JSON object.")


def _clean_scored_line(line: str) -> str:
    cleaned = str(line or "").replace("\u202f", " ").replace("\xa0", " ").strip()
    cleaned = re.sub(r"^[\-\*\u2022]+\s*", "", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "")
    return cleaned.strip()


def _title_lookup_keys(title: str) -> List[str]:
    title = str(title or "").strip()
    if not title:
        return []

    candidates = [title]
    ampersand_variant = title.replace("&", " and ").strip()
    if ampersand_variant and ampersand_variant != title:
        candidates.append(ampersand_variant)
    stripped_numeric_prefix = re.sub(
        r"^\s*(?:section\s+)?(?:[ivxlcdm]+|\d+)(?:\.\d+)*[\)\.\-: ]+\s*",
        "",
        title,
        flags=re.IGNORECASE,
    ).strip()
    if stripped_numeric_prefix and stripped_numeric_prefix != title:
        candidates.append(stripped_numeric_prefix)
    stripped_parenthetical = re.sub(r"\([^)]*\)", "", title).strip()
    if stripped_parenthetical and stripped_parenthetical != title:
        candidates.append(stripped_parenthetical)

    expanded_candidates: List[str] = []
    for candidate in candidates:
        if candidate not in expanded_candidates:
            expanded_candidates.append(candidate)
        article_stripped = re.sub(r"\b(?:a|an|the)\b", " ", candidate, flags=re.IGNORECASE)
        article_stripped = re.sub(r"\s+", " ", article_stripped).strip()
        if article_stripped and article_stripped != candidate and article_stripped not in expanded_candidates:
            expanded_candidates.append(article_stripped)

    lookup_keys: List[str] = []
    for candidate in expanded_candidates:
        normalized = normalized_key(candidate)
        if normalized and normalized not in lookup_keys:
            lookup_keys.append(normalized)
    return lookup_keys


def build_section_child_lookup(
    section_schema: Dict[str, Any],
    path_prefix: Optional[List[str]] = None,
    lookup: Optional[Dict[Tuple[str, ...], Dict[str, str]]] = None,
) -> Dict[Tuple[str, ...], Dict[str, str]]:
    if lookup is None:
        lookup = {}

    current_path = tuple(path_prefix or [])
    children = section_schema if isinstance(section_schema, dict) else {}
    child_lookup: Dict[str, str] = {}
    for title in children:
        title_str = str(title)
        for key in _title_lookup_keys(title_str):
            child_lookup.setdefault(key, title_str)
    lookup[current_path] = child_lookup

    for title, child_schema in children.items():
        build_section_child_lookup(
            child_schema if isinstance(child_schema, dict) else {},
            path_prefix=list(current_path) + [str(title)],
            lookup=lookup,
        )

    return lookup


def extract_text_fallback_paragraph_payloads(
    text: str,
    section_schema: Dict[str, Any],
    paragraph_inventory: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    child_lookup = build_section_child_lookup(section_schema)
    paragraph_lookup: Dict[Tuple[Tuple[str, ...], int], Dict[str, Any]] = {}
    paragraphs_by_section: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for leaf in paragraph_inventory:
        section_path = tuple(str(part) for part in leaf.get("section_path", []))
        section_paragraphs = list(leaf.get("paragraphs", []))
        if section_paragraphs:
            paragraphs_by_section[section_path] = section_paragraphs
        for paragraph in section_paragraphs:
            paragraph_index = int(safe_float(paragraph.get("paragraph_index"), 0.0))
            if paragraph_index > 0:
                paragraph_lookup[(section_path, paragraph_index)] = paragraph

    def resolve_document_path(title: str, current_path: Sequence[str]) -> Optional[List[str]]:
        lookup_keys = _title_lookup_keys(title)
        current_tuple = tuple(str(part) for part in current_path)
        candidate_parents = [current_tuple[:idx] for idx in range(len(current_tuple), -1, -1)]
        for parent_path in candidate_parents:
            children = child_lookup.get(parent_path, {})
            for key in lookup_keys:
                child_title = children.get(key)
                if child_title is not None:
                    return list(parent_path) + [child_title]
        return None

    def looks_like_citation_child(title: str) -> bool:
        stripped = str(title or "").strip()
        if not stripped:
            return False
        if (stripped.startswith("(") and stripped.endswith(")")) or (stripped.startswith("[") and stripped.endswith("]")):
            return True
        return False

    raw_paragraph_scores: List[Dict[str, Any]] = []
    raw_paragraph_citation_scores: List[Dict[str, Any]] = []
    current_section_path: List[str] = []
    current_paragraph_key: Optional[Tuple[Tuple[str, ...], int]] = None
    in_citation_section = False

    for raw_line in str(text or "").splitlines():
        line = _clean_scored_line(raw_line)
        if not line:
            continue

        normalized_line = normalize_for_match(line).lower()
        if normalized_line == normalize_for_match("Citation Contributions").lower():
            in_citation_section = True
            current_paragraph_key = None
            continue
        if in_citation_section:
            continue

        node_match = parse_node_score_line(line)
        if node_match:
            title, total_score, citation_score, has_explicit_citation_score = node_match
            if current_paragraph_key is not None and looks_like_citation_child(title):
                paragraph_payload = paragraph_lookup.get(current_paragraph_key)
                if paragraph_payload is not None:
                    raw_paragraph_citation_scores.append(
                        {
                            "paragraph_id": str(paragraph_payload.get("paragraph_id", "")),
                            "section_path": list(current_paragraph_key[0]),
                            "paragraph_index": current_paragraph_key[1],
                            "citation": title,
                            "citation_score": max(total_score, citation_score),
                        }
                    )
                continue
            paragraph_match = PARAGRAPH_TITLE_RE.fullmatch(title)
            if paragraph_match:
                paragraph_index = int(paragraph_match.group("index"))
                paragraph_key = (tuple(current_section_path), paragraph_index)
                paragraph_payload = paragraph_lookup.get(paragraph_key)
                if paragraph_payload is None:
                    current_paragraph_key = None
                    continue
                raw_paragraph_scores.append(
                    {
                        "paragraph_id": str(paragraph_payload.get("paragraph_id", "")),
                        "section_path": list(current_section_path),
                        "paragraph_index": paragraph_index,
                        "technical_score": max(0.0, total_score - citation_score),
                        "citation_score": citation_score,
                        "raw_total_score": total_score,
                        "has_explicit_citation_score": has_explicit_citation_score,
                    }
                )
                current_paragraph_key = paragraph_key
                continue

            current_paragraph_key = None
            if normalize_for_match(title).lower().startswith(normalize_for_match("Paper").lower()):
                current_section_path = []
                continue

            resolved_path = resolve_document_path(title, current_section_path)
            if resolved_path is not None:
                current_section_path = resolved_path
                section_paragraphs = paragraphs_by_section.get(tuple(current_section_path), [])
                if len(section_paragraphs) == 1:
                    only_paragraph = section_paragraphs[0]
                    paragraph_key = (tuple(current_section_path), int(safe_float(only_paragraph.get("paragraph_index"), 0.0)))
                    raw_paragraph_scores.append(
                        {
                            "paragraph_id": str(only_paragraph.get("paragraph_id", "")),
                            "section_path": list(current_section_path),
                            "paragraph_index": paragraph_key[1],
                            "technical_score": max(0.0, total_score - citation_score),
                            "citation_score": citation_score,
                            "raw_total_score": total_score,
                            "has_explicit_citation_score": has_explicit_citation_score,
                        }
                    )
                    current_paragraph_key = paragraph_key
            continue

        if current_paragraph_key is None:
            continue

        citation_match = CITATION_SCORE_TABLE_ROW_RE.match(line)
        if not citation_match:
            citation_match = CITATION_SCORE_LINE_RE.match(line)
        if not citation_match:
            continue

        paragraph_payload = paragraph_lookup.get(current_paragraph_key)
        if paragraph_payload is None:
            continue
        raw_paragraph_citation_scores.append(
            {
                "paragraph_id": str(paragraph_payload.get("paragraph_id", "")),
                "section_path": list(current_paragraph_key[0]),
                "paragraph_index": current_paragraph_key[1],
                "citation": str(citation_match.group("citation")).strip(),
                "citation_score": max(0.0, safe_float(citation_match.group("score"), 0.0)),
            }
        )

    return raw_paragraph_scores, raw_paragraph_citation_scores


def parse_contribution_tree_text_response(
    text: str,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], float]:
    node_scores: Dict[str, Dict[str, float]] = {}
    citation_scores: Dict[str, Dict[str, float]] = {}
    in_citation_section = False

    for raw_line in str(text or "").splitlines():
        line = _clean_scored_line(raw_line)
        if not line:
            continue

        normalized_line = normalize_for_match(line).lower()
        if normalized_line == normalize_for_match("Citation Contributions").lower():
            in_citation_section = True
            continue
        if normalized_line in {
            normalize_for_match("Output format").lower(),
            normalize_for_match("Node Title: Total Score | Citation Score").lower(),
            normalize_for_match("Citation Identifier: Score").lower(),
            normalize_for_match("Paper").lower(),
            normalize_for_match("Contribution Tree").lower(),
        }:
            continue

        if in_citation_section:
            match = CITATION_SCORE_TABLE_ROW_RE.match(line)
            if match and normalize_for_match(match.group("citation")).lower() == normalize_for_match("Citation Identifier").lower():
                continue
            if match and re.fullmatch(r"-+", match.group("citation").strip()):
                continue
            if not match:
                match = CITATION_SCORE_LINE_RE.match(line)
            if not match:
                continue
            citation = str(match.group("citation")).strip()
            if not citation:
                continue
            citation_scores.setdefault(citation, {"citation_score": 0.0})
            citation_scores[citation]["citation_score"] += max(0.0, safe_float(match.group("score"), 0.0))
            continue

        match = parse_node_score_line(line)
        if not match:
            continue
        title, total_score, citation_score, _ = match
        entry = node_scores.setdefault(title, {"total_score": 0.0, "citation_score": 0.0})
        entry["total_score"] = max(entry["total_score"], total_score)
        entry["citation_score"] = max(entry["citation_score"], citation_score)

    if not node_scores and not citation_scores:
        raise ValueError("Model response did not contain parseable contribution-tree lines.")

    root_citation_score = 0.0
    if node_scores:
        _, root_payload = max(node_scores.items(), key=lambda item: safe_float(item[1].get("total_score"), 0.0))
        root_citation_score = max(0.0, safe_float(root_payload.get("citation_score"), 0.0))

    return node_scores, citation_scores, root_citation_score


def read_score_value(raw_value: Any) -> float:
    if isinstance(raw_value, dict):
        for field in ("total_score", "citation_score", "score", "value", "weight", "allocation", "credit"):
            if field in raw_value:
                return max(0.0, safe_float(raw_value.get(field), 0.0))
        return 0.0
    return max(0.0, safe_float(raw_value, 0.0))


def canonicalize_inventory_citation(citation: str) -> str:
    canonical = canonicalize_citation_key(str(citation))
    bracket_numeric = re.fullmatch(r"\[\s*([0-9,\-;– ]+)\s*\]", canonical)
    paren_numeric = re.fullmatch(r"\(\s*([0-9,\-;– ]+)\s*\)", canonical)
    match = bracket_numeric or paren_numeric
    if not match:
        return canonical

    inner = match.group(1)
    pieces = re.split(r"([,;–-])", inner)
    normalized_parts: List[str] = []
    for piece in pieces:
        if piece in {",", ";", "–", "-"}:
            normalized_parts.append(piece)
        else:
            cleaned = re.sub(r"\s+", "", piece)
            if cleaned:
                normalized_parts.append(cleaned)
    merged = "".join(normalized_parts)
    if bracket_numeric:
        return f"[{merged}]"
    return f"({merged})"


def normalize_section_scores_to_schema(
    schema: Dict[str, Any],
    raw_section_scores: Any,
    total_score: float = 1.0,
    fill_missing_uniform: bool = True,
) -> Dict[str, Any]:
    raw_dict = raw_section_scores if isinstance(raw_section_scores, dict) else {}
    sibling_weights = {
        section_name: read_score_value(raw_dict.get(section_name, 0.0))
        for section_name in schema
    }
    if sum(sibling_weights.values()) <= 0.0 and sibling_weights and fill_missing_uniform:
        sibling_weights = {section_name: 1.0 for section_name in schema}
    normalized_scores = (
        normalize_distribution(sibling_weights, total_score)
        if sibling_weights and sum(sibling_weights.values()) > 0.0
        else {section_name: 0.0 for section_name in schema}
    )

    normalized_tree: Dict[str, Any] = {}
    for section_name, child_schema in schema.items():
        raw_child = raw_dict.get(section_name, {})
        if isinstance(raw_child, dict):
            child_payload = raw_child.get("subsections")
            if not isinstance(child_payload, dict):
                child_payload = {
                    key: value
                    for key, value in raw_child.items()
                    if key not in {"total_score", "citation_score", "score", "value", "weight", "allocation", "credit"}
                }
        else:
            child_payload = {}
        subsections = (
            normalize_section_scores_to_schema(
                child_schema,
                child_payload,
                total_score=normalized_scores.get(section_name, 0.0),
                fill_missing_uniform=fill_missing_uniform,
            )
            if isinstance(child_schema, dict) and child_schema
            else {}
        )
        normalized_tree[section_name] = {
            "total_score": normalized_scores.get(section_name, 0.0),
            "citation_score": 0.0,
            "subsections": subsections,
        }
    return normalized_tree


def enforce_section_score_conservation(
    section_tree: Dict[str, Any],
    expected_total: float = 1.0,
    tol: float = 1e-8,
) -> float:
    if not isinstance(section_tree, dict):
        return 0.0

    sibling_totals = {
        section_name: max(0.0, safe_float(payload.get("total_score"), 0.0))
        for section_name, payload in section_tree.items()
        if isinstance(payload, dict)
    }
    actual_total = sum(sibling_totals.values())

    if actual_total <= 0.0:
        normalized_totals = {section_name: 0.0 for section_name in sibling_totals}
    elif abs(actual_total - expected_total) > tol:
        normalized_totals = normalize_distribution(sibling_totals, expected_total)
    else:
        normalized_totals = sibling_totals

    for section_name, payload in section_tree.items():
        if not isinstance(payload, dict):
            continue
        payload["total_score"] = normalized_totals.get(section_name, 0.0)
        subsections = payload.get("subsections", {})
        if isinstance(subsections, dict) and subsections:
            enforce_section_score_conservation(
                subsections,
                expected_total=max(0.0, safe_float(payload.get("total_score"), 0.0)),
                tol=tol,
            )

    return sum(
        max(0.0, safe_float(payload.get("total_score"), 0.0))
        for payload in section_tree.values()
        if isinstance(payload, dict)
    )


def build_raw_section_tree_from_flat_node_scores(
    schema: Dict[str, Any],
    flat_node_scores: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    lookup: Dict[str, Dict[str, float]] = {}
    for title, payload in flat_node_scores.items():
        for key in _title_lookup_keys(title):
            existing = lookup.get(key)
            if existing is None or read_score_value(payload) > read_score_value(existing):
                lookup[key] = payload

    def recurse(subschema: Dict[str, Any]) -> Dict[str, Any]:
        raw_tree: Dict[str, Any] = {}
        for section_name, child_schema in subschema.items():
            payload: Optional[Dict[str, float]] = None
            for key in _title_lookup_keys(section_name):
                payload = lookup.get(key)
                if payload is not None:
                    break
            raw_tree[section_name] = {
                "total_score": read_score_value(payload or 0.0),
                "citation_score": max(0.0, safe_float((payload or {}).get("citation_score"), 0.0)),
                "subsections": recurse(child_schema) if isinstance(child_schema, dict) and child_schema else {},
            }
        return raw_tree

    return recurse(schema)


def apply_scaled_section_citation_scores(
    normalized_section_tree: Dict[str, Any],
    raw_section_tree: Dict[str, Any],
) -> float:
    total = 0.0
    raw_tree = raw_section_tree if isinstance(raw_section_tree, dict) else {}

    for section_name, normalized_payload in normalized_section_tree.items():
        raw_payload = raw_tree.get(section_name, {})
        raw_subsections = raw_payload.get("subsections", {}) if isinstance(raw_payload, dict) else {}
        child_total = 0.0
        subsections = normalized_payload.get("subsections", {})
        if isinstance(subsections, dict) and subsections:
            child_total = apply_scaled_section_citation_scores(subsections, raw_subsections)

        raw_total = read_score_value(raw_payload if isinstance(raw_payload, dict) else 0.0)
        raw_citation = max(0.0, safe_float((raw_payload or {}).get("citation_score"), 0.0)) if isinstance(raw_payload, dict) else 0.0
        normalized_total = max(0.0, safe_float(normalized_payload.get("total_score"), 0.0))

        if raw_total > 0.0 and raw_citation > 0.0:
            citation_score = min(normalized_total, raw_citation * (normalized_total / raw_total))
        elif raw_citation > 0.0:
            citation_score = min(normalized_total, raw_citation)
        elif child_total > 0.0:
            citation_score = min(normalized_total, child_total)
        else:
            citation_score = 0.0

        normalized_payload["citation_score"] = citation_score
        total += citation_score

    return total


def scale_section_citation_scores(
    section_tree: Dict[str, Any],
    target_total: float,
) -> float:
    current_total = sum(
        max(0.0, safe_float(node.get("citation_score"), 0.0))
        for node in section_tree.values()
    )
    if current_total <= 0.0 or target_total < 0.0:
        return current_total

    scale = target_total / current_total

    def recurse(tree: Dict[str, Any]) -> None:
        for payload in tree.values():
            scaled = max(0.0, safe_float(payload.get("citation_score"), 0.0)) * scale
            payload["citation_score"] = min(
                max(0.0, safe_float(payload.get("total_score"), 0.0)),
                scaled,
            )
            subsections = payload.get("subsections", {})
            if isinstance(subsections, dict) and subsections:
                recurse(subsections)

    recurse(section_tree)
    return sum(max(0.0, safe_float(node.get("citation_score"), 0.0)) for node in section_tree.values())


def total_section_citation_score(section_tree: Dict[str, Any]) -> float:
    if not isinstance(section_tree, dict):
        return 0.0
    return sum(
        max(0.0, safe_float(node.get("citation_score"), 0.0))
        for node in section_tree.values()
        if isinstance(node, dict)
    )


def collect_citation_inventory(
    citation_tree: Any,
    path_prefix: Optional[List[str]] = None,
    inventory: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    if inventory is None:
        inventory = {}
    if not isinstance(citation_tree, dict):
        return inventory

    for key, value in citation_tree.items():
        if isinstance(value, dict):
            collect_citation_inventory(value, path_prefix=(path_prefix or []) + [str(key)], inventory=inventory)
            continue
        split_keys = split_citation_block(str(key))
        if not split_keys:
            split_keys = [str(key)]
        mention_increment = len(value) if isinstance(value, list) else (1 if value is not None else 0)
        section_path = " > ".join(path_prefix or [])
        for split_key in split_keys:
            citation_key = canonicalize_inventory_citation(split_key)
            entry = inventory.setdefault(citation_key, {"mention_count": 0, "sections": []})
            entry["mention_count"] += mention_increment
            if section_path and section_path not in entry["sections"]:
                entry["sections"].append(section_path)

    return inventory


def build_leaf_paragraph_inventory(
    content_tree: Any,
    citation_tree: Any,
    paper_id: str,
    path_prefix: Optional[List[str]] = None,
    inventory: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if inventory is None:
        inventory = []
    if not isinstance(content_tree, dict):
        return inventory

    for section_name, section_content in content_tree.items():
        section_path = list(path_prefix or []) + [str(section_name)]
        section_citations = citation_tree.get(section_name, {}) if isinstance(citation_tree, dict) else {}

        if isinstance(section_content, dict) and section_content:
            build_leaf_paragraph_inventory(
                section_content,
                section_citations,
                paper_id=paper_id,
                path_prefix=section_path,
                inventory=inventory,
            )
            continue

        raw_text = section_content if isinstance(section_content, str) else flatten_content_to_text(section_content, 6000)
        paragraphs = split_text_into_paragraphs(raw_text)
        if not paragraphs:
            normalized_text = normalize_for_match(raw_text)
            if normalized_text:
                paragraphs = [normalized_text]
        if not paragraphs:
            continue

        paragraph_items = {f"Paragraph {idx + 1}": paragraph for idx, paragraph in enumerate(paragraphs)}
        paragraph_norms = {name: normalize_for_match(text) for name, text in paragraph_items.items()}
        paragraph_tokens = {
            name: set(re.findall(r"[a-z0-9]+", paragraph_norms[name].lower()))
            for name in paragraph_items
        }
        mention_buckets: Dict[str, List[Tuple[str, str, str]]] = {name: [] for name in paragraph_items}

        section_citation_dict = section_citations if isinstance(section_citations, dict) else {}
        for citation_block, context_value in section_citation_dict.items():
            contexts = context_value if isinstance(context_value, list) else [context_value]
            for context in contexts:
                context_str = str(context)
                context_norm = normalize_for_match(context_str)
                target_paragraph = None

                if context_norm:
                    for paragraph_name, paragraph_norm in paragraph_norms.items():
                        if context_norm in paragraph_norm:
                            target_paragraph = paragraph_name
                            break

                if target_paragraph is None and context_norm:
                    context_tokens = set(re.findall(r"[a-z0-9]+", context_norm.lower()))
                    if context_tokens:
                        best_name = None
                        best_overlap = -1
                        for paragraph_name, p_tokens in paragraph_tokens.items():
                            overlap = len(context_tokens & p_tokens)
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_name = paragraph_name
                        if best_name is not None and best_overlap > 0:
                            target_paragraph = best_name

                if target_paragraph is None:
                    target_paragraph = max(paragraph_items, key=lambda name: len(paragraph_items[name]))

                for citation in split_citation_block(str(citation_block)):
                    mention_buckets[target_paragraph].append(
                        (canonicalize_inventory_citation(citation), str(citation_block), context_str)
                    )

        paragraph_payloads: List[Dict[str, Any]] = []
        for idx, paragraph_name in enumerate(paragraph_items, start=1):
            paragraph_text = paragraph_items[paragraph_name]
            mentions = mention_buckets.get(paragraph_name, [])
            if not mentions:
                mentions = infer_paragraph_citation_mentions(paragraph_text)
            citation_contexts: Dict[str, List[str]] = defaultdict(list)
            for citation, _, context in mentions:
                normalized_context = normalize_for_match(context)
                if normalized_context:
                    citation_contexts[citation].append(normalized_context)

            citation_entries = []
            for citation, contexts in citation_contexts.items():
                unique_contexts = [ctx for ctx in dict.fromkeys(contexts) if ctx]
                context_blob = " ".join(unique_contexts)[:700]
                citation_entries.append(
                    {
                        "citation": citation,
                        "mention_count": len(contexts),
                        "context": context_blob if context_blob else paragraph_text[:700],
                    }
                )

            paragraph_payloads.append(
                {
                    "paragraph_id": f"{paper_id}::{' > '.join(section_path)}::p{idx}",
                    "section_path": list(section_path),
                    "paragraph_index": idx,
                    "text": paragraph_text,
                    "has_citations": bool(mentions),
                    "citation_focus_text": build_citation_focus_text(paragraph_text, mentions),
                    "citations": citation_entries,
                }
            )

        inventory.append(
            {
                "section_path": list(section_path),
                "paragraphs": paragraph_payloads,
            }
        )

    return inventory


def collect_citation_inventory_from_paragraph_inventory(
    paragraph_inventory: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    inventory: Dict[str, Dict[str, Any]] = {}
    for leaf in paragraph_inventory:
        section_path = " > ".join(str(part) for part in leaf.get("section_path", []))
        for paragraph in leaf.get("paragraphs", []):
            for citation_entry in paragraph.get("citations", []):
                citation = canonicalize_inventory_citation(str(citation_entry.get("citation", "")))
                if not citation:
                    continue
                mention_increment = max(1, int(safe_float(citation_entry.get("mention_count", 1), 1.0)))
                entry = inventory.setdefault(citation, {"mention_count": 0, "sections": []})
                entry["mention_count"] += mention_increment
                if section_path and section_path not in entry["sections"]:
                    entry["sections"].append(section_path)
    return inventory


def section_schema_from_content(content: Any) -> Dict[str, Any]:
    if not isinstance(content, dict):
        return {}
    schema: Dict[str, Any] = {}
    for key, value in content.items():
        schema[str(key)] = section_schema_from_content(value) if isinstance(value, dict) else {}
    return schema


def render_authoritative_hierarchy(
    section_schema: Dict[str, Any],
    paragraph_inventory: Sequence[Dict[str, Any]],
) -> str:
    tree: Dict[str, Any] = {"children": {}}

    def ensure_section_nodes(parent: Dict[str, Any], schema: Dict[str, Any]) -> None:
        for title, child_schema in schema.items():
            node = parent.setdefault(str(title), {"children": {}})
            if isinstance(child_schema, dict) and child_schema:
                ensure_section_nodes(node["children"], child_schema)

    ensure_section_nodes(tree["children"], section_schema if isinstance(section_schema, dict) else {})

    path_to_node: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    def index_paths(children: Dict[str, Any], prefix: Optional[List[str]] = None) -> None:
        for title, node in children.items():
            current_path = tuple(list(prefix or []) + [str(title)])
            path_to_node[current_path] = node
            node_children = node.get("children", {})
            if isinstance(node_children, dict) and node_children:
                index_paths(node_children, list(current_path))

    index_paths(tree["children"])

    for leaf in paragraph_inventory:
        section_path = tuple(str(part) for part in leaf.get("section_path", []))
        parent_node = path_to_node.get(section_path)
        if parent_node is None:
            continue

        parent_children = parent_node.setdefault("children", {})
        for paragraph in leaf.get("paragraphs", []):
            paragraph_index = max(1, int(safe_float(paragraph.get("paragraph_index"), 1)))
            paragraph_title = f"Paragraph {paragraph_index}"
            paragraph_node = parent_children.setdefault(paragraph_title, {"children": {}})
            citation_children = paragraph_node.setdefault("children", {})
            for citation_entry in paragraph.get("citations", []):
                citation = str(citation_entry.get("citation", "")).strip()
                if citation and citation not in citation_children:
                    citation_children[citation] = {"children": {}}

    lines = ["Paper"]

    def emit(children: Dict[str, Any], depth: int) -> None:
        indent = "  " * depth
        for title, node in children.items():
            lines.append(f"{indent}- {title}")
            node_children = node.get("children", {})
            if isinstance(node_children, dict) and node_children:
                emit(node_children, depth + 1)

    emit(tree["children"], depth=1)
    return "\n".join(lines)


def section_totals_by_path(section_tree: Dict[str, Any], path_prefix: Optional[List[str]] = None) -> Dict[Tuple[str, ...], float]:
    totals: Dict[Tuple[str, ...], float] = {}
    for section_name, payload in section_tree.items():
        current_path = tuple(list(path_prefix or []) + [section_name])
        totals[current_path] = max(0.0, safe_float(payload.get("total_score"), 0.0))
        subsections = payload.get("subsections", {})
        if isinstance(subsections, dict) and subsections:
            totals.update(section_totals_by_path(subsections, list(current_path)))
    return totals


def annotate_section_citation_scores(
    section_tree: Dict[str, Any],
    paragraph_scores: Sequence[Dict[str, Any]],
    path_prefix: Optional[List[str]] = None,
) -> float:
    total = 0.0
    for section_name, payload in section_tree.items():
        current_path = list(path_prefix or []) + [section_name]
        subsections = payload.get("subsections", {})
        if isinstance(subsections, dict) and subsections:
            citation_total = annotate_section_citation_scores(subsections, paragraph_scores, current_path)
        else:
            citation_total = sum(
                max(0.0, safe_float(item.get("citation_score"), 0.0))
                for item in paragraph_scores
                if list(item.get("section_path", [])) == current_path
            )
        payload["citation_score"] = citation_total
        total += citation_total
    return total


def _paragraph_lookup_key(section_path: Sequence[str], paragraph_index: int) -> Tuple[Tuple[str, ...], int]:
    return (tuple(str(part) for part in section_path), int(paragraph_index))


def _extract_channel_value(raw_item: Dict[str, Any], field: str, aliases: Sequence[str]) -> float:
    if field in raw_item:
        return max(0.0, safe_float(raw_item.get(field), 0.0))
    for alias in aliases:
        if alias in raw_item:
            return max(0.0, safe_float(raw_item.get(alias), 0.0))
    return 0.0


def normalize_openai_paragraph_scores(
    paragraph_inventory: Sequence[Dict[str, Any]],
    raw_paragraph_scores: Any,
    normalized_section_scores: Dict[str, Any],
    raw_paragraph_citation_scores: Any = None,
) -> List[Dict[str, Any]]:
    raw_items = raw_paragraph_scores if isinstance(raw_paragraph_scores, list) else []
    raw_citation_items = raw_paragraph_citation_scores if isinstance(raw_paragraph_citation_scores, list) else []
    by_id: Dict[str, Dict[str, Any]] = {}
    by_key: Dict[Tuple[Tuple[str, ...], int], Dict[str, Any]] = {}
    paragraphs_with_raw_citation_children: set[str] = set()
    paragraph_keys_with_raw_citation_children: set[Tuple[Tuple[str, ...], int]] = set()
    child_citation_total_by_id: Dict[str, float] = defaultdict(float)
    child_citation_total_by_key: Dict[Tuple[Tuple[str, ...], int], float] = defaultdict(float)
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        paragraph_id = str(item.get("paragraph_id", "")).strip()
        if paragraph_id:
            by_id[paragraph_id] = item
        section_path = item.get("section_path")
        paragraph_index = safe_float(item.get("paragraph_index"), -1)
        if isinstance(section_path, list) and paragraph_index >= 0:
            by_key[_paragraph_lookup_key(section_path, int(paragraph_index))] = item
    for item in raw_citation_items:
        if not isinstance(item, dict):
            continue
        citation_score = max(0.0, safe_float(item.get("citation_score", item.get("score", 0.0)), 0.0))
        paragraph_id = str(item.get("paragraph_id", "")).strip()
        if paragraph_id:
            paragraphs_with_raw_citation_children.add(paragraph_id)
            child_citation_total_by_id[paragraph_id] += citation_score
        section_path = item.get("section_path")
        paragraph_index = safe_float(item.get("paragraph_index"), -1)
        if isinstance(section_path, list) and paragraph_index >= 0:
            paragraph_key = _paragraph_lookup_key(section_path, int(paragraph_index))
            paragraph_keys_with_raw_citation_children.add(paragraph_key)
            child_citation_total_by_key[paragraph_key] += citation_score

    section_total_lookup = section_totals_by_path(normalized_section_scores)
    normalized_paragraphs: List[Dict[str, Any]] = []

    for leaf in paragraph_inventory:
        section_path = list(leaf.get("section_path", []))
        section_total = section_total_lookup.get(tuple(section_path), 0.0)
        expected_paragraphs = leaf.get("paragraphs", [])

        raw_totals: Dict[str, float] = {}
        channel_ratios: Dict[str, Tuple[float, float]] = {}
        for paragraph in expected_paragraphs:
            paragraph_id = str(paragraph.get("paragraph_id", ""))
            paragraph_index = int(paragraph.get("paragraph_index", 0))
            paragraph_key = _paragraph_lookup_key(section_path, paragraph_index)
            raw_item = by_id.get(paragraph_id) or by_key.get(paragraph_key)
            raw_technical = _extract_channel_value(raw_item or {}, "technical_score", ("technical",))
            raw_citation = _extract_channel_value(raw_item or {}, "citation_score", ("citation",))
            explicit_raw_total = max(
                0.0,
                safe_float((raw_item or {}).get("raw_total_score", (raw_item or {}).get("total_score", 0.0)), 0.0),
            )
            has_explicit_citation_score = bool((raw_item or {}).get("has_explicit_citation_score"))
            has_citations = bool(paragraph.get("has_citations"))
            has_explicit_citation_children = (
                paragraph_id in paragraphs_with_raw_citation_children
                or paragraph_key in paragraph_keys_with_raw_citation_children
            )
            child_citation_total = child_citation_total_by_id.get(paragraph_id, 0.0) + child_citation_total_by_key.get(paragraph_key, 0.0)
            if has_explicit_citation_children and raw_citation <= 0.0 and child_citation_total > 0.0:
                raw_citation = child_citation_total

            if explicit_raw_total > 0.0:
                raw_total = explicit_raw_total
                if not has_explicit_citation_score:
                    raw_technical = max(0.0, raw_total - raw_citation)
                elif raw_technical + raw_citation > raw_total > 0.0:
                    scale = raw_total / (raw_technical + raw_citation)
                    raw_technical *= scale
                    raw_citation *= scale
            else:
                raw_total = raw_technical + raw_citation

            if raw_total <= 0.0:
                raw_total = float(max(1, token_count(str(paragraph.get("text", "")))))

            if not has_citations and not has_explicit_citation_children:
                tech_ratio, cit_ratio = 1.0, 0.0
            elif raw_technical + raw_citation > 0.0:
                tech_ratio = raw_technical / (raw_technical + raw_citation)
                cit_ratio = raw_citation / (raw_technical + raw_citation)
            else:
                mention_total = sum(int(entry.get("mention_count", 0)) for entry in paragraph.get("citations", []))
                cit_ratio = min(0.40, max(0.10, 0.12 * max(1, mention_total)))
                tech_ratio = 1.0 - cit_ratio

            raw_totals[paragraph_id] = raw_total
            channel_ratios[paragraph_id] = (tech_ratio, cit_ratio)

        paragraph_totals = normalize_distribution(raw_totals, section_total) if raw_totals else {}
        for paragraph in expected_paragraphs:
            paragraph_id = str(paragraph.get("paragraph_id", ""))
            paragraph_total = max(0.0, safe_float(paragraph_totals.get(paragraph_id), 0.0))
            tech_ratio, cit_ratio = channel_ratios.get(paragraph_id, (1.0, 0.0))
            technical_score = paragraph_total * tech_ratio
            citation_score = paragraph_total * cit_ratio
            normalized_paragraphs.append(
                {
                    "section_path": section_path,
                    "paragraph_index": int(paragraph.get("paragraph_index", 0)),
                    "paragraph": str(paragraph.get("text", "")),
                    "paragraph_id": paragraph_id,
                    "technical_score": technical_score,
                    "citation_score": citation_score,
                }
            )

    return normalized_paragraphs


def normalize_openai_paragraph_citation_scores(
    paragraph_inventory: Sequence[Dict[str, Any]],
    paragraph_scores: Sequence[Dict[str, Any]],
    raw_paragraph_citation_scores: Any,
    global_citation_scores: Any = None,
) -> List[Dict[str, Any]]:
    raw_items = raw_paragraph_citation_scores if isinstance(raw_paragraph_citation_scores, list) else []
    grouped_raw: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        paragraph_id = str(item.get("paragraph_id", "")).strip()
        if paragraph_id:
            grouped_raw[paragraph_id].append(item)
            continue
        section_path = item.get("section_path")
        paragraph_index = safe_float(item.get("paragraph_index"), -1)
        if isinstance(section_path, list) and paragraph_index >= 0:
            grouped_raw[str(_paragraph_lookup_key(section_path, int(paragraph_index)))].append(item)

    paragraph_score_lookup = {
        str(item.get("paragraph_id", "")): item for item in paragraph_scores if str(item.get("paragraph_id", ""))
    }
    normalized_outputs: List[Dict[str, Any]] = []

    for leaf in paragraph_inventory:
        section_path = list(leaf.get("section_path", []))
        for paragraph in leaf.get("paragraphs", []):
            paragraph_id = str(paragraph.get("paragraph_id", ""))
            paragraph_index = int(paragraph.get("paragraph_index", 0))
            paragraph_score_payload = paragraph_score_lookup.get(paragraph_id)
            if paragraph_score_payload is None:
                continue
            paragraph_citation_budget = max(0.0, safe_float(paragraph_score_payload.get("citation_score"), 0.0))
            raw_group = grouped_raw.get(paragraph_id, []) + grouped_raw.get(str(_paragraph_lookup_key(section_path, paragraph_index)), [])
            citation_entries = list(paragraph.get("citations", []))
            if not citation_entries and raw_group:
                inferred_counts: Dict[str, int] = defaultdict(int)
                for item in raw_group:
                    citation = item.get("citation") or item.get("id") or item.get("key")
                    if citation is None:
                        continue
                    split_keys = split_citation_block(str(citation))
                    if not split_keys:
                        split_keys = [str(citation)]
                    for split_key in split_keys:
                        canonical = canonicalize_inventory_citation(split_key)
                        if canonical:
                            inferred_counts[canonical] += 1
                citation_entries = [
                    {
                        "citation": citation,
                        "mention_count": mention_count,
                        "context": str(paragraph.get("text", ""))[:700],
                    }
                    for citation, mention_count in inferred_counts.items()
                ]
            if paragraph_citation_budget <= 0.0 or not citation_entries:
                continue

            candidates = {
                canonicalize_inventory_citation(str(entry.get("citation", ""))): int(entry.get("mention_count", 0))
                for entry in citation_entries
            }
            candidate_norm_to_id = {
                normalized_key(citation): citation
                for citation in candidates
                if normalized_key(citation)
            }
            raw_weights: Dict[str, float] = {}
            candidate_ids = set(candidates.keys())
            for item in raw_group:
                citation = item.get("citation") or item.get("id") or item.get("key")
                if citation is None:
                    continue
                value = max(0.0, safe_float(item.get("citation_score", item.get("score", 0.0)), 0.0))
                split_keys = split_citation_block(str(citation))
                if len(split_keys) > 1:
                    per_value = value / len(split_keys)
                    for split_key in split_keys:
                        canonical = canonicalize_inventory_citation(split_key)
                        target_key = canonical if canonical in candidate_ids else candidate_norm_to_id.get(normalized_key(split_key))
                        if target_key is not None:
                            raw_weights[target_key] = raw_weights.get(target_key, 0.0) + per_value
                    continue
                canonical = canonicalize_inventory_citation(str(citation))
                target_key = canonical if canonical in candidate_ids else candidate_norm_to_id.get(normalized_key(str(citation)))
                if target_key is not None:
                    raw_weights[target_key] = raw_weights.get(target_key, 0.0) + value

            if sum(raw_weights.values()) <= 0.0:
                global_items: List[Tuple[str, float]] = []
                if isinstance(global_citation_scores, list):
                    for item in global_citation_scores:
                        if not isinstance(item, dict):
                            continue
                        citation = item.get("citation") or item.get("id") or item.get("key")
                        if citation is None:
                            continue
                        global_items.append(
                            (
                                str(citation),
                                max(0.0, safe_float(item.get("citation_score", item.get("score", 0.0)), 0.0)),
                            )
                        )
                elif isinstance(global_citation_scores, dict):
                    for citation, payload in global_citation_scores.items():
                        global_items.append((str(citation), read_score_value(payload)))

                for citation, value in global_items:
                    split_keys = split_citation_block(str(citation))
                    if len(split_keys) > 1:
                        per_value = value / len(split_keys)
                        for split_key in split_keys:
                            canonical = canonicalize_inventory_citation(split_key)
                            target_key = canonical if canonical in candidate_ids else candidate_norm_to_id.get(normalized_key(split_key))
                            if target_key is not None:
                                raw_weights[target_key] = raw_weights.get(target_key, 0.0) + per_value
                        continue
                    canonical = canonicalize_inventory_citation(str(citation))
                    target_key = canonical if canonical in candidate_ids else candidate_norm_to_id.get(normalized_key(str(citation)))
                    if target_key is not None:
                        raw_weights[target_key] = raw_weights.get(target_key, 0.0) + value

            if sum(raw_weights.values()) <= 0.0:
                raw_weights = {citation: float(max(1, mention_count)) for citation, mention_count in candidates.items()}
            if sum(raw_weights.values()) <= 0.0:
                raw_weights = {citation: 1.0 for citation in candidates}

            normalized = normalize_distribution(raw_weights, paragraph_citation_budget)
            for citation in candidates:
                normalized_outputs.append(
                    {
                        "section_path": section_path,
                        "paragraph_index": paragraph_index,
                        "paragraph": str(paragraph.get("text", "")),
                        "paragraph_id": paragraph_id,
                        "citation": citation,
                        "citation_score": max(0.0, safe_float(normalized.get(citation), 0.0)),
                    }
                )

    return normalized_outputs


def aggregate_citation_scores_from_paragraphs(
    paragraph_citation_scores: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    aggregated: Dict[str, Dict[str, float]] = {}
    for item in paragraph_citation_scores:
        citation = canonicalize_inventory_citation(str(item.get("citation", "")))
        if not citation:
            continue
        aggregated.setdefault(citation, {"citation_score": 0.0})
        aggregated[citation]["citation_score"] += max(0.0, safe_float(item.get("citation_score"), 0.0))
    return aggregated


def normalize_citation_scores(
    citation_inventory: Dict[str, Dict[str, Any]],
    raw_citation_scores: Any,
    total_score: float = 1.0,
    fallback_to_mentions: bool = True,
) -> Dict[str, Dict[str, float]]:
    expected_ids = list(citation_inventory.keys())
    if not expected_ids:
        return {}

    norm_to_id = {normalized_key(citation): citation for citation in expected_ids}
    canonical_to_id = {canonicalize_inventory_citation(citation): citation for citation in expected_ids}
    raw_weights: Dict[str, float] = {}

    def add_weight(raw_key: Any, raw_value: Any) -> None:
        value = read_score_value(raw_value)
        split_keys = split_citation_block(str(raw_key))
        if len(split_keys) > 1:
            per_key_value = value / len(split_keys) if split_keys else 0.0
            for split_key in split_keys:
                canonical_key = canonicalize_inventory_citation(split_key)
                target_key = (
                    canonical_to_id.get(canonical_key)
                    or (split_key if split_key in citation_inventory else None)
                    or norm_to_id.get(normalized_key(split_key))
                )
                if target_key is None:
                    continue
                raw_weights[target_key] = raw_weights.get(target_key, 0.0) + per_key_value
            return

        canonical_key = canonicalize_inventory_citation(str(raw_key))
        target_key = (
            canonical_to_id.get(canonical_key)
            or (str(raw_key) if str(raw_key) in citation_inventory else None)
            or norm_to_id.get(normalized_key(str(raw_key)))
        )
        if target_key is None:
            return
        raw_weights[target_key] = raw_weights.get(target_key, 0.0) + value

    if isinstance(raw_citation_scores, list):
        for item in raw_citation_scores:
            if not isinstance(item, dict):
                continue
            citation_key = item.get("citation") or item.get("id") or item.get("key")
            if citation_key is None:
                continue
            add_weight(citation_key, item)
    else:
        raw_dict = raw_citation_scores if isinstance(raw_citation_scores, dict) else {}
        for raw_key, raw_value in raw_dict.items():
            add_weight(raw_key, raw_value)

    if sum(raw_weights.values()) <= 0.0 and fallback_to_mentions:
        raw_weights = {citation: float(citation_inventory[citation]["mention_count"]) for citation in expected_ids}
    if sum(raw_weights.values()) <= 0.0 and fallback_to_mentions:
        raw_weights = {citation: 1.0 for citation in expected_ids}
    if sum(raw_weights.values()) <= 0.0:
        return {citation: {"citation_score": 0.0} for citation in expected_ids}

    normalized_scores = normalize_distribution(raw_weights, total_score)
    return {
        citation: {"citation_score": normalized_scores.get(citation, 0.0)}
        for citation in expected_ids
    }


def build_section_records(
    content: Dict[str, Any],
    citations: Dict[str, Any],
    path_prefix: Optional[List[str]] = None,
) -> List[SectionRecord]:
    records: List[SectionRecord] = []
    for section_name, section_content in content.items():
        path = list(path_prefix or []) + [section_name]
        section_citations = citations.get(section_name, {}) if isinstance(citations, dict) else {}
        if isinstance(section_content, dict):
            children = build_section_records(section_content, section_citations, path)
            text = flatten_content_to_text(section_content, limit=0)
            mention_counter = Counter()
            for child in children:
                mention_counter.update(child.citation_mentions)
            record = SectionRecord(
                name=section_name,
                path=path,
                text=text,
                citation_contexts=flatten_citation_contexts(section_citations),
                children=children,
                token_count=max(1, token_count(text)),
                citation_mentions=mention_counter,
            )
        else:
            text = str(section_content or "")
            flattened = flatten_citation_contexts(section_citations)
            mention_counter = Counter({citation: len(contexts) for citation, contexts in flattened.items()})
            record = SectionRecord(
                name=section_name,
                path=path,
                text=text,
                citation_contexts=flattened,
                children=[],
                token_count=max(1, token_count(text)),
                citation_mentions=mention_counter,
            )
        records.append(record)
    return records


def iter_records(records: Iterable[SectionRecord]) -> Iterable[SectionRecord]:
    for record in records:
        yield record
        yield from iter_records(record.children)


def scan_paragraph_citations(paragraph_text: str, expected_citations: Sequence[str]) -> Counter:
    expected_set = set(expected_citations)
    if not expected_set:
        return Counter()

    dominant_style = detect_dominant_citation_style(paragraph_text or "")
    mention_counter: Counter = Counter()
    for match in re.finditer(CITATION_BLOCK_PATTERN, paragraph_text or ""):
        block = match.group(0)
        prefix = (paragraph_text or "")[max(0, match.start() - 24) : match.start()]
        suffix = (paragraph_text or "")[match.end() : min(len(paragraph_text or ""), match.end() + 24)]
        style = classify_citation_block(block, prefix_text=prefix, suffix_text=suffix)
        if style is None or style != dominant_style:
            continue
        for citation in split_citation_block(block):
            key = canonicalize_citation_key(citation)
            if key in expected_set:
                mention_counter[key] += 1
    return mention_counter


def infer_paragraph_citation_mentions(paragraph_text: str) -> List[Tuple[str, str, str]]:
    mentions: List[Tuple[str, str, str]] = []
    for match in re.finditer(CITATION_BLOCK_PATTERN, paragraph_text or ""):
        citation_block = match.group(0)
        prefix_text = (paragraph_text or "")[max(0, match.start() - 24) : match.start()]
        suffix_text = (paragraph_text or "")[match.end() : min(len(paragraph_text or ""), match.end() + 24)]
        citation_style = classify_citation_block(
            citation_block,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
        )
        if citation_style is None:
            continue
        for citation in split_citation_block(citation_block):
            canonical = canonicalize_inventory_citation(citation)
            if canonical:
                mentions.append((canonical, citation_block, citation_block))
    return mentions


def format_section_excerpt(record: SectionRecord, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", record.text).strip()
    if limit > 0:
        text = text[:limit]
    return f"Section name: {record.name}\nExcerpt:\n{text}"


class BaselineModel(ABC):
    """Shared interface for deterministic and single-pass baseline models."""
    model_tag = "baseline"
    uses_llm = False
    citation_only_output = False

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_retries: int = 3,
        debug_log_path: str = "",
    ) -> None:
        self.model = model
        self.host = host
        self.temperature = temperature
        self.max_retries = max(1, max_retries)
        self.debug_log_path = debug_log_path
        if self.uses_llm:
            if Client is None:
                raise ImportError(
                    "The 'ollama' Python package is required for the single_pass_llm baseline."
                )
            self.client = Client(host=host)
        else:
            self.client = None

    @abstractmethod
    def raw_section_weights(
        self,
        records: Sequence[SectionRecord],
        parent_path: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def leaf_citation_fraction(self, record: SectionRecord) -> float:
        raise NotImplementedError

    @abstractmethod
    def raw_citation_weight(self, record: SectionRecord, citation: str, mentions: int) -> float:
        raise NotImplementedError

    def raw_paragraph_weight(self, paragraph_text: str, mention_count: int) -> float:
        return float(max(1, token_count(paragraph_text)) + 50 * max(0, mention_count))

    def assign_section_scores(
        self,
        records: Sequence[SectionRecord],
        total_score: float = 1.0,
        parent_path: Optional[List[str]] = None,
    ) -> None:
        raw_weights = self.raw_section_weights(records, parent_path=parent_path)
        weights = normalize_distribution(raw_weights, total_score)
        for record in records:
            record.total_score = weights.get(record.name, 0.0)
            if record.children:
                self.assign_section_scores(record.children, record.total_score, parent_path=record.path)

    def build_outputs(
        self,
        content_dict: Dict[str, Any],
        citations_dict: Dict[str, Any],
        paper_id: str,  # noqa: ARG002
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        records = build_section_records(content_dict, citations_dict)
        self.assign_section_scores(records, total_score=1.0, parent_path=None)

        citation_scores: Dict[str, Dict[str, float]] = {}
        paragraph_scores: List[Dict[str, Any]] = []
        paragraph_citation_scores: List[Dict[str, Any]] = []

        def process_leaf(record: SectionRecord) -> None:
            raw_text = record.text.strip()
            paragraphs = split_text_into_paragraphs(raw_text)
            if not paragraphs:
                paragraphs = [raw_text] if raw_text else [record.name]

            paragraph_mentions = [scan_paragraph_citations(p, list(record.citation_mentions.keys())) for p in paragraphs]
            paragraph_raw = {
                f"Paragraph {idx + 1}": self.raw_paragraph_weight(paragraphs[idx], sum(paragraph_mentions[idx].values()))
                for idx in range(len(paragraphs))
            }
            paragraph_total = normalize_distribution(paragraph_raw, record.total_score)

            citation_fraction = max(0.0, min(0.95, self.leaf_citation_fraction(record)))
            record.citation_score = record.total_score * citation_fraction

            citation_raw = {
                citation: self.raw_citation_weight(record, citation, mentions)
                for citation, mentions in record.citation_mentions.items()
                if mentions > 0
            }
            citation_budget = normalize_distribution(citation_raw, record.citation_score) if citation_raw else {}

            per_paragraph_citation_allocations: Dict[int, Dict[str, float]] = {}
            for idx, mention_counter in enumerate(paragraph_mentions):
                local_raw = {
                    citation: float(count) * citation_budget.get(citation, 0.0)
                    for citation, count in mention_counter.items()
                    if citation_budget.get(citation, 0.0) > 0.0
                }
                if local_raw:
                    total_local = sum(local_raw.values())
                    alloc = {
                        citation: (value / total_local) * min(paragraph_total[f"Paragraph {idx + 1}"] * citation_fraction, sum(local_raw.values()))
                        for citation, value in local_raw.items()
                    }
                else:
                    alloc = {}
                per_paragraph_citation_allocations[idx] = alloc

            leaf_citation_total = 0.0
            for idx, paragraph_text in enumerate(paragraphs):
                paragraph_name = f"Paragraph {idx + 1}"
                paragraph_citation_score = sum(per_paragraph_citation_allocations[idx].values())
                if paragraph_citation_score > paragraph_total[paragraph_name]:
                    scale = paragraph_total[paragraph_name] / paragraph_citation_score
                    per_paragraph_citation_allocations[idx] = {
                        key: value * scale for key, value in per_paragraph_citation_allocations[idx].items()
                    }
                    paragraph_citation_score = sum(per_paragraph_citation_allocations[idx].values())

                paragraph_scores.append(
                    {
                        "section_path": list(record.path),
                        "paragraph_index": idx + 1,
                        "paragraph": paragraph_text,
                        "technical_score": max(0.0, paragraph_total[paragraph_name] - paragraph_citation_score),
                        "citation_score": max(0.0, paragraph_citation_score),
                    }
                )
                leaf_citation_total += paragraph_citation_score

                for citation, value in per_paragraph_citation_allocations[idx].items():
                    paragraph_citation_scores.append(
                        {
                            "section_path": list(record.path),
                            "paragraph_index": idx + 1,
                            "paragraph": paragraph_text,
                            "citation": citation,
                            "citation_score": value,
                        }
                    )
                    citation_scores.setdefault(citation, {"citation_score": 0.0})
                    citation_scores[citation]["citation_score"] += value

            record.citation_score = leaf_citation_total

        for record in iter_records(records):
            if record.is_leaf:
                process_leaf(record)

        def finalize_section_tree(record: SectionRecord) -> Dict[str, Any]:
            if record.children:
                subsections = {child.name: finalize_section_tree(child) for child in record.children}
                citation_score = sum(safe_float(subsections[name]["citation_score"], 0.0) for name in subsections)
                record.citation_score = citation_score
            else:
                subsections = {}
            return {
                "total_score": record.total_score,
                "citation_score": record.citation_score,
                "subsections": subsections,
            }

        section_scores = {record.name: finalize_section_tree(record) for record in records}
        return citation_scores, section_scores, paragraph_scores, paragraph_citation_scores


class LengthHeuristicBaseline(BaselineModel):
    """Length-only section weighting with paragraph-mediated citation density.

    Section scores are proportional to token count. Each leaf section is split
    into paragraphs weighted by token count + 50 * citation mentions; citations
    inherit scores through the paragraph layer using mention density.
    No LLM calls.
    """
    model_tag = "length_weighted_frequency"

    def raw_section_weights(
        self,
        records: Sequence[SectionRecord],
        parent_path: Optional[List[str]] = None,  # noqa: ARG002
    ) -> Dict[str, float]:
        return {record.name: float(max(1, record.token_count)) for record in records}

    def leaf_citation_fraction(self, record: SectionRecord) -> float:
        del record
        return 0.45

    def raw_citation_weight(self, record: SectionRecord, citation: str, mentions: int) -> float:
        del citation
        return float(mentions) / max(1.0, float(record.token_count))


class UniformBaseline(BaselineModel):
    """Uniform section and paragraph weights with raw mention-count citation scoring.

    Every section at each level receives equal score regardless of length.
    Within each leaf section, paragraphs are also weighted uniformly. Citations
    are scored by raw mention count. No LLM calls.
    """
    model_tag = "citation_frequency"

    def raw_section_weights(
        self,
        records: Sequence[SectionRecord],
        parent_path: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        del parent_path
        return {record.name: 1.0 for record in records}

    def raw_paragraph_weight(self, paragraph_text: str, mention_count: int) -> float:
        del paragraph_text, mention_count
        return 1.0

    def leaf_citation_fraction(self, record: SectionRecord) -> float:
        del record
        return 0.45

    def raw_citation_weight(self, record: SectionRecord, citation: str, mentions: int) -> float:
        del record, citation
        return float(mentions)



class TechnicalSectionPriorBaseline(BaselineModel):
    """Citation-density baseline with fixed section-title technicality priors."""
    model_tag = "technical_section_prior"

    def section_prior(self, title: str) -> float:
        name = re.sub(r"[^a-z0-9]+", "", title.lower())
        if any(token in name for token in ("method", "algorithm", "approach", "framework", "model", "proof", "theorem", "implementation")):
            return 1.0
        if any(token in name for token in ("experiment", "result", "evaluation", "analysis", "benchmark")):
            return 0.9
        if any(token in name for token in ("problem", "preliminar", "definition", "setup", "overview")):
            return 0.65
        if any(token in name for token in ("introduction", "relatedwork", "background", "survey", "conclusion", "discussion", "futurework", "limitation")):
            return 0.25
        return 0.5

    def raw_section_weights(
        self,
        records: Sequence[SectionRecord],
        parent_path: Optional[List[str]] = None,  # noqa: ARG002
    ) -> Dict[str, float]:
        return {
            record.name: float(max(1, record.token_count)) * self.section_prior(record.name)
            for record in records
        }

    def leaf_citation_fraction(self, record: SectionRecord) -> float:
        prior = self.section_prior(record.name)
        if prior >= 0.9:
            return 0.65
        if prior >= 0.6:
            return 0.45
        return 0.25

    def raw_citation_weight(self, record: SectionRecord, citation: str, mentions: int) -> float:
        return self.section_prior(record.name) * float(mentions) / max(1.0, float(record.token_count))


class SinglePassLLMSectionBaseline(BaselineModel):
    """One-pass LLM baseline for top-level section salience only."""
    model_tag = "single_pass_llm"
    uses_llm = True

    SYSTEM_PROMPT = (
        "You are an expert academic reviewer. "
        "Given the top-level sections of one paper, distribute 100 points across them "
        "based on how much each contributes to the paper's main scientific contribution. "
        "Higher scores should go to sections that carry the core scientific contribution "
        "through methods, theory, algorithms, proofs, experimental findings, analysis, "
        "or important problem formulation. Lower scores should usually go to sections "
        "that mainly provide background, motivation, transitions, setup details, or "
        "lower-impact narrative. Return plain text lines of the form 'Section Name: value'. "
        "Use non-negative values that sum to 100."
    )

    def raw_section_weights(
        self,
        records: Sequence[SectionRecord],
        parent_path: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        if parent_path:
            return {record.name: float(max(1, record.token_count)) for record in records}
        return self._query_top_level_scores(records)

    def _query_top_level_scores(self, records: Sequence[SectionRecord]) -> Dict[str, float]:
        assert self.client is not None
        expected_ids = [record.name for record in records]
        section_blocks = "\n\n".join(
            format_section_excerpt(record, limit=900) for record in records
        )
        user_prompt = (
            "Allocate importance across the following top-level sections.\n\n"
            f"{section_blocks}\n\n"
            "Guidelines:\n"
            "- Higher scores: segments that carry the core scientific contribution through\n"
            "  methods, theory, algorithms, proofs, experimental findings, analysis, or\n"
            "  important problem formulation.\n"
            "- Lower scores: segments that mainly provide background, motivation, transitions,\n"
            "  setup details, or lower-impact narrative.\n"
            "- If one segment contributes about 3x as much as another, assign about 3x the score.\n"
            "- If two segments contribute equally, assign equal scores.\n"
            "- Do not copy decimal values that appear inside the excerpts. Infer new percentages.\n\n"
            "Return one line per section exactly in the form:\n"
            "Section Name: percentage"
        )

        append_debug_log(self.debug_log_path, f"[single_pass_llm_prompt]\n{user_prompt}")
        last_error = "no_valid_response"
        for attempt in range(1, self.max_retries + 1):
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": self.temperature},
            )
            response_text = response["message"]["content"]
            append_debug_log(
                self.debug_log_path,
                f"[single_pass_llm_response] attempt={attempt}/{self.max_retries}\n{response_text}",
            )
            parsed = parse_score_map_from_response(
                response_text,
                expected_ids=expected_ids,
                allow_percentage=True,
            )
            if set(parsed) == set(expected_ids):
                return {key: max(0.0, safe_float(value, 0.0)) for key, value in parsed.items()}
            last_error = f"incomplete_parse_keys={sorted(parsed.keys())}"

        append_debug_log(self.debug_log_path, f"[single_pass_llm_fallback] reason={last_error}")
        return {record.name: float(max(1, record.token_count)) for record in records}

    def leaf_citation_fraction(self, record: SectionRecord) -> float:
        del record
        return 0.5

    def raw_citation_weight(self, record: SectionRecord, citation: str, mentions: int) -> float:
        del citation
        return float(mentions) / max(1.0, float(record.token_count))


class OpenAIFullPaperBaseline(BaselineModel):
    """One-request full-paper annotator via an OpenAI-compatible chat endpoint."""

    model_tag = "openai_full_paper"

    SYSTEM_PROMPT = (
        "You are a scientific contribution-scoring assistant.\n\n"
        "Score the provided hierarchy using the paper text.\n\n"
        "Use citation identifiers exactly as they appear in the paper.\n\n"
        "Return only the requested scores and format. Do not provide explanations, reasoning traces, commentary, or summaries unless explicitly requested."
    )

    def __init__(
        self,
        model: str = "",
        host: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_retries: int = 3,
        debug_log_path: str = "",
        pdf_path: str = "",
        api_key: str = "",
        api_key_env: str = "OPENAI_API_KEY",
        api_endpoint: str = "",
        request_timeout: int = 600,
        max_output_tokens: int = 12000,
        api_response_format: str = "none",
    ) -> None:
        if not model.strip():
            raise ValueError(
                "openai_full_paper requires an explicit model name; pass it via --model."
            )
        super().__init__(
            model=model,
            host=host,
            temperature=temperature,
            max_retries=max_retries,
            debug_log_path=debug_log_path,
        )
        self.pdf_path = pdf_path
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.api_endpoint = api_endpoint
        self.request_timeout = max(30, request_timeout)
        self.max_output_tokens = max(512, max_output_tokens)
        self.api_response_format = api_response_format

    def raw_section_weights(
        self,
        records: Sequence[SectionRecord],
        parent_path: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        del records, parent_path
        raise NotImplementedError("OpenAIFullPaperBaseline scores sections directly from the full-paper response.")

    def leaf_citation_fraction(self, record: SectionRecord) -> float:
        del record
        raise NotImplementedError("OpenAIFullPaperBaseline does not allocate citation mass via paragraph fractions.")

    def raw_citation_weight(self, record: SectionRecord, citation: str, mentions: int) -> float:
        del record, citation, mentions
        raise NotImplementedError("OpenAIFullPaperBaseline scores citations directly from the full-paper response.")

    def _resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        env_value = os.environ.get(self.api_key_env, "")
        if env_value:
            return env_value
        raise ValueError(
            f"Missing API key. Pass --api-key directly or set the {self.api_key_env} environment variable."
        )

    def _endpoint(self) -> str:
        if self.api_endpoint:
            return self.api_endpoint
        if self._host_looks_like_bedrock_mantle():
            base = self.host.rstrip("/")
            if base.endswith("/v1"):
                return base + "/chat/completions"
            return base + "/v1/chat/completions"
        return self.host.rstrip("/") + "/chat/completions"

    def _host_looks_like_bedrock(self) -> bool:
        for candidate in (self.api_endpoint, self.host):
            if candidate and "bedrock" in candidate.lower():
                return True
        return False

    def _host_looks_like_bedrock_mantle(self) -> bool:
        for candidate in (self.api_endpoint, self.host):
            if candidate and "bedrock-mantle" in candidate.lower():
                return True
        return False

    def _model_looks_like_bedrock(self) -> bool:
        model_name = self.model.strip().lower()
        return model_name.startswith(
            (
                "anthropic.",
                "us.anthropic.",
                "openai.",
                "us.openai.",
            )
        )

    def _uses_bedrock_converse_transport(self) -> bool:
        if self._host_looks_like_bedrock_mantle():
            return False
        return (
            self.api_key_env == "AWS_BEARER_TOKEN_BEDROCK"
            or self._host_looks_like_bedrock()
            or self._model_looks_like_bedrock()
        )

    def _resolved_aws_region(self) -> str:
        for env_name in ("AWS_REGION", "AWS_DEFAULT_REGION"):
            env_value = os.environ.get(env_name, "").strip()
            if env_value:
                return env_value

        for candidate in (self.api_endpoint, self.host):
            if not candidate:
                continue
            match = re.search(r"bedrock-(?:runtime|mantle)\.([a-z0-9-]+)\.", candidate)
            if not match:
                match = re.search(r"bedrock\.([a-z0-9-]+)\.", candidate)
            if match:
                return match.group(1)

        return "us-east-1"

    def _configure_bedrock_api_key(self) -> None:
        if self.api_key:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = self.api_key
            return
        if self.api_key_env and self.api_key_env != "AWS_BEARER_TOKEN_BEDROCK":
            env_value = os.environ.get(self.api_key_env, "").strip()
            if env_value:
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = env_value

    def _resolved_bedrock_model_id(self) -> str:
        return self.model.strip()

    def _call_bedrock_converse_api(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - depends on local optional install
            raise RuntimeError(
                "Bedrock Converse support requires the 'boto3' package. "
                "Install it with 'python3 -m pip install boto3'."
            ) from exc

        self._configure_bedrock_api_key()
        try:
            client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self._resolved_aws_region(),
            )
            response = client.converse(
                modelId=self._resolved_bedrock_model_id(),
                system=[{"text": system_prompt}],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": user_prompt}],
                    }
                ],
                inferenceConfig={
                    "maxTokens": self.max_output_tokens,
                    "temperature": self.temperature,
                },
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Bedrock Converse request failed: {exc}") from exc

        try:
            content = response["output"]["message"]["content"]
            if not isinstance(content, list):
                raise TypeError("content is not a list")
            text_blocks = [
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("text")
            ]
            joined = "\n".join(block for block in text_blocks if block.strip()).strip()
            if not joined:
                raise ValueError("response did not contain any text blocks")
            return joined
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Unexpected Bedrock Converse response shape: {response}") from exc

    def _call_full_paper_api(self, system_prompt: str, user_prompt: str) -> str:
        if self._uses_bedrock_converse_transport():
            return self._call_bedrock_converse_api(system_prompt, user_prompt)
        return self._call_openai_compatible_api(system_prompt, user_prompt)

    def _build_prompt(
        self,
        paper_id: str,
        full_paper_text: str,
        section_schema: Dict[str, Any],
        content_dict: Dict[str, Any],
        citation_inventory: Dict[str, Dict[str, Any]],
        paragraph_inventory: List[Dict[str, Any]],
    ) -> str:
        del paper_id, content_dict, citation_inventory
        hierarchy_text = render_authoritative_hierarchy(section_schema, paragraph_inventory)
        return (
            "Assign contribution scores to the provided hierarchy using the paper text.\n\n"
            "Use the provided hierarchy exactly as given.\n"
            "Do not add, remove, rename, merge, split, reorder, or invent nodes.\n"
            "Output every node exactly once, including nodes with score 0.\n"
            "Output nodes in the exact order they appear in the hierarchy.\n\n"
            "Scoring rules\n\n"
            "* The root paper node has total score 1.0.\n"
            "* For each parent node, distribute its score only among its immediate children.\n"
            "* Do not skip hierarchy levels or assign a parent's score directly to deeper descendants.\n"
            "* Child scores should approximately sum to the parent score. Exact normalization is handled after output.\n"
            "* Scores should be global scores relative to the entire paper, not local percentages.\n"
            "* Score scientific contribution, not length.\n"
            "* Assign higher scores to content central to the paper's technical contribution.\n"
            "* Assign lower scores to background, motivation, related work, summaries, and organizational text.\n\n"
            "Citation attribution\n\n"
            "* For a paragraph without citations, Citation Score = 0.\n"
            "* For a paragraph with citations, estimate what fraction of the paragraph's contribution depends on prior work. This is the paragraph Citation Score.\n"
            "* Allocate that citation-derived contribution across the cited references according to their importance in that paragraph.\n"
            "* If multiple citations play the same role, split them evenly.\n"
            "* Aggregate repeated appearances of the same citation into one final citation score.\n"
            "* Use citation identifiers exactly as they appear in the paper.\n\n"
            "Output Format\n\n"
            "For each document node output:\n\n"
            "Node Title: Total Score | Citation Score\n\n"
            "After all document nodes output:\n\n"
            "Citation Contributions\n\n"
            "Citation Identifier: Score\n\n"
            "Paper Text\n\n"
            f"{full_paper_text}\n\n"
            "Hierarchy To Score\n\n"
            f"{hierarchy_text}"
        )

    def _call_openai_compatible_api(self, system_prompt: str, user_prompt: str) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        if self.api_response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}

        encoded_payload = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint(),
            data=encoded_payload,
            headers={
                "Authorization": f"Bearer {self._resolved_api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        network_attempts = 5
        backoff_seconds = (2, 5, 10, 20)
        response_data: Dict[str, Any]
        for network_attempt in range(1, network_attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                    response_data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:  # pragma: no cover - exercised only against live endpoints
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"OpenAI-compatible request failed with HTTP {exc.code}: {body[:2000]}"
                ) from exc
            except urllib.error.URLError as exc:  # pragma: no cover - exercised only against live endpoints
                if network_attempt >= network_attempts or not is_retryable_openai_network_error(exc):
                    raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc
                delay = backoff_seconds[min(network_attempt - 1, len(backoff_seconds) - 1)]
                append_debug_log(
                    self.debug_log_path,
                    (
                        f"[{self.model_tag}_network_retry] attempt={network_attempt}/{network_attempts} "
                        f"delay={delay}s error={exc}"
                    ),
                )
                time.sleep(delay)

        try:
            return str(response_data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected API response shape: {json.dumps(response_data)[:2000]}") from exc

    def build_outputs(
        self,
        content_dict: Dict[str, Any],
        citations_dict: Dict[str, Any],
        paper_id: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not self.pdf_path:
            raise ValueError("OpenAIFullPaperBaseline requires a PDF path.")

        section_schema = section_schema_from_content(content_dict)
        citation_inventory = collect_citation_inventory(citations_dict)
        paragraph_inventory = build_leaf_paragraph_inventory(content_dict, citations_dict, paper_id=paper_id)
        full_paper_text = read_pdf_text(self.pdf_path)
        user_prompt = self._build_prompt(
            paper_id=paper_id,
            full_paper_text=full_paper_text,
            section_schema=section_schema,
            content_dict=content_dict,
            citation_inventory=citation_inventory,
            paragraph_inventory=paragraph_inventory,
        )
        append_debug_log(self.debug_log_path, f"[{self.model_tag}_prompt]\n{user_prompt}")

        last_error = "no_valid_response"
        for attempt in range(1, self.max_retries + 1):
            try:
                response_text = self._call_full_paper_api(self.SYSTEM_PROMPT, user_prompt)
            except Exception as api_exc:  # noqa: BLE001
                last_error = f"api_error={api_exc}"
                append_debug_log(
                    self.debug_log_path,
                    f"[{self.model_tag}_api_error] attempt={attempt}/{self.max_retries}\n{api_exc}",
                )
                continue
            append_debug_log(
                self.debug_log_path,
                f"[{self.model_tag}_response] attempt={attempt}/{self.max_retries}\n{response_text}",
            )
            try:
                payload = extract_json_payload(response_text)
                section_scores = normalize_section_scores_to_schema(
                    section_schema,
                    payload.get("section_scores", {}),
                    total_score=1.0,
                )
                enforce_section_score_conservation(section_scores, expected_total=1.0)
                paragraph_scores = normalize_openai_paragraph_scores(
                    paragraph_inventory=paragraph_inventory,
                    raw_paragraph_scores=payload.get("paragraph_scores", []),
                    normalized_section_scores=section_scores,
                    raw_paragraph_citation_scores=payload.get("paragraph_citation_scores", []),
                )
                paragraph_citation_scores = normalize_openai_paragraph_citation_scores(
                    paragraph_inventory=paragraph_inventory,
                    paragraph_scores=paragraph_scores,
                    raw_paragraph_citation_scores=payload.get("paragraph_citation_scores", []),
                )
                citation_scores = aggregate_citation_scores_from_paragraphs(paragraph_citation_scores)
                annotate_section_citation_scores(section_scores, paragraph_scores)
                return citation_scores, section_scores, paragraph_scores, paragraph_citation_scores
            except Exception as json_exc:  # noqa: BLE001
                try:
                    flat_node_scores, raw_citation_scores, root_citation_score = parse_contribution_tree_text_response(
                        response_text
                    )
                    raw_section_scores = build_raw_section_tree_from_flat_node_scores(section_schema, flat_node_scores)
                    section_scores = normalize_section_scores_to_schema(
                        section_schema,
                        raw_section_scores,
                        total_score=1.0,
                        fill_missing_uniform=False,
                    )
                    enforce_section_score_conservation(section_scores, expected_total=1.0)
                    apply_scaled_section_citation_scores(section_scores, raw_section_scores)

                    # Reconstruct the paper citation total bottom-up from child nodes instead of
                    # trusting the model's root Paper citation line, which can be inconsistent.
                    target_citation_total = total_section_citation_score(section_scores)
                    if target_citation_total <= 0.0:
                        target_citation_total = sum(
                            max(0.0, safe_float(item.get("citation_score"), 0.0))
                            for item in raw_citation_scores.values()
                        )

                    raw_paragraph_scores, raw_paragraph_citation_scores = extract_text_fallback_paragraph_payloads(
                        response_text,
                        section_schema=section_schema,
                        paragraph_inventory=paragraph_inventory,
                    )
                    if raw_paragraph_scores:
                        paragraph_scores = normalize_openai_paragraph_scores(
                            paragraph_inventory=paragraph_inventory,
                            raw_paragraph_scores=raw_paragraph_scores,
                            normalized_section_scores=section_scores,
                            raw_paragraph_citation_scores=raw_paragraph_citation_scores,
                        )
                        paragraph_citation_scores = normalize_openai_paragraph_citation_scores(
                            paragraph_inventory=paragraph_inventory,
                            paragraph_scores=paragraph_scores,
                            raw_paragraph_citation_scores=raw_paragraph_citation_scores,
                            global_citation_scores=raw_citation_scores,
                        )
                        citation_scores = aggregate_citation_scores_from_paragraphs(paragraph_citation_scores)
                        annotate_section_citation_scores(section_scores, paragraph_scores)
                        return citation_scores, section_scores, paragraph_scores, paragraph_citation_scores

                    citation_scores = normalize_citation_scores(
                        citation_inventory,
                        raw_citation_scores,
                        total_score=target_citation_total,
                        fallback_to_mentions=False,
                    )
                    return citation_scores, section_scores, [], []
                except Exception as text_exc:  # noqa: BLE001
                    last_error = f"json_error={json_exc}; text_error={text_exc}"

        raise RuntimeError(f"Failed to parse full-paper response after {self.max_retries} attempts: {last_error}")


class AnthropicFullPaperBaseline(OpenAIFullPaperBaseline):
    """One-request full-paper annotator via Anthropic's direct Messages API."""

    model_tag = "anthropic_full_paper"
    ANTHROPIC_VERSION = "2023-06-01"
    DIRECT_MODEL_ALIASES = {
        "anthropic.claude-sonnet-4-6": "claude-sonnet-4-6",
        "anthropic.claude-sonnet-4-6-20260217-v1": "claude-sonnet-4-6",
        "anthropic.claude-sonnet-4-6-20260217-v1:0": "claude-sonnet-4-6",
    }

    def __init__(
        self,
        model: str = "",
        host: str = "https://api.anthropic.com",
        temperature: float = 0.0,
        max_retries: int = 3,
        debug_log_path: str = "",
        pdf_path: str = "",
        api_key: str = "",
        api_key_env: str = "ANTHROPIC_API_KEY",
        api_endpoint: str = "",
        request_timeout: int = 600,
        max_output_tokens: int = 12000,
        api_response_format: str = "none",
    ) -> None:
        if not model.strip():
            raise ValueError(
                "anthropic_full_paper requires an explicit model name; pass it via --model."
            )
        super().__init__(
            model=model,
            host=host,
            temperature=temperature,
            max_retries=max_retries,
            debug_log_path=debug_log_path,
            pdf_path=pdf_path,
            api_key=api_key,
            api_key_env=api_key_env,
            api_endpoint=api_endpoint,
            request_timeout=request_timeout,
            max_output_tokens=max_output_tokens,
            api_response_format=api_response_format,
        )

    def _resolved_direct_model_name(self) -> str:
        return self.DIRECT_MODEL_ALIASES.get(self.model.strip(), self.model.strip())

    def _endpoint(self) -> str:
        if self.api_endpoint:
            return self.api_endpoint
        return self.host.rstrip("/") + "/v1/messages"

    def _call_messages_api_transport(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - depends on local optional install
            raise RuntimeError(
                "Anthropic full-paper baseline requires the 'anthropic' package. "
                "Install it with 'python3 -m pip install anthropic'."
            ) from exc

        try:
            client = Anthropic(
                api_key=self._resolved_api_key(),
                timeout=self.request_timeout,
                max_retries=max(1, self.max_retries),
            )
            response = client.messages.create(
                model=self._resolved_direct_model_name(),
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Anthropic direct Messages request failed: {exc}") from exc

        text_blocks = []
        for block in getattr(response, "content", []):
            if getattr(block, "type", "") == "text":
                text = getattr(block, "text", "")
                if text and str(text).strip():
                    text_blocks.append(str(text))
        joined = "\n".join(text_blocks).strip()
        if not joined:
            raise RuntimeError(f"Unexpected Anthropic response shape: {response}")
        return joined

    def _call_full_paper_api(self, system_prompt: str, user_prompt: str) -> str:
        return self._call_messages_api_transport(system_prompt, user_prompt)


SINGLE_SHOT_CITATION_LINE_RE = re.compile(
    r"^(?P<citation>.+?)\s*(?::|=|=>|->|\s-\s)\s*(?P<score>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


class SingleShotCitationAPIBaseline(AnthropicFullPaperBaseline):
    """One full-paper API pass that outputs citation scores only."""

    model_tag = "single_shot_citation_api"
    citation_only_output = True

    SYSTEM_PROMPT = (
        "You are a scientific contribution-scoring assistant.\n\n"
        "Your task is to assign contribution scores directly to the cited references in a paper.\n\n"
        "A citation's contribution score should reflect how much that cited work supports, grounds, shapes, "
        "or is built upon by the paper's core scientific contribution.\n\n"
        "Use higher scores for cited works that provide a method the paper builds on, provide a baseline "
        "central to the evaluation, supply a theory or formulation the paper depends on, or strongly shape "
        "the paper's technical design or claims.\n\n"
        "Use lower scores for cited works that are broad background, generic motivation, mentioned only for "
        "completeness, peripheral comparisons, or cited only briefly in passing.\n\n"
        "Use citation identifiers exactly as they appear in the paper.\n\n"
        "Return only the requested scores and format. Do not provide explanations, reasoning traces, commentary, or summaries."
    )

    def __init__(
        self,
        model: str = "",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_retries: int = 3,
        debug_log_path: str = "",
        pdf_path: str = "",
        api_key: str = "",
        api_key_env: str = "OPENAI_API_KEY",
        api_endpoint: str = "",
        request_timeout: int = 600,
        max_output_tokens: int = 12000,
        api_response_format: str = "none",
        api_provider: str = "",
    ) -> None:
        if not model.strip():
            raise ValueError(
                "single_shot_citation_api requires an explicit model name; pass it via --model."
            )
        provider = api_provider.strip().lower()
        if provider not in {"openai", "anthropic"}:
            raise ValueError(
                "single_shot_citation_api requires --api-provider to be either 'openai' or 'anthropic'."
            )

        resolved_host = host
        resolved_api_key_env = api_key_env
        if provider == "openai":
            if host == "http://localhost:11434":
                resolved_host = "https://api.openai.com/v1"
            if not resolved_api_key_env.strip():
                resolved_api_key_env = "OPENAI_API_KEY"
        else:
            if host == "http://localhost:11434":
                resolved_host = "https://bedrock-runtime.us-east-1.amazonaws.com"
            if not resolved_api_key_env.strip() or resolved_api_key_env == "OPENAI_API_KEY":
                resolved_api_key_env = "AWS_BEARER_TOKEN_BEDROCK"

        OpenAIFullPaperBaseline.__init__(
            self,
            model=model,
            host=resolved_host,
            temperature=temperature,
            max_retries=max_retries,
            debug_log_path=debug_log_path,
            pdf_path=pdf_path,
            api_key=api_key,
            api_key_env=resolved_api_key_env,
            api_endpoint=api_endpoint,
            request_timeout=request_timeout,
            max_output_tokens=max_output_tokens,
            api_response_format=api_response_format,
        )
        self.api_provider = provider
        self.model_tag = f"single_shot_citation_{provider}"

    def raw_section_weights(
        self,
        records: Sequence[SectionRecord],
        parent_path: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        del records, parent_path
        raise NotImplementedError("SingleShotCitationAPIBaseline outputs citation scores only.")

    def leaf_citation_fraction(self, record: SectionRecord) -> float:
        del record
        raise NotImplementedError("SingleShotCitationAPIBaseline outputs citation scores only.")

    def raw_citation_weight(self, record: SectionRecord, citation: str, mentions: int) -> float:
        del record, citation, mentions
        raise NotImplementedError("SingleShotCitationAPIBaseline outputs citation scores only.")

    def _endpoint(self) -> str:
        if self.api_endpoint:
            return self.api_endpoint
        if self.api_provider == "openai":
            return self.host.rstrip("/") + "/chat/completions"
        return self.host.rstrip("/") + "/anthropic/v1/messages"

    def _build_prompt(
        self,
        paper_id: str,
        full_paper_text: str,
        citation_inventory: Dict[str, Dict[str, Any]],
    ) -> str:
        del paper_id, citation_inventory
        return (
            "Assign direct citation contribution scores for the following paper.\n\n"
            "Goal:\n"
            "Score the cited references by how foundational they are to the paper's actual scientific contribution.\n\n"
            "Scoring rules:\n"
            "- Score all cited references that appear in the paper.\n"
            "- Scores must be non-negative.\n"
            "- Scores must sum to 1.0 across all cited references.\n"
            "- Use relative scoring: if one citation is about twice as foundational as another, its score should be about twice as large.\n"
            "- Do not output any citation that does not appear in the paper.\n"
            "- Use citation identifiers exactly as they appear in the paper.\n\n"
            "Output format:\n"
            "Return one JSON object with this exact shape:\n\n"
            "{\n"
            '  "citation_scores": {\n'
            '    "<citation identifier>": <score>,\n'
            '    "<citation identifier>": <score>\n'
            "  }\n"
            "}\n\n"
            "Paper text:\n"
            f"{full_paper_text}"
        )

    def _extract_json_citation_scores(self, response_text: str) -> Any:
        payload = extract_json_payload(response_text)
        if "citation_scores" in payload:
            return payload["citation_scores"]
        if "citations" in payload:
            return payload["citations"]
        if "scores" in payload:
            return payload["scores"]
        if all(
            isinstance(value, (int, float, str, dict))
            for value in payload.values()
        ):
            return payload
        raise ValueError("JSON payload did not contain a citation_scores mapping.")

    def _extract_text_citation_scores(self, response_text: str) -> Dict[str, Dict[str, float]]:
        parsed: Dict[str, Dict[str, float]] = {}
        for raw_line in str(response_text or "").splitlines():
            line = _clean_scored_line(raw_line)
            line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if not line:
                continue

            normalized_line = normalize_for_match(line).lower()
            if normalized_line in {
                normalize_for_match("citation contributions").lower(),
                normalize_for_match("citation scores").lower(),
                normalize_for_match("citation identifier").lower(),
                normalize_for_match("output format").lower(),
            }:
                continue

            match = CITATION_SCORE_TABLE_ROW_RE.match(line)
            if match and normalize_for_match(match.group("citation")).lower() == normalize_for_match("Citation Identifier").lower():
                continue
            if match and re.fullmatch(r"-+", match.group("citation").strip()):
                continue
            if not match:
                match = SINGLE_SHOT_CITATION_LINE_RE.match(line)
            if not match:
                match = CITATION_SCORE_LINE_RE.match(line)
            if not match:
                continue

            citation = str(match.group("citation")).strip()
            if not citation:
                continue
            parsed.setdefault(citation, {"citation_score": 0.0})
            parsed[citation]["citation_score"] += max(0.0, safe_float(match.group("score"), 0.0))

        if not parsed:
            raise ValueError("Response did not contain parseable citation score lines.")
        return parsed

    def _parse_citation_scores(
        self,
        response_text: str,
        citation_inventory: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:
        raw_scores: Any
        try:
            raw_scores = self._extract_json_citation_scores(response_text)
        except Exception:
            raw_scores = self._extract_text_citation_scores(response_text)

        normalized_scores = normalize_citation_scores(
            citation_inventory,
            raw_scores,
            total_score=1.0,
            fallback_to_mentions=False,
        )
        if sum(value["citation_score"] for value in normalized_scores.values()) <= 0.0:
            raise ValueError("Parsed citation scores were empty after normalization.")
        return normalized_scores

    def _call_full_paper_api(self, system_prompt: str, user_prompt: str) -> str:
        if self._uses_bedrock_converse_transport():
            return self._call_bedrock_converse_api(system_prompt, user_prompt)
        if self.api_provider == "openai":
            return self._call_openai_compatible_api(system_prompt, user_prompt)
        if self._uses_messages_api_transport():
            return self._call_messages_api_transport(system_prompt, user_prompt)
        return self._call_legacy_bedrock_transport(system_prompt, user_prompt)

    def build_outputs(
        self,
        content_dict: Dict[str, Any],
        citations_dict: Dict[str, Any],
        paper_id: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not self.pdf_path:
            raise ValueError("SingleShotCitationAPIBaseline requires a PDF path.")

        citation_inventory = collect_citation_inventory(citations_dict)
        paragraph_inventory = build_leaf_paragraph_inventory(content_dict, citations_dict, paper_id=paper_id)
        if not citation_inventory:
            citation_inventory = collect_citation_inventory_from_paragraph_inventory(paragraph_inventory)
        if not citation_inventory:
            raise ValueError(f"No citations were extracted for paper '{paper_id}'.")

        full_paper_text = read_pdf_text(self.pdf_path)
        user_prompt = self._build_prompt(
            paper_id=paper_id,
            full_paper_text=full_paper_text,
            citation_inventory=citation_inventory,
        )
        append_debug_log(self.debug_log_path, f"[{self.model_tag}_prompt]\n{user_prompt}")

        last_error = "no_valid_response"
        for attempt in range(1, self.max_retries + 1):
            try:
                response_text = self._call_full_paper_api(self.SYSTEM_PROMPT, user_prompt)
            except Exception as api_exc:  # noqa: BLE001
                last_error = f"api_error={api_exc}"
                append_debug_log(
                    self.debug_log_path,
                    f"[{self.model_tag}_api_error] attempt={attempt}/{self.max_retries}\n{api_exc}",
                )
                continue

            append_debug_log(
                self.debug_log_path,
                f"[{self.model_tag}_response] attempt={attempt}/{self.max_retries}\n{response_text}",
            )
            try:
                citation_scores = self._parse_citation_scores(response_text, citation_inventory)
                return citation_scores, {}, [], []
            except Exception as parse_exc:  # noqa: BLE001
                last_error = f"parse_error={parse_exc}"
                append_debug_log(
                    self.debug_log_path,
                    f"[{self.model_tag}_parse_error] attempt={attempt}/{self.max_retries}\n{parse_exc}",
                )

        raise RuntimeError(
            f"Failed to parse single-shot citation response after {self.max_retries} attempts: {last_error}"
        )


def build_baseline_model(
    baseline_name: str,
    model: str = "",
    host: str = "http://localhost:11434",
    temperature: float = 0.0,
    max_retries: int = 3,
    debug_log_path: str = "",
    pdf_path: str = "",
    api_key: str = "",
    api_key_env: str = "OPENAI_API_KEY",
    api_endpoint: str = "",
    request_timeout: int = 600,
    max_output_tokens: int = 12000,
    api_response_format: str = "none",
    api_provider: str = "",
) -> BaselineModel:
    baseline_name = baseline_name.strip().lower()
    resolved_model = model
    resolved_host = host
    resolved_api_key_env = api_key_env
    if (
        baseline_name == "anthropic_full_paper"
        or (baseline_name == "single_shot_citation_api" and api_provider.strip().lower() == "anthropic")
    ) and api_key_env == "OPENAI_API_KEY":
        resolved_api_key_env = "ANTHROPIC_API_KEY" if baseline_name == "anthropic_full_paper" else "AWS_BEARER_TOKEN_BEDROCK"
    if baseline_name == "anthropic_full_paper":
        if host == "http://localhost:11434":
            resolved_host = "https://api.anthropic.com"
    if baseline_name in {"single_pass_llm", "openai_full_paper", "anthropic_full_paper", "single_shot_citation_api"} and not resolved_model.strip():
        raise ValueError(
            f"{baseline_name} requires an explicit model name; pass it via --model."
        )
    if baseline_name == "citation_frequency":
        return UniformBaseline(
            model=resolved_model,
            host=resolved_host,
            temperature=temperature,
            max_retries=max_retries,
            debug_log_path=debug_log_path,
        )
    if baseline_name == "length_weighted_frequency":
        return LengthHeuristicBaseline(
            model=resolved_model,
            host=resolved_host,
            temperature=temperature,
            max_retries=max_retries,
            debug_log_path=debug_log_path,
        )
    if baseline_name == "technical_section_prior":
        return TechnicalSectionPriorBaseline(
            model=resolved_model,
            host=resolved_host,
            temperature=temperature,
            max_retries=max_retries,
            debug_log_path=debug_log_path,
        )
    if baseline_name == "single_pass_llm":
        return SinglePassLLMSectionBaseline(
            model=resolved_model,
            host=resolved_host,
            temperature=temperature,
            max_retries=max_retries,
            debug_log_path=debug_log_path,
        )
    if baseline_name == "openai_full_paper":
        return OpenAIFullPaperBaseline(
            model=resolved_model,
            host=resolved_host,
            temperature=temperature,
            max_retries=max_retries,
            debug_log_path=debug_log_path,
            pdf_path=pdf_path,
            api_key=api_key,
            api_key_env=resolved_api_key_env,
            api_endpoint=api_endpoint,
            request_timeout=request_timeout,
            max_output_tokens=max_output_tokens,
            api_response_format=api_response_format,
        )
    if baseline_name == "anthropic_full_paper":
        return AnthropicFullPaperBaseline(
            model=resolved_model,
            host=resolved_host,
            temperature=temperature,
            max_retries=max_retries,
            debug_log_path=debug_log_path,
            pdf_path=pdf_path,
            api_key=api_key,
            api_key_env=resolved_api_key_env,
            api_endpoint=api_endpoint,
            request_timeout=request_timeout,
            max_output_tokens=max_output_tokens,
            api_response_format=api_response_format,
        )
    if baseline_name == "single_shot_citation_api":
        return SingleShotCitationAPIBaseline(
            model=resolved_model,
            host=resolved_host,
            temperature=temperature,
            max_retries=max_retries,
            debug_log_path=debug_log_path,
            pdf_path=pdf_path,
            api_key=api_key,
            api_key_env=resolved_api_key_env,
            api_endpoint=api_endpoint,
            request_timeout=request_timeout,
            max_output_tokens=max_output_tokens,
            api_response_format=api_response_format,
            api_provider=api_provider,
        )
    raise ValueError(f"Unknown baseline '{baseline_name}'.")


def run_baseline_from_args(args: argparse.Namespace) -> Tuple[str, str, str, str]:
    text = read_pdf_text(args.pdf)
    sections = load_sections_from_file(args.sections_file, args.sections_var)
    citations, content = extract_citations_by_section(text, sections)

    model_tag = args.model_tag.strip() if args.model_tag else build_baseline_model(
        args.baseline,
        model=args.model,
        host=args.host,
        temperature=args.temperature,
        max_retries=args.max_retries,
        debug_log_path=args.debug_log,
        pdf_path=args.pdf,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        api_endpoint=args.api_endpoint,
        request_timeout=args.request_timeout,
        max_output_tokens=args.max_output_tokens,
        api_response_format=args.api_response_format,
        api_provider=args.api_provider,
    ).model_tag

    debug_log_path = args.debug_log
    if debug_log_path:
        debug_path = Path(debug_log_path)
        debug_log_path = str(debug_path.with_name(f"{debug_path.stem}_{model_tag}{debug_path.suffix}"))

    baseline = build_baseline_model(
        args.baseline,
        model=args.model,
        host=args.host,
        temperature=args.temperature,
        max_retries=args.max_retries,
        debug_log_path=debug_log_path,
        pdf_path=args.pdf,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        api_endpoint=args.api_endpoint,
        request_timeout=args.request_timeout,
        max_output_tokens=args.max_output_tokens,
        api_response_format=args.api_response_format,
        api_provider=args.api_provider,
    )
    citation_scores, section_scores, paragraph_scores, paragraph_citation_scores = baseline.build_outputs(
        content_dict=content,
        citations_dict=citations,
        paper_id=args.paper_id,
    )

    output1_prefix = f"{args.output1}_{model_tag}"
    output2_prefix = f"{args.output2}_{model_tag}"
    paragraph_prefix_base = args.output3 if args.output3 else args.output2
    paragraph_prefix = f"{paragraph_prefix_base}_{model_tag}"

    citation_path = f"{output1_prefix}_citation_scores.json"
    section_path = f"{output2_prefix}_section_scores.json"
    paragraph_path = f"{paragraph_prefix}_paragraph_scores.json"
    paragraph_citation_path = f"{paragraph_prefix}_paragraph_citation_scores.json"

    output_paths = [citation_path]
    if not baseline.citation_only_output:
        output_paths.extend([section_path, paragraph_path, paragraph_citation_path])

    for path in output_paths:
        ensure_parent(path)

    with open(citation_path, "w", encoding="utf-8") as f:
        json.dump(citation_scores, f, indent=2)
    if not baseline.citation_only_output:
        with open(section_path, "w", encoding="utf-8") as f:
            json.dump(section_scores, f, indent=2)
        with open(paragraph_path, "w", encoding="utf-8") as f:
            json.dump(paragraph_scores, f, indent=2)
        with open(paragraph_citation_path, "w", encoding="utf-8") as f:
            json.dump(paragraph_citation_scores, f, indent=2)
        return citation_path, section_path, paragraph_path, paragraph_citation_path

    return citation_path, "", "", ""
