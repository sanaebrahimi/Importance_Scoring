from __future__ import annotations

"""Baseline models for section and citation importance scoring.

The goal of this module is to provide lightweight alternatives to the full
hierarchical LLM pipeline while keeping the output format compatible with the
existing evaluation and graph tooling.
"""

import argparse
import json
import re
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
    canonicalize_citation_key,
    classify_citation_block,
    detect_dominant_citation_style,
    extract_citations_by_section,
    flatten_content_to_text,
    load_sections_from_file,
    normalize_distribution,
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


def format_section_excerpt(record: SectionRecord, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", record.text).strip()
    if limit > 0:
        text = text[:limit]
    return f"Section name: {record.name}\nExcerpt:\n{text}"


class BaselineModel(ABC):
    """Shared interface for deterministic and single-pass baseline models."""
    model_tag = "baseline"
    uses_llm = False

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


def build_baseline_model(
    baseline_name: str,
    model: str = "llama3.2",
    host: str = "http://localhost:11434",
    temperature: float = 0.0,
    max_retries: int = 3,
    debug_log_path: str = "",
) -> BaselineModel:
    baseline_name = baseline_name.strip().lower()
    if baseline_name == "citation_frequency":
        return UniformBaseline(
            model=model,
            host=host,
            temperature=temperature,
            max_retries=max_retries,
            debug_log_path=debug_log_path,
        )
    if baseline_name == "length_weighted_frequency":
        return LengthHeuristicBaseline(
            model=model,
            host=host,
            temperature=temperature,
            max_retries=max_retries,
            debug_log_path=debug_log_path,
        )
    if baseline_name == "technical_section_prior":
        return TechnicalSectionPriorBaseline(
            model=model,
            host=host,
            temperature=temperature,
            max_retries=max_retries,
            debug_log_path=debug_log_path,
        )
    if baseline_name == "single_pass_llm":
        return SinglePassLLMSectionBaseline(
            model=model,
            host=host,
            temperature=temperature,
            max_retries=max_retries,
            debug_log_path=debug_log_path,
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

    for path in (citation_path, section_path, paragraph_path, paragraph_citation_path):
        ensure_parent(path)

    with open(citation_path, "w", encoding="utf-8") as f:
        json.dump(citation_scores, f, indent=2)
    with open(section_path, "w", encoding="utf-8") as f:
        json.dump(section_scores, f, indent=2)
    with open(paragraph_path, "w", encoding="utf-8") as f:
        json.dump(paragraph_scores, f, indent=2)
    with open(paragraph_citation_path, "w", encoding="utf-8") as f:
        json.dump(paragraph_citation_scores, f, indent=2)

    return citation_path, section_path, paragraph_path, paragraph_citation_path
