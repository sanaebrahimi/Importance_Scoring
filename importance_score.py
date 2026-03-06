import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import PyPDF2
from ollama import Client


DEFAULT_PDF_PATH = "adv_res_paper.pdf"
DEFAULT_PAPER_ID = "target_paper"

DEFAULT_SECTIONS = {
    "Introduction": None,
    "System Overview": None,
    "Team of Agents": None,
    "CrS-Aware Aggregation": None,
    "Learning Credibility Scores On-The-Fly": None,
    "Experiment Results": {
        "Experiments Setting": None,
        "Collaboration Setup": None,
        "Insights from Experimental Observations": {
            "Credibility Scores Drive Consistent Gains": None,
            "Reasoning vs Multi-Choice Tasks": None,
            "Model Capacity Matters But Only With Coordination": None,
            "Judge-Computed CrS Imitates the Shapley Value": None,
            "Judge Alters the Outcome": None,
            "Topology and Link Density": None,
            "Adversary Proportion": None,
        },
    },
    "Conclusion": None,
    "Limitations": None,
}

SECTION_PAIRWISE_SYSTEM_PROMPT = (
    "You are an expert academic reviewer. "
    "Compare two items from the same paper by contribution to the paper's main scientific contribution. "
    "Distribute a total credit of 1.0 between item A and item B. "
    "Use both technical contribution and citation-supported contribution. "
    "Return JSON only."
)

SECTION_PAIRWISE_USER_PROMPT_TEMPLATE = """Parent node: "{parent_name}"
Task: distribute a credit of 1.0 between A and B based on contribution to the paper's main contribution.

Scoring rubric:
- Higher credit: core technical contribution (method/theory/algorithm/findings) and meaningful use of prior work.
- Lower credit: background context, transitions, setup details, or low-impact narrative.
- If one item contributes 3x the other, assign 3x credit.
- If both contribute equally, assign equal credit.
- Do not copy placeholder values; infer from the provided excerpts.

Item A name: "{item_a}"
Item A excerpt:
{excerpt_a}

Item B name: "{item_b}"
Item B excerpt:
{excerpt_b}

Return ONLY JSON:
{{
  "a_credit": <float>,
  "b_credit": <float>
}}

Constraints:
- 0.0 <= a_credit <= 1.0
- 0.0 <= b_credit <= 1.0
- a_credit + b_credit = 1.0
"""

SECTION_DIRECT_SYSTEM_PROMPT = (
    "You are an expert academic reviewer. "
    "Distribute a parent score among child items from the same paper by contribution. "
    "Use both technical contribution and citation-supported contribution. "
    "Return JSON only."
)

SECTION_DIRECT_USER_PROMPT_TEMPLATE = """Parent node: "{parent_name}"
Parent score to distribute: {parent_score}
Task: divide the parent score among all child items together (not pairwise).

Scoring rubric:
- Higher score: core technical contribution (method/theory/algorithm/findings) and meaningful citation-supported value.
- Lower score: background context, transitions, setup detail, or low-impact narrative.
- Use non-negative scores; larger means more important.
- The children scores must sum to exactly the parent score.

Items (item_id -> {{"name": ..., "excerpt": ...}}):
{items_json}

Return ONLY JSON mapping every item_id to a non-negative score.
Do NOT use item names as keys.
Example:
{{
  "I1": 0.32,
  "I2": 0.18
}}
"""

CITATION_SPLIT_SYSTEM_PROMPT = (
    "You are an expert academic reviewer. "
    "Distribute a paragraph citation score among citations appearing in that paragraph. "
    "Return JSON only."
)

CITATION_SPLIT_USER_PROMPT_TEMPLATE = """Paragraph id: "{paragraph_id}"
Paragraph citation score to distribute: {paragraph_citation_score}

Paragraph text:
{paragraph_text}

Task:
- Divide the paragraph citation score among the citations below.
- Higher share: citation contributes more to the paragraph's claims, evidence, grounding, or comparison.
- Lower share: citation is peripheral or weakly connected.
- Use non-negative scores and make them sum to exactly the paragraph citation score.

Citation entries (citation_id -> {{"citation": ..., "context": ...}}):
{citations_json}

Return ONLY JSON mapping every citation_id to a non-negative score.
Do NOT use citation strings as keys.
Example:
{{
  "C1": 0.06,
  "C2": 0.02
}}
"""

PARAGRAPH_TECHNICAL_SYSTEM_PROMPT = (
    "You are an expert academic reviewer. "
    "Score each paragraph only for technical contribution. "
    "Ignore citation popularity or influence. "
    "Return JSON only."
)

PARAGRAPH_TECHNICAL_USER_PROMPT_TEMPLATE = """Task: assign a technical contribution score T(p) in [0, 1] for each paragraph.

Scoring rubric:
- Higher T(p): introduces a novel method, key algorithm, theorem, model, or core experimental insight.
- Lower T(p): background context, transitions, setup details, or non-technical narrative.
- Evaluate novelty, methodological contribution, and technical specificity.
- Ignore citation influence, citation counts, and venue prestige.

Paragraphs (id -> text snippet):
{paragraphs_json}

Return ONLY JSON mapping every paragraph id to a float in [0, 1].
Example:
{{
  "p1": 0.82,
  "p2": 0.27
}}
"""

PARAGRAPH_CITATION_SYSTEM_PROMPT = (
    "You are an expert academic reviewer. "
    "Score each paragraph only for citation-added contribution. "
    "Measure value added by cited prior work used in the paragraph. "
    "Do not score intrinsic technical novelty in this channel. "
    "Return JSON only."
)

PARAGRAPH_CITATION_USER_PROMPT_TEMPLATE = """Task: assign a citation-added contribution score C(p) in [0, 1] for each paragraph.

Scoring rubric:
- Higher C(p): cited prior work is used substantively (comparison, evidence, grounding, or dependency).
- Lower C(p): no citations, perfunctory citation mentions, or weak linkage to the claim.
- Evaluate citation relevance and how much cited work strengthens this paragraph's contribution.
- Ignore global citation popularity and venue prestige.

Paragraphs (id -> text snippet):
{paragraphs_json}

Return ONLY JSON mapping every paragraph id to a float in [0, 1].
Example:
{{
  "p1": 0.62,
  "p2": 0.05
}}
"""

PROMPT_CATALOG = {
    "section_pairwise_system_prompt": SECTION_PAIRWISE_SYSTEM_PROMPT,
    "section_pairwise_user_prompt_template": SECTION_PAIRWISE_USER_PROMPT_TEMPLATE,
    "section_direct_system_prompt": SECTION_DIRECT_SYSTEM_PROMPT,
    "section_direct_user_prompt_template": SECTION_DIRECT_USER_PROMPT_TEMPLATE,
    "citation_split_system_prompt": CITATION_SPLIT_SYSTEM_PROMPT,
    "citation_split_user_prompt_template": CITATION_SPLIT_USER_PROMPT_TEMPLATE,
    "paragraph_technical_system_prompt": PARAGRAPH_TECHNICAL_SYSTEM_PROMPT,
    "paragraph_technical_user_prompt_template": PARAGRAPH_TECHNICAL_USER_PROMPT_TEMPLATE,
    "paragraph_citation_system_prompt": PARAGRAPH_CITATION_SYSTEM_PROMPT,
    "paragraph_citation_user_prompt_template": PARAGRAPH_CITATION_USER_PROMPT_TEMPLATE,
}


def read_pdf_text(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def extract_citations_by_section(text: str, sections: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract citation blocks with context and section content based on a nested section schema.
    """
    citation_pattern = r"\([A-Z][^)]*\d{4}[a-z]?\)"

    def process_section_text(section_text: str) -> Dict[str, List[str]]:
        citations_dict: Dict[str, List[str]] = {}
        section_text = section_text.replace("\n", "")
        section_text = re.sub(r"\s+", " ", section_text)

        for match in re.finditer(citation_pattern, section_text):
            citation_block = match.group(0)
            citation_start = match.start()
            citation_end = match.end()

            text_before = section_text[:citation_start]
            paragraph_start = text_before.rfind(".")
            paragraph_start = paragraph_start + 1 if paragraph_start != -1 else 0

            text_after = section_text[citation_end:]
            next_period = text_after.find(".")
            if next_period != -1:
                sentence = section_text[paragraph_start : citation_end + next_period + 1].strip()
            else:
                sentence = section_text[paragraph_start:].strip()

            citations_dict.setdefault(citation_block, []).append(sentence)

        return citations_dict

    def extract_section_content(full_text: str, section_name: str, next_section_name: str = None) -> str:
        section_start = full_text.find(section_name)
        if section_start == -1:
            return ""

        if next_section_name:
            section_end = full_text.find(next_section_name, section_start + len(section_name))
            if section_end == -1:
                section_end = len(full_text)
        else:
            section_end = len(full_text)

        return full_text[section_start + len(section_name) : section_end]

    def process_sections(
        section_text: str, sections_dict: Dict[str, Any], section_names_list: List[str]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        local_citations: Dict[str, Any] = {}
        local_content: Dict[str, Any] = {}

        for i, section_name in enumerate(section_names_list):
            next_section = section_names_list[i + 1] if i + 1 < len(section_names_list) else None
            content = extract_section_content(section_text, section_name, next_section)
            if not content:
                continue

            subsections = sections_dict[section_name]
            if isinstance(subsections, dict) and subsections:
                subsection_names = list(subsections.keys())
                nested_citations, nested_content = process_sections(content, subsections, subsection_names)
                local_citations[section_name] = nested_citations
                local_content[section_name] = nested_content
            else:
                local_citations[section_name] = process_section_text(content)
                # Keep line breaks so we can score paragraph-level importance later.
                normalized_content = re.sub(r"[ \t]+", " ", content).strip()
                local_content[section_name] = normalized_content

        return local_citations, local_content

    section_names = list(sections.keys())
    citations, content = process_sections(text, sections, section_names)
    return citations, content


def parse_json_response(response_text: str) -> Dict[str, Any]:
    response_text = response_text.strip()
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
        response_text = response_text.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(response_text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def strip_code_fences(response_text: str) -> str:
    text = (response_text or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        text = text.replace("```json", "").replace("```", "").strip()
    return text


def parse_json_loose(response_text: str) -> Any:
    text = strip_code_fences(response_text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (key or "").lower())


def coerce_non_negative_number(value: Any, allow_percentage: bool = False) -> Optional[float]:
    number: Optional[float] = None

    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if match:
            number = safe_float(match.group(0), None)

    if number is None:
        return None
    if allow_percentage and number > 1.0 and number <= 100.0:
        number = number / 100.0
    if number < 0.0:
        return None
    return float(number)


def parse_score_map_from_response(
    response_text: str,
    expected_ids: List[str],
    allow_percentage: bool = False,
    alias_to_id: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    if not expected_ids:
        return {}

    expected_set = set(expected_ids)
    norm_to_id = {normalized_key(item_id): item_id for item_id in expected_ids}
    if alias_to_id:
        for alias, target_id in alias_to_id.items():
            if target_id in expected_set:
                norm_to_id[normalized_key(alias)] = target_id
    parsed_scores: Dict[str, float] = {}

    def assign_from_pair(key: Any, value: Any) -> None:
        key_str = str(key)
        target_id = key_str if key_str in expected_set else norm_to_id.get(normalized_key(key_str))
        if target_id is None:
            return

        score_val = coerce_non_negative_number(value, allow_percentage=allow_percentage)
        if score_val is None and isinstance(value, dict):
            for field in ("score", "value", "weight", "allocation", "credit"):
                if field in value:
                    score_val = coerce_non_negative_number(value.get(field), allow_percentage=allow_percentage)
                    if score_val is not None:
                        break

        if score_val is not None:
            parsed_scores[target_id] = score_val

    json_payload = parse_json_loose(response_text)
    if isinstance(json_payload, dict):
        for key, value in json_payload.items():
            assign_from_pair(key, value)

        for bucket_key in ("scores", "items", "allocations", "distribution", "results"):
            bucket = json_payload.get(bucket_key)
            if not isinstance(bucket, list):
                continue
            for entry in bucket:
                if not isinstance(entry, dict):
                    continue
                key_candidate = (
                    entry.get("id")
                    or entry.get("item_id")
                    or entry.get("name")
                    or entry.get("item")
                    or entry.get("key")
                    or entry.get("label")
                )
                if key_candidate is None:
                    continue
                value_candidate = (
                    entry.get("score")
                    if "score" in entry
                    else entry.get("value")
                    if "value" in entry
                    else entry.get("weight")
                    if "weight" in entry
                    else entry.get("allocation")
                    if "allocation" in entry
                    else entry.get("credit")
                )
                assign_from_pair(key_candidate, value_candidate)

    elif isinstance(json_payload, list):
        for entry in json_payload:
            if not isinstance(entry, dict):
                continue
            key_candidate = (
                entry.get("id")
                or entry.get("item_id")
                or entry.get("name")
                or entry.get("item")
                or entry.get("key")
                or entry.get("label")
            )
            if key_candidate is None:
                continue
            value_candidate = (
                entry.get("score")
                if "score" in entry
                else entry.get("value")
                if "value" in entry
                else entry.get("weight")
                if "weight" in entry
                else entry.get("allocation")
                if "allocation" in entry
                else entry.get("credit")
            )
            assign_from_pair(key_candidate, value_candidate)

    text = strip_code_fences(response_text)
    for item_id in expected_ids:
        if item_id in parsed_scores:
            continue
        patterns = [
            rf'"{re.escape(item_id)}"\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
            rf"{re.escape(item_id)}\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            rf"{re.escape(item_id)}\s*[-–]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            value = coerce_non_negative_number(match.group(1), allow_percentage=allow_percentage)
            if value is not None:
                parsed_scores[item_id] = value
                break

    return parsed_scores


def append_debug_log(debug_log_path: str, entry: str) -> None:
    if not debug_log_path:
        return
    path = Path(debug_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry.rstrip() + "\n")


def append_run_separator(debug_log_path: str, args: argparse.Namespace) -> None:
    if not debug_log_path:
        return
    timestamp = datetime.now().isoformat(timespec="seconds")
    separator = "=" * 100
    entry = (
        "\n"
        f"{separator}\n"
        f"============================== RUN START ==============================\n"
        f"[run_start] time={timestamp} model={args.model} host={args.host} "
        f"paper_id={args.paper_id} "
        f"n_samples={max(1, args.n_samples)} temperature={max(0.0, args.temperature)} "
        f"max_retries={max(1, args.max_retries)} "
        f"paragraph_direct_max_tokens={max(0, args.paragraph_direct_max_tokens)} "
        f"paragraph_compressed_snippet_limit={max(60, args.paragraph_compressed_snippet_limit)}\n"
        f"============================== RUN START ==============================\n"
        f"{separator}"
    )
    append_debug_log(debug_log_path, entry)


def append_run_end(debug_log_path: str) -> None:
    if not debug_log_path:
        return
    timestamp = datetime.now().isoformat(timespec="seconds")
    separator = "=" * 100
    entry = (
        f"{separator}\n"
        f"=============================== RUN END ===============================\n"
        f"[run_end] time={timestamp}\n"
        f"=============================== RUN END ===============================\n"
        f"{separator}\n"
    )
    append_debug_log(debug_log_path, entry)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_distribution(raw_scores: Dict[str, float], total: float) -> Dict[str, float]:
    if not raw_scores:
        return {}

    cleaned = {k: max(0.0, safe_float(v, 0.0)) for k, v in raw_scores.items()}
    score_sum = sum(cleaned.values())

    if score_sum <= 0:
        raise ValueError(
            "normalize_distribution received non-positive total raw mass; "
            "refusing equal-score fallback."
        )

    normalized = {k: (v / score_sum) * total for k, v in cleaned.items()}

    residual = total - sum(normalized.values())
    best_key = max(normalized, key=normalized.get)
    normalized[best_key] += residual
    return normalized


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def combine_scores(technical_score: float, citation_score: float) -> float:
    return max(0.0, safe_float(technical_score, 0.0)) + max(0.0, safe_float(citation_score, 0.0))


def assert_close(actual: float, expected: float, context: str, tol: float = 1e-8) -> None:
    if abs(actual - expected) > tol:
        raise ValueError(
            f"Score conservation failed at {context}: expected {expected:.12f}, got {actual:.12f}, "
            f"diff={abs(actual - expected):.12f}"
        )


def flatten_content_to_text(content: Any, limit: Optional[int] = 700) -> str:
    if isinstance(content, str):
        if limit is None or limit <= 0:
            return content
        return content[:limit]
    if isinstance(content, dict):
        chunks: List[str] = []
        remaining = None if limit is None or limit <= 0 else limit
        for value in content.values():
            child_limit = remaining if remaining is not None else None
            chunk = flatten_content_to_text(value, limit=child_limit)
            if chunk:
                chunks.append(chunk)
                if remaining is not None:
                    remaining -= len(chunk)
            if remaining is not None and remaining <= 0:
                break
        merged = " ".join(chunks)
        if limit is None or limit <= 0:
            return merged
        return merged[:limit]
    return ""


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_text_into_paragraphs(text: str) -> List[str]:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    paragraphs = [normalize_for_match(p) for p in re.split(r"\n\s*\n+", normalized) if normalize_for_match(p)]
    if len(paragraphs) > 1:
        return paragraphs

    # PDF extraction often collapses paragraphs; use sentence chunks as a fallback.
    single = normalize_for_match(normalized)
    if not single:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", single) if s.strip()]
    if len(sentences) <= 3:
        return [single]

    chunk_size = 5
    return [" ".join(sentences[i : i + chunk_size]).strip() for i in range(0, len(sentences), chunk_size)]


def estimate_tokens_approx(text: str) -> int:
    # Rough rule-of-thumb for English text in many tokenizers.
    return max(1, (len(text) + 3) // 4)


def estimate_direct_allocation_tokens(
    parent_name: str,
    item_to_content: Dict[str, Any],
    snippet_limit: Optional[int] = 0,
) -> int:
    items = list(item_to_content.keys())
    snippets = {name: flatten_content_to_text(item_to_content[name], limit=snippet_limit) for name in items}
    item_payload = {
        f"I{idx + 1}": {"name": item_name, "excerpt": snippets[item_name]}
        for idx, item_name in enumerate(items)
    }
    prompt = SECTION_DIRECT_USER_PROMPT_TEMPLATE.format(
        parent_name=parent_name,
        parent_score=1.0,
        items_json=json.dumps(item_payload, indent=2),
    )
    system_prompt = SECTION_DIRECT_SYSTEM_PROMPT
    return estimate_tokens_approx(system_prompt) + estimate_tokens_approx(prompt)


def extract_probability_from_response(response_text: str) -> float:
    parsed = parse_json_response(response_text)
    if isinstance(parsed, dict):
        a_credit = parsed.get("a_credit", parsed.get("credit_a", parsed.get("item_a_credit")))
        b_credit = parsed.get("b_credit", parsed.get("credit_b", parsed.get("item_b_credit")))
        if a_credit is not None and b_credit is not None:
            a_val = safe_float(a_credit, -1.0)
            b_val = safe_float(b_credit, -1.0)
            if a_val >= 0.0 and b_val >= 0.0 and (a_val + b_val) > 0.0:
                return a_val / (a_val + b_val)

        for key in ("p", "probability", "score", "item_a_prob"):
            if key in parsed:
                value = safe_float(parsed.get(key), -1.0)
                if 0.0 <= value <= 1.0:
                    return value
        if len(parsed) == 1:
            only_value = safe_float(next(iter(parsed.values())), -1.0)
            if 0.0 <= only_value <= 1.0:
                return only_value

    match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", response_text)
    if match:
        value = safe_float(match.group(1), -1.0)
        if 0.0 <= value <= 1.0:
            return value
    return -1.0


def is_template_credit_response(response_text: str) -> bool:
    parsed = parse_json_response(response_text)
    if not isinstance(parsed, dict) or len(parsed) > 2:
        return False

    if "a_credit" not in parsed or "b_credit" not in parsed:
        return False

    a_val = safe_float(parsed.get("a_credit"), -1.0)
    b_val = safe_float(parsed.get("b_credit"), -1.0)
    return abs(a_val - 0.75) < 1e-9 and abs(b_val - 0.25) < 1e-9


def query_pair_probability(
    client: Client,
    parent_name: str,
    model: str,
    temperature: float,
    item_a: str,
    item_b: str,
    excerpt_a: str,
    excerpt_b: str,
    max_retries: int,
    debug_log_path: str,
    sample_idx: int,
) -> Tuple[float, bool]:
    system_prompt = SECTION_PAIRWISE_SYSTEM_PROMPT
    prompt = SECTION_PAIRWISE_USER_PROMPT_TEMPLATE.format(
        parent_name=parent_name,
        item_a=item_a,
        item_b=item_b,
        excerpt_a=excerpt_a,
        excerpt_b=excerpt_b,
    )

    for attempt in range(max(1, max_retries)):
        response = client.chat(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        raw_response = response.message.content if getattr(response, "message", None) else ""
        append_debug_log(
            debug_log_path,
            (
                f"[pair] parent={parent_name} sample={sample_idx} "
                f"attempt={attempt + 1}/{max(1, max_retries)} key={item_a}|{item_b}\n"
                f"{raw_response}\n"
            ),
        )

        if is_template_credit_response(raw_response):
            append_debug_log(
                debug_log_path,
                (
                    f"[pair_reject] parent={parent_name} sample={sample_idx} "
                    f"attempt={attempt + 1}/{max(1, max_retries)} key={item_a}|{item_b} "
                    "reason=template_075_025"
                ),
            )
            continue

        p = extract_probability_from_response(raw_response)
        if 0.0 <= p <= 1.0:
            return p, True

    return 0.5, False


def direct_allocate_scores(
    client: Client,
    item_to_content: Dict[str, Any],
    total_score: float,
    parent_name: str,
    model: str,
    temperature: float,
    max_retries: int,
    debug_log_path: str,
    sample_idx: int,
    snippet_limit: int = 0,
    log_tag: str = "all_together",
) -> Dict[str, float]:
    items = list(item_to_content.keys())
    use_limit = None if snippet_limit <= 0 else snippet_limit
    snippets = {name: flatten_content_to_text(item_to_content[name], limit=use_limit) for name in items}
    if not items:
        return {}
    if len(items) == 1:
        return {items[0]: total_score}

    item_ids = [f"I{idx + 1}" for idx in range(len(items))]
    item_id_to_name = {item_ids[idx]: items[idx] for idx in range(len(items))}
    item_payload = {
        item_id: {
            "name": item_id_to_name[item_id],
            "excerpt": snippets[item_id_to_name[item_id]],
        }
        for item_id in item_ids
    }

    system_prompt = SECTION_DIRECT_SYSTEM_PROMPT
    prompt = SECTION_DIRECT_USER_PROMPT_TEMPLATE.format(
        parent_name=parent_name,
        parent_score=total_score,
        items_json=json.dumps(item_payload, indent=2),
    )

    parsed_scores: Dict[str, float] = {}
    for attempt in range(max(1, max_retries)):
        response = client.chat(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        raw_response = response.message.content if getattr(response, "message", None) else ""
        append_debug_log(
            debug_log_path,
            (
                f"[{log_tag}] parent={parent_name} sample={sample_idx} "
                f"attempt={attempt + 1}/{max(1, max_retries)}\n{raw_response}\n"
            ),
        )
        alias_to_id = {item_id_to_name[item_id]: item_id for item_id in item_ids}
        parsed_by_id = parse_score_map_from_response(
            raw_response,
            item_ids,
            allow_percentage=False,
            alias_to_id=alias_to_id,
        )
        parsed_scores = {
            item_id_to_name[item_id]: max(0.0, safe_float(parsed_by_id.get(item_id), 0.0))
            for item_id in item_ids
        }

        if any(v > 0.0 for v in parsed_scores.values()):
            return normalize_distribution(parsed_scores, total_score)

    raise ValueError(
        f"Model failed to produce usable child scores for parent '{parent_name}' "
        f"after {max(1, max_retries)} retries; refusing equal-score fallback."
    )


def all_together_allocate_scores(
    client: Client,
    item_to_content: Dict[str, Any],
    total_score: float,
    parent_name: str,
    model: str,
    n_samples: int,
    temperature: float,
    max_retries: int,
    debug_log_path: str,
    snippet_limit: int = 0,
    log_tag: str = "all_together",
) -> Dict[str, float]:
    items = list(item_to_content.keys())
    if not items:
        return {}
    if len(items) == 1:
        return {items[0]: total_score}

    sample_count = max(1, n_samples)
    sample_distributions: List[Dict[str, float]] = []
    for s in range(sample_count):
        sample_distributions.append(
            direct_allocate_scores(
                client=client,
                item_to_content=item_to_content,
                total_score=total_score,
                parent_name=parent_name,
                model=model,
                temperature=min(1.0, temperature + (0.05 * s)),
                max_retries=max(1, max_retries),
                debug_log_path=debug_log_path,
                sample_idx=s + 1,
                snippet_limit=snippet_limit,
                log_tag=log_tag,
            )
        )

    averaged = {item: 0.0 for item in items}
    for dist in sample_distributions:
        for item, value in dist.items():
            averaged[item] += value
    averaged = {item: value / sample_count for item, value in averaged.items()}
    return normalize_distribution(averaged, total_score)


def direct_allocate_citation_scores(
    client: Client,
    paragraph_id: str,
    paragraph_text: str,
    citation_to_context: Dict[str, str],
    total_score: float,
    model: str,
    temperature: float,
    max_retries: int,
    debug_log_path: str,
    sample_idx: int,
) -> Dict[str, float]:
    citations = list(citation_to_context.keys())
    if not citations:
        return {}
    if len(citations) == 1:
        return {citations[0]: total_score}

    citation_ids = [f"C{idx + 1}" for idx in range(len(citations))]
    citation_id_to_name = {citation_ids[idx]: citations[idx] for idx in range(len(citations))}
    citation_payload = {
        citation_id: {
            "citation": citation_id_to_name[citation_id],
            "context": citation_to_context[citation_id_to_name[citation_id]],
        }
        for citation_id in citation_ids
    }

    prompt = CITATION_SPLIT_USER_PROMPT_TEMPLATE.format(
        paragraph_id=paragraph_id,
        paragraph_citation_score=total_score,
        paragraph_text=paragraph_text[:1200],
        citations_json=json.dumps(citation_payload, indent=2),
    )

    parsed_scores: Dict[str, float] = {}
    for attempt in range(max(1, max_retries)):
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": CITATION_SPLIT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": temperature},
        )
        raw_response = response.message.content if getattr(response, "message", None) else ""
        append_debug_log(
            debug_log_path,
            (
                f"[citation_split] paragraph_id={paragraph_id} sample={sample_idx} "
                f"attempt={attempt + 1}/{max(1, max_retries)}\n{raw_response}\n"
            ),
        )
        alias_to_id = {citation_id_to_name[citation_id]: citation_id for citation_id in citation_ids}
        parsed_by_id = parse_score_map_from_response(
            raw_response,
            citation_ids,
            allow_percentage=False,
            alias_to_id=alias_to_id,
        )
        parsed_scores = {
            citation_id_to_name[citation_id]: max(0.0, safe_float(parsed_by_id.get(citation_id), 0.0))
            for citation_id in citation_ids
        }

        if any(v > 0.0 for v in parsed_scores.values()):
            return normalize_distribution(parsed_scores, total_score)

    raise ValueError(
        f"Model failed to split citation score for paragraph '{paragraph_id}' "
        f"after {max(1, max_retries)} retries; refusing equal-score fallback."
    )


def allocate_citation_scores_for_paragraph(
    client: Client,
    paragraph_id: str,
    paragraph_text: str,
    citation_to_context: Dict[str, str],
    total_score: float,
    model: str,
    n_samples: int,
    temperature: float,
    max_retries: int,
    debug_log_path: str,
) -> Dict[str, float]:
    citations = list(citation_to_context.keys())
    if not citations:
        return {}
    if len(citations) == 1:
        return {citations[0]: total_score}

    sample_count = max(1, n_samples)
    sample_distributions: List[Dict[str, float]] = []
    for s in range(sample_count):
        sample_distributions.append(
            direct_allocate_citation_scores(
                client=client,
                paragraph_id=paragraph_id,
                paragraph_text=paragraph_text,
                citation_to_context=citation_to_context,
                total_score=total_score,
                model=model,
                temperature=min(1.0, temperature + (0.05 * s)),
                max_retries=max(1, max_retries),
                debug_log_path=debug_log_path,
                sample_idx=s + 1,
            )
        )

    averaged = {citation: 0.0 for citation in citations}
    for dist in sample_distributions:
        for citation, value in dist.items():
            averaged[citation] += value
    averaged = {citation: value / sample_count for citation, value in averaged.items()}
    return normalize_distribution(averaged, total_score)


def allocate_scores_by_mode(
    client: Client,
    item_to_content: Dict[str, Any],
    total_score: float,
    parent_name: str,
    model: str,
    n_samples: int,
    temperature: float,
    max_retries: int,
    debug_log_path: str,
    scoring_mode: str,
    snippet_limit: int = 0,
    log_tag: str = "all_together",
) -> Dict[str, float]:
    if scoring_mode == "pairwise":
        return pairwise_allocate_scores(
            client=client,
            item_to_content=item_to_content,
            total_score=total_score,
            parent_name=parent_name,
            model=model,
            n_samples=n_samples,
            temperature=temperature,
            max_retries=max_retries,
            debug_log_path=debug_log_path,
        )

    return all_together_allocate_scores(
        client=client,
        item_to_content=item_to_content,
        total_score=total_score,
        parent_name=parent_name,
        model=model,
        n_samples=n_samples,
        temperature=temperature,
        max_retries=max_retries,
        debug_log_path=debug_log_path,
        snippet_limit=snippet_limit,
        log_tag=log_tag,
    )


def pairwise_allocate_scores(
    client: Client,
    item_to_content: Dict[str, Any],
    total_score: float,
    parent_name: str,
    model: str,
    n_samples: int,
    temperature: float,
    max_retries: int = 3,
    debug_log_path: str = "",
) -> Dict[str, float]:
    items = list(item_to_content.keys())
    if not items:
        return {}
    if len(items) == 1:
        return {items[0]: total_score}

    snippets = {name: flatten_content_to_text(item_to_content[name], limit=None) for name in items}
    pairs: List[Tuple[str, str]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            pairs.append((items[i], items[j]))

    sample_distributions: List[Dict[str, float]] = []
    sample_count = max(1, n_samples)
    retry_count = max(1, max_retries)

    for s in range(sample_count):
        raw_scores = {item: 0.0 for item in items}
        fallback_pairs = 0
        pair_ps: List[float] = []

        for a, b in pairs:
            p_ab, ok_ab = query_pair_probability(
                client=client,
                parent_name=parent_name,
                model=model,
                temperature=min(1.0, temperature + (0.05 * s)),
                item_a=a,
                item_b=b,
                excerpt_a=snippets[a],
                excerpt_b=snippets[b],
                max_retries=retry_count,
                debug_log_path=debug_log_path,
                sample_idx=s + 1,
            )

            p_ba, ok_ba = query_pair_probability(
                client=client,
                parent_name=parent_name,
                model=model,
                temperature=min(1.0, temperature + (0.05 * s)),
                item_a=b,
                item_b=a,
                excerpt_a=snippets[b],
                excerpt_b=snippets[a],
                max_retries=retry_count,
                debug_log_path=debug_log_path,
                sample_idx=s + 1,
            )

            if ok_ab and ok_ba:
                # Symmetric estimate removes fixed-order bias.
                p = (p_ab + (1.0 - p_ba)) / 2.0
            elif ok_ab:
                p = p_ab
            elif ok_ba:
                p = 1.0 - p_ba
            else:
                p = 0.5
                fallback_pairs += 1

            raw_scores[a] += p
            raw_scores[b] += 1.0 - p
            pair_ps.append(p)

        if fallback_pairs > 0:
            print(
                f"[WARN] Pairwise fallback used for '{parent_name}' (sample {s + 1}): "
                f"{fallback_pairs}/{len(pairs)} pairs defaulted to 0.5."
            )

        degenerate_pairwise = len(pair_ps) >= 3 and (max(pair_ps) - min(pair_ps)) <= 0.02
        if degenerate_pairwise:
            print(
                f"[WARN] Degenerate pairwise outputs for '{parent_name}' (sample {s + 1}); "
                "switching to direct allocation fallback."
            )

        if fallback_pairs == len(pairs) or degenerate_pairwise:
            if fallback_pairs == len(pairs):
                print(
                    f"[WARN] All pairwise calls failed for '{parent_name}' (sample {s + 1}). "
                    "Using direct allocation fallback."
                )
            else:
                print(
                    f"[WARN] Pairwise outputs lacked variation for '{parent_name}' (sample {s + 1}). "
                    "Using direct allocation fallback."
                )
            sample_distributions.append(
                direct_allocate_scores(
                    client=client,
                    item_to_content=item_to_content,
                    total_score=total_score,
                    parent_name=parent_name,
                    model=model,
                    temperature=min(1.0, temperature + (0.05 * s)),
                    max_retries=retry_count,
                    debug_log_path=debug_log_path,
                    sample_idx=s + 1,
                    log_tag="fallback",
                )
            )
        else:
            sample_distributions.append(normalize_distribution(raw_scores, total_score))

    averaged = {item: 0.0 for item in items}
    for dist in sample_distributions:
        for item, value in dist.items():
            averaged[item] += value
    averaged = {item: value / sample_count for item, value in averaged.items()}
    return normalize_distribution(averaged, total_score)


def enforce_top_level_constraints(scores: Dict[str, float], total: float) -> Dict[str, float]:
    constrained = dict(scores)

    # Domain preference from user: Introduction should not exceed Experiment Results.
    if "Introduction" in constrained and "Experiment Results" in constrained:
        intro = constrained["Introduction"]
        exp = constrained["Experiment Results"]
        if intro > exp:
            transfer = (intro - exp) / 2.0
            constrained["Introduction"] -= transfer
            constrained["Experiment Results"] += transfer

    return normalize_distribution(constrained, total)


def score_paragraph_channel(
    client: Client,
    paragraph_items: Dict[str, str],
    model: str,
    n_samples: int,
    temperature: float,
    max_retries: int,
    snippet_limit: int,
    debug_log_path: str,
    system_prompt: str,
    user_prompt_template: str,
    log_tag: str,
) -> Dict[str, float]:
    if not paragraph_items:
        return {}

    paragraph_ids = list(paragraph_items.keys())
    sample_count = max(1, n_samples)
    per_sample_scores: List[Dict[str, float]] = []

    for sample_idx in range(sample_count):
        sample_scores: Dict[str, float] = {}
        current_temp = min(1.0, max(0.0, temperature) + (0.05 * sample_idx))

        if snippet_limit <= 0:
            snippets = {paragraph_id: paragraph_items[paragraph_id] for paragraph_id in paragraph_ids}
        else:
            snippets = {
                paragraph_id: paragraph_items[paragraph_id][: max(80, snippet_limit)]
                for paragraph_id in paragraph_ids
            }
        user_prompt = user_prompt_template.format(paragraphs_json=json.dumps(snippets, indent=2))

        parsed_scores: Dict[str, float] = {}
        for attempt in range(max(1, max_retries)):
            try:
                response = client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    options={"temperature": current_temp},
                )
                raw_response = response.message.content if getattr(response, "message", None) else ""
            except Exception as exc:
                raw_response = f"[exception] {exc}"

            append_debug_log(
                debug_log_path,
                (
                    f"[{log_tag}] sample={sample_idx + 1}/{sample_count} "
                    f"attempt={attempt + 1}/{max(1, max_retries)} "
                    f"temperature={current_temp}\n{raw_response}\n"
                ),
            )
            parsed_scores = parse_score_map_from_response(
                raw_response,
                paragraph_ids,
                allow_percentage=True,
            )
            if parsed_scores:
                break

        for paragraph_id in paragraph_ids:
            raw_score = safe_float(parsed_scores.get(paragraph_id), -1.0)
            if not (0.0 <= raw_score <= 1.0):
                raise ValueError(
                    f"Model failed to produce valid paragraph channel score for '{paragraph_id}' "
                    f"(sample {sample_idx + 1}); refusing 0.5 fallback."
                )
            sample_scores[paragraph_id] = clamp_unit(raw_score)

        per_sample_scores.append(sample_scores)

    averaged: Dict[str, float] = {}
    for paragraph_id in paragraph_ids:
        avg_score = sum(sample[paragraph_id] for sample in per_sample_scores) / sample_count
        averaged[paragraph_id] = clamp_unit(avg_score)
    return averaged


def compute_paragraph_channel_scores(
    section_score: float,
    paragraph_ids: List[str],
    paragraph_total_scores: Dict[str, float],
    mention_buckets: Dict[str, List[Tuple[str, str, str]]],
    technical_raw_scores: Dict[str, float],
    citation_raw_scores: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    if not paragraph_ids:
        return {}, {}, {}

    cleaned_t_raw: Dict[str, float] = {}
    cleaned_c_raw: Dict[str, float] = {}
    cleaned_total: Dict[str, float] = {}

    for paragraph_id in paragraph_ids:
        t_raw = max(0.0, safe_float(technical_raw_scores.get(paragraph_id), 0.0))
        c_raw = max(0.0, safe_float(citation_raw_scores.get(paragraph_id), 0.0))
        if not mention_buckets.get(paragraph_id):
            c_raw = 0.0

        cleaned_t_raw[paragraph_id] = t_raw
        cleaned_c_raw[paragraph_id] = c_raw
        cleaned_total[paragraph_id] = max(0.0, safe_float(paragraph_total_scores.get(paragraph_id), 0.0))

    total_sum = sum(cleaned_total.values())
    if total_sum <= 0.0:
        raise ValueError(
            "Paragraph total score allocation has zero mass; refusing equal-score fallback."
        )
    paragraph_total = normalize_distribution(cleaned_total, section_score)

    paragraph_technical: Dict[str, float] = {}
    paragraph_citation: Dict[str, float] = {}
    for paragraph_id in paragraph_ids:
        total_val = max(0.0, safe_float(paragraph_total.get(paragraph_id), 0.0))
        t_raw = cleaned_t_raw.get(paragraph_id, 0.0)
        c_raw = cleaned_c_raw.get(paragraph_id, 0.0)

        if not mention_buckets.get(paragraph_id):
            t_val = total_val
            c_val = 0.0
        else:
            denom = t_raw + c_raw
            if denom <= 0.0:
                raise ValueError(
                    f"Paragraph channel split has zero raw mass for '{paragraph_id}'; "
                    "refusing equal technical/citation fallback."
                )
            t_val = total_val * (t_raw / denom)
            c_val = total_val * (c_raw / denom)
            t_val += total_val - (t_val + c_val)

        paragraph_technical[paragraph_id] = t_val
        paragraph_citation[paragraph_id] = c_val

    return paragraph_technical, paragraph_citation, paragraph_total


def split_citation_block(citation_block: str) -> List[str]:
    block = citation_block.strip()
    left_delim = ""
    right_delim = ""
    if block.startswith("(") and block.endswith(")"):
        left_delim, right_delim = "(", ")"
        inner = block[1:-1]
    elif block.startswith("[") and block.endswith("]"):
        left_delim, right_delim = "[", "]"
        inner = block[1:-1]
    else:
        inner = block

    if ";" in inner:
        raw_parts = inner.split(";")
    elif left_delim == "[" and right_delim == "]" and "," in inner:
        raw_parts = inner.split(",")
    elif re.fullmatch(r"\s*\d+(?:\s*,\s*\d+)+\s*", inner):
        raw_parts = inner.split(",")
    else:
        raw_parts = [inner]

    parts = [re.sub(r"\s+", " ", p).strip() for p in raw_parts if p.strip()]
    if not parts:
        return [block]
    if left_delim and right_delim:
        return [f"{left_delim}{p}{right_delim}" for p in parts]
    return parts


def assign_importance_scores(
    content_dict: Dict[str, Any],
    citations_dict: Dict[str, Any],
    model: str = "llama3.2",
    host: str = "localhost:11434",
    n_samples: int = 3,
    temperature: float = 0.2,
    max_retries: int = 3,
    paragraph_direct_max_tokens: int = 0,
    paragraph_compressed_snippet_limit: int = 180,
    debug_log_path: str = "",
    paper_id: str = DEFAULT_PAPER_ID,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Assign hierarchical section, paragraph, and citation scores with:
    - level-wise all-children-together scoring (parent sees all child nodes in one prompt),
    - repeated LLM sampling + averaging,
    - strict normalization at each tree level,
    - paragraph-level technical and citation channel decomposition.
    """
    client = Client(host=host)
    citation_scores: Dict[str, Any] = {}
    section_scores: Dict[str, Any] = {}
    paragraph_scores: Dict[str, Any] = {}

    top_level_scores = all_together_allocate_scores(
        client=client,
        item_to_content=content_dict,
        total_score=1.0,
        parent_name="Whole Paper",
        model=model,
        n_samples=n_samples,
        temperature=temperature,
        max_retries=max_retries,
        debug_log_path=debug_log_path,
        snippet_limit=0,
        log_tag="all_together_top",
    )
    top_level_scores = enforce_top_level_constraints(top_level_scores, total=1.0)

    for section_name, score in top_level_scores.items():
        section_scores[section_name] = {
            "technical_score": 0.0,
            "citation_score": 0.0,
            "subsections": {},
        }

    def assign_citation_scores(
        section_name: str,
        section_path: List[str],
        section_score: float,
        section_content: Any,
        section_citations: Dict[str, Any],
        node_ref: Dict[str, Any],
    ) -> None:
        raw_text = section_content if isinstance(section_content, str) else flatten_content_to_text(section_content, 6000)
        paragraphs = split_text_into_paragraphs(raw_text)
        if not paragraphs:
            normalized_text = normalize_for_match(raw_text)
            if not normalized_text:
                return
            paragraphs = [normalized_text]

        paragraph_items = {f"Paragraph {idx + 1}": paragraph for idx, paragraph in enumerate(paragraphs)}
        paragraph_meta = {
            name: {
                "paragraph_id": f"{paper_id}::{' > '.join(section_path)}::p{idx + 1}",
                "paragraph_index": idx + 1,
                "text": paragraph,
            }
            for idx, (name, paragraph) in enumerate(paragraph_items.items())
        }
        paragraph_parent_name = f"{section_name}::paragraphs"
        est_tokens = estimate_direct_allocation_tokens(paragraph_parent_name, paragraph_items, snippet_limit=0)
        use_compressed = paragraph_direct_max_tokens > 0 and est_tokens > paragraph_direct_max_tokens
        snippet_limit = max(60, paragraph_compressed_snippet_limit) if use_compressed else 0
        method = "channel_compressed" if use_compressed else "channel_full_parent"
        append_debug_log(
            debug_log_path,
            (
                f"[paragraph_scoring] parent={paragraph_parent_name} method={method} "
                f"estimated_tokens={est_tokens} threshold={paragraph_direct_max_tokens} "
                f"paragraphs={len(paragraph_items)} snippet_limit={snippet_limit} "
                f"n_samples={max(1, n_samples)} max_retries={max(1, max_retries)}"
            ),
        )

        paragraph_total_scores = all_together_allocate_scores(
            client=client,
            item_to_content=paragraph_items,
            total_score=section_score,
            parent_name=paragraph_parent_name,
            model=model,
            n_samples=max(1, n_samples),
            temperature=temperature,
            max_retries=max(1, max_retries),
            debug_log_path=debug_log_path,
            snippet_limit=snippet_limit,
            log_tag="all_together_paragraphs",
        )

        technical_raw_scores = score_paragraph_channel(
            client=client,
            paragraph_items=paragraph_items,
            model=model,
            n_samples=max(1, n_samples),
            temperature=temperature,
            max_retries=max(1, max_retries),
            snippet_limit=snippet_limit,
            debug_log_path=debug_log_path,
            system_prompt=PARAGRAPH_TECHNICAL_SYSTEM_PROMPT,
            user_prompt_template=PARAGRAPH_TECHNICAL_USER_PROMPT_TEMPLATE,
            log_tag="paragraph_technical_batch",
        )
        citation_raw_scores = score_paragraph_channel(
            client=client,
            paragraph_items=paragraph_items,
            model=model,
            n_samples=max(1, n_samples),
            temperature=temperature,
            max_retries=max(1, max_retries),
            snippet_limit=snippet_limit,
            debug_log_path=debug_log_path,
            system_prompt=PARAGRAPH_CITATION_SYSTEM_PROMPT,
            user_prompt_template=PARAGRAPH_CITATION_USER_PROMPT_TEMPLATE,
            log_tag="paragraph_citation_batch",
        )

        paragraph_norms = {name: normalize_for_match(text) for name, text in paragraph_items.items()}
        paragraph_tokens = {
            name: set(re.findall(r"[a-z0-9]+", paragraph_norms[name].lower())) for name in paragraph_items
        }
        mention_buckets: Dict[str, List[Tuple[str, str, str]]] = {name: [] for name in paragraph_items}

        section_citation_dict = section_citations if isinstance(section_citations, dict) else {}
        for citation_block, context_value in section_citation_dict.items():
            if isinstance(context_value, list):
                contexts = context_value
            else:
                contexts = [context_value]

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
                    target_paragraph = max(
                        paragraph_items,
                        key=lambda name: technical_raw_scores.get(name, 0.0) + citation_raw_scores.get(name, 0.0),
                    )

                for citation in split_citation_block(citation_block):
                    mention_buckets[target_paragraph].append((citation, citation_block, context_str))

        paragraph_names = list(paragraph_items.keys())
        paragraph_technical, paragraph_citation, paragraph_total = compute_paragraph_channel_scores(
            section_score=section_score,
            paragraph_ids=paragraph_names,
            paragraph_total_scores=paragraph_total_scores,
            mention_buckets=mention_buckets,
            technical_raw_scores=technical_raw_scores,
            citation_raw_scores=citation_raw_scores,
        )

        for paragraph_name in paragraph_names:
            meta = paragraph_meta[paragraph_name]
            paragraph_id = meta["paragraph_id"]
            mentions = mention_buckets.get(paragraph_name, [])
            paragraph_t = max(0.0, paragraph_technical.get(paragraph_name, 0.0))
            paragraph_c = max(0.0, paragraph_citation.get(paragraph_name, 0.0))
            paragraph_total_score = combine_scores(paragraph_t, paragraph_c)
            assert_close(
                paragraph_total_score,
                max(0.0, safe_float(paragraph_total.get(paragraph_name), 0.0)),
                f"paragraph {paragraph_id}",
            )

            paragraph_scores[paragraph_id] = {
                "technical_score": paragraph_t,
                "citation_score": paragraph_c,
            }

            if not mentions:
                continue

            if paragraph_c <= 0.0:
                continue

            citation_contexts: Dict[str, List[str]] = {}
            for citation, _, context in mentions:
                citation_contexts.setdefault(citation, []).append(normalize_for_match(str(context)))

            citation_to_context: Dict[str, str] = {}
            for citation, contexts in citation_contexts.items():
                unique_contexts = [ctx for ctx in dict.fromkeys(contexts) if ctx]
                context_blob = " ".join(unique_contexts)[:700]
                citation_to_context[citation] = context_blob if context_blob else meta["text"][:700]

            citation_split = allocate_citation_scores_for_paragraph(
                client=client,
                paragraph_id=paragraph_id,
                paragraph_text=meta["text"],
                citation_to_context=citation_to_context,
                total_score=paragraph_c,
                model=model,
                n_samples=max(1, n_samples),
                temperature=temperature,
                max_retries=max(1, max_retries),
                debug_log_path=debug_log_path,
            )
            split_total = sum(citation_split.values())
            assert_close(split_total, paragraph_c, f"citation split for {paragraph_id}")

            for citation, citation_value in citation_split.items():
                if citation not in citation_scores:
                    citation_scores[citation] = {"citation_score": 0.0}

                citation_scores[citation]["citation_score"] += citation_value

        leaf_t = sum(paragraph_technical.values())
        leaf_c = sum(paragraph_citation.values())
        leaf_total = combine_scores(leaf_t, leaf_c)
        assert_close(leaf_total, section_score, f"leaf section {' > '.join(section_path)}")
        node_ref["technical_score"] = leaf_t
        node_ref["citation_score"] = leaf_c

    def process_section(
        section_name: str,
        section_content: Any,
        section_score: float,
        section_citations: Any,
        parent_ref: Optional[Dict[str, Any]] = None,
        section_path: Optional[List[str]] = None,
    ) -> None:
        current_path = section_path or [section_name]
        if parent_ref is None:
            current_ref = section_scores[section_name]
        else:
            parent_ref[section_name] = {
                "technical_score": 0.0,
                "citation_score": 0.0,
                "subsections": {},
            }
            current_ref = parent_ref[section_name]

        if isinstance(section_content, dict) and section_content:
            subsection_scores = all_together_allocate_scores(
                client=client,
                item_to_content=section_content,
                total_score=section_score,
                parent_name=section_name,
                model=model,
                n_samples=n_samples,
                temperature=temperature,
                max_retries=max_retries,
                debug_log_path=debug_log_path,
                snippet_limit=0,
                log_tag="all_together_subsections",
            )

            for subsection_name, subsection_content in section_content.items():
                subsection_score = subsection_scores.get(
                    subsection_name, section_score / max(1, len(section_content))
                )
                if isinstance(section_citations, dict):
                    subsection_citations = section_citations.get(subsection_name, {})
                else:
                    subsection_citations = {}

                process_section(
                    subsection_name,
                    subsection_content,
                    subsection_score,
                    subsection_citations,
                    parent_ref=current_ref["subsections"],
                    section_path=current_path + [subsection_name],
                )
            child_total = sum(
                combine_scores(child["technical_score"], child["citation_score"])
                for child in current_ref["subsections"].values()
            )
            assert_close(child_total, section_score, f"internal section {' > '.join(current_path)}")
            current_ref["technical_score"] = sum(
                child["technical_score"] for child in current_ref["subsections"].values()
            )
            current_ref["citation_score"] = sum(
                child["citation_score"] for child in current_ref["subsections"].values()
            )
        else:
            assign_citation_scores(
                section_name=section_name,
                section_path=current_path,
                section_score=section_score,
                section_content=section_content,
                section_citations=section_citations,
                node_ref=current_ref,
            )

    for section_name, section_content in content_dict.items():
        score = top_level_scores.get(section_name, 1.0 / max(1, len(content_dict)))
        section_citations = citations_dict.get(section_name, {}) if isinstance(citations_dict, dict) else {}
        process_section(
            section_name,
            section_content,
            score,
            section_citations,
            parent_ref=None,
            section_path=[section_name],
        )

    top_level_total = sum(
        combine_scores(payload["technical_score"], payload["citation_score"])
        for payload in section_scores.values()
    )
    assert_close(top_level_total, 1.0, "top-level sections")

    return citation_scores, section_scores, paragraph_scores


def print_section_hierarchy(section_scores: Dict[str, Any], indent: int = 0) -> None:
    for section_name, section_data in section_scores.items():
        technical_score = max(0.0, safe_float(section_data.get("technical_score"), 0.0))
        citation_score = max(0.0, safe_float(section_data.get("citation_score"), 0.0))
        total_score = combine_scores(technical_score, citation_score)
        print(
            f"{'  ' * indent}{section_name}: "
            f"total={total_score:.4f}, technical={technical_score:.4f}, citation={citation_score:.4f}"
        )
        subsections = section_data.get("subsections", {})
        if subsections:
            print_section_hierarchy(subsections, indent + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Importance scoring for paper sections and citations.")
    parser.add_argument("--output1", required=True, help="Prefix for citation output file")
    parser.add_argument("--output2", required=True, help="Prefix for section output file")
    parser.add_argument(
        "--output3",
        default="",
        help="Optional prefix for paragraph output file (default: use output2 prefix).",
    )
    parser.add_argument("--pdf", default=DEFAULT_PDF_PATH, help="Path to PDF file")
    parser.add_argument("--paper-id", default=DEFAULT_PAPER_ID, help="Paper identifier used in paragraph ids")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name")
    parser.add_argument("--host", default="localhost:11434", help="Ollama host")
    parser.add_argument("--n-samples", type=int, default=3, help="Number of LLM samples to average")
    parser.add_argument("--temperature", type=float, default=0.2, help="Base sampling temperature")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per sample for JSON parsing")
    parser.add_argument(
        "--paragraph-direct-max-tokens",
        type=int,
        default=0,
        help=(
            "Token threshold for optional paragraph compression; "
            "set 0 to always send full paragraph text."
        ),
    )
    parser.add_argument(
        "--paragraph-compressed-snippet-limit",
        type=int,
        default=180,
        help="Per-paragraph snippet length used by compressed paragraph channel scoring.",
    )
    parser.add_argument(
        "--debug-log",
        default="pairwise_debug.log",
        help="Path to write raw model responses for debugging",
    )
    parser.add_argument(
        "--prompts-output",
        default="",
        help="Optional path to save the exact prompt strings as JSON.",
    )
    args = parser.parse_args()
    append_run_separator(args.debug_log, args)

    text = read_pdf_text(args.pdf)
    citations, content = extract_citations_by_section(text, DEFAULT_SECTIONS)
    citation_importance, section_importance, paragraph_importance = assign_importance_scores(
        content_dict=content,
        citations_dict=citations,
        model=args.model,
        host=args.host,
        n_samples=max(1, args.n_samples),
        temperature=max(0.0, args.temperature),
        max_retries=max(1, args.max_retries),
        paragraph_direct_max_tokens=max(0, args.paragraph_direct_max_tokens),
        paragraph_compressed_snippet_limit=max(60, args.paragraph_compressed_snippet_limit),
        debug_log_path=args.debug_log,
        paper_id=args.paper_id,
    )

    citation_path = f"{args.output1}_citation_scores.json"
    section_path = f"{args.output2}_section_scores.json"
    paragraph_prefix = args.output3 if args.output3 else args.output2
    paragraph_path = f"{paragraph_prefix}_paragraph_scores.json"

    with open(citation_path, "w", encoding="utf-8") as f:
        json.dump(citation_importance, f, indent=2)

    with open(section_path, "w", encoding="utf-8") as f:
        json.dump(section_importance, f, indent=2)

    with open(paragraph_path, "w", encoding="utf-8") as f:
        json.dump(paragraph_importance, f, indent=2)

    if args.prompts_output:
        with open(args.prompts_output, "w", encoding="utf-8") as f:
            json.dump(PROMPT_CATALOG, f, indent=2)

    print("\n" + "=" * 50)
    print("FINAL CITATION SCORES (AGGREGATED ACROSS SECTIONS):")
    print("=" * 50)
    sorted_citations = sorted(
        citation_importance.items(),
        key=lambda x: safe_float(x[1].get("citation_score"), 0.0),
        reverse=True,
    )
    for citation, info in sorted_citations[:10]:
        print(f"{citation}: citation={safe_float(info.get('citation_score'), 0.0):.4f}")

    print("\n" + "=" * 50)
    print("SECTION HIERARCHY WITH SCORES:")
    print("=" * 50)
    print_section_hierarchy(section_importance)

    print("\n" + "=" * 50)
    print("TOP PARAGRAPH SCORES (TECHNICAL / CITATION):")
    print("=" * 50)
    sorted_paragraphs = sorted(
        paragraph_importance.items(),
        key=lambda x: combine_scores(
            safe_float(x[1].get("technical_score"), 0.0),
            safe_float(x[1].get("citation_score"), 0.0),
        ),
        reverse=True,
    )
    for paragraph_id, payload in sorted_paragraphs[:10]:
        technical_score = safe_float(payload.get("technical_score"), 0.0)
        citation_score = safe_float(payload.get("citation_score"), 0.0)
        print(
            f"{paragraph_id}: technical={technical_score:.4f}, citation={citation_score:.4f}"
        )

    append_run_end(args.debug_log)


if __name__ == "__main__":
    main()
