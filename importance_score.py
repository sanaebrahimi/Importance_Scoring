import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import PyPDF2
from ollama import Client


DEFAULT_PDF_PATH = "adv_res_paper.pdf"

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
    separator = "=" * 80
    entry = (
        f"{separator}\n"
        f"[run_start] time={timestamp} model={args.model} host={args.host} "
        f"n_samples={max(1, args.n_samples)} temperature={max(0.0, args.temperature)} "
        f"max_retries={max(1, args.max_retries)} "
        f"paragraph_direct_max_tokens={max(0, args.paragraph_direct_max_tokens)} "
        f"paragraph_pairwise_max_items={max(1, args.paragraph_pairwise_max_items)} "
        f"paragraph_pairwise_n_samples={max(1, args.paragraph_pairwise_n_samples)} "
        f"paragraph_pairwise_max_retries={max(1, args.paragraph_pairwise_max_retries)} "
        f"paragraph_compressed_snippet_limit={max(60, args.paragraph_compressed_snippet_limit)}\n"
        f"{separator}"
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
        equal = total / len(cleaned)
        normalized = {k: equal for k in cleaned}
    else:
        normalized = {k: (v / score_sum) * total for k, v in cleaned.items()}

    residual = total - sum(normalized.values())
    best_key = max(normalized, key=normalized.get)
    normalized[best_key] += residual
    return normalized


def flatten_content_to_text(content: Any, limit: int = 700) -> str:
    if isinstance(content, str):
        return content[:limit]
    if isinstance(content, dict):
        chunks: List[str] = []
        for value in content.values():
            chunk = flatten_content_to_text(value, limit=limit)
            if chunk:
                chunks.append(chunk)
            if sum(len(c) for c in chunks) >= limit:
                break
        merged = " ".join(chunks)
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


def estimate_direct_allocation_tokens(parent_name: str, item_to_content: Dict[str, Any], snippet_limit: int = 500) -> int:
    snippets = {name: flatten_content_to_text(item_to_content[name], limit=snippet_limit) for name in item_to_content}
    prompt = f"""Parent node: "{parent_name}"
Items:
{json.dumps(snippets, indent=2)}

Return ONLY JSON mapping each item to a non-negative raw score:
{{
  "item_name_1": 12.0,
  "item_name_2": 5.0
}}
"""
    system_prompt = "You score academic paper sections by importance. Return JSON only."
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
    system_prompt = (
        "You are an expert academic reviewer. "
        "Compare two items from the same paper by contribution to the paper's main scientific contribution. "
        "Distribute a total credit of 1.0 between item A and item B. "
        "Return JSON only."
    )
    prompt = f"""Parent node: "{parent_name}"
Task: distribute a credit of 1.0 between A and B based on contribution to the paper's main contribution.

Scoring guidance:
- Higher credit: defines core problem, novel method, key theory/algorithm, primary validation.
- Lower credit: background, minor details, auxiliary discussion, transitions.
- If one item contributes 3x the other, credits should be 0.75 and 0.25.
- If equal, credits should be 0.5 and 0.5.

Item A name: "{item_a}"
Item A excerpt:
{excerpt_a}

Item B name: "{item_b}"
Item B excerpt:
{excerpt_b}

Return ONLY JSON:
{{
  "a_credit": 0.75,
  "b_credit": 0.25
}}

Constraints:
- 0.0 <= a_credit <= 1.0
- 0.0 <= b_credit <= 1.0
- a_credit + b_credit = 1.0
"""

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

        p = extract_probability_from_response(raw_response)
        if 0.0 <= p <= 1.0:
            return p, True

    return 0.5, False


def direct_allocate_scores_fallback(
    client: Client,
    item_to_content: Dict[str, Any],
    total_score: float,
    parent_name: str,
    model: str,
    temperature: float,
    max_retries: int,
    debug_log_path: str,
    sample_idx: int,
    snippet_limit: int = 500,
) -> Dict[str, float]:
    items = list(item_to_content.keys())
    snippets = {name: flatten_content_to_text(item_to_content[name], limit=snippet_limit) for name in items}
    system_prompt = (
       "You are an expert academic reviewer. Your task is to read two sections, subsections, or paragraphs from the same \
        research paper and  estimate the importance of each item that is defined strictly as contribution to the paper’s central \
        claim, novel method, primary findings, or core empirical validation, not writing quality, length, or stylistic emphasis."
        "Return JSON only."
    )
    prompt = f"""Parent node: "{parent_name}"
    Assess the importance of each item relative to the paper’s main scientific contribution.
    Importance should reflect how strongly the item contributes to:
    • Defining the central research problem
    • Presenting the novel method, model, or theoretical contribution
    • Reporting core empirical results or primary validation
    • Distinguishing the work from prior state-of-the-art
    • Explaining key findings that support the main claim
    Lower importance includes:
    • Background or contextual material
    • Minor implementation details
    • Auxiliary or secondary experiments
    • Transitional or structural text
    Items:
    "{json.dumps(snippets, indent=2)}"
    Assign each item a non-negative raw importance score.
    Higher scores indicate greater contribution to the paper’s main contribution.
    Scores are relative within this set (they need not sum to any fixed value).
    Return ONLY JSON mapping each item to its raw score:
    {{
    "item_name_1": a float between 0.0 and 1.0 representing P(Item A > Item B),
    "item_name_2": a float between 0.0 and 1.0 representing P(Item A > Item B)
    }}
    """

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
                f"[fallback] parent={parent_name} sample={sample_idx} "
                f"attempt={attempt + 1}/{max(1, max_retries)}\n{raw_response}\n"
            ),
        )
        parsed = parse_json_response(raw_response)
        if not isinstance(parsed, dict) or not parsed:
            continue

        parsed_scores = {}
        for item in items:
            parsed_scores[item] = max(0.0, safe_float(parsed.get(item), 0.0))

        if any(v > 0 for v in parsed_scores.values()):
            return normalize_distribution(parsed_scores, total_score)

    # Final neutral fallback if LLM fallback also fails.
    equal_raw = {item: 1.0 for item in items}
    return normalize_distribution(equal_raw, total_score)


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

    snippets = {name: flatten_content_to_text(item_to_content[name], limit=450) for name in items}
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

        if fallback_pairs > 0:
            print(
                f"[WARN] Pairwise fallback used for '{parent_name}' (sample {s + 1}): "
                f"{fallback_pairs}/{len(pairs)} pairs defaulted to 0.5."
            )

        if fallback_pairs == len(pairs):
            print(
                f"[WARN] All pairwise calls failed for '{parent_name}' (sample {s + 1}). "
                "Using direct allocation fallback."
            )
            sample_distributions.append(
                direct_allocate_scores_fallback(
                    client=client,
                    item_to_content=item_to_content,
                    total_score=total_score,
                    parent_name=parent_name,
                    model=model,
                    temperature=min(1.0, temperature + (0.05 * s)),
                    max_retries=retry_count,
                    debug_log_path=debug_log_path,
                    sample_idx=s + 1,
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
    paragraph_direct_max_tokens: int = 4000,
    paragraph_pairwise_max_items: int = 8,
    paragraph_pairwise_n_samples: int = 1,
    paragraph_pairwise_max_retries: int = 1,
    paragraph_compressed_snippet_limit: int = 180,
    debug_log_path: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Assign hierarchical section and citation scores with:
    - pairwise scoring,
    - repeated LLM sampling + averaging,
    - strict normalization at each tree level.
    """
    client = Client(host=host)
    citation_scores: Dict[str, Any] = {}
    section_scores: Dict[str, Any] = {}

    top_level_scores = pairwise_allocate_scores(
        client=client,
        item_to_content=content_dict,
        total_score=1.0,
        parent_name="Whole Paper",
        model=model,
        n_samples=n_samples,
        temperature=temperature,
        max_retries=max_retries,
        debug_log_path=debug_log_path,
    )
    top_level_scores = enforce_top_level_constraints(top_level_scores, total=1.0)

    for section_name, score in top_level_scores.items():
        section_scores[section_name] = {
            "score": score,
            "type": "section",
            "level": 1,
            "subsections": {},
        }

    def assign_citation_scores(
        section_name: str,
        section_score: float,
        section_content: Any,
        section_citations: Dict[str, Any],
        level: int,
    ) -> None:
        if not section_citations or not isinstance(section_citations, dict):
            return

        raw_text = section_content if isinstance(section_content, str) else flatten_content_to_text(section_content, 6000)
        paragraphs = split_text_into_paragraphs(raw_text)
        if not paragraphs:
            normalized_text = normalize_for_match(raw_text)
            if not normalized_text:
                return
            paragraphs = [normalized_text]

        paragraph_items = {f"Paragraph {idx + 1}": paragraph for idx, paragraph in enumerate(paragraphs)}
        paragraph_parent_name = f"{section_name}::paragraphs"
        est_tokens = estimate_direct_allocation_tokens(paragraph_parent_name, paragraph_items, snippet_limit=500)

        if paragraph_direct_max_tokens > 0 and est_tokens <= paragraph_direct_max_tokens:
            append_debug_log(
                debug_log_path,
                (
                    f"[paragraph_scoring] parent={paragraph_parent_name} method=direct "
                    f"estimated_tokens={est_tokens} threshold={paragraph_direct_max_tokens}"
                ),
            )
            paragraph_scores = direct_allocate_scores_fallback(
                client=client,
                item_to_content=paragraph_items,
                total_score=section_score,
                parent_name=paragraph_parent_name,
                model=model,
                temperature=temperature,
                max_retries=1,
                debug_log_path=debug_log_path,
                sample_idx=1,
                snippet_limit=500,
            )
        else:
            if len(paragraph_items) <= max(1, paragraph_pairwise_max_items):
                append_debug_log(
                    debug_log_path,
                    (
                        f"[paragraph_scoring] parent={paragraph_parent_name} method=pairwise "
                        f"estimated_tokens={est_tokens} threshold={paragraph_direct_max_tokens} "
                        f"paragraphs={len(paragraph_items)} pairwise_max_items={paragraph_pairwise_max_items} "
                        f"n_samples={max(1, paragraph_pairwise_n_samples)} "
                        f"max_retries={max(1, paragraph_pairwise_max_retries)}"
                    ),
                )
                paragraph_scores = pairwise_allocate_scores(
                    client=client,
                    item_to_content=paragraph_items,
                    total_score=section_score,
                    parent_name=paragraph_parent_name,
                    model=model,
                    n_samples=max(1, paragraph_pairwise_n_samples),
                    temperature=temperature,
                    max_retries=max(1, paragraph_pairwise_max_retries),
                    debug_log_path=debug_log_path,
                )
            else:
                append_debug_log(
                    debug_log_path,
                    (
                        f"[paragraph_scoring] parent={paragraph_parent_name} method=direct_compressed "
                        f"estimated_tokens={est_tokens} threshold={paragraph_direct_max_tokens} "
                        f"paragraphs={len(paragraph_items)} pairwise_max_items={paragraph_pairwise_max_items} "
                        f"snippet_limit={max(60, paragraph_compressed_snippet_limit)}"
                    ),
                )
                paragraph_scores = direct_allocate_scores_fallback(
                    client=client,
                    item_to_content=paragraph_items,
                    total_score=section_score,
                    parent_name=paragraph_parent_name,
                    model=model,
                    temperature=temperature,
                    max_retries=1,
                    debug_log_path=debug_log_path,
                    sample_idx=1,
                    snippet_limit=max(60, paragraph_compressed_snippet_limit),
                )

        paragraph_norms = {name: normalize_for_match(text) for name, text in paragraph_items.items()}
        paragraph_tokens = {
            name: set(re.findall(r"[a-z0-9]+", paragraph_norms[name].lower())) for name in paragraph_items
        }
        mention_buckets: Dict[str, List[Tuple[str, str, str]]] = {name: [] for name in paragraph_items}

        for citation_block, context_value in section_citations.items():
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
                    target_paragraph = max(paragraph_scores, key=paragraph_scores.get)

                for citation in split_citation_block(citation_block):
                    mention_buckets[target_paragraph].append((citation, citation_block, context_str))

        for paragraph_name, mentions in mention_buckets.items():
            if not mentions:
                continue

            paragraph_score = paragraph_scores.get(paragraph_name, 0.0)
            if paragraph_score <= 0:
                continue

            score_per_mention = paragraph_score / len(mentions)
            for citation, source_block, context in mentions:
                if citation not in citation_scores:
                    citation_scores[citation] = {"score": 0.0, "mentions": 0, "occurrences": []}

                citation_scores[citation]["score"] += score_per_mention
                citation_scores[citation]["mentions"] += 1
                citation_scores[citation]["occurrences"].append(
                    {
                        "section": section_name,
                        "level": level,
                        "source_block": source_block,
                        "context": context,
                        "paragraph": paragraph_name,
                    }
                )

    def process_section(
        section_name: str,
        section_content: Any,
        section_score: float,
        section_citations: Any,
        level: int = 1,
        parent_ref: Dict[str, Any] = None,
    ) -> None:
        if parent_ref is None:
            current_ref = section_scores[section_name]
        else:
            parent_ref[section_name] = {
                "score": section_score,
                "type": "subsection",
                "level": level,
                "subsections": {},
            }
            current_ref = parent_ref[section_name]

        if isinstance(section_content, dict) and section_content:
            subsection_scores = pairwise_allocate_scores(
                client=client,
                item_to_content=section_content,
                total_score=section_score,
                parent_name=section_name,
                model=model,
                n_samples=n_samples,
                temperature=temperature,
                max_retries=max_retries,
                debug_log_path=debug_log_path,
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
                    level=level + 1,
                    parent_ref=current_ref["subsections"],
                )
        else:
            assign_citation_scores(section_name, section_score, section_content, section_citations, level)

    for section_name, section_content in content_dict.items():
        score = top_level_scores.get(section_name, 1.0 / max(1, len(content_dict)))
        section_citations = citations_dict.get(section_name, {}) if isinstance(citations_dict, dict) else {}
        process_section(section_name, section_content, score, section_citations, level=1, parent_ref=None)

    return citation_scores, section_scores


def print_section_hierarchy(section_scores: Dict[str, Any], indent: int = 0) -> None:
    for section_name, section_data in section_scores.items():
        print(
            f"{'  ' * indent}{section_name} "
            f"[{section_data['type']}, Level {section_data['level']}]: "
            f"{section_data['score']:.4f}"
        )
        if section_data["subsections"]:
            print_section_hierarchy(section_data["subsections"], indent + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Importance scoring for paper sections and citations.")
    parser.add_argument("--output1", required=True, help="Prefix for citation output file")
    parser.add_argument("--output2", required=True, help="Prefix for section output file")
    parser.add_argument("--pdf", default=DEFAULT_PDF_PATH, help="Path to PDF file")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name")
    parser.add_argument("--host", default="localhost:11434", help="Ollama host")
    parser.add_argument("--n-samples", type=int, default=3, help="Number of LLM samples to average")
    parser.add_argument("--temperature", type=float, default=0.2, help="Base sampling temperature")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per sample for pairwise JSON")
    parser.add_argument(
        "--paragraph-direct-max-tokens",
        type=int,
        default=4000,
        help=(
            "Use direct paragraph scoring when estimated prompt tokens are <= this value; "
            "otherwise use pairwise paragraph scoring."
        ),
    )
    parser.add_argument(
        "--paragraph-pairwise-max-items",
        type=int,
        default=8,
        help=(
            "If paragraph count exceeds this value, skip paragraph pairwise and use compressed direct scoring."
        ),
    )
    parser.add_argument(
        "--paragraph-pairwise-n-samples",
        type=int,
        default=1,
        help="Number of samples to use for paragraph pairwise scoring (when enabled).",
    )
    parser.add_argument(
        "--paragraph-pairwise-max-retries",
        type=int,
        default=1,
        help="Retries per paragraph pairwise query (when enabled).",
    )
    parser.add_argument(
        "--paragraph-compressed-snippet-limit",
        type=int,
        default=180,
        help="Per-paragraph snippet length used by compressed direct paragraph scoring.",
    )
    parser.add_argument(
        "--debug-log",
        default="pairwise_debug.log",
        help="Path to write raw pairwise model responses for debugging",
    )
    args = parser.parse_args()
    append_run_separator(args.debug_log, args)

    text = read_pdf_text(args.pdf)
    citations, content = extract_citations_by_section(text, DEFAULT_SECTIONS)
    citation_importance, section_importance = assign_importance_scores(
        content_dict=content,
        citations_dict=citations,
        model=args.model,
        host=args.host,
        n_samples=max(1, args.n_samples),
        temperature=max(0.0, args.temperature),
        max_retries=max(1, args.max_retries),
        paragraph_direct_max_tokens=max(0, args.paragraph_direct_max_tokens),
        paragraph_pairwise_max_items=max(1, args.paragraph_pairwise_max_items),
        paragraph_pairwise_n_samples=max(1, args.paragraph_pairwise_n_samples),
        paragraph_pairwise_max_retries=max(1, args.paragraph_pairwise_max_retries),
        paragraph_compressed_snippet_limit=max(60, args.paragraph_compressed_snippet_limit),
        debug_log_path=args.debug_log,
    )

    citation_path = f"{args.output1}_citation_scores.json"
    section_path = f"{args.output2}_section_scores.json"

    with open(citation_path, "w") as f:
        json.dump(citation_importance, f, indent=2)

    with open(section_path, "w") as f:
        json.dump(section_importance, f, indent=2)

    print("\n" + "=" * 50)
    print("FINAL CITATION SCORES (AGGREGATED ACROSS SECTIONS):")
    print("=" * 50)
    sorted_citations = sorted(citation_importance.items(), key=lambda x: x[1]["score"], reverse=True)
    for citation, info in sorted_citations[:10]:
        print(f"{citation}: {info['score']:.4f} ({info['mentions']} mentions)")

    print("\n" + "=" * 50)
    print("SECTION HIERARCHY WITH SCORES:")
    print("=" * 50)
    print_section_hierarchy(section_importance)


if __name__ == "__main__":
    main()
