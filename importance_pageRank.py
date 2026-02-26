import argparse
import json
import re
from collections import defaultdict
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

TECHNICAL_SYSTEM_PROMPT = (
    "You are an expert academic reviewer. "
    "Score each paragraph only for technical contribution. "
    "Ignore citation popularity or influence. "
    "Return JSON only."
)

OVERALL_SYSTEM_PROMPT = (
    "You are an expert academic reviewer. "
    "Score each paragraph for total contribution to the paper. "
    "Include both intrinsic technical contribution and citation-added value. "
    "Return JSON only."
)

TECHNICAL_USER_PROMPT_TEMPLATE = """Task: assign a technical contribution score T(p) in [0, 1] for each paragraph.

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

OVERALL_USER_PROMPT_TEMPLATE = """Task: assign an overall paragraph score S(p) in [0, 1].

Scoring rubric:
- Higher S(p): paragraph materially advances the paper's contribution.
- Consider both:
  1) intrinsic technical contribution from the paragraph text itself, and
  2) extra value added by citations used in that paragraph.
- Use lower scores for background, transitions, and low-impact detail.

Paragraphs (id -> text snippet):
{paragraphs_json}

Return ONLY JSON mapping every paragraph id to a float in [0, 1].
Example:
{{
  "p1": 0.88,
  "p2": 0.41
}}
"""

PROMPT_CATALOG = {
    "technical_system_prompt": TECHNICAL_SYSTEM_PROMPT,
    "technical_user_prompt_template": TECHNICAL_USER_PROMPT_TEMPLATE,
    "overall_system_prompt": OVERALL_SYSTEM_PROMPT,
    "overall_user_prompt_template": OVERALL_USER_PROMPT_TEMPLATE,
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
        section_text: str,
        sections_dict: Dict[str, Any],
        section_names_list: List[str],
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
        f"paper_id={args.paper_id} alpha={args.alpha} beta={args.beta} "
        f"lambda_weight={args.lambda_weight} n_samples={max(1, args.n_samples)} "
        f"temperature={max(0.0, args.temperature)} max_retries={max(1, args.max_retries)} "
        f"{separator}"
    )
    append_debug_log(debug_log_path, entry)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_text_into_paragraphs(text: str) -> List[str]:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    paragraphs = [normalize_for_match(p) for p in re.split(r"\n\s*\n+", normalized) if normalize_for_match(p)]
    if len(paragraphs) > 1:
        return paragraphs

    single = normalize_for_match(normalized)
    if not single:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", single) if s.strip()]
    if len(sentences) <= 3:
        return [single]

    chunk_size = 5
    return [" ".join(sentences[i : i + chunk_size]).strip() for i in range(0, len(sentences), chunk_size)]


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


def flatten_content_to_text(content: Any, limit: int = 700) -> str:
    if isinstance(content, str):
        return content[:limit]
    if isinstance(content, dict):
        chunks: List[str] = []
        total_chars = 0
        for value in content.values():
            chunk = flatten_content_to_text(value, limit=limit)
            if chunk:
                chunks.append(chunk)
                total_chars += len(chunk)
            if total_chars >= limit:
                break
        merged = " ".join(chunks)
        return merged[:limit]
    return ""


def chunked(items: List[Any], chunk_size: int) -> List[List[Any]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), max(1, chunk_size))]


def collect_leaf_sections(
    content_tree: Dict[str, Any],
    citations_tree: Dict[str, Any],
    path: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    leaves: List[Dict[str, Any]] = []
    current_path = path or []

    for section_name, section_content in content_tree.items():
        next_path = current_path + [section_name]
        section_citations = citations_tree.get(section_name, {}) if isinstance(citations_tree, dict) else {}

        if isinstance(section_content, dict) and section_content:
            leaves.extend(collect_leaf_sections(section_content, section_citations, next_path))
        else:
            leaves.append(
                {
                    "section_path": next_path,
                    "section_name": " > ".join(next_path),
                    "content": section_content if isinstance(section_content, str) else flatten_content_to_text(section_content, 6000),
                    "citations": section_citations if isinstance(section_citations, dict) else {},
                }
            )

    return leaves


def match_citations_to_paragraphs(
    paragraphs: List[str],
    section_citations: Dict[str, Any],
) -> Dict[int, List[Tuple[str, str, str]]]:
    if not paragraphs:
        return {}

    paragraph_norms = [normalize_for_match(text) for text in paragraphs]
    paragraph_tokens = [set(re.findall(r"[a-z0-9]+", p.lower())) for p in paragraph_norms]
    mention_buckets: Dict[int, List[Tuple[str, str, str]]] = {i: [] for i in range(len(paragraphs))}

    for citation_block, context_value in section_citations.items():
        contexts = context_value if isinstance(context_value, list) else [context_value]
        for context in contexts:
            context_str = str(context)
            context_norm = normalize_for_match(context_str)
            target_idx: Optional[int] = None

            if context_norm:
                for idx, para_norm in enumerate(paragraph_norms):
                    if context_norm in para_norm:
                        target_idx = idx
                        break

            if target_idx is None and context_norm:
                context_tokens = set(re.findall(r"[a-z0-9]+", context_norm.lower()))
                if context_tokens:
                    best_idx = None
                    best_overlap = -1
                    for idx, p_tokens in enumerate(paragraph_tokens):
                        overlap = len(context_tokens & p_tokens)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_idx = idx
                    if best_idx is not None and best_overlap > 0:
                        target_idx = best_idx

            if target_idx is None:
                target_idx = 0

            for citation in split_citation_block(citation_block):
                mention_buckets[target_idx].append((citation, citation_block, context_str))

    return mention_buckets


def build_paragraph_records(
    content_dict: Dict[str, Any],
    citations_dict: Dict[str, Any],
    paper_id: str,
) -> List[Dict[str, Any]]:
    paragraph_records: List[Dict[str, Any]] = []
    leaves = collect_leaf_sections(content_dict, citations_dict)

    for leaf in leaves:
        section_name = leaf["section_name"]
        raw_text = leaf["content"] or ""
        section_citations = leaf["citations"] if isinstance(leaf["citations"], dict) else {}

        paragraphs = split_text_into_paragraphs(raw_text)
        if not paragraphs:
            normalized = normalize_for_match(raw_text)
            if normalized:
                paragraphs = [normalized]
            else:
                continue

        mention_buckets = match_citations_to_paragraphs(paragraphs, section_citations)

        for idx, paragraph_text in enumerate(paragraphs):
            paragraph_id = f"{paper_id}::{section_name}::p{idx + 1}"
            mentions = mention_buckets.get(idx, [])
            citations = [citation for citation, _, _ in mentions]
            paragraph_records.append(
                {
                    "paragraph_id": paragraph_id,
                    "paper_id": paper_id,
                    "section_path": leaf["section_path"],
                    "section_name": section_name,
                    "paragraph_index": idx + 1,
                    "text": paragraph_text,
                    "citation_count": len(citations),
                    "citations": citations,
                    "mentions": mentions,
                }
            )

    return paragraph_records


def score_paragraph_signal(
    client: Client,
    paragraphs: List[Dict[str, Any]],
    model: str,
    n_samples: int,
    temperature: float,
    max_retries: int,
    batch_size: int,
    snippet_limit: int,
    debug_log_path: str,
    system_prompt: str,
    user_prompt_template: str,
    log_tag: str,
) -> Dict[str, float]:
    if not paragraphs:
        return {}

    sample_count = max(1, n_samples)
    batches = chunked(paragraphs, max(1, batch_size))
    per_sample_scores: List[Dict[str, float]] = []

    for sample_idx in range(sample_count):
        sample_scores: Dict[str, float] = {}
        current_temp = min(1.0, max(0.0, temperature) + 0.05 * sample_idx)

        for batch_idx, batch in enumerate(batches):
            snippets = {
                item["paragraph_id"]: item["text"][: max(80, snippet_limit)]
                for item in batch
            }
            user_prompt = user_prompt_template.format(paragraphs_json=json.dumps(snippets, indent=2))

            parsed: Dict[str, Any] = {}
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
                        f"batch={batch_idx + 1}/{len(batches)} "
                        f"attempt={attempt + 1}/{max(1, max_retries)} "
                        f"temperature={current_temp}\n{raw_response}\n"
                    ),
                )

                parsed = parse_json_response(raw_response)
                if parsed:
                    break

            for item in batch:
                pid = item["paragraph_id"]
                raw_score = safe_float(parsed.get(pid), -1.0)
                if 0.0 <= raw_score <= 1.0:
                    sample_scores[pid] = raw_score
                else:
                    # Neutral fallback if this paragraph is missing/invalid in the model response.
                    sample_scores[pid] = 0.5

        per_sample_scores.append(sample_scores)

    averaged_scores: Dict[str, float] = {}
    for item in paragraphs:
        pid = item["paragraph_id"]
        avg = sum(sample.get(pid, 0.5) for sample in per_sample_scores) / sample_count
        averaged_scores[pid] = clamp_unit(avg)

    return averaged_scores


def score_technical_contributions(
    client: Client,
    paragraphs: List[Dict[str, Any]],
    model: str,
    n_samples: int,
    temperature: float,
    max_retries: int,
    batch_size: int,
    snippet_limit: int,
    debug_log_path: str,
) -> Dict[str, float]:
    return score_paragraph_signal(
        client=client,
        paragraphs=paragraphs,
        model=model,
        n_samples=n_samples,
        temperature=temperature,
        max_retries=max_retries,
        batch_size=batch_size,
        snippet_limit=snippet_limit,
        debug_log_path=debug_log_path,
        system_prompt=TECHNICAL_SYSTEM_PROMPT,
        user_prompt_template=TECHNICAL_USER_PROMPT_TEMPLATE,
        log_tag="technical_batch",
    )


def score_overall_contributions(
    client: Client,
    paragraphs: List[Dict[str, Any]],
    model: str,
    n_samples: int,
    temperature: float,
    max_retries: int,
    batch_size: int,
    snippet_limit: int,
    debug_log_path: str,
) -> Dict[str, float]:
    return score_paragraph_signal(
        client=client,
        paragraphs=paragraphs,
        model=model,
        n_samples=n_samples,
        temperature=temperature,
        max_retries=max_retries,
        batch_size=batch_size,
        snippet_limit=snippet_limit,
        debug_log_path=debug_log_path,
        system_prompt=OVERALL_SYSTEM_PROMPT,
        user_prompt_template=OVERALL_USER_PROMPT_TEMPLATE,
        log_tag="overall_batch",
    )


def compute_citation_local_scores(
    paragraphs: List[Dict[str, Any]],
    technical_scores: Dict[str, float],
    overall_scores: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Split each paragraph total score into:
    - T(p): intrinsic technical contribution (from technical_scores)
    - C(p): citation-added value (from total minus technical)
    with T(p) + C(p) = total paragraph score.
    Then distribute C(p) to citation blocks (importance_score.py style).
    """
    if not paragraphs:
        return {}, {}, {}

    paragraph_citation_value: Dict[str, float] = {}
    paragraph_total: Dict[str, float] = {}
    citation_credit: Dict[str, float] = {}

    for paragraph in paragraphs:
        pid = paragraph["paragraph_id"]
        mentions = paragraph.get("mentions", [])

        t_val = clamp_unit(safe_float(technical_scores.get(pid), 0.0))
        overall_val = clamp_unit(safe_float(overall_scores.get(pid), t_val))
        total_val = max(t_val, overall_val)

        # Citation-added value is the part beyond intrinsic technical value.
        c_val = max(0.0, total_val - t_val)
        # No citation mentions -> no citation-added component for this paragraph.
        if not mentions:
            c_val = 0.0
            total_val = t_val

        paragraph_citation_value[pid] = c_val
        paragraph_total[pid] = total_val

        if not mentions or c_val <= 0.0:
            continue

        per_mention = c_val / len(mentions)
        for citation, _, _ in mentions:
            citation_credit[citation] = citation_credit.get(citation, 0.0) + per_mention

    return paragraph_citation_value, citation_credit, paragraph_total


def aggregate_section_local_scores(
    paragraphs: List[Dict[str, Any]],
    technical_scores: Dict[str, float],
    citation_scores: Dict[str, float],
) -> Dict[str, Any]:
    section_to_paragraphs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for paragraph in paragraphs:
        section_to_paragraphs[paragraph["section_name"]].append(paragraph)

    section_scores: Dict[str, Any] = {}
    for section_name, section_items in section_to_paragraphs.items():
        if not section_items:
            continue
        weight = 1.0 / len(section_items)
        weights = {item["paragraph_id"]: weight for item in section_items}

        t_section = sum(weights[p["paragraph_id"]] * technical_scores.get(p["paragraph_id"], 0.0) for p in section_items)
        c_section = sum(weights[p["paragraph_id"]] * citation_scores.get(p["paragraph_id"], 0.0) for p in section_items)

        section_scores[section_name] = {
            "weights": weights,
            "paragraph_count": len(section_items),
            "technical_local": t_section,
            "citation_local": c_section,
        }

    return section_scores


def aggregate_paper_local_scores(
    paragraphs: List[Dict[str, Any]],
    technical_scores: Dict[str, float],
    citation_scores: Dict[str, float],
    paper_id: str,
) -> Dict[str, float]:
    paper_paragraphs = [p for p in paragraphs if p.get("paper_id") == paper_id]
    if not paper_paragraphs:
        return {"technical_local": 0.0, "citation_local": 0.0}

    weight = 1.0 / len(paper_paragraphs)
    t_paper = sum(weight * technical_scores.get(p["paragraph_id"], 0.0) for p in paper_paragraphs)
    c_paper = sum(weight * citation_scores.get(p["paragraph_id"], 0.0) for p in paper_paragraphs)
    return {"technical_local": t_paper, "citation_local": c_paper}


def load_citation_graph(
    graph_path: str,
    paper_id: str,
) -> Tuple[List[str], List[Tuple[str, str]], Dict[str, float], Dict[str, float]]:
    if not graph_path:
        return [paper_id], [(paper_id, paper_id)], {}, {}

    with open(graph_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    nodes = set()
    edges: List[Tuple[str, str]] = []

    if isinstance(graph_data, dict) and "edges" in graph_data:
        raw_edges = graph_data.get("edges", [])
        for edge in raw_edges:
            if isinstance(edge, list) and len(edge) == 2:
                src, dst = str(edge[0]), str(edge[1])
                edges.append((src, dst))
                nodes.add(src)
                nodes.add(dst)
        for node in graph_data.get("nodes", []):
            nodes.add(str(node))
    elif isinstance(graph_data, dict) and "adjacency" in graph_data:
        adjacency = graph_data.get("adjacency", {})
        if isinstance(adjacency, dict):
            for src, dsts in adjacency.items():
                src_s = str(src)
                nodes.add(src_s)
                if isinstance(dsts, list):
                    for dst in dsts:
                        dst_s = str(dst)
                        nodes.add(dst_s)
                        edges.append((src_s, dst_s))
    elif isinstance(graph_data, dict):
        # Also accept plain adjacency mapping {"paper_a": ["paper_b", ...], ...}.
        for src, dsts in graph_data.items():
            if src in ("technical", "citation", "nodes"):
                continue
            src_s = str(src)
            nodes.add(src_s)
            if isinstance(dsts, list):
                for dst in dsts:
                    dst_s = str(dst)
                    nodes.add(dst_s)
                    edges.append((src_s, dst_s))

    technical_seed = {}
    citation_seed = {}
    if isinstance(graph_data, dict):
        if isinstance(graph_data.get("technical"), dict):
            technical_seed = {str(k): safe_float(v, 0.0) for k, v in graph_data["technical"].items()}
        if isinstance(graph_data.get("citation"), dict):
            citation_seed = {str(k): safe_float(v, 0.0) for k, v in graph_data["citation"].items()}

    if paper_id not in nodes:
        nodes.add(paper_id)
    if not edges:
        edges = [(paper_id, paper_id)]

    return sorted(nodes), edges, technical_seed, citation_seed


def propagate_rank(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    local_scores: Dict[str, float],
    damping: float,
    max_iter: int,
    tol: float,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    damping = max(0.0, min(0.999999, damping))
    n = len(nodes)
    if n == 0:
        return {}, {}, {}

    incoming: Dict[str, List[str]] = {u: [] for u in nodes}
    out_degree: Dict[str, int] = {u: 0 for u in nodes}

    for src, dst in edges:
        if src not in incoming:
            incoming[src] = []
            out_degree[src] = 0
        if dst not in incoming:
            incoming[dst] = []
            out_degree[dst] = 0
        incoming[dst].append(src)
        out_degree[src] = out_degree.get(src, 0) + 1

    scores = {u: safe_float(local_scores.get(u), 0.0) for u in incoming}

    for _ in range(max(1, max_iter)):
        dangling_mass = sum(scores[v] for v, deg in out_degree.items() if deg == 0)
        next_scores: Dict[str, float] = {}

        for u in incoming:
            inherited = 0.0
            for v in incoming[u]:
                deg_v = out_degree.get(v, 0)
                if deg_v > 0:
                    inherited += scores.get(v, 0.0) / deg_v
            inherited += dangling_mass / len(incoming)
            next_scores[u] = (1.0 - damping) * safe_float(local_scores.get(u), 0.0) + damping * inherited

        delta = max(abs(next_scores[u] - scores.get(u, 0.0)) for u in next_scores)
        scores = next_scores
        if delta <= tol:
            break

    # Final decomposition terms from Eq. style update at the converged point.
    dangling_mass = sum(scores[v] for v, deg in out_degree.items() if deg == 0)
    kept = {u: (1.0 - damping) * safe_float(local_scores.get(u), 0.0) for u in incoming}
    inherited_terms: Dict[str, float] = {}
    for u in incoming:
        inherited = 0.0
        for v in incoming[u]:
            deg_v = out_degree.get(v, 0)
            if deg_v > 0:
                inherited += scores.get(v, 0.0) / deg_v
        inherited += dangling_mass / len(incoming)
        inherited_terms[u] = damping * inherited

    return scores, kept, inherited_terms


def redistribute_paper_rank_to_paragraphs(
    paragraphs: List[Dict[str, Any]],
    paper_id: str,
    paper_rank: float,
    local_scores: Dict[str, float],
) -> Dict[str, float]:
    paper_paragraphs = [p for p in paragraphs if p.get("paper_id") == paper_id]
    if not paper_paragraphs:
        return {}

    denominator = sum(max(0.0, local_scores.get(p["paragraph_id"], 0.0)) for p in paper_paragraphs)
    redistributed: Dict[str, float] = {}

    if denominator <= 0.0:
        equal = paper_rank / len(paper_paragraphs)
        for p in paper_paragraphs:
            redistributed[p["paragraph_id"]] = equal
        return redistributed

    for p in paper_paragraphs:
        pid = p["paragraph_id"]
        redistributed[pid] = paper_rank * (max(0.0, local_scores.get(pid, 0.0)) / denominator)
    return redistributed


def aggregate_citation_scores(
    paragraphs: List[Dict[str, Any]],
    tr_scores: Dict[str, float],
    cr_scores: Dict[str, float],
    lambda_weight: float,
) -> Dict[str, Any]:
    citation_scores: Dict[str, Any] = {}

    for paragraph in paragraphs:
        pid = paragraph["paragraph_id"]
        mentions = paragraph.get("mentions", [])
        if not mentions:
            continue

        mention_count = len(mentions)
        tr_share = tr_scores.get(pid, 0.0) / mention_count
        cr_share = cr_scores.get(pid, 0.0) / mention_count

        for citation, source_block, context in mentions:
            if citation not in citation_scores:
                citation_scores[citation] = {
                    "technical_rank": 0.0,
                    "citation_rank": 0.0,
                    "combined_score": 0.0,
                    "mentions": 0,
                    "occurrences": [],
                }

            citation_scores[citation]["technical_rank"] += tr_share
            citation_scores[citation]["citation_rank"] += cr_share
            citation_scores[citation]["mentions"] += 1
            citation_scores[citation]["occurrences"].append(
                {
                    "section": paragraph.get("section_name"),
                    "paragraph_id": pid,
                    "source_block": source_block,
                    "context": context,
                }
            )

    for citation in citation_scores:
        tr_val = citation_scores[citation]["technical_rank"]
        cr_val = citation_scores[citation]["citation_rank"]
        citation_scores[citation]["combined_score"] = lambda_weight * tr_val + (1.0 - lambda_weight) * cr_val

    return citation_scores


def run_two_channel_pagerank(
    content_dict: Dict[str, Any],
    citations_dict: Dict[str, Any],
    model: str,
    host: str,
    paper_id: str,
    citation_graph_path: str,
    alpha: float,
    beta: float,
    lambda_weight: float,
    n_samples: int,
    temperature: float,
    max_retries: int,
    batch_size: int,
    snippet_limit: int,
    pagerank_max_iter: int,
    pagerank_tol: float,
    debug_log_path: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    client = Client(host=host)

    paragraphs = build_paragraph_records(content_dict, citations_dict, paper_id=paper_id)
    if not paragraphs:
        return {}, {}, {}, PROMPT_CATALOG

    local_t = score_technical_contributions(
        client=client,
        paragraphs=paragraphs,
        model=model,
        n_samples=n_samples,
        temperature=temperature,
        max_retries=max_retries,
        batch_size=batch_size,
        snippet_limit=snippet_limit,
        debug_log_path=debug_log_path,
    )
    local_overall = score_overall_contributions(
        client=client,
        paragraphs=paragraphs,
        model=model,
        n_samples=n_samples,
        temperature=temperature,
        max_retries=max_retries,
        batch_size=batch_size,
        snippet_limit=snippet_limit,
        debug_log_path=debug_log_path,
    )
    local_c, citation_local_credit, local_total = compute_citation_local_scores(paragraphs, local_t, local_overall)

    section_local = aggregate_section_local_scores(paragraphs, local_t, local_c)
    paper_local = aggregate_paper_local_scores(paragraphs, local_t, local_c, paper_id=paper_id)

    nodes, edges, graph_t_seed, graph_c_seed = load_citation_graph(citation_graph_path, paper_id=paper_id)
    paper_local_t = {node: graph_t_seed.get(node, 0.0) for node in nodes}
    paper_local_c = {node: graph_c_seed.get(node, 0.0) for node in nodes}
    paper_local_t[paper_id] = paper_local["technical_local"]
    paper_local_c[paper_id] = paper_local["citation_local"]

    tr_paper, tr_kept, tr_inherited = propagate_rank(
        nodes=nodes,
        edges=edges,
        local_scores=paper_local_t,
        damping=alpha,
        max_iter=pagerank_max_iter,
        tol=pagerank_tol,
    )
    cr_paper, cr_kept, cr_inherited = propagate_rank(
        nodes=nodes,
        edges=edges,
        local_scores=paper_local_c,
        damping=beta,
        max_iter=pagerank_max_iter,
        tol=pagerank_tol,
    )

    paragraph_tr = redistribute_paper_rank_to_paragraphs(
        paragraphs=paragraphs,
        paper_id=paper_id,
        paper_rank=tr_paper.get(paper_id, 0.0),
        local_scores=local_t,
    )
    paragraph_cr = redistribute_paper_rank_to_paragraphs(
        paragraphs=paragraphs,
        paper_id=paper_id,
        paper_rank=cr_paper.get(paper_id, 0.0),
        local_scores=local_c,
    )

    lambda_weight = max(0.0, min(1.0, lambda_weight))

    paragraph_scores: Dict[str, Any] = {}
    for p in paragraphs:
        pid = p["paragraph_id"]
        tr_val = paragraph_tr.get(pid, 0.0)
        cr_val = paragraph_cr.get(pid, 0.0)
        paragraph_scores[pid] = {
            "paper_id": p["paper_id"],
            "section": p["section_name"],
            "paragraph_index": p["paragraph_index"],
            "text": p["text"],
            "local_total": local_total.get(pid, local_t.get(pid, 0.0) + local_c.get(pid, 0.0)),
            "local_technical": local_t.get(pid, 0.0),
            "local_citation": local_c.get(pid, 0.0),
            "technical_rank": tr_val,
            "citation_rank": cr_val,
            "combined_score": lambda_weight * tr_val + (1.0 - lambda_weight) * cr_val,
            "citation_count": p.get("citation_count", 0),
            "citations": p.get("citations", []),
        }

    for section_name, section_payload in section_local.items():
        pids = list(section_payload["weights"].keys())
        section_payload["technical_rank"] = sum(paragraph_scores[pid]["technical_rank"] for pid in pids)
        section_payload["citation_rank"] = sum(paragraph_scores[pid]["citation_rank"] for pid in pids)
        section_payload["combined_score"] = (
            lambda_weight * section_payload["technical_rank"]
            + (1.0 - lambda_weight) * section_payload["citation_rank"]
        )

    paper_scores = {
        "paper_id": paper_id,
        "alpha": alpha,
        "beta": beta,
        "lambda_weight": lambda_weight,
        "local": {
            "technical": paper_local["technical_local"],
            "citation": paper_local["citation_local"],
            "overall_total": paper_local["technical_local"] + paper_local["citation_local"],
        },
        "technical_rank": tr_paper.get(paper_id, 0.0),
        "citation_rank": cr_paper.get(paper_id, 0.0),
        "combined_score": (
            lambda_weight * tr_paper.get(paper_id, 0.0)
            + (1.0 - lambda_weight) * cr_paper.get(paper_id, 0.0)
        ),
        "decomposition": {
            "technical": {
                "kept_local": tr_kept.get(paper_id, 0.0),
                "inherited_flow": tr_inherited.get(paper_id, 0.0),
            },
            "citation": {
                "kept_local": cr_kept.get(paper_id, 0.0),
                "inherited_flow": cr_inherited.get(paper_id, 0.0),
            },
        },
        "graph_nodes": nodes,
        "graph_edges": edges,
        "local_citation_credit_by_citation": citation_local_credit,
    }

    citation_scores = aggregate_citation_scores(
        paragraphs=paragraphs,
        tr_scores=paragraph_tr,
        cr_scores=paragraph_cr,
        lambda_weight=lambda_weight,
    )

    return citation_scores, section_local, {"paper": paper_scores, "paragraphs": paragraph_scores}, PROMPT_CATALOG


def print_top_scores(citation_scores: Dict[str, Any], paragraph_scores: Dict[str, Any], limit: int = 10) -> None:
    print("\n" + "=" * 50)
    print("TOP CITATIONS (BY COMBINED SCORE):")
    print("=" * 50)
    sorted_citations = sorted(citation_scores.items(), key=lambda x: x[1].get("combined_score", 0.0), reverse=True)
    for citation, payload in sorted_citations[:limit]:
        print(
            f"{citation}: combined={payload.get('combined_score', 0.0):.4f}, "
            f"TR={payload.get('technical_rank', 0.0):.4f}, "
            f"CR={payload.get('citation_rank', 0.0):.4f}, "
            f"mentions={payload.get('mentions', 0)}"
        )

    print("\n" + "=" * 50)
    print("TOP PARAGRAPHS (BY COMBINED SCORE):")
    print("=" * 50)
    sorted_paragraphs = sorted(paragraph_scores.items(), key=lambda x: x[1].get("combined_score", 0.0), reverse=True)
    for paragraph_id, payload in sorted_paragraphs[:limit]:
        print(
            f"{paragraph_id}: combined={payload.get('combined_score', 0.0):.4f}, "
            f"TR={payload.get('technical_rank', 0.0):.4f}, "
            f"CR={payload.get('citation_rank', 0.0):.4f}, "
            f"section={payload.get('section', '')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-channel PageRank-style contribution scoring.")
    parser.add_argument("--output1", required=True, help="Prefix for citation output file")
    parser.add_argument("--output2", required=True, help="Prefix for section output file")
    parser.add_argument("--pdf", default=DEFAULT_PDF_PATH, help="Path to PDF file")
    parser.add_argument("--paper-id", default=DEFAULT_PAPER_ID, help="Paper node id for graph propagation")
    parser.add_argument("--citation-graph", default="", help="Optional JSON graph file (nodes/edges or adjacency)")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name")
    parser.add_argument("--host", default="localhost:11434", help="Ollama host")
    parser.add_argument("--alpha", type=float, default=0.85, help="Technical propagation parameter")
    parser.add_argument("--beta", type=float, default=0.85, help="Citation propagation parameter")
    parser.add_argument("--lambda-weight", type=float, default=0.5, help="Blend weight for combined score")
    parser.add_argument("--n-samples", type=int, default=3, help="Number of LLM samples for T(p)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Base LLM temperature")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries for each LLM batch")
    parser.add_argument("--batch-size", type=int, default=12, help="Paragraphs per LLM scoring batch")
    parser.add_argument("--snippet-limit", type=int, default=420, help="Max chars per paragraph snippet")
    parser.add_argument("--pagerank-max-iter", type=int, default=100, help="Max PageRank iterations")
    parser.add_argument("--pagerank-tol", type=float, default=1e-9, help="PageRank convergence tolerance")
    parser.add_argument("--debug-log", default="pagerank_debug.log", help="Path for raw model/debug logs")
    parser.add_argument(
        "--prompts-output",
        default="",
        help="Optional path to save the exact prompt strings as JSON",
    )
    args = parser.parse_args()

    append_run_separator(args.debug_log, args)

    text = read_pdf_text(args.pdf)
    citations, content = extract_citations_by_section(text, DEFAULT_SECTIONS)

    citation_scores, section_scores, rank_payload, prompt_catalog = run_two_channel_pagerank(
        content_dict=content,
        citations_dict=citations,
        model=args.model,
        host=args.host,
        paper_id=args.paper_id,
        citation_graph_path=args.citation_graph,
        alpha=max(0.0, min(0.999999, args.alpha)),
        beta=max(0.0, min(0.999999, args.beta)),
        lambda_weight=max(0.0, min(1.0, args.lambda_weight)),
        n_samples=max(1, args.n_samples),
        temperature=max(0.0, args.temperature),
        max_retries=max(1, args.max_retries),
        batch_size=max(1, args.batch_size),
        snippet_limit=max(80, args.snippet_limit),
        pagerank_max_iter=max(1, args.pagerank_max_iter),
        pagerank_tol=max(1e-15, args.pagerank_tol),
        debug_log_path=args.debug_log,
    )

    citation_path = f"{args.output1}_citation_scores.json"
    section_path = f"{args.output2}_section_scores.json"
    paragraph_path = f"{args.output2}_paragraph_scores.json"
    paper_path = f"{args.output2}_paper_scores.json"

    with open(citation_path, "w", encoding="utf-8") as f:
        json.dump(citation_scores, f, indent=2)

    with open(section_path, "w", encoding="utf-8") as f:
        json.dump(section_scores, f, indent=2)

    with open(paragraph_path, "w", encoding="utf-8") as f:
        json.dump(rank_payload.get("paragraphs", {}), f, indent=2)

    with open(paper_path, "w", encoding="utf-8") as f:
        json.dump(rank_payload.get("paper", {}), f, indent=2)

    if args.prompts_output:
        with open(args.prompts_output, "w", encoding="utf-8") as f:
            json.dump(prompt_catalog, f, indent=2)

    print(f"Saved citation scores: {citation_path}")
    print(f"Saved section scores:  {section_path}")
    print(f"Saved paragraph scores:{paragraph_path}")
    print(f"Saved paper scores:    {paper_path}")

    print_top_scores(citation_scores, rank_payload.get("paragraphs", {}), limit=10)


if __name__ == "__main__":
    main()
