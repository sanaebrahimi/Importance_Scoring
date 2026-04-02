import argparse
import ast
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import PyPDF2
except ImportError:  # pragma: no cover - fallback for environments with pypdf only
    import pypdf as PyPDF2
from ollama import Client


DEFAULT_PDF_PATH = "adv_res_paper.pdf"
DEFAULT_PAPER_ID = "target_paper"

DEFAULT_SECTIONS = {
    "Introduction": None,
    "System Overview": None,
    "Team of Agents": None,
    "CrS-Aware Aggregation": None,
    "Learning Credibility Scores On-The-Fly": {
        "Calculating the agent contributions": None,
        "Updating the CrS values": None,
    },
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
    "You are an expert academic reviewer evaluating the structure of a scientific paper. "
    "Your task is to allocate a parent score across its child segments based only on their technical contribution "
    "to the paper. "
    "Evaluate how much each segment contributes to the core scientific content of the paper, such as methods, "
    "algorithms, theory, experimental design, results, and technical analysis. "
    "Do not consider citation value at this stage. Only evaluate the intrinsic technical importance of the segment. "
    "Use the entire parent content as context when evaluating children. "
    "Treat the parent score as a fixed budget that must be fully distributed. "
    "Scores must be non-negative. "
    "Segments containing the core technical ideas or results should receive higher scores. "
    "Segments that are mainly background, motivation, transitions, or narrative should receive lower scores. "
    "Return plain text lines only. Do not output JSON."
)

SECTION_DIRECT_USER_PROMPT_TEMPLATE = """Parent segment:
{parent_name}

Parent score to distribute:
{parent_score}

Parent content (context):
{parent_content}

Task:
Divide the parent score among the following child segments according to their technical importance to the paper.

Guidelines:
- Prioritize technical novelty, algorithmic description, experimental findings, or key analysis.
- Lower scores should go to background explanations, transitions, or descriptive text.
- Consider how much the technical understanding of the paper would suffer if the segment were removed.

Child segments (id -> name and excerpt):
{items}

Output format (plain text only):
Any clear one-line-per-child list is acceptable.
Separators such as `:`, `-`, `=`, or `->` are all fine.
Line order matters more than exact counter formatting.

Example:
1: 0.32
2: 0.18

The scores must sum to {parent_score}.

Do not output JSON.
Do not include explanations.
"""

CITATION_SPLIT_SYSTEM_PROMPT = (
    "You are an expert academic reviewer. "
    "Your task is to distribute a paragraph's citation score among the citations that appear in that paragraph, "
    "based on how much each cited work contributes to the paragraph's claims or grounding. "
    "Treat the parent score as a fixed budget that must be fully distributed. "
    "Return plain text lines only."
)

CITATION_SPLIT_USER_PROMPT_TEMPLATE = """Paragraph id:
{paragraph_id}

Paragraph citation score to distribute: {paragraph_citation_score}

Paragraph text:
{paragraph_text}

Task:
- Divide the paragraph citation score among the citations below.
- Higher share:
  citation directly supports the paragraph's claim
  citation provides key comparison or baseline
  citation introduces a method the paragraph builds upon
- Lower share:
  citation is peripheral or only mentioned briefly
- Scores must be non-negative.
- Prefer scores that sum to the paragraph citation score.

Citation entries (citation_id -> citation and context):
{citations_json}

Output format (plain text only):
Any clear one-line-per-citation list is acceptable.
Separators such as `:`, `-`, `=`, or `->` are all fine.
Line order matters more than exact counter formatting.

Example:
1: 0.06
2: 0.02

Do not output JSON.
Do not include explanations.
"""

PARAGRAPH_CHANNEL_SPLIT_SYSTEM_PROMPT = (
    "You are an expert academic reviewer. "
    "For each paragraph, split its given total score into technical and citation-added components. "
    "Technical = intrinsic technical value of the paragraph while ignoring citations. "
    "Citation = added value contributed by cited prior work in that paragraph. "
    "Treat the parent score as a fixed budget that must be fully distributed. "
    "Return plain text lines only."
)

PARAGRAPH_CHANNEL_SPLIT_USER_PROMPT_TEMPLATE = """Task:
For each paragraph below, split the provided total paragraph score into:
- technical score
- citation score

Rules:
- For each paragraph, technical + citation should equal the provided total_score.
- Higher technical score:
  paragraph introduces method, algorithm, theory, dataset, experiment, or key result
- Higher citation score:
  cited work materially strengthens grounding, comparison, dependency, or evidence
- If has_citations is false, citation should be 0 and technical should equal total_score.
- Use non-negative values.

Paragraph entries (paragraph_id -> details):
{paragraphs_json}

Output format (plain text only):
paragraph_id: technical=<float>, citation=<float>

Example:
Paragraph1: technical=0.21, citation=0.04
Paragraph2: technical=0.13, citation=0.00

Do not output JSON.
Do not include explanations.
"""

PROMPT_CATALOG = {
    "section_pairwise_system_prompt": SECTION_PAIRWISE_SYSTEM_PROMPT,
    "section_pairwise_user_prompt_template": SECTION_PAIRWISE_USER_PROMPT_TEMPLATE,
    "section_direct_system_prompt": SECTION_DIRECT_SYSTEM_PROMPT,
    "section_direct_user_prompt_template": SECTION_DIRECT_USER_PROMPT_TEMPLATE,
    "citation_split_system_prompt": CITATION_SPLIT_SYSTEM_PROMPT,
    "citation_split_user_prompt_template": CITATION_SPLIT_USER_PROMPT_TEMPLATE,
    "paragraph_channel_split_system_prompt": PARAGRAPH_CHANNEL_SPLIT_SYSTEM_PROMPT,
    "paragraph_channel_split_user_prompt_template": PARAGRAPH_CHANNEL_SPLIT_USER_PROMPT_TEMPLATE,
}

AUTHOR_YEAR_CITATION_PATTERN = r"\([A-Z][^)]*\d{4}[a-z]?\)"
NUMERIC_BRACKET_CITATION_PATTERN = r"\[(?:\s*\d+\s*(?:[-,;–]\s*\d+\s*)*)\]"
CITATION_BLOCK_PATTERN = rf"{AUTHOR_YEAR_CITATION_PATTERN}|{NUMERIC_BRACKET_CITATION_PATTERN}"


def load_section_assignments(assignments_path: str) -> Dict[str, Dict[str, Any]]:
    path = Path(assignments_path)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    assignments: Dict[str, Dict[str, Any]] = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = ast.literal_eval(node.value)
        if isinstance(value, dict):
            assignments[target.id] = value

    return assignments


def load_sections_from_file(assignments_path: str, variable_name: str) -> Dict[str, Any]:
    assignments = load_section_assignments(assignments_path)
    if variable_name not in assignments:
        available = ", ".join(sorted(assignments))
        raise KeyError(
            f"Sections variable '{variable_name}' not found in {assignments_path}. "
            f"Available variables: {available}"
        )
    return assignments[variable_name]


def read_pdf_text(pdf_path: str) -> str:
    pages: List[str] = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def normalize_heading_text(text: str) -> str:
    normalized = (text or "").lower()
    normalized = normalized.replace("𝜀", "epsilon").replace("ϵ", "epsilon").replace("ε", "epsilon")
    normalized = normalized.replace("&", "and")
    normalized = re.sub(r"[-‐‑–—]", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def heading_tokens_match(target: str, candidate: str) -> bool:
    if not target or not candidate:
        return False
    if candidate == target:
        return True
    if len(candidate.split()) >= 4 and target.startswith(candidate):
        return True
    if len(target.split()) >= 4 and candidate.startswith(target):
        return True

    target_tokens = target.split()
    candidate_tokens = candidate.split()
    if not target_tokens or not candidate_tokens:
        return False

    j = 0
    for token in candidate_tokens:
        if j < len(target_tokens) and token == target_tokens[j]:
            j += 1
    return j == len(target_tokens)


def trim_text_before_references(text: str) -> str:
    lines = (text or "").splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped == "references" or re.match(r"^references\d+$", stripped):
            return "\n".join(lines[:idx])
    return text


def extract_citations_by_section(text: str, sections: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract citation blocks with context and section content based on a nested section schema.
    """
    def process_section_text(section_text: str) -> Dict[str, List[str]]:
        citations_dict: Dict[str, List[str]] = {}
        section_text = section_text.replace("\n", "")
        section_text = re.sub(r"\s+", " ", section_text)

        for match in re.finditer(CITATION_BLOCK_PATTERN, section_text):
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

    trimmed_text = trim_text_before_references(text)

    def find_heading_line_offsets(full_text: str, heading_name: str) -> Tuple[int, int]:
        direct_idx = full_text.find(heading_name)
        if direct_idx != -1:
            return direct_idx, direct_idx + len(heading_name)

        target = normalize_heading_text(heading_name)
        if not target:
            return -1, -1

        lines = full_text.splitlines(keepends=True)
        offsets: List[int] = []
        running = 0
        for line in lines:
            offsets.append(running)
            running += len(line)

        for idx in range(len(lines)):
            combined = ""
            for width in range(1, 4):
                end_idx = idx + width
                if end_idx > len(lines):
                    break
                chunk = "".join(lines[idx:end_idx]).replace("\n", " ").replace("\r", " ")
                normalized_line = normalize_heading_text(chunk)
                normalized_line = re.sub(r"^\d+(?:\.\d+)*\s*", "", normalized_line)
                normalized_line = re.sub(r"^[a-z]\s+", "", normalized_line)

                combined = f"{combined} {normalized_line}".strip() if combined else normalized_line

                if heading_tokens_match(target, normalized_line) or heading_tokens_match(target, combined):
                    return offsets[idx], offsets[end_idx - 1] + len(lines[end_idx - 1])

        return -1, -1

    def extract_section_content(
        full_text: str,
        section_name: str,
        next_section_name: Optional[str] = None,
        child_section_names: Optional[List[str]] = None,
    ) -> str:
        section_start, content_start = find_heading_line_offsets(full_text, section_name)
        used_child_anchor = False

        if section_start == -1 and child_section_names:
            for child_name in child_section_names:
                child_start, child_content_start = find_heading_line_offsets(full_text, child_name)
                if child_start != -1:
                    section_start = child_start
                    content_start = child_start
                    used_child_anchor = True
                    break

        if section_start == -1:
            return ""

        if next_section_name:
            relative_next_start, _ = find_heading_line_offsets(full_text[content_start:], next_section_name)
            if relative_next_start == -1:
                section_end = len(full_text)
            else:
                section_end = content_start + relative_next_start
        else:
            section_end = len(full_text)

        if used_child_anchor:
            return full_text[content_start:section_end]
        return full_text[content_start:section_end]

    def process_sections(
        section_text: str, sections_dict: Dict[str, Any], section_names_list: List[str]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        local_citations: Dict[str, Any] = {}
        local_content: Dict[str, Any] = {}

        for i, section_name in enumerate(section_names_list):
            next_section = section_names_list[i + 1] if i + 1 < len(section_names_list) else None
            subsections = sections_dict[section_name]
            child_names = list(subsections.keys()) if isinstance(subsections, dict) and subsections else None
            content = extract_section_content(section_text, section_name, next_section, child_names)
            if not content:
                continue

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
    citations, content = process_sections(trimmed_text, sections, section_names)
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


def parse_plaintext_score_lines(response_text: str) -> List[Tuple[str, float]]:
    text = strip_code_fences(response_text)
    pairs: List[Tuple[str, float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().strip(",")
        if not line:
            continue
        line = re.sub(r"^[-*•]\s*", "", line)
        match = re.match(
            r'^"?(.{1,120}?)"?\s*(?::|=|->|=>|-|–)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\s*%?)\s*$',
            line,
        )
        if not match:
            continue
        key = match.group(1).strip()
        value = safe_float(match.group(2), -1.0)
        if value < 0.0:
            continue
        pairs.append((key, value))
    return pairs


def parse_ordered_score_values(response_text: str, allow_percentage: bool = False) -> List[float]:
    text = strip_code_fences(response_text)
    values: List[float] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().strip(",")
        if not line:
            continue
        line = re.sub(r"^[-*•]\s*", "", line)

        score_val: Optional[float] = None
        for pattern in (
            r"^\d+[.)]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\s*%?)\s*$",
            r"^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\s*%?)\s*$",
        ):
            match = re.match(pattern, line)
            if not match:
                continue
            score_val = coerce_non_negative_number(match.group(1), allow_percentage=allow_percentage)
            if score_val is not None:
                break

        if score_val is None:
            continue
        values.append(score_val)
    return values


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

    # Prefer explicit plain-text score lines like "I1: 0.2" / "Paragraph 3: 0.4".
    plaintext_pairs = parse_plaintext_score_lines(response_text)
    ordered_plaintext_values = [value for _, value in plaintext_pairs]
    ordered_score_values = parse_ordered_score_values(response_text, allow_percentage=allow_percentage)
    if len(ordered_plaintext_values) == len(expected_ids):
        return {
            expected_ids[idx]: ordered_plaintext_values[idx]
            for idx in range(len(expected_ids))
        }
    if len(ordered_score_values) == len(expected_ids):
        return {
            expected_ids[idx]: ordered_score_values[idx]
            for idx in range(len(expected_ids))
        }

    for key, value in plaintext_pairs:
        assign_from_pair(key, value)

    json_payload = parse_json_loose(response_text)
    if isinstance(json_payload, dict):
        for key, value in json_payload.items():
            assign_from_pair(key, value)

        # Accept simple array forms under common wrapper keys.
        for seq_key in ("scores", "values", "allocations", "distribution"):
            seq = json_payload.get(seq_key)
            if not isinstance(seq, list):
                continue
            if not seq:
                continue
            if all(not isinstance(entry, dict) for entry in seq):
                for idx, entry in enumerate(seq):
                    if idx >= len(expected_ids):
                        break
                    score_val = coerce_non_negative_number(entry, allow_percentage=allow_percentage)
                    if score_val is not None:
                        parsed_scores[expected_ids[idx]] = score_val

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
        if json_payload and all(not isinstance(entry, dict) for entry in json_payload):
            for idx, entry in enumerate(json_payload):
                if idx >= len(expected_ids):
                    break
                score_val = coerce_non_negative_number(entry, allow_percentage=allow_percentage)
                if score_val is not None:
                    parsed_scores[expected_ids[idx]] = score_val
        else:
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
    # Accept loose "label: value" lines.
    for match in re.finditer(
        r"([A-Za-z][A-Za-z0-9 _-]{0,80}|[A-Za-z]\d+)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        text,
    ):
        assign_from_pair(match.group(1), match.group(2))

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


def parse_paragraph_channel_split_response(
    response_text: str,
    paragraph_ids: List[str],
    alias_to_id: Optional[Dict[str, str]] = None,
) -> Dict[str, Tuple[float, float]]:
    if not paragraph_ids:
        return {}

    expected_set = set(paragraph_ids)
    norm_to_id = {normalized_key(item_id): item_id for item_id in paragraph_ids}
    if alias_to_id:
        for alias, target_id in alias_to_id.items():
            if target_id in expected_set:
                norm_to_id[normalized_key(alias)] = target_id

    def resolve_id(key: Any) -> Optional[str]:
        key_str = str(key).strip()
        if key_str in expected_set:
            return key_str
        return norm_to_id.get(normalized_key(key_str))

    def coerce_pair(technical_value: Any, citation_value: Any) -> Optional[Tuple[float, float]]:
        technical = coerce_non_negative_number(technical_value, allow_percentage=True)
        citation = coerce_non_negative_number(citation_value, allow_percentage=True)
        if technical is None or citation is None:
            return None
        return max(0.0, technical), max(0.0, citation)

    parsed_pairs: Dict[str, Tuple[float, float]] = {}

    payload = parse_json_loose(response_text)
    if isinstance(payload, dict):
        for key, value in payload.items():
            paragraph_id = resolve_id(key)
            if paragraph_id is None:
                continue
            if isinstance(value, dict):
                pair = coerce_pair(
                    value.get("technical", value.get("technical_score", value.get("tech", value.get("t")))),
                    value.get("citation", value.get("citation_score", value.get("cite", value.get("c")))),
                )
                if pair is not None:
                    parsed_pairs[paragraph_id] = pair
            elif isinstance(value, list) and len(value) >= 2:
                pair = coerce_pair(value[0], value[1])
                if pair is not None:
                    parsed_pairs[paragraph_id] = pair
            elif isinstance(value, str):
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
                if len(nums) >= 2:
                    pair = coerce_pair(nums[0], nums[1])
                    if pair is not None:
                        parsed_pairs[paragraph_id] = pair

        for list_key in ("scores", "items", "allocations", "results", "distribution"):
            entries = payload.get(list_key)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                paragraph_id = resolve_id(
                    entry.get("id")
                    or entry.get("item_id")
                    or entry.get("paragraph_id")
                    or entry.get("name")
                    or entry.get("label")
                )
                if paragraph_id is None:
                    continue
                pair = coerce_pair(
                    entry.get("technical", entry.get("technical_score", entry.get("tech", entry.get("t")))),
                    entry.get("citation", entry.get("citation_score", entry.get("cite", entry.get("c")))),
                )
                if pair is not None:
                    parsed_pairs[paragraph_id] = pair

    elif isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            paragraph_id = resolve_id(
                entry.get("id")
                or entry.get("item_id")
                or entry.get("paragraph_id")
                or entry.get("name")
                or entry.get("label")
            )
            if paragraph_id is None:
                continue
            pair = coerce_pair(
                entry.get("technical", entry.get("technical_score", entry.get("tech", entry.get("t")))),
                entry.get("citation", entry.get("citation_score", entry.get("cite", entry.get("c")))),
            )
            if pair is not None:
                parsed_pairs[paragraph_id] = pair

    text = strip_code_fences(response_text)
    for raw_line in text.splitlines():
        line = re.sub(r"^[-*•]\s*", "", raw_line.strip()).strip(",")
        if not line:
            continue

        paragraph_id: Optional[str] = None
        body = line

        if ":" in line:
            left, right = line.split(":", 1)
            maybe_id = resolve_id(left)
            if maybe_id is not None:
                paragraph_id = maybe_id
                body = right
        if paragraph_id is None:
            prefix_match = re.match(r'^"?(.+?)"?\s+(technical|tech)\b(.*)$', line, flags=re.IGNORECASE)
            if prefix_match:
                maybe_id = resolve_id(prefix_match.group(1))
                if maybe_id is not None:
                    paragraph_id = maybe_id
                    body = f"{prefix_match.group(2)} {prefix_match.group(3)}"

        if paragraph_id is None:
            continue

        technical_match = re.search(
            r"\b(?:technical|tech)\s*[:=]?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*%?",
            body,
            flags=re.IGNORECASE,
        )
        citation_match = re.search(
            r"\b(?:citation|cite|cit)\s*[:=]?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*%?",
            body,
            flags=re.IGNORECASE,
        )

        pair: Optional[Tuple[float, float]] = None
        if technical_match and citation_match:
            pair = coerce_pair(technical_match.group(1), citation_match.group(1))
        else:
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", body)
            if len(nums) >= 2:
                pair = coerce_pair(nums[0], nums[1])

        if pair is not None:
            parsed_pairs[paragraph_id] = pair

    return parsed_pairs


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
        # Soft fallback: deterministic non-uniform seed weights.
        cleaned = {key: float(idx + 1) for idx, key in enumerate(cleaned.keys())}
        score_sum = sum(cleaned.values())

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
        print(
            f"[WARN] Score conservation drift at {context}: expected {expected:.12f}, "
            f"got {actual:.12f}, diff={abs(actual - expected):.12f}"
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


def is_page_artifact_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.fullmatch(r"\d+", stripped):
        return True
    if re.fullmatch(r"page\s+\d+(?:\s+of\s+\d+)?", stripped.lower()):
        return True
    return False


def is_probable_heading_line(line: str) -> bool:
    stripped = normalize_for_match(line)
    if not stripped or len(stripped) > 120:
        return False
    if is_page_artifact_line(stripped):
        return False
    if stripped.endswith((".", "?", "!", ";")):
        return False

    heading_number = re.match(r"^(?:\d+(?:\.\d+)*|[A-Z])(?:[\.\)])?\s+(.+)$", stripped)
    if heading_number:
        remainder = heading_number.group(1).strip()
        if remainder and len(remainder.split()) <= 14:
            return True

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", stripped)
    if not words or len(words) > 14:
        return False

    uppercase_like = sum(1 for word in words if word.isupper() and len(word) > 1)
    title_like = sum(1 for word in words if word[:1].isupper())
    connector_words = {"and", "or", "of", "the", "on", "in", "with", "for", "to", "a", "an", "vs"}

    if uppercase_like >= max(1, len(words) - 2):
        return True
    if title_like >= max(1, len(words) - 2):
        return True
    if all(word.lower() in connector_words or word[:1].isupper() for word in words):
        return True
    return False


def is_list_item_line(line: str) -> bool:
    stripped = line.strip()
    return bool(re.match(r"^(?:[-*•]|\(?\d+[\.\)]|[A-Za-z][\.\)])\s+", stripped))


def median_line_length(lines: List[str]) -> int:
    lengths = sorted(len(line.strip()) for line in lines if len(line.strip()) >= 20)
    if not lengths:
        return 80
    mid = len(lengths) // 2
    if len(lengths) % 2 == 1:
        return lengths[mid]
    return (lengths[mid - 1] + lengths[mid]) // 2


def merge_paragraph_lines(lines: List[str]) -> str:
    return normalize_for_match(" ".join(line.strip() for line in lines if line.strip()))


def rebuild_paragraphs_from_lines(text: str) -> List[str]:
    lines = [line.rstrip() for line in text.split("\n")]
    content_lines = [line for line in lines if line.strip() and not is_page_artifact_line(line)]
    if not content_lines:
        return []

    typical_length = median_line_length(content_lines)
    paragraphs: List[str] = []
    current: List[str] = []

    def flush_current() -> None:
        if not current:
            return
        paragraph = merge_paragraph_lines(current)
        current.clear()
        if paragraph:
            paragraphs.append(paragraph)

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or is_page_artifact_line(stripped):
            flush_current()
            continue

        if is_probable_heading_line(stripped):
            flush_current()
            continue

        if not current:
            current.append(stripped)
            continue

        previous = current[-1].strip()
        previous_ends_sentence = previous.endswith((".", "?", "!", ":"))
        previous_is_short = len(previous) <= max(48, int(0.65 * typical_length))
        current_starts_fresh = bool(re.match(r'^[\(\["“]?[A-Z0-9]', stripped))
        current_is_substantial = len(stripped) >= max(35, int(0.45 * typical_length))
        break_before = False

        if is_list_item_line(stripped):
            break_before = True
        elif previous_ends_sentence and previous_is_short and current_starts_fresh and current_is_substantial:
            break_before = True
        elif previous.endswith(":") and current_starts_fresh:
            break_before = True

        if break_before:
            flush_current()

        current.append(stripped)

    flush_current()

    merged: List[str] = []
    for paragraph in paragraphs:
        if (
            merged
            and len(paragraph) < 30
            and not is_list_item_line(paragraph)
            and not re.search(r"[.!?]\s*$", merged[-1])
        ):
            merged[-1] = normalize_for_match(f"{merged[-1]} {paragraph}")
        else:
            merged.append(paragraph)
    return merged


def split_text_into_paragraphs(text: str) -> List[str]:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    paragraphs = [normalize_for_match(p) for p in re.split(r"\n\s*\n+", normalized) if normalize_for_match(p)]
    if len(paragraphs) > 1:
        return paragraphs

    rebuilt = rebuild_paragraphs_from_lines(normalized)
    if rebuilt:
        return rebuilt

    # PDF extraction often collapses paragraphs; use sentence chunks as a last fallback.
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
    use_limit = None if snippet_limit is None or snippet_limit <= 0 else snippet_limit
    snippets = {name: flatten_content_to_text(item_to_content[name], limit=use_limit) for name in items}
    parent_content = flatten_content_to_text(item_to_content, limit=None)
    item_payload = {str(idx + 1): {"name": item_name, "excerpt": snippets[item_name]} for idx, item_name in enumerate(items)}
    prompt = SECTION_DIRECT_USER_PROMPT_TEMPLATE.format(
        parent_name=parent_name,
        parent_score=1.0,
        parent_content=parent_content,
        items=json.dumps(item_payload, indent=2),
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

    item_ids = [str(idx + 1) for idx in range(len(items))]
    item_id_to_name = {item_ids[idx]: items[idx] for idx in range(len(items))}
    item_payload = {
        item_id: {
            "name": item_id_to_name[item_id],
            "excerpt": snippets[item_id_to_name[item_id]],
        }
        for item_id in item_ids
    }
    parent_limit = None
    if snippet_limit > 0:
        parent_limit = max(1200, min(6000, len(items) * max(80, snippet_limit)))
    parent_content = flatten_content_to_text(item_to_content, limit=parent_limit)

    system_prompt = SECTION_DIRECT_SYSTEM_PROMPT
    base_prompt = SECTION_DIRECT_USER_PROMPT_TEMPLATE.format(
        parent_name=parent_name,
        parent_score=total_score,
        parent_content=parent_content,
        items=json.dumps(item_payload, indent=2),
    )
    prompt = base_prompt
    accumulated_scores: Dict[str, float] = {}
    alias_to_id = {item_id_to_name[item_id]: item_id for item_id in item_ids}

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
        remaining_item_ids = [item_id for item_id in item_ids if item_id not in accumulated_scores]
        remaining_aliases = {
            item_id_to_name[item_id]: item_id for item_id in remaining_item_ids
        }
        parsed_full = parse_score_map_from_response(
            raw_response,
            item_ids,
            allow_percentage=False,
            alias_to_id=alias_to_id,
        )
        parsed_remaining = parse_score_map_from_response(
            raw_response,
            remaining_item_ids,
            allow_percentage=False,
            alias_to_id=remaining_aliases,
        )
        parsed_update = {
            item_id: value
            for item_id, value in parsed_full.items()
            if item_id in remaining_item_ids
        }
        parsed_update.update(parsed_remaining)

        for item_id, value in parsed_update.items():
            accumulated_scores[item_id] = max(0.0, safe_float(value, 0.0))

        if len(accumulated_scores) == len(item_ids) and any(v > 0.0 for v in accumulated_scores.values()):
            parsed_scores = {
                item_id_to_name[item_id]: accumulated_scores[item_id]
                for item_id in item_ids
            }
            return normalize_distribution(parsed_scores, total_score)

        missing_item_ids = [item_id for item_id in item_ids if item_id not in accumulated_scores]
        prompt = (
            f"{base_prompt}\n\n"
            "Your previous answer was incomplete or invalid.\n"
            f"You must provide one score for every missing child.\n"
            f"Missing counters: {', '.join(missing_item_ids) if missing_item_ids else ', '.join(item_ids)}\n"
            "Any readable one-line-per-item list is fine.\n"
            "Separators such as :, -, =, or -> are all acceptable.\n"
            "If the numbering drifts, line order will be used.\n"
            "You may answer with only the missing items."
        )

    append_debug_log(
        debug_log_path,
        (
            f"[{log_tag}_error] parent={parent_name} sample={sample_idx} "
            f"reason=incomplete_or_invalid_model_scores expected_ids={item_ids}"
        ),
    )
    raise ValueError(
        f"Model failed to produce complete child scores for parent '{parent_name}' "
        f"after {max(1, max_retries)} retries."
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
        try:
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
        except ValueError as exc:
            append_debug_log(
                debug_log_path,
                (
                    f"[{log_tag}_sample_skip] parent={parent_name} sample={s + 1}/{sample_count} "
                    f"reason={exc}"
                ),
            )

    if not sample_distributions:
        retry_snippet_limits: List[int] = []
        if snippet_limit <= 0:
            retry_snippet_limits = [180, 120, 80]
        elif snippet_limit > 120:
            retry_snippet_limits = [120, 80]
        elif snippet_limit > 80:
            retry_snippet_limits = [80]

        for retry_snippet_limit in retry_snippet_limits:
            append_debug_log(
                debug_log_path,
                (
                    f"[{log_tag}_compressed_retry] parent={parent_name} "
                    f"reason=no_complete_samples retry_snippet_limit={retry_snippet_limit}"
                ),
            )
            try:
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
                    snippet_limit=retry_snippet_limit,
                    log_tag=f"{log_tag}_compressed_retry",
                )
            except ValueError:
                continue

        if len(items) > 2:
            if len(items) <= 4:
                chunk_size = 2
            elif len(items) <= 8:
                chunk_size = 3
            else:
                chunk_size = 5
            compressed_limit = max(80, snippet_limit if snippet_limit > 0 else 120)
            append_debug_log(
                debug_log_path,
                (
                    f"[{log_tag}_chunked_retry] parent={parent_name} "
                    f"reason=no_complete_samples items={len(items)} chunk_size={chunk_size} "
                    f"snippet_limit={compressed_limit}"
                ),
            )
            chunk_item_names = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
            chunk_to_content: Dict[str, str] = {}
            chunk_to_items: Dict[str, List[str]] = {}
            for idx, chunk_names in enumerate(chunk_item_names, start=1):
                chunk_key = f"Chunk {idx}"
                chunk_to_items[chunk_key] = chunk_names
                chunk_to_content[chunk_key] = " ".join(
                    f"{item_name}: {flatten_content_to_text(item_to_content[item_name], limit=compressed_limit)}"
                    for item_name in chunk_names
                )

            chunk_scores = all_together_allocate_scores(
                client=client,
                item_to_content=chunk_to_content,
                total_score=total_score,
                parent_name=f"{parent_name}::chunks",
                model=model,
                n_samples=n_samples,
                temperature=temperature,
                max_retries=max_retries,
                debug_log_path=debug_log_path,
                snippet_limit=compressed_limit,
                log_tag=f"{log_tag}_chunks",
            )

            final_scores: Dict[str, float] = {}
            for chunk_key, chunk_names in chunk_to_items.items():
                nested_items = {item_name: item_to_content[item_name] for item_name in chunk_names}
                nested_scores = all_together_allocate_scores(
                    client=client,
                    item_to_content=nested_items,
                    total_score=chunk_scores[chunk_key],
                    parent_name=f"{parent_name}::{chunk_key}",
                    model=model,
                    n_samples=n_samples,
                    temperature=temperature,
                    max_retries=max_retries,
                    debug_log_path=debug_log_path,
                    snippet_limit=compressed_limit,
                    log_tag=f"{log_tag}_within_chunk",
                )
                final_scores.update(nested_scores)

            return normalize_distribution(final_scores, total_score)

        if len(items) == 2:
            append_debug_log(
                debug_log_path,
                f"[{log_tag}_pairwise_retry] parent={parent_name} reason=no_complete_samples",
            )
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

        raise ValueError(
            f"Model failed to produce any complete child-score sample for parent '{parent_name}' "
            f"after compression retries."
        )

    averaged = {item: 0.0 for item in items}
    for dist in sample_distributions:
        for item, value in dist.items():
            averaged[item] += value
    averaged = {item: value / len(sample_distributions) for item, value in averaged.items()}
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

    citation_ids = [str(idx + 1) for idx in range(len(citations))]
    citation_id_to_name = {citation_ids[idx]: citations[idx] for idx in range(len(citations))}
    citation_payload = {
        citation_id: {
            "citation": citation_id_to_name[citation_id],
            "context": citation_to_context[citation_id_to_name[citation_id]],
        }
        for citation_id in citation_ids
    }

    base_prompt = CITATION_SPLIT_USER_PROMPT_TEMPLATE.format(
        paragraph_id=paragraph_id,
        paragraph_citation_score=total_score,
        paragraph_text=paragraph_text[:1200],
        citations_json=json.dumps(citation_payload, indent=2),
    )
    prompt = base_prompt
    accumulated_scores: Dict[str, float] = {}
    alias_to_id = {citation_id_to_name[citation_id]: citation_id for citation_id in citation_ids}

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
        remaining_citation_ids = [citation_id for citation_id in citation_ids if citation_id not in accumulated_scores]
        remaining_aliases = {
            citation_id_to_name[citation_id]: citation_id for citation_id in remaining_citation_ids
        }
        parsed_full = parse_score_map_from_response(
            raw_response,
            citation_ids,
            allow_percentage=False,
            alias_to_id=alias_to_id,
        )
        parsed_remaining = parse_score_map_from_response(
            raw_response,
            remaining_citation_ids,
            allow_percentage=False,
            alias_to_id=remaining_aliases,
        )
        parsed_update = {
            citation_id: value
            for citation_id, value in parsed_full.items()
            if citation_id in remaining_citation_ids
        }
        parsed_update.update(parsed_remaining)

        for citation_id, value in parsed_update.items():
            accumulated_scores[citation_id] = max(0.0, safe_float(value, 0.0))

        if len(accumulated_scores) == len(citation_ids) and any(v > 0.0 for v in accumulated_scores.values()):
            parsed_scores = {
                citation_id_to_name[citation_id]: accumulated_scores[citation_id]
                for citation_id in citation_ids
            }
            return normalize_distribution(parsed_scores, total_score)

        missing_citation_ids = [citation_id for citation_id in citation_ids if citation_id not in accumulated_scores]
        prompt = (
            f"{base_prompt}\n\n"
            "Your previous answer was incomplete or invalid.\n"
            f"You must provide one score for every missing citation.\n"
            f"Missing counters: {', '.join(missing_citation_ids) if missing_citation_ids else ', '.join(citation_ids)}\n"
            "Any readable one-line-per-item list is fine.\n"
            "Separators such as :, -, =, or -> are all acceptable.\n"
            "If the numbering drifts, line order will be used.\n"
            "You may answer with only the missing items."
        )

    append_debug_log(
        debug_log_path,
        (
            f"[citation_split_error] paragraph_id={paragraph_id} sample={sample_idx} "
            f"reason=incomplete_or_invalid_model_scores expected_ids={citation_ids}"
        ),
    )
    raise ValueError(
        f"Model failed to produce complete citation scores for paragraph '{paragraph_id}' "
        f"after {max(1, max_retries)} retries."
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


def split_paragraph_channel_scores(
    client: Client,
    paragraph_items: Dict[str, str],
    paragraph_total_scores: Dict[str, float],
    mention_buckets: Dict[str, List[Tuple[str, str, str]]],
    model: str,
    n_samples: int,
    temperature: float,
    max_retries: int,
    snippet_limit: int,
    debug_log_path: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not paragraph_items:
        return {}, {}

    paragraph_ids = list(paragraph_items.keys())
    sample_count = max(1, n_samples)

    cleaned_totals: Dict[str, float] = {
        paragraph_id: max(0.0, safe_float(paragraph_total_scores.get(paragraph_id), 0.0))
        for paragraph_id in paragraph_ids
    }

    paragraph_aliases: Dict[str, str] = {}
    for idx, paragraph_id in enumerate(paragraph_ids, start=1):
        paragraph_aliases[paragraph_id] = paragraph_id
        paragraph_aliases[f"p{idx}"] = paragraph_id
        paragraph_aliases[f"paragraph {idx}"] = paragraph_id
        paragraph_aliases[f"paragraph{idx}"] = paragraph_id
        paragraph_aliases[str(idx)] = paragraph_id

    if snippet_limit <= 0:
        snippets = {paragraph_id: paragraph_items[paragraph_id] for paragraph_id in paragraph_ids}
    else:
        snippets = {
            paragraph_id: paragraph_items[paragraph_id][: max(80, snippet_limit)] for paragraph_id in paragraph_ids
        }

    payload = {
        paragraph_id: {
            "total_score": cleaned_totals[paragraph_id],
            "has_citations": bool(mention_buckets.get(paragraph_id)),
            "text": snippets[paragraph_id],
        }
        for paragraph_id in paragraph_ids
    }
    user_prompt = PARAGRAPH_CHANNEL_SPLIT_USER_PROMPT_TEMPLATE.format(
        paragraphs_json=json.dumps(payload, indent=2)
    )

    technical_samples: List[Dict[str, float]] = []
    citation_samples: List[Dict[str, float]] = []

    for sample_idx in range(sample_count):
        current_temp = min(1.0, max(0.0, temperature) + (0.05 * sample_idx))
        parsed_pairs: Dict[str, Tuple[float, float]] = {}

        for attempt in range(max(1, max_retries)):
            try:
                response = client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": PARAGRAPH_CHANNEL_SPLIT_SYSTEM_PROMPT},
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
                    f"[paragraph_channel_split] sample={sample_idx + 1}/{sample_count} "
                    f"attempt={attempt + 1}/{max(1, max_retries)} "
                    f"temperature={current_temp}\n{raw_response}\n"
                ),
            )
            parsed_pairs = parse_paragraph_channel_split_response(
                raw_response,
                paragraph_ids,
                alias_to_id=paragraph_aliases,
            )
            if len(parsed_pairs) == len(paragraph_ids):
                break

        sample_technical: Dict[str, float] = {}
        sample_citation: Dict[str, float] = {}
        for paragraph_id in paragraph_ids:
            total_val = cleaned_totals.get(paragraph_id, 0.0)
            mentions = mention_buckets.get(paragraph_id, [])
            has_citations = bool(mentions)

            if paragraph_id in parsed_pairs:
                t_raw, c_raw = parsed_pairs[paragraph_id]
            else:
                if has_citations:
                    fallback_ratio = min(0.4, max(0.1, 0.12 * len(mentions)))
                    t_raw = 1.0 - fallback_ratio
                    c_raw = fallback_ratio
                else:
                    t_raw = 1.0
                    c_raw = 0.0

            t_raw = max(0.0, t_raw)
            c_raw = max(0.0, c_raw)
            if not has_citations:
                t_val = total_val
                c_val = 0.0
            else:
                channel_sum = t_raw + c_raw
                if channel_sum <= 0.0:
                    t_val = total_val
                    c_val = 0.0
                else:
                    t_val = total_val * (t_raw / channel_sum)
                    c_val = total_val * (c_raw / channel_sum)

            sample_technical[paragraph_id] = t_val
            sample_citation[paragraph_id] = c_val

        technical_samples.append(sample_technical)
        citation_samples.append(sample_citation)

    paragraph_technical: Dict[str, float] = {}
    paragraph_citation: Dict[str, float] = {}
    effective_samples = max(1, len(technical_samples))
    for paragraph_id in paragraph_ids:
        avg_t = sum(sample[paragraph_id] for sample in technical_samples) / effective_samples
        avg_c = sum(sample[paragraph_id] for sample in citation_samples) / effective_samples
        total_val = cleaned_totals.get(paragraph_id, 0.0)
        has_citations = bool(mention_buckets.get(paragraph_id))

        if not has_citations:
            avg_t = total_val
            avg_c = 0.0
        else:
            channel_sum = max(0.0, avg_t) + max(0.0, avg_c)
            if channel_sum <= 0.0:
                avg_t = total_val
                avg_c = 0.0
            else:
                avg_t = total_val * max(0.0, avg_t) / channel_sum
                avg_c = total_val * max(0.0, avg_c) / channel_sum

        paragraph_technical[paragraph_id] = max(0.0, avg_t)
        paragraph_citation[paragraph_id] = max(0.0, avg_c)

    return paragraph_technical, paragraph_citation


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
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
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
    paragraph_scores: List[Dict[str, Any]] = []

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
            "total_score": max(0.0, safe_float(score, 0.0)),
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
                "internal_paragraph_id": f"{paper_id}::{' > '.join(section_path)}::p{idx + 1}",
                "paragraph_index": idx + 1,
                "text": paragraph,
            }
            for idx, (name, paragraph) in enumerate(paragraph_items.items())
        }
        paragraph_parent_name = f"{section_name}::paragraphs"
        est_tokens = estimate_direct_allocation_tokens(paragraph_parent_name, paragraph_items, snippet_limit=0)
        effective_threshold = paragraph_direct_max_tokens if paragraph_direct_max_tokens > 0 else 7000
        use_compressed = est_tokens > effective_threshold
        snippet_limit = max(60, paragraph_compressed_snippet_limit) if use_compressed else 0
        method = "channel_compressed" if use_compressed else "channel_full_parent"
        append_debug_log(
            debug_log_path,
            (
                f"[paragraph_scoring] parent={paragraph_parent_name} method={method} "
                f"estimated_tokens={est_tokens} threshold={effective_threshold} "
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
        paragraph_total = normalize_distribution(paragraph_total_scores, section_score)

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
                        key=lambda name: paragraph_total.get(name, 0.0),
                    )

                for citation in split_citation_block(citation_block):
                    mention_buckets[target_paragraph].append((citation, citation_block, context_str))

        paragraph_names = list(paragraph_items.keys())
        paragraph_technical, paragraph_citation = split_paragraph_channel_scores(
            client=client,
            paragraph_items=paragraph_items,
            paragraph_total_scores=paragraph_total,
            mention_buckets=mention_buckets,
            model=model,
            n_samples=max(1, n_samples),
            temperature=temperature,
            max_retries=max(1, max_retries),
            snippet_limit=snippet_limit,
            debug_log_path=debug_log_path,
        )

        for paragraph_name in paragraph_names:
            meta = paragraph_meta[paragraph_name]
            internal_paragraph_id = meta["internal_paragraph_id"]
            mentions = mention_buckets.get(paragraph_name, [])
            paragraph_t = max(0.0, paragraph_technical.get(paragraph_name, 0.0))
            paragraph_c = max(0.0, paragraph_citation.get(paragraph_name, 0.0))
            paragraph_total_score = combine_scores(paragraph_t, paragraph_c)
            assert_close(
                paragraph_total_score,
                max(0.0, safe_float(paragraph_total.get(paragraph_name), 0.0)),
                f"paragraph {internal_paragraph_id}",
            )

            paragraph_scores.append(
                {
                    "section_path": list(section_path),
                    "paragraph_index": meta["paragraph_index"],
                    "paragraph": meta["text"],
                    "technical_score": paragraph_t,
                    "citation_score": paragraph_c,
                }
            )

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
                paragraph_id=f"{' > '.join(section_path)}::p{meta['paragraph_index']}",
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
            assert_close(split_total, paragraph_c, f"citation split for {internal_paragraph_id}")

            for citation, citation_value in citation_split.items():
                if citation not in citation_scores:
                    citation_scores[citation] = {"citation_score": 0.0}

                citation_scores[citation]["citation_score"] += citation_value

        leaf_c = sum(
            max(0.0, safe_float(paragraph_citation.get(paragraph_name, 0.0), 0.0))
            for paragraph_name in paragraph_names
        )
        leaf_total = sum(
            combine_scores(
                paragraph_technical.get(paragraph_name, 0.0),
                paragraph_citation.get(paragraph_name, 0.0),
            )
            for paragraph_name in paragraph_names
        )
        assert_close(leaf_total, section_score, f"leaf section {' > '.join(section_path)}")
        node_ref["total_score"] = max(0.0, safe_float(section_score, 0.0))
        node_ref["citation_score"] = max(0.0, safe_float(leaf_c, 0.0))

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
                "total_score": max(0.0, safe_float(section_score, 0.0)),
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
                max(0.0, safe_float(child.get("total_score"), 0.0))
                for child in current_ref["subsections"].values()
            )
            assert_close(child_total, section_score, f"internal section {' > '.join(current_path)}")
            current_ref["total_score"] = max(0.0, safe_float(section_score, 0.0))
            current_ref["citation_score"] = sum(
                max(0.0, safe_float(child.get("citation_score"), 0.0))
                for child in current_ref["subsections"].values()
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
        max(0.0, safe_float(payload.get("total_score"), 0.0))
        for payload in section_scores.values()
    )
    assert_close(top_level_total, 1.0, "top-level sections")

    return citation_scores, section_scores, paragraph_scores


def print_section_hierarchy(section_scores: Dict[str, Any], indent: int = 0) -> None:
    for section_name, section_data in section_scores.items():
        total_score = max(0.0, safe_float(section_data.get("total_score"), 0.0))
        citation_score = max(0.0, safe_float(section_data.get("citation_score"), 0.0))
        print(f"{'  ' * indent}{section_name}: total={total_score:.4f}, citation={citation_score:.4f}")
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
    parser.add_argument(
        "--sections-file",
        default="",
        help="Optional path to a text file containing section-tree assignments.",
    )
    parser.add_argument(
        "--sections-var",
        default="",
        help="Assignment name inside --sections-file to use for this paper.",
    )
    parser.add_argument("--model", default="llama3.2", help="Ollama model name")
    parser.add_argument("--host", default="localhost:11434", help="Ollama host")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of LLM samples to average")
    parser.add_argument("--temperature", type=float, default=0.2, help="Base sampling temperature")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries per sample for JSON parsing")
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
    if bool(args.sections_file) != bool(args.sections_var):
        parser.error("--sections-file and --sections-var must be provided together.")
    append_run_separator(args.debug_log, args)

    text = read_pdf_text(args.pdf)
    sections = DEFAULT_SECTIONS
    if args.sections_file and args.sections_var:
        sections = load_sections_from_file(args.sections_file, args.sections_var)
    citations, content = extract_citations_by_section(text, sections)
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
        paragraph_importance,
        key=lambda payload: combine_scores(
            safe_float(payload.get("technical_score"), 0.0),
            safe_float(payload.get("citation_score"), 0.0),
        ),
        reverse=True,
    )
    for payload in sorted_paragraphs[:10]:
        technical_score = safe_float(payload.get("technical_score"), 0.0)
        citation_score = safe_float(payload.get("citation_score"), 0.0)
        section_path = payload.get("section_path", [])
        section_label = " > ".join(section_path) if isinstance(section_path, list) else str(section_path)
        paragraph_index = payload.get("paragraph_index", "?")
        print(
            f"{section_label}::p{paragraph_index}: technical={technical_score:.4f}, citation={citation_score:.4f}"
        )

    append_run_end(args.debug_log)


if __name__ == "__main__":
    main()
