import argparse
import ast
import difflib
import json
import random
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import PyPDF2
except ImportError:  # pragma: no cover - fallback for environments with pypdf only
    import pypdf as PyPDF2
try:
    from ollama import Client
except ImportError:  # pragma: no cover - optional for non-LLM workflows
    Client = None  # type: ignore[assignment]


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


def sanitize_model_tag(model: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").lower()
    return cleaned or "model"


DEFAULT_SAMPLE_TEMPERATURE_JITTER = 0.03
_SAMPLE_TEMPERATURE_JITTER = DEFAULT_SAMPLE_TEMPERATURE_JITTER
_SAMPLE_TEMPERATURE_RNG = random


def sample_temperature(base_temperature: float, _sample_idx: int) -> float:
    # Give each sample a small independent temperature jitter.
    base = max(0.0, base_temperature)
    offset = _SAMPLE_TEMPERATURE_RNG.uniform(0.0, _SAMPLE_TEMPERATURE_JITTER)
    return min(1.0, base + offset)

SECTION_PAIRWISE_SYSTEM_PROMPT = (
    "You are an expert academic reviewer. "
    "Compare two items from the same paper by contribution to the paper's main scientific contribution. "
    "Distribute a total credit of 1.0 between item A and item B. "
    "Use both technical contribution and citation-supported contribution. "
    "Return JSON only."
)

SECTION_PAIRWISE_USER_PROMPT_TEMPLATE = """Parent node: "{parent_name}"
Task: distribute a credit of 1.0 between A and B based on contribution to the paper's main contribution.

Parent context: these items are section/subsection children of "{parent_name}".
Treat their names as meaningful context, not just labels.

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
    "Given the child sections or subsections of one paper segment, distribute 100 points "
    "across them based on how much each contributes to the parent segment's main scientific "
    "contribution. "
    "Treat section names as meaningful context. Use the excerpts and parent context to judge "
    "which children contribute most strongly through methods, theory, algorithms, proofs, "
    "experimental findings, analysis, problem formulation, or other core scientific content. "
    "Background, transitions, setup details, summaries, and lower-impact narrative should "
    "usually receive less weight. "
    "Return percentages, not copied numbers from the excerpts. "
    "Percentages must be non-negative and sum to 100. "
    "Return plain text lines only. Do not output JSON."
)

SECTION_DIRECT_USER_PROMPT_TEMPLATE = """Parent segment:
{parent_name}

Parent score to distribute:
{parent_score}

Parent context: these items are section/subsection children of "{parent_name}".
Treat their names as meaningful context, not just labels.

Parent content (context):
{parent_content}

Task:
Assign percentage importance to the following child segments according to how much
each contributes to the parent segment's main scientific contribution. The percentages
will be rescaled in code to the parent score.

Guidelines:
- Higher scores: segments that carry the core scientific contribution through
  methods, theory, algorithms, proofs, experimental findings, analysis, or
  important problem formulation.
- Lower scores: segments that mainly provide background, motivation, transitions,
  setup details, or lower-impact narrative.
- If one segment contributes about 3x as much as another, assign about 3x the score.
- If two segments contribute equally, assign equal scores.
- Do not copy decimal values that appear inside the excerpts. Infer new percentages.

Child segments (id -> name and excerpt):
{items}

Output format (plain text only):
One line per child.
You may write either:
- <id>: <percentage>
or just one percentage line per child in the same order as the child list above.

Example:
1: 60
2: 40

The percentages must sum to 100.
Do not output JSON.
Do not include explanations.
"""

CITATION_SPLIT_SYSTEM_PROMPT = (
    "You are an expert academic reviewer. "
    "Your task is to distribute percentage importance among the citations that appear in a paragraph, "
    "based on how much each cited work contributes to the paragraph's claims or grounding. "
    "Return percentages, not copied numbers from the paragraph or contexts. "
    "Percentages must be non-negative and sum to 100. "
    "Return plain text lines only."
)

CITATION_SPLIT_USER_PROMPT_TEMPLATE = """Paragraph id:
{paragraph_id}

Paragraph citation score to distribute later in code: {paragraph_citation_score}

Paragraph text:
{paragraph_text}

Task:
- Divide percentage importance among the citations below.
- Higher share:
  citation directly supports the paragraph's claim
  citation provides key comparison or baseline
  citation introduces a method the paragraph builds upon
- Lower share:
  citation is peripheral or only mentioned briefly
- Percentages must be non-negative.
- Percentages must sum to 100.
- Do not copy decimal values from the paragraph or citation contexts.

Citation entries (citation_id -> citation and context):
{citations_json}

Output format (plain text only):
Any clear one-line-per-citation list is acceptable.
Separators such as `:`, `-`, `=`, or `->` are all fine.
Line order matters more than exact counter formatting.

Example:
1: 60
2: 40

Do not output JSON.
Do not include explanations.
"""

PARAGRAPH_CHANNEL_SPLIT_SYSTEM_PROMPT = (
"""You are an expert academic reviewer. For each paragraph, split its total score into two components:

- technical_percentage: the percentage of value this paragraph contributes through the authors' own 
  original ideas — new definitions, algorithms, proofs, experimental design, 
  results, or analysis that would survive even if all cited works were removed.

- citation_percentage: the percentage of value this paragraph derives from cited prior work — 
  including summarizing, comparing, contextualizing, or building upon existing 
  work. If the paragraph's main purpose is to describe what others have done, 
  most of its value is citation-derived.

Return percentages, not copied numbers from the paragraph text. technical + citation must equal 
100 for each paragraph."""
)

PARAGRAPH_CHANNEL_SPLIT_USER_PROMPT_TEMPLATE = """Task:
For each paragraph, split the value into technical_percentage and citation_percentage.

Section context: these paragraphs are from the "{section_name}" section of the paper.
Use this to calibrate your expectations - a paragraph in Related Work behaves
very differently from one in Methods, even if the text looks similar.

Ask yourself: "If I deleted all content describing, comparing, or bridging prior
work - what fraction of this paragraph's value survives?" That surviving fraction
is technical_percentage. The rest is citation_percentage.

Calibration guidance:
- A paragraph in Related Work that summarizes prior methods:
  citation_percentage ≈ 80–95, technical_percentage ≈ 5–20
- A paragraph in Related Work that draws a novel connection between prior works
  and the current paper's gap:
  citation_percentage ≈ 50–70, technical_percentage ≈ 30–50
- A Background paragraph that defines a known concept from prior work:
  citation_percentage ≈ 60–80
- A Methods paragraph describing the authors' new algorithm:
  citation_percentage ≈ 0–15, technical_percentage ≈ 85–100
- A paragraph comparing experiment results to a baseline from prior work:
  citation_percentage ≈ 20–40
- A Conclusion paragraph summarizing the paper's own contributions:
  citation_percentage ≈ 0–10, technical_percentage ≈ 90–100
- If has_citations is false, citation_percentage = 0 and technical_percentage = 100.

Rules:
- technical + citation = 100 for every paragraph
- Both values must be non-negative
- Do not assign citation_percentage > 0 if has_citations is false
- Use `citation_focus_text` as the only evidence for citation-derived value.
  Text outside `citation_focus_text` should not increase citation_percentage.
- If `citation_focus_text` is empty or minimal, citation_percentage should stay low
  even if the paragraph has citation markers elsewhere.
- In citation-heavy sections such as Related Work, if most of the paragraph's
  citation-focused sentences are in `citation_focus_text`, citation_percentage
  should usually be the majority share.

Paragraph entries (paragraph_id -> details):
{paragraphs_json}

Output format (plain text only):
paragraph_id: technical=<float>, citation=<float>

Example:
Paragraph1: technical=80, citation=20
Paragraph2: technical=100, citation=0

Do not output JSON. Do not include explanations.
"""

PARAGRAPH_CHANNEL_SPLIT_STRICT_SYSTEM_PROMPT = (
"""You are an expert academic reviewer. For each paragraph, split its total score into two components:

- technical_percentage: the percentage of value this paragraph contributes through the authors' own
  original ideas — new definitions, algorithms, proofs, experimental design,
  results, or analysis that would survive even if all cited works were removed.

- citation_percentage: the percentage of value this paragraph derives from cited prior work —
  including summarizing, comparing, contextualizing, or building upon existing
  work. If the paragraph's main purpose is to describe what others have done,
  most of its value is citation-derived.

Return percentages, not copied numbers from the paragraph text. technical + citation must equal
100 for each paragraph. If a paragraph has citations (has_citations is true), citation_percentage
must be greater than 0 — a cited work always contributes some value, even if the paragraph is
mostly technical."""
)

PARAGRAPH_CHANNEL_SPLIT_STRICT_USER_PROMPT_TEMPLATE = """Task:
For each paragraph, split the value into technical_percentage and citation_percentage.

Section context: these paragraphs are from the "{section_name}" section of the paper.
Use this to calibrate your expectations - a paragraph in Related Work behaves
very differently from one in Methods, even if the text looks similar.

Ask yourself: "If I deleted all content describing, comparing, or bridging prior
work - what fraction of this paragraph's value survives?" That surviving fraction
is technical_percentage. The rest is citation_percentage.

Calibration guidance:
- A paragraph in Related Work that summarizes prior methods:
  citation_percentage ≈ 80–95, technical_percentage ≈ 5–20
- A paragraph in Related Work that draws a novel connection between prior works
  and the current paper's gap:
  citation_percentage ≈ 50–70, technical_percentage ≈ 30–50
- A Background paragraph that defines a known concept from prior work:
  citation_percentage ≈ 60–80
- A Methods paragraph describing the authors' new algorithm:
  citation_percentage ≈ 0–15, technical_percentage ≈ 85–100
- A paragraph comparing experiment results to a baseline from prior work:
  citation_percentage ≈ 20–40
- A Conclusion paragraph summarizing the paper's own contributions:
  citation_percentage ≈ 0–10, technical_percentage ≈ 90–100
- If has_citations is false, citation_percentage = 0 and technical_percentage = 100.

Rules:
- technical + citation = 100 for every paragraph
- Both values must be non-negative
- Do not assign citation_percentage > 0 if has_citations is false
- If has_citations is true, citation_percentage must be greater than 0.
- Use `citation_focus_text` as the only evidence for citation-derived value.
  Text outside `citation_focus_text` should not increase citation_percentage.
- If `citation_focus_text` is empty or minimal, citation_percentage should stay low
  even if the paragraph has citation markers elsewhere.
- In citation-heavy sections such as Related Work, if most of the paragraph's
  citation-focused sentences are in `citation_focus_text`, citation_percentage
  should usually be the majority share.

Paragraph entries (paragraph_id -> details):
{paragraphs_json}

Output format (plain text only):
paragraph_id: technical=<float>, citation=<float>

Example:
Paragraph1: technical=80, citation=20
Paragraph2: technical=100, citation=0

Do not output JSON. Do not include explanations.
"""

SUBSECTION_TITLE_RECOVERY_SYSTEM_PROMPT = (
    "You are helping recover subsection headings from noisy PDF text. "
    "Given a parent section's extracted text and a list of expected child subsection names, "
    "identify the best matching heading title that actually appears in the text for each expected child. "
    "Do not invent headings that are not supported by the text. "
    "Return plain text only."
)

SUBSECTION_TITLE_RECOVERY_USER_PROMPT_TEMPLATE = """Parent section:
{parent_name}

Expected child subsection names:
{expected_children_json}

Candidate heading titles found in the parent text:
{candidate_titles_json}

Parent section text:
{parent_text}

Task:
- For each expected child subsection name, choose the best matching heading title from the candidate list above.
- Prefer exact or near-exact matches in meaning.
- If no credible match exists, write none for that child.
- Only recover titles. Do not extract subsection bodies.

Output format (plain text only):
One line per expected child.
Use any readable separator such as `:`, `=`, `->`, or `-`.

Example:
Lattice Based Solution: Lattice Based Solution
Compute Bounds for Anytime Answer -> Compute Bounds for Anytime Answer
Bound Computation Algorithms: none
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
    "subsection_title_recovery_system_prompt": SUBSECTION_TITLE_RECOVERY_SYSTEM_PROMPT,
    "subsection_title_recovery_user_prompt_template": SUBSECTION_TITLE_RECOVERY_USER_PROMPT_TEMPLATE,
}

AUTHOR_YEAR_CITATION_PATTERN = r"\([A-Z][^()]*\d{4}[a-z]?\)"
AUTHOR_NAME_TOKEN_PATTERN = r"[A-Z][A-Za-z`'’\-]*"
AUTHOR_NAME_CONTINUATION_PATTERN = r"[A-Za-z`'’\-]+"
AUTHOR_NAME_PHRASE_PATTERN = (
    rf"{AUTHOR_NAME_TOKEN_PATTERN}"
    rf"(?:\s+{AUTHOR_NAME_CONTINUATION_PATTERN}){{0,3}}"
)
NARRATIVE_AUTHOR_YEAR_CITATION_PATTERN = (
    rf"(?:"
    rf"{AUTHOR_NAME_PHRASE_PATTERN}\s+et al\.\s*\(\d{{4}}[a-z]?\)"
    rf"|"
    rf"{AUTHOR_NAME_PHRASE_PATTERN}\s+(?:and|&)\s+{AUTHOR_NAME_PHRASE_PATTERN}\s*\(\d{{4}}[a-z]?\)"
    rf"|"
    rf"[A-Z][A-Za-z`''\-]{{2,}}(?:\s+[A-Z][A-Za-z`''\-]+)?\s*\(\d{{4}}[a-z]?\)"
    rf")"
)
NUMERIC_BRACKET_CITATION_PATTERN = r"\[(?:\s*\d+\s*(?:[-,;–]\s*\d+\s*)*)\]"
NUMERIC_PAREN_CITATION_PATTERN = r"\((?:\s*\d+\s*(?:[-,;–]\s*\d+\s*)*)\)"
CITATION_BLOCK_PATTERN = (
    rf"{NARRATIVE_AUTHOR_YEAR_CITATION_PATTERN}|"
    rf"{AUTHOR_YEAR_CITATION_PATTERN}|"
    rf"{NUMERIC_BRACKET_CITATION_PATTERN}|"
    rf"{NUMERIC_PAREN_CITATION_PATTERN}"
)
BACK_MATTER_HEADINGS = (
    "acknowledgments",
    "acknowledgements",
    "acknowledgment",
    "acknowledgement",
    "references",
    "bibliography",
    "appendix",
    "appendices",
    "supplementary material",
    "supplementary materials",
    "supplementary",
    "broader impact",
    "ethics statement",
)
SECTION_LEAD_IN_NODE = "[Section Lead-in]"
MONTH_NAME_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
NUMERIC_CITATION_CONTEXT_TERMS = (
    "shape",
    "hidden",
    "layer",
    "layers",
    "grid",
    "kan",
    "mlp",
    "neuron",
    "neurons",
    "width",
    "batch",
    "dataset",
    "train",
    "training",
    "test",
    "accuracy",
    "loss",
    "error",
    "optimizer",
    "step",
    "steps",
    "size",
    "architecture",
    "parameter",
    "parameters",
    "tensor",
    "embedding",
    "channel",
    "channels",
)
AUTHOR_YEAR_LEAD_IN_TOKENS = {
    "a",
    "an",
    "and",
    "by",
    "cf",
    "eg",
    "e.g.",
    "for",
    "from",
    "in",
    "of",
    "on",
    "see",
    "the",
    "using",
    "via",
    "with",
}

CITATION_STYLE_NUMERIC = "numeric"
CITATION_STYLE_NUMERIC_BRACKET = "numeric_bracket"
CITATION_STYLE_NUMERIC_PAREN = "numeric_paren"
CITATION_STYLE_AUTHOR_YEAR = "author_year"


def normalize_author_year_authors(authors: str) -> str:
    normalized = normalize_for_match(authors)
    normalized = re.sub(r"\s+", " ", normalized).strip(" ,;:")
    normalized = re.sub(r"\bet\s+al\s*\.", "et al.", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s*([&])\s*", r" \1 ", normalized)
    normalized = re.sub(r"\s+([`'’])", r"\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = trim_author_year_lead_in(normalized)
    return normalized


def trim_author_year_lead_in(authors: str) -> str:
    normalized = re.sub(r"\s+", " ", authors or "").strip(" ,;:")
    if not normalized:
        return normalized

    tokens = normalized.split()
    while tokens and tokens[0].lower().strip(".,;:") in AUTHOR_YEAR_LEAD_IN_TOKENS:
        tokens.pop(0)
    normalized = " ".join(tokens).strip()
    if not normalized:
        return normalized

    et_al_match = re.search(r"\bet al\.", normalized, flags=re.IGNORECASE)
    if et_al_match:
        prefix = normalized[: et_al_match.start()].strip()
        prefix_tokens = prefix.split()
        if len(prefix_tokens) > 1:
            prefix = prefix_tokens[-1]
        normalized = f"{prefix} et al."
        return normalized.strip()

    pair_match = re.search(r"\s(?:&|and)\s", normalized)
    if pair_match:
        left, right = re.split(r"\s(?:&|and)\s", normalized, maxsplit=1)
        left_tokens = left.strip().split()
        right_tokens = right.strip().split()
        if len(left_tokens) > 2:
            left = left_tokens[-1]
        else:
            left = " ".join(left_tokens)
        if len(right_tokens) > 2:
            right = right_tokens[-1]
        else:
            right = " ".join(right_tokens)
        return f"{left} & {right}".strip()

    return normalized


def canonicalize_citation_key(citation: str) -> str:
    raw = normalize_for_match(citation)
    raw = re.sub(r"\s+", " ", raw).strip()
    if not raw:
        return citation.strip()

    narrative_match = re.fullmatch(r"(.+?)\s*\((\d{4}[a-z]?)\)", raw)
    if narrative_match and not raw.startswith("("):
        authors = normalize_author_year_authors(narrative_match.group(1))
        year = narrative_match.group(2)
        if authors:
            return f"({authors}, {year})"

    parenthetical_match = re.fullmatch(r"\((.+?)\)", raw)
    if parenthetical_match:
        inner = parenthetical_match.group(1).strip()
        if ";" not in inner:
            author_year_match = re.fullmatch(r"(.+?)(?:,\s*|\s+)(\d{4}[a-z]?)", inner)
            if author_year_match:
                authors = normalize_author_year_authors(author_year_match.group(1))
                year = author_year_match.group(2)
                if authors:
                    return f"({authors}, {year})"

    return raw


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
    normalized = normalize_for_match(text).lower()
    normalized = re.sub(r"[εϵ]", " epsilon ", normalized)
    normalized = normalized.replace("&", "and")
    normalized = re.sub(r"[-‐‑–—]", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def heading_similarity_score(target: str, candidate: str) -> float:
    target_norm = normalize_heading_text(target)
    candidate_norm = normalize_heading_text(candidate)
    if not target_norm or not candidate_norm:
        return 0.0
    if heading_tokens_match(target_norm, candidate_norm) or heading_tokens_match(candidate_norm, target_norm):
        return 1.0
    return difflib.SequenceMatcher(None, target_norm, candidate_norm).ratio()


def strip_heading_prefix(text: str) -> str:
    stripped = normalize_for_match(text)
    stripped = re.sub(r"^(?:\d+(?:\.\d+)*|[A-Z])(?:[\.\)])?\s+", "", stripped)
    return stripped.strip()


def strip_heading_label_prefix(text: str) -> str:
    stripped = normalize_for_match(text)
    stripped = re.sub(
        r"^(?:\d+(?:\.\d+)*|[IVXLCDM]+|[A-Z])(?:[\.\)])?\s+",
        "",
        stripped,
        flags=re.IGNORECASE,
    )
    return normalize_heading_text(stripped)


def heading_core_key(text: str) -> str:
    tokens = strip_heading_label_prefix(text).split()
    tokens = [token for token in tokens if token != "epsilon"]
    return " ".join(tokens)


def heading_matches_expected_title(expected_title: str, candidate_title: str) -> bool:
    expected_tokens = strip_heading_label_prefix(expected_title).split()
    candidate_tokens = strip_heading_label_prefix(candidate_title).split()
    if not expected_tokens or not candidate_tokens:
        return False

    exp_idx = 0
    cand_idx = 0
    while exp_idx < len(expected_tokens) and cand_idx < len(candidate_tokens):
        if candidate_tokens[cand_idx] == expected_tokens[exp_idx]:
            exp_idx += 1
            cand_idx += 1
            continue

        if candidate_tokens[cand_idx] == "epsilon":
            cand_idx += 1
            continue
        if expected_tokens[exp_idx] == "epsilon":
            exp_idx += 1
            continue

        if cand_idx + 1 < len(candidate_tokens):
            merged_token = candidate_tokens[cand_idx] + candidate_tokens[cand_idx + 1]
            if merged_token == expected_tokens[exp_idx]:
                exp_idx += 1
                cand_idx += 2
                continue

        return False

    while exp_idx < len(expected_tokens) and expected_tokens[exp_idx] == "epsilon":
        exp_idx += 1
    while cand_idx < len(candidate_tokens) and candidate_tokens[cand_idx] == "epsilon":
        cand_idx += 1

    return exp_idx == len(expected_tokens) and cand_idx == len(candidate_tokens)


def parse_heading_label_and_title(text: str) -> Tuple[str, str]:
    stripped = normalize_for_match(text)
    match = re.match(r"^((?:\d+(?:\.\d+)*|[A-Z])(?:[\.\)])?)\s+(.+)$", stripped)
    if not match:
        return "", stripped
    label = match.group(1).rstrip(".)")
    title = match.group(2).strip()
    return label, title


def heading_level(label: str) -> int:
    if not label:
        return 0
    if re.fullmatch(r"\d+(?:\.\d+)*", label):
        return label.count(".") + 1
    return 1


def collect_heading_candidates_with_offsets(section_text: str) -> List[Dict[str, Any]]:
    lines = section_text.splitlines()
    raw_lines = section_text.splitlines(keepends=True)
    candidates: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int]] = set()
    offsets: List[int] = []
    running = 0
    for line in raw_lines:
        offsets.append(running)
        running += len(line)

    def add_candidate(raw_text: str, start_idx: int, end_idx: int) -> None:
        label, titled_text = parse_heading_label_and_title(raw_text)
        candidate = titled_text if titled_text else strip_heading_prefix(raw_text)
        if not candidate:
            return
        if is_page_artifact_line(candidate) or is_back_matter_heading(candidate):
            return
        normalized = normalize_heading_text(candidate)
        if not normalized:
            return
        if len(normalized.split()) > 18:
            return
        start_offset = offsets[start_idx]
        content_start = offsets[end_idx] + len(raw_lines[end_idx])
        dedupe_key = (normalized, start_offset)
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        candidates.append(
            {
                "label": label,
                "level": heading_level(label),
                "title": candidate,
                "start": start_offset,
                "content_start": content_start,
            }
        )

    for idx, line in enumerate(lines):
        stripped = normalize_for_match(line)
        if not stripped or is_page_artifact_line(stripped):
            continue

        if is_probable_heading_line(stripped):
            add_candidate(stripped, idx, idx)

        combined_parts = [stripped]
        base_label, _ = parse_heading_label_and_title(stripped)
        lookahead = idx + 1
        while lookahead < len(lines) and len(combined_parts) < 3:
            next_line = normalize_for_match(lines[lookahead])
            lookahead += 1
            if not next_line or is_page_artifact_line(next_line):
                continue
            if len(next_line) > 90:
                break
            if next_line.endswith((".", "?", "!", ";", ":")):
                break
            if not is_probable_heading_line(next_line):
                break
            next_label, _ = parse_heading_label_and_title(next_line)
            if next_label:
                if base_label and next_label != base_label:
                    break
                if not base_label:
                    break
            combined_parts.append(next_line)
            combined_text = " ".join(combined_parts)
            if is_probable_heading_line(combined_text):
                add_candidate(combined_text, idx, lookahead - 1)

    return candidates


def collect_heading_candidates(section_text: str) -> List[str]:
    return [candidate["title"] for candidate in collect_heading_candidates_with_offsets(section_text)]


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


def extract_numeric_citation_numbers(citation_block: str) -> List[int]:
    return [int(value) for value in re.findall(r"\d+", citation_block)]


def is_math_like_numeric_citation(
    citation_block: str,
    prefix_text: str = "",
    suffix_text: str = "",
) -> bool:
    numbers = extract_numeric_citation_numbers(citation_block)
    if not numbers:
        return True
    if any(number <= 0 for number in numbers):
        return True

    prefix = normalize_for_match(prefix_text)
    suffix = normalize_for_match(suffix_text)
    math_prefix_pattern = re.compile(r"(?:∈|=|≤|≥|<|>|\\in|\\subseteq|\\subset)\s*$")
    if math_prefix_pattern.search(prefix_text):
        return True
    if re.search(r"[A-Za-z0-9_]\s*(?:∈|=|≤|≥|<|>)\s*$", prefix_text):
        return True

    joined = f"{prefix} {normalize_for_match(citation_block)} {suffix}".strip().lower()
    if any(token in joined for token in ("probability", "reward", "score", "interval", "range")) and len(numbers) <= 2:
        return True

    has_shape_context = any(term in joined for term in NUMERIC_CITATION_CONTEXT_TERMS)
    is_bracket_block = citation_block.startswith("[") and citation_block.endswith("]")
    inner_block = citation_block[1:-1] if is_bracket_block else citation_block[1:-1] if citation_block.startswith("(") and citation_block.endswith(")") else citation_block

    if is_bracket_block:
        comma_only = "," in inner_block and ";" not in inner_block and "-" not in inner_block and "–" not in inner_block
        if comma_only:
            # Filter tensor shapes / architecture tuples such as [784,100,10] or [2,2,1].
            if max(numbers, default=0) >= 50:
                return True
            if len(numbers) >= 3 and (numbers.count(1) >= 1 or has_shape_context):
                return True
            if len(numbers) >= 2 and has_shape_context and all(number <= 20 for number in numbers):
                return True

    if citation_block.startswith("(") and citation_block.endswith(")"):
        if len(numbers) == 1 and numbers[0] >= 1000:
            return True
        if len(numbers) == 1 and has_shape_context:
            return True
        if any(token in joined for token in ("equation", "figure", "table", "observation", "dataset", "batch size")):
            return True

    return False


def classify_citation_block(
    citation_block: str,
    prefix_text: str = "",
    suffix_text: str = "",
) -> Optional[str]:
    if re.fullmatch(AUTHOR_YEAR_CITATION_PATTERN, citation_block) or re.fullmatch(
        NARRATIVE_AUTHOR_YEAR_CITATION_PATTERN,
        citation_block,
    ):
        return CITATION_STYLE_AUTHOR_YEAR
    if re.fullmatch(NUMERIC_BRACKET_CITATION_PATTERN, citation_block):
        if is_math_like_numeric_citation(citation_block, prefix_text=prefix_text, suffix_text=suffix_text):
            return None
        return CITATION_STYLE_NUMERIC_BRACKET
    if re.fullmatch(NUMERIC_PAREN_CITATION_PATTERN, citation_block):
        if is_math_like_numeric_citation(citation_block, prefix_text=prefix_text, suffix_text=suffix_text):
            return None
        return CITATION_STYLE_NUMERIC_PAREN
    return None


def is_weak_numeric_paren_style_signal(citation_block: str) -> bool:
    """
    Parenthesized numeric blocks are often equation/step labels rather than citations.
    Treat low-information blocks like (1), (2), or short tuples like (1, 0, 2)
    as too weak to determine the paper's global citation style.
    """
    if not re.fullmatch(NUMERIC_PAREN_CITATION_PATTERN, citation_block):
        return False

    numbers = extract_numeric_citation_numbers(citation_block)
    if not numbers:
        return True
    if len(numbers) == 1:
        return True
    if len(numbers) <= 3 and all(number <= 10 for number in numbers):
        return True
    return False


def detect_dominant_citation_style(text: str) -> str:
    numeric_bracket_count = 0
    numeric_paren_count = 0
    author_year_count = 0

    for match in re.finditer(CITATION_BLOCK_PATTERN, text or ""):
        citation_block = match.group(0)
        prefix_text = (text or "")[max(0, match.start() - 24) : match.start()]
        suffix_text = (text or "")[match.end() : min(len(text or ""), match.end() + 24)]
        citation_style = classify_citation_block(
            citation_block,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
        )
        if citation_style == CITATION_STYLE_NUMERIC_BRACKET:
            numeric_bracket_count += 1
        elif citation_style == CITATION_STYLE_NUMERIC_PAREN:
            if is_weak_numeric_paren_style_signal(citation_block):
                continue
            numeric_paren_count += 1
        elif citation_style == CITATION_STYLE_AUTHOR_YEAR:
            author_year_count += 1

    numeric_count = numeric_bracket_count + numeric_paren_count
    if author_year_count > numeric_count:
        return CITATION_STYLE_AUTHOR_YEAR
    if numeric_bracket_count >= numeric_paren_count:
        return CITATION_STYLE_NUMERIC_BRACKET
    return CITATION_STYLE_NUMERIC_PAREN


def is_back_matter_heading(text: str) -> bool:
    normalized = normalize_heading_text(text)
    normalized = re.sub(r"^\d+(?:\.\d+)*\s*", "", normalized).strip()
    return normalized in {normalize_heading_text(name) for name in BACK_MATTER_HEADINGS}


def trim_text_before_back_matter(text: str) -> str:
    lines = (text or "").splitlines()
    for idx, line in enumerate(lines):
        if is_back_matter_heading(line):
            return "\n".join(lines[:idx])
    return text


def find_earliest_heading_start(full_text: str, heading_names: List[str]) -> int:
    earliest = -1
    for heading_name in heading_names:
        heading_start, _ = find_heading_line_offsets_global(full_text, heading_name)
        if heading_start == -1:
            continue
        if earliest == -1 or heading_start < earliest:
            earliest = heading_start
    return earliest


def find_heading_line_offsets_global(full_text: str, heading_name: str) -> Tuple[int, int]:
    def is_heading_scan_noise_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return True
        if re.fullmatch(r"\d+", stripped):
            return True
        if re.fullmatch(r"page\s+\d+(?:\s+of\s+\d+)?", stripped.lower()):
            return True
        if re.search(rf"\b(?:{MONTH_NAME_PATTERN})\b", stripped, flags=re.IGNORECASE) and re.search(
            r"\b(?:19|20)\d{2}\b", stripped
        ):
            if stripped.count(",") >= 2 or "conference" in stripped.lower():
                return True
        if re.search(r"\b[A-Z]{2,}\s*[’']?\d{2}\b", stripped) and re.search(r"\b(?:19|20)\d{2}\b", stripped):
            return True
        if re.search(r"\bFigure\s+\d+\s*:", stripped, flags=re.IGNORECASE):
            return True
        if re.search(r"\bTable\s+\d+\s*:", stripped, flags=re.IGNORECASE):
            return True
        return False

    target = normalize_heading_text(heading_name)
    if not target:
        return -1, -1

    target_core_key = heading_core_key(heading_name)
    candidate_entries = collect_heading_candidates_with_offsets(full_text)
    best_match: Optional[Dict[str, Any]] = None
    best_rank: Optional[Tuple[int, float, int, int]] = None
    for candidate_entry in candidate_entries:
        score = heading_similarity_score(heading_name, candidate_entry["title"])
        core_equal = heading_core_key(candidate_entry["title"]) == target_core_key
        if not core_equal and score < 0.75:
            continue

        candidate_core_tokens = heading_core_key(candidate_entry["title"]).split()
        target_core_tokens = target_core_key.split()
        rank = (
            1 if core_equal else 0,
            score,
            -abs(len(candidate_core_tokens) - len(target_core_tokens)),
            candidate_entry["start"],
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_match = candidate_entry

    if best_match is not None:
        return best_match["start"], best_match["content_start"]

    numbered_heading_pattern = re.compile(
        rf"(?<!\w)((?:\d+(?:\.\d+)*|[IVXLCDM]+))[\.\)]?\s+{re.escape(heading_name)}\b",
        flags=re.IGNORECASE,
    )

    direct_idx = full_text.find(heading_name)
    if direct_idx != -1:
        line_start = full_text.rfind("\n", 0, direct_idx) + 1
        line_end = full_text.find("\n", direct_idx)
        if line_end == -1:
            line_end = len(full_text)
        line_text = full_text[line_start:line_end]
        if not is_probable_heading_line(line_text):
            direct_idx = -1
        else:
            if heading_matches_expected_title(heading_name, line_text):
                return line_start, line_end

    lines = full_text.splitlines(keepends=True)
    offsets: List[int] = []
    running = 0
    for line in lines:
        offsets.append(running)
        running += len(line)

    for idx in range(len(lines)):
        if is_heading_scan_noise_line(lines[idx]):
            continue
        embedded_match = numbered_heading_pattern.search(lines[idx])
        if embedded_match:
            return offsets[idx] + embedded_match.start(1), offsets[idx] + len(lines[idx])
        if not is_probable_heading_line(lines[idx]):
            continue
        combined = ""
        consumed = 0
        end_idx = idx
        while end_idx < len(lines) and consumed < 6:
            if is_heading_scan_noise_line(lines[end_idx]):
                end_idx += 1
                continue
            if end_idx > idx and not is_probable_heading_line(lines[end_idx]):
                break

            consumed += 1
            chunk_lines = [line for line in lines[idx : end_idx + 1] if not is_heading_scan_noise_line(line)]
            if not chunk_lines:
                end_idx += 1
                break

            chunk = "".join(chunk_lines).replace("\n", " ").replace("\r", " ")
            normalized_line = strip_heading_label_prefix(chunk)
            combined = f"{combined} {normalized_line}".strip() if combined else normalized_line

            if heading_matches_expected_title(heading_name, chunk) or heading_matches_expected_title(
                heading_name, combined
            ):
                return offsets[idx], offsets[end_idx] + len(lines[end_idx])

            end_idx += 1

    return -1, -1


def extract_citations_by_section(
    text: str,
    sections: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract citation blocks with context and section content based on a nested section schema.
    """
    dominant_citation_style = detect_dominant_citation_style(text)

    def process_section_text(section_text: str) -> Dict[str, List[str]]:
        citations_dict: Dict[str, List[str]] = {}
        section_text = section_text.replace("\n", " ")
        # Repair words split across line breaks in PDF extraction, e.g. McK-\ninstry.
        section_text = re.sub(r"(?<=\w)-\s+(?=\w)", "-", section_text)
        section_text = re.sub(r"\s+", " ", section_text)

        for match in re.finditer(CITATION_BLOCK_PATTERN, section_text):
            citation_block = match.group(0)
            prefix_text = section_text[max(0, match.start() - 24) : match.start()]
            suffix_text = section_text[match.end() : min(len(section_text), match.end() + 24)]
            citation_style = classify_citation_block(
                citation_block,
                prefix_text=prefix_text,
                suffix_text=suffix_text,
            )
            if citation_style is None or citation_style != dominant_citation_style:
                continue
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

            canonical_citation = canonicalize_citation_key(citation_block)
            citations_dict.setdefault(canonical_citation, []).append(sentence)

        return citations_dict

    # Preserve appendix sections that may appear after the references.
    trimmed_text = text

    def extract_section_content(
        full_text: str,
        section_name: str,
        next_section_name: Optional[str] = None,
        child_section_names: Optional[List[str]] = None,
    ) -> str:
        section_start, content_start = find_heading_line_offsets_global(full_text, section_name)
        used_child_anchor = False

        if section_start == -1 and child_section_names:
            for child_name in child_section_names:
                child_start, child_content_start = find_heading_line_offsets_global(full_text, child_name)
                if child_start != -1:
                    section_start = child_start
                    content_start = child_start
                    used_child_anchor = True
                    break

        if section_start == -1:
            return ""

        if next_section_name:
            relative_next_start, _ = find_heading_line_offsets_global(full_text[content_start:], next_section_name)
            if relative_next_start == -1:
                section_end = len(full_text)
            else:
                section_end = content_start + relative_next_start
        else:
            section_end = len(full_text)

        back_matter_start = find_earliest_heading_start(full_text[content_start:section_end], list(BACK_MATTER_HEADINGS))
        if back_matter_start != -1:
            section_end = min(section_end, content_start + back_matter_start)

        if used_child_anchor:
            return full_text[content_start:section_end]
        return full_text[content_start:section_end]

    def recover_declared_children_from_titles(
        parent_name: str,
        section_text: str,
        subsections_dict: Dict[str, Any],
        title_hints: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        recovered_citations: Dict[str, Any] = {}
        recovered_content: Dict[str, Any] = {}
        candidate_entries = collect_heading_candidates_with_offsets(section_text)
        if not candidate_entries:
            return recovered_citations, recovered_content

        parent_matches = [
            entry
            for entry in candidate_entries
            if heading_similarity_score(parent_name, entry["title"]) >= 0.75
        ]
        parent_level = min((entry["level"] for entry in parent_matches if entry["level"] > 0), default=0)
        if parent_level > 0:
            filtered_candidates = [
                entry
                for entry in candidate_entries
                if entry["level"] == parent_level + 1 and heading_similarity_score(parent_name, entry["title"]) < 0.9
            ]
            if filtered_candidates:
                candidate_entries = filtered_candidates
        else:
            candidate_entries = [
                entry
                for entry in candidate_entries
                if heading_similarity_score(parent_name, entry["title"]) < 0.9
            ]

        found_children: List[Tuple[str, int, int]] = []
        next_candidate_idx = 0

        child_names = list(subsections_dict.keys())
        remaining_children = len(child_names)
        for child_name in child_names:
            title_options: List[str] = [child_name]
            hinted_title = (title_hints or {}).get(child_name)
            if hinted_title and normalize_heading_text(hinted_title) != normalize_heading_text(child_name):
                title_options.insert(0, hinted_title)

            max_start_idx = len(candidate_entries) - remaining_children
            best_match: Optional[Tuple[int, Dict[str, Any], float]] = None
            for idx in range(next_candidate_idx, max_start_idx + 1):
                candidate_entry = candidate_entries[idx]
                score = max(
                    heading_similarity_score(option, candidate_entry["title"])
                    for option in title_options
                )
                if best_match is None or score > best_match[2]:
                    best_match = (idx, candidate_entry, score)

            remaining_children -= 1
            if best_match is None or best_match[2] < 0.55:
                continue

            next_candidate_idx = best_match[0] + 1
            found_children.append((child_name, best_match[1]["start"], best_match[1]["content_start"]))

        found_children.sort(key=lambda item: item[1])

        for idx, (child_name, _, child_content_start) in enumerate(found_children):
            next_child_start = found_children[idx + 1][1] if idx + 1 < len(found_children) else len(section_text)
            child_end = next_child_start
            back_matter_start = find_earliest_heading_start(
                section_text[child_content_start:child_end],
                list(BACK_MATTER_HEADINGS),
            )
            if back_matter_start != -1:
                child_end = min(child_end, child_content_start + back_matter_start)

            child_text = section_text[child_content_start:child_end].strip()
            if not child_text:
                continue

            child_schema = subsections_dict.get(child_name)
            if isinstance(child_schema, dict) and child_schema:
                nested_citations, nested_content = process_sections(
                    child_text,
                    child_schema,
                    list(child_schema.keys()),
                )
                if nested_content:
                    recovered_citations[child_name] = nested_citations
                    recovered_content[child_name] = nested_content
                    continue

            recovered_citations[child_name] = process_section_text(child_text)
            recovered_content[child_name] = re.sub(r"[ \t]+", " ", child_text).strip()

        return recovered_citations, recovered_content

    def recover_declared_children(
        parent_name: str,
        section_text: str,
        subsections_dict: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return recover_declared_children_from_titles(
            parent_name=parent_name,
            section_text=section_text,
            subsections_dict=subsections_dict,
            title_hints=None,
        )

    def extract_section_lead_in(
        section_text: str,
        subsections_dict: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, List[str]]], Optional[str]]:
        child_names = list(subsections_dict.keys())
        if not child_names:
            return None, None

        child_starts = [
            start
            for child_name in child_names
            for start, _ in [find_heading_line_offsets_global(section_text, child_name)]
            if start != -1
        ]
        if not child_starts:
            return None, None

        first_child_start = min(child_starts)
        if first_child_start <= 0:
            return None, None

        lead_text = section_text[:first_child_start]
        sanitized_lead_text = sanitize_section_scoring_text(lead_text).strip()
        if not sanitized_lead_text or not re.search(r"[A-Za-z]", sanitized_lead_text):
            return None, None

        normalized_lead_text = re.sub(r"[ \t]+", " ", sanitized_lead_text).strip()
        if len(normalized_lead_text) < 40:
            return None, None

        return process_section_text(normalized_lead_text), normalized_lead_text

    def attach_section_lead_in(
        nested_citations: Dict[str, Any],
        nested_content: Dict[str, Any],
        content: str,
        subsections_dict: Dict[str, Any],
    ) -> None:
        if SECTION_LEAD_IN_NODE in nested_content:
            return

        lead_citations, lead_content = extract_section_lead_in(content, subsections_dict)
        if not lead_content:
            return

        ordered_content = {SECTION_LEAD_IN_NODE: lead_content}
        ordered_content.update(nested_content)
        nested_content.clear()
        nested_content.update(ordered_content)

        ordered_citations = {SECTION_LEAD_IN_NODE: lead_citations or {}}
        ordered_citations.update(nested_citations)
        nested_citations.clear()
        nested_citations.update(ordered_citations)

    def match_declared_sections_in_order(
        section_text: str,
        expected_names: List[str],
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        candidate_entries = collect_heading_candidates_with_offsets(section_text)
        if not candidate_entries or not expected_names:
            return []

        numbered_levels = sorted({entry["level"] for entry in candidate_entries if entry["level"] > 0})
        if numbered_levels:
            best_pool = candidate_entries
            best_pool_score = -1.0
            for level in numbered_levels:
                candidate_pool = [entry for entry in candidate_entries if entry["level"] == level]
                if not candidate_pool:
                    continue

                pool_score = 0.0
                next_candidate_idx = 0
                remaining_expected = len(expected_names)
                for expected_name in expected_names:
                    if next_candidate_idx >= len(candidate_pool):
                        break

                    max_start_idx = len(candidate_pool) - remaining_expected
                    if max_start_idx < next_candidate_idx:
                        max_start_idx = len(candidate_pool) - 1

                    best_local_idx: Optional[int] = None
                    best_local_score = 0.0
                    for idx in range(next_candidate_idx, max_start_idx + 1):
                        score = heading_similarity_score(expected_name, candidate_pool[idx]["title"])
                        if best_local_idx is None or score > best_local_score:
                            best_local_idx = idx
                            best_local_score = score

                    remaining_expected -= 1
                    if best_local_idx is None:
                        continue

                    next_candidate_idx = best_local_idx + 1
                    pool_score += best_local_score

                if pool_score > best_pool_score:
                    best_pool_score = pool_score
                    best_pool = candidate_pool

            if best_pool_score > 0:
                candidate_entries = best_pool

        matches: List[Tuple[str, Dict[str, Any], float]] = []
        next_candidate_idx = 0
        remaining_expected = len(expected_names)

        for expected_name in expected_names:
            if next_candidate_idx >= len(candidate_entries):
                break

            max_start_idx = len(candidate_entries) - remaining_expected
            if max_start_idx < next_candidate_idx:
                max_start_idx = len(candidate_entries) - 1

            best_match: Optional[Tuple[int, Dict[str, Any], float]] = None
            for idx in range(next_candidate_idx, max_start_idx + 1):
                candidate_entry = candidate_entries[idx]
                score = heading_similarity_score(expected_name, candidate_entry["title"])
                if best_match is None or score > best_match[2]:
                    best_match = (idx, candidate_entry, score)

            remaining_expected -= 1
            if best_match is None or best_match[2] < 0.55:
                continue

            next_candidate_idx = best_match[0] + 1
            matches.append((expected_name, best_match[1], best_match[2]))

        return matches

    def process_sections(
        section_text: str, sections_dict: Dict[str, Any], section_names_list: List[str]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        local_citations: Dict[str, Any] = {}
        local_content: Dict[str, Any] = {}

        matched_sections = match_declared_sections_in_order(section_text, section_names_list)
        if len(matched_sections) == len(section_names_list):
            for idx, (section_name, match_entry, _) in enumerate(matched_sections):
                subsections = sections_dict[section_name]
                section_start = match_entry["start"]
                content_start = match_entry["content_start"]
                next_start = (
                    matched_sections[idx + 1][1]["start"]
                    if idx + 1 < len(matched_sections)
                    else len(section_text)
                )
                section_end = next_start
                back_matter_start = find_earliest_heading_start(
                    section_text[content_start:section_end],
                    list(BACK_MATTER_HEADINGS),
                )
                if back_matter_start != -1:
                    section_end = min(section_end, content_start + back_matter_start)

                content = section_text[content_start:section_end].strip()
                if not content:
                    continue

                if isinstance(subsections, dict) and subsections:
                    subsection_names = list(subsections.keys())
                    nested_citations, nested_content = process_sections(content, subsections, subsection_names)
                    if len(nested_content) < len(subsection_names):
                        recovered_citations, recovered_content = recover_declared_children(
                            section_name,
                            content,
                            subsections,
                        )
                        for child_name, child_content in recovered_content.items():
                            if child_name in nested_content:
                                continue
                            nested_content[child_name] = child_content
                            nested_citations[child_name] = recovered_citations.get(child_name, {})

                    if nested_content:
                        attach_section_lead_in(nested_citations, nested_content, content, subsections)
                        local_citations[section_name] = nested_citations
                        local_content[section_name] = nested_content
                    else:
                        local_citations[section_name] = process_section_text(content)
                        normalized_content = re.sub(r"[ \t]+", " ", content).strip()
                        local_content[section_name] = normalized_content
                else:
                    local_citations[section_name] = process_section_text(content)
                    normalized_content = re.sub(r"[ \t]+", " ", content).strip()
                    local_content[section_name] = normalized_content

            if local_content:
                return local_citations, local_content

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
                if len(nested_content) < len(subsection_names):
                    recovered_citations, recovered_content = recover_declared_children(
                        section_name,
                        content,
                        subsections,
                    )
                    for child_name, child_content in recovered_content.items():
                        if child_name in nested_content:
                            continue
                        nested_content[child_name] = child_content
                        nested_citations[child_name] = recovered_citations.get(child_name, {})

                if nested_content:
                    attach_section_lead_in(nested_citations, nested_content, content, subsections)
                    local_citations[section_name] = nested_citations
                    local_content[section_name] = nested_content
                else:
                    # Preserve parent text if child extraction fails instead of
                    # collapsing the section into an empty subtree.
                    local_citations[section_name] = process_section_text(content)
                    normalized_content = re.sub(r"[ \t]+", " ", content).strip()
                    local_content[section_name] = normalized_content
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


def strip_reasoning_blocks(response_text: str) -> str:
    text = strip_code_fences(response_text)
    # Prefer the final answer block over reflective preambles emitted by some models.
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


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


def parse_plaintext_score_lines(
    response_text: str,
    allow_percentage: bool = False,
) -> List[Tuple[str, float]]:
    text = strip_reasoning_blocks(response_text)
    blocks: List[List[Tuple[str, float]]] = []
    current_block: List[Tuple[str, float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().strip(",")
        if not line:
            if current_block:
                blocks.append(current_block)
                current_block = []
            continue
        line = re.sub(r"^[-*•]\s*", "", line)
        match = re.match(
            r'^"?(.{1,120}?)"?\s*(?::|=|->|=>|-|–)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\s*%?)\s*$',
            line,
        )
        if not match:
            if current_block:
                blocks.append(current_block)
                current_block = []
            continue
        key = match.group(1).strip()
        value = coerce_non_negative_number(match.group(2), allow_percentage=allow_percentage)
        if value is None:
            if current_block:
                blocks.append(current_block)
                current_block = []
            continue
        current_block.append((key, value))
    if current_block:
        blocks.append(current_block)

    if not blocks:
        return []

    best_block = max(
        enumerate(blocks),
        key=lambda item: (len(item[1]), item[0]),
    )[1]
    return best_block


def parse_plaintext_title_lines(response_text: str) -> List[Tuple[str, str]]:
    text = strip_code_fences(response_text)
    pairs: List[Tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().strip(",")
        if not line:
            continue
        line = re.sub(r"^[-*•]\s*", "", line)
        match = re.match(r'^"?(.{1,200}?)"?\s*(?::|=|->|=>|-|–)\s*(.+?)\s*$', line)
        if not match:
            continue
        key = match.group(1).strip().strip('"').strip("'")
        value = match.group(2).strip().strip('"').strip("'")
        if not key or not value:
            continue
        pairs.append((key, value))
    return pairs


def parse_ordered_score_values(response_text: str, allow_percentage: bool = False) -> List[float]:
    text = strip_reasoning_blocks(response_text)
    blocks: List[List[float]] = []
    current_block: List[float] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().strip(",")
        if not line:
            if current_block:
                blocks.append(current_block)
                current_block = []
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
            if current_block:
                blocks.append(current_block)
                current_block = []
            continue
        current_block.append(score_val)
    if current_block:
        blocks.append(current_block)

    if not blocks:
        return []

    best_block = max(
        enumerate(blocks),
        key=lambda item: (len(item[1]), item[0]),
    )[1]
    return best_block


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

    def resolve_target_id(key: Any) -> Optional[str]:
        key_str = str(key)
        return key_str if key_str in expected_set else norm_to_id.get(normalized_key(key_str))

    def assign_from_pair(key: Any, value: Any) -> None:
        target_id = resolve_target_id(key)
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
    plaintext_pairs = parse_plaintext_score_lines(
        response_text,
        allow_percentage=allow_percentage,
    )
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

    resolved_plaintext_pairs: List[Tuple[str, float]] = []
    unresolved_plaintext = 0
    for key, value in plaintext_pairs:
        target_id = resolve_target_id(key)
        if target_id is None:
            unresolved_plaintext += 1
            continue
        resolved_plaintext_pairs.append((target_id, value))

    if resolved_plaintext_pairs and unresolved_plaintext == 0:
        resolved_ids = [target_id for target_id, _ in resolved_plaintext_pairs]
        if len(resolved_plaintext_pairs) == len(expected_ids) and len(set(resolved_ids)) == len(expected_ids):
            return {
                target_id: value
                for target_id, value in resolved_plaintext_pairs
            }
    elif resolved_plaintext_pairs:
        for target_id, value in resolved_plaintext_pairs:
            parsed_scores[target_id] = value

    json_payload = parse_json_loose(response_text)
    if isinstance(json_payload, dict):
        for key, value in json_payload.items():
            assign_from_pair(key, value)

        # Accept simple array forms under common wrapper keys.
        for seq_key in ("scores", "values", "allocations", "distribution"):
            seq = json_payload.get(seq_key)
            if not isinstance(seq, list):
                continue
            if not seq or len(seq) != len(expected_ids):
                continue
            if all(not isinstance(entry, dict) for entry in seq):
                for idx, entry in enumerate(seq):
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
        if json_payload and len(json_payload) == len(expected_ids) and all(not isinstance(entry, dict) for entry in json_payload):
            for idx, entry in enumerate(json_payload):
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


def parse_title_map_from_response(
    response_text: str,
    expected_children: List[str],
) -> Dict[str, str]:
    if not expected_children:
        return {}

    expected_set = set(expected_children)
    norm_to_child = {normalized_key(child_name): child_name for child_name in expected_children}
    recovered: Dict[str, str] = {}
    null_tokens = {"none", "null", "n/a", "na", "not found", "missing", "unknown"}

    def coerce_title(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().strip('"').strip("'")
        if not text:
            return None
        if normalized_key(text) in {normalized_key(token) for token in null_tokens}:
            return None
        return text

    json_payload = parse_json_loose(response_text)
    if isinstance(json_payload, dict):
        for key, value in json_payload.items():
            child_name = key if key in expected_set else norm_to_child.get(normalized_key(str(key)))
            if not child_name:
                continue
            title = coerce_title(value)
            if title is not None:
                recovered[child_name] = title
        if recovered:
            return recovered

    for key, value in parse_plaintext_title_lines(response_text):
        child_name = key if key in expected_set else norm_to_child.get(normalized_key(key))
        if not child_name:
            continue
        title = coerce_title(value)
        if title is not None:
            recovered[child_name] = title
    if recovered:
        return recovered

    ordered_values: List[str] = []
    text = strip_code_fences(response_text)
    for raw_line in text.splitlines():
        line = raw_line.strip().strip(",")
        if not line:
            continue
        line = re.sub(r"^[-*•]\s*", "", line)
        title = coerce_title(line)
        if title is not None:
            ordered_values.append(title)

    for child_name, title in zip(expected_children, ordered_values):
        recovered[child_name] = title
    return recovered


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
        f"sample_temperature_jitter={max(0.0, args.sample_temperature_jitter)} "
        f"seed={args.seed if args.seed is not None else 'none'} "
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


def apply_minimum_positive_floor(
    scores: Dict[str, float],
    total: float,
    min_fraction: float = 0.005,
) -> Dict[str, float]:
    if not scores:
        return {}
    if len(scores) == 1:
        return dict(scores)

    floor_value = max(1e-12, total * max(0.0, min_fraction))
    adjusted = {key: max(floor_value, safe_float(value, 0.0)) for key, value in scores.items()}
    return normalize_distribution(adjusted, total)


def counterbalanced_item_order(items: List[str], sample_idx: int) -> List[str]:
    if len(items) <= 1:
        return list(items)

    rotation = sample_idx % len(items)
    ordered = list(items[rotation:]) + list(items[:rotation])

    # Alternate direction so head/tail items do not consistently benefit
    # from the same left-to-right exposure pattern across samples.
    if sample_idx % 2 == 1:
        ordered.reverse()
    return ordered


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def combine_scores(technical_score: float, citation_score: float) -> float:
    return max(0.0, safe_float(technical_score, 0.0)) + max(0.0, safe_float(citation_score, 0.0))


def citation_fallback_token_count(text: str) -> int:
    return max(1, len(re.findall(r"[A-Za-z0-9_]+", text or "")))


def assert_close(actual: float, expected: float, context: str, tol: float = 1e-8) -> None:
    if abs(actual - expected) > tol:
        print(
            f"[WARN] Score conservation drift at {context}: expected {expected:.12f}, "
            f"got {actual:.12f}, diff={abs(actual - expected):.12f}"
        )


def validate_allocation_distribution(
    parsed_scores: Dict[str, float],
    item_ids: List[str],
    mode: str = "strict",
) -> Optional[str]:
    if set(parsed_scores.keys()) != set(item_ids):
        return "incomplete_keys"

    values = [max(0.0, safe_float(parsed_scores[item_id], 0.0)) for item_id in item_ids]
    if not any(value > 0.0 for value in values):
        return "all_zero"

    score_sum = sum(values)
    if score_sum <= 0.0:
        return "non_positive_sum"

    if mode == "relaxed":
        return None

    normalized = [value / score_sum for value in values]
    positive_count = sum(1 for value in normalized if value > 1e-9)
    zero_count = len(normalized) - positive_count
    max_share = max(normalized)
    sorted_shares = sorted(normalized, reverse=True)
    second_share = sorted_shares[1] if len(sorted_shares) > 1 else 0.0

    if len(item_ids) > 1 and zero_count > 0:
        return "zero_child_not_allowed"
    if len(item_ids) >= 4 and positive_count < max(3, len(item_ids) // 2):
        return "too_many_zero_children"
    if len(item_ids) >= 4 and max_share >= 0.75 and zero_count >= 2:
        return "pathological_concentration"
    if len(item_ids) >= 5 and max_share >= 0.65 and second_share <= 0.20 and zero_count >= 1:
        return "dominant_head_with_sparse_tail"

    return None


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


def flatten_content_with_names(content: Any, limit: Optional[int] = 700) -> str:
    if isinstance(content, str):
        if limit is None or limit <= 0:
            return content
        return content[:limit]

    if isinstance(content, dict):
        chunks: List[str] = []
        remaining = None if limit is None or limit <= 0 else limit
        for name, value in content.items():
            child_limit = remaining if remaining is not None else None
            child_text = flatten_content_with_names(value, limit=child_limit)
            labeled_chunk = f"{name}\n{child_text}".strip() if child_text else str(name)
            chunks.append(labeled_chunk)
            if remaining is not None:
                remaining -= len(labeled_chunk)
            if remaining is not None and remaining <= 0:
                break

        merged = "\n\n".join(chunks)
        if limit is None or limit <= 0:
            return merged
        return merged[:limit]

    return ""


def sanitize_section_scoring_text(text: str) -> str:
    if not text:
        return ""

    kept_lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if kept_lines and kept_lines[-1]:
                kept_lines.append("")
            continue
        if is_page_artifact_line(line):
            continue

        numeric_tokens = len(re.findall(r"\b\d+(?:\.\d+)?(?:e[-+]?\d+)?%?\b", line))
        alpha_tokens = len(re.findall(r"[A-Za-z]+", line))

        # Drop short table-like or metric-heavy lines that tend to leak numbers into score outputs.
        if numeric_tokens >= 4 and (alpha_tokens <= numeric_tokens or len(line) <= 140):
            continue
        if numeric_tokens >= 3 and alpha_tokens <= 2:
            continue

        kept_lines.append(line)

    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def format_scored_segment_excerpt(segment_name: str, segment_content: Any, limit: Optional[int] = 700) -> str:
    body = sanitize_section_scoring_text(flatten_content_with_names(segment_content, limit=limit))
    if body:
        return f"Section/subsection name: {segment_name}\nContent:\n{body}"
    return f"Section/subsection name: {segment_name}"


def normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = re.sub(
        r"^(\d+(?:\.\d+)*[\.\)]?)(?=[A-Za-zεϵ])",
        r"\1 ",
        normalized,
    )
    return re.sub(r"\s+", " ", normalized).strip()


def is_page_artifact_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.fullmatch(r"\d+", stripped):
        return True
    if re.fullmatch(r"page\s+\d+(?:\s+of\s+\d+)?", stripped.lower()):
        return True
    if re.search(rf"\b(?:{MONTH_NAME_PATTERN})\b", stripped, flags=re.IGNORECASE) and re.search(r"\b(?:19|20)\d{2}\b", stripped):
        if stripped.count(",") >= 2 or "proceedings" in stripped.lower() or "conference" in stripped.lower():
            return True
    if re.search(r"\b[A-Z]{2,}\s*[’']?\d{2}\b", stripped) and re.search(r"\b(?:19|20)\d{2}\b", stripped):
        return True
    if re.fullmatch(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:,\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})+(?:,\s+and\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})?", stripped):
        return True
    if re.search(r"\bFigure\s+\d+\s*:", stripped, flags=re.IGNORECASE):
        return True
    if re.search(r"\bTable\s+\d+\s*:", stripped, flags=re.IGNORECASE):
        return True
    return False


def is_probable_heading_line(line: str) -> bool:
    stripped = normalize_for_match(line)
    if not stripped or len(stripped) > 120:
        return False
    if is_page_artifact_line(stripped):
        return False
    if stripped.endswith((".", "!", ";")):
        return False
    if "@" in stripped:
        return False
    if stripped.count(",") >= 2:
        return False

    heading_number = re.match(r"^(?:\d+(?:\.\d+)*|[A-Z])(?:[\.\)])?\s+(.+)$", stripped)
    if heading_number:
        remainder = heading_number.group(1).strip()
        if remainder and len(remainder.split()) <= 14:
            return True

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", stripped)
    if not words or len(words) > 14:
        return False

    if len(words) >= 6 and all(word[:1].isupper() for word in words[: min(len(words), 6)]):
        lowered = stripped.lower()
        if any(
            marker in lowered
            for marker in (
                "university",
                "institute",
                "department",
                "arlington",
                "newark",
                "mumbai",
                "united states",
                "open access",
                "support provided by",
            )
        ):
            return False

    uppercase_like = sum(1 for word in words if word.isupper() and len(word) > 1)
    title_like = sum(1 for word in words if word[:1].isupper())
    connector_words = {"and", "or", "of", "the", "on", "in", "with", "for", "to", "a", "an", "vs"}
    lower_nonconnector = sum(
        1
        for word in words
        if word[:1].islower() and word.lower() not in connector_words
    )

    if lower_nonconnector >= 2:
        return False

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


def is_plot_or_caption_line(line: str) -> bool:
    stripped = normalize_for_match(line)
    if not stripped:
        return False

    figure_refs = len(re.findall(r"\b(?:Figure|Table)\s+\d+\s*:", stripped, flags=re.IGNORECASE))
    panel_refs = len(re.findall(r"\([a-z]\)", stripped, flags=re.IGNORECASE))
    chart_terms = len(
        re.findall(
            r"\b(?:precision|recall|ndcg|runtime|running time|scalability|baseline|synthetic dataset|indexed|exact|approx(?:imate)?)\b",
            stripped,
            flags=re.IGNORECASE,
        )
    )
    numeric_tokens = len(re.findall(r"\b\d+(?:\.\d+)?\b", stripped))

    if figure_refs >= 1:
        return True
    if panel_refs >= 2 and (chart_terms >= 1 or numeric_tokens >= 2):
        return True
    if panel_refs >= 3:
        return True
    if chart_terms >= 3 and numeric_tokens >= 2 and len(stripped.split()) <= 40:
        return True
    return False


def is_non_prose_artifact_paragraph(text: str) -> bool:
    normalized = normalize_for_match(text)
    if not normalized:
        return False

    figure_refs = len(re.findall(r"\b(?:Figure|Table)\s+\d+\s*:", normalized, flags=re.IGNORECASE))
    panel_refs = len(re.findall(r"\([a-z]\)", normalized, flags=re.IGNORECASE))
    chart_terms = len(
        re.findall(
            r"\b(?:precision|recall|ndcg|runtime|running time|scalability|baseline|synthetic dataset|indexed|exact|approx(?:imate)?|dataset)\b",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    numeric_tokens = len(re.findall(r"\b\d+(?:\.\d+)?\b", normalized))
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", normalized)
    title_like = bool(
        re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){3,}\b", normalized)
        and re.search(r"\b(?:19|20)\d{2}\b", normalized)
    )
    if figure_refs >= 1 and (panel_refs >= 1 or title_like):
        return True
    if figure_refs >= 2:
        return True
    if panel_refs >= 4:
        return True
    if normalized.lower().startswith("(a)") and panel_refs >= 2:
        return True
    if panel_refs >= 2 and len(words) <= 24:
        return True
    if chart_terms >= 2 and len(words) <= 18 and numeric_tokens <= 4:
        return True
    if re.fullmatch(r"lines on .*dataset", normalized, flags=re.IGNORECASE):
        return True
    return False


def is_inline_subheading_sentence(sentence: str) -> bool:
    stripped = normalize_for_match(sentence)
    if not stripped or len(stripped) > 90:
        return False
    if not stripped.endswith((".", ":")):
        return False
    core = stripped[:-1].strip()
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", core)
    if not words or len(words) > 10:
        return False
    if any(word.isdigit() for word in words):
        return False

    title_like = sum(1 for word in words if word[:1].isupper() or word.lower() in {"and", "or", "of", "the", "for", "in"})
    return title_like >= max(1, len(words) - 1)


def starts_with_inline_subheading_label(sentence: str) -> bool:
    stripped = normalize_for_match(sentence)
    if not stripped:
        return False

    return bool(
        re.match(
            r"^(?:[a-z]|[ivxlcdm]+|\d+)[\.\)]\s+[A-Z][^:]{0,80}:",
            stripped,
            flags=re.IGNORECASE,
        )
    )


def split_paragraph_on_inline_subheadings(paragraph: str) -> List[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalize_for_match(paragraph)) if s.strip()]
    if len(sentences) <= 1:
        return [normalize_for_match(paragraph)] if normalize_for_match(paragraph) else []

    chunks: List[str] = []
    current: List[str] = []
    for idx, sentence in enumerate(sentences):
        if current and (
            is_inline_subheading_sentence(sentence)
            or starts_with_inline_subheading_label(sentence)
        ):
            merged_current = normalize_for_match(" ".join(current))
            if merged_current:
                chunks.append(merged_current)
            current = [sentence]
            continue
        current.append(sentence)

    merged_current = normalize_for_match(" ".join(current))
    if merged_current:
        chunks.append(merged_current)
    return chunks


def merge_lowercase_continuation_paragraphs(paragraphs: List[str]) -> List[str]:
    merged: List[str] = []
    for paragraph in paragraphs:
        normalized = normalize_for_match(paragraph)
        if not normalized:
            continue

        if (
            merged
            and re.match(r'^[\(\["“]?[a-z]', normalized)
            and not starts_with_inline_subheading_label(normalized)
        ):
            merged[-1] = normalize_for_match(f"{merged[-1]} {normalized}")
            continue

        merged.append(normalized)
    return merged


def clean_text_for_paragraph_split(text: str) -> str:
    cleaned_lines: List[str] = []
    for raw_line in text.split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if is_page_artifact_line(stripped) or is_plot_or_caption_line(stripped):
            cleaned_lines.append("")
            continue
        cleaned_lines.append(raw_line.rstrip())
    return "\n".join(cleaned_lines)


def median_line_length(lines: List[str]) -> int:
    lengths = sorted(len(line.strip()) for line in lines if len(line.strip()) >= 20)
    if not lengths:
        return 80
    mid = len(lengths) // 2
    if len(lengths) % 2 == 1:
        return lengths[mid]
    return (lengths[mid - 1] + lengths[mid]) // 2


def merge_paragraph_lines(lines: List[str]) -> str:
    merged = normalize_for_match(" ".join(line.strip() for line in lines if line.strip()))
    merged = re.sub(r"\s+\d{1,4}\s*$", "", merged).strip()
    return merged


def rebuild_paragraphs_from_lines(text: str) -> List[str]:
    lines = [line.rstrip() for line in text.split("\n")]
    content_lines = [
        line for line in lines if line.strip() and not is_page_artifact_line(line) and not is_plot_or_caption_line(line)
    ]
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
        if paragraph and not is_page_artifact_line(paragraph):
            paragraphs.append(paragraph)

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or is_page_artifact_line(stripped) or is_plot_or_caption_line(stripped):
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
        if is_non_prose_artifact_paragraph(paragraph):
            continue
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
    normalized = trim_text_before_back_matter(normalized).strip()
    normalized = clean_text_for_paragraph_split(normalized).strip()
    if not normalized:
        return []

    paragraphs = [
        normalize_for_match(p)
        for p in re.split(r"\n\s*\n+", normalized)
        if normalize_for_match(p) and not is_non_prose_artifact_paragraph(normalize_for_match(p))
    ]
    if len(paragraphs) > 1:
        refined: List[str] = []
        for paragraph in paragraphs:
            refined.extend(split_paragraph_on_inline_subheadings(paragraph))
        refined = merge_lowercase_continuation_paragraphs(refined)
        return [p for p in refined if p and not is_non_prose_artifact_paragraph(p)]

    rebuilt = rebuild_paragraphs_from_lines(normalized)
    if rebuilt:
        refined = []
        for paragraph in rebuilt:
            refined.extend(split_paragraph_on_inline_subheadings(paragraph))
        refined = merge_lowercase_continuation_paragraphs(refined)
        return [p for p in refined if p and not is_non_prose_artifact_paragraph(p)]

    # PDF extraction often collapses paragraphs; use sentence chunks as a last fallback.
    single = normalize_for_match(normalized)
    if not single:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", single) if s.strip()]
    if len(sentences) <= 3:
        return [single]

    chunk_size = 5
    chunks = [" ".join(sentences[i : i + chunk_size]).strip() for i in range(0, len(sentences), chunk_size)]
    refined = []
    for chunk in chunks:
        refined.extend(split_paragraph_on_inline_subheadings(chunk))
    refined = merge_lowercase_continuation_paragraphs(refined)
    return [p for p in refined if p and not is_non_prose_artifact_paragraph(p)]


def split_into_sentences(text: str) -> List[str]:
    normalized = normalize_for_match(text)
    if not normalized:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalized) if s.strip()]


def allocate_leaf_citation_scores_length_fallback(
    paragraph_items: Dict[str, str],
    mention_buckets: Dict[str, List[Tuple[str, str, str]]],
    section_score: float,
    citation_fraction: float = 0.45,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]]:
    """Fallback leaf citation allocation mirroring the length-based baseline."""
    paragraph_names = list(paragraph_items.keys())
    paragraph_raw = {
        paragraph_name: float(citation_fallback_token_count(paragraph_items[paragraph_name]))
        + 50.0 * len(mention_buckets.get(paragraph_name, []))
        for paragraph_name in paragraph_names
    }
    paragraph_total = normalize_distribution(paragraph_raw, section_score)

    global_mentions: Dict[str, int] = {}
    for paragraph_mentions in mention_buckets.values():
        for citation, _, _ in paragraph_mentions:
            global_mentions[citation] = global_mentions.get(citation, 0) + 1

    if not global_mentions:
        return (
            paragraph_total,
            {paragraph_name: paragraph_total[paragraph_name] for paragraph_name in paragraph_names},
            {paragraph_name: 0.0 for paragraph_name in paragraph_names},
            {paragraph_name: {} for paragraph_name in paragraph_names},
        )

    section_token_total = citation_fallback_token_count(" ".join(paragraph_items.values()))
    section_citation_score = max(0.0, safe_float(section_score, 0.0)) * max(0.0, min(0.95, citation_fraction))
    citation_raw = {
        citation: float(mentions) / float(section_token_total)
        for citation, mentions in global_mentions.items()
        if mentions > 0
    }
    citation_budget = normalize_distribution(citation_raw, section_citation_score) if citation_raw else {}

    paragraph_technical: Dict[str, float] = {}
    paragraph_citation: Dict[str, float] = {}
    paragraph_citation_allocations: Dict[str, Dict[str, float]] = {}

    for paragraph_name in paragraph_names:
        mention_counter: Dict[str, int] = {}
        for citation, _, _ in mention_buckets.get(paragraph_name, []):
            mention_counter[citation] = mention_counter.get(citation, 0) + 1

        local_raw = {
            citation: float(count) * citation_budget.get(citation, 0.0)
            for citation, count in mention_counter.items()
            if citation_budget.get(citation, 0.0) > 0.0
        }
        if local_raw:
            target_local_total = min(
                paragraph_total[paragraph_name] * max(0.0, min(0.95, citation_fraction)),
                sum(local_raw.values()),
            )
            alloc = normalize_distribution(local_raw, target_local_total)
        else:
            alloc = {}

        paragraph_citation_score = sum(alloc.values())
        if paragraph_citation_score > paragraph_total[paragraph_name]:
            scale = paragraph_total[paragraph_name] / max(paragraph_citation_score, 1e-12)
            alloc = {citation: value * scale for citation, value in alloc.items()}
            paragraph_citation_score = sum(alloc.values())

        paragraph_citation_allocations[paragraph_name] = alloc
        paragraph_citation[paragraph_name] = max(0.0, paragraph_citation_score)
        paragraph_technical[paragraph_name] = max(0.0, paragraph_total[paragraph_name] - paragraph_citation_score)

    return paragraph_total, paragraph_technical, paragraph_citation, paragraph_citation_allocations


def allocate_paragraph_citation_scores_length_fallback(
    mentions: List[Tuple[str, str, str]],
    total_score: float,
) -> Dict[str, float]:
    citation_counts: Dict[str, int] = {}
    for citation, _, _ in mentions:
        citation_counts[citation] = citation_counts.get(citation, 0) + 1
    if not citation_counts:
        return {}
    raw = {citation: float(count) for citation, count in citation_counts.items() if count > 0}
    return normalize_distribution(raw, total_score)


def is_citation_heavy_section_name(section_name: str) -> bool:
    norm = normalized_key(section_name)
    return any(
        token in norm
        for token in (
            "relatedwork",
            "priorwork",
            "literaturereview",
            "background",
            "survey",
        )
    )


def extract_citation_focus_sentences(
    paragraph_text: str,
    mentions: List[Tuple[str, str, str]],
) -> List[str]:
    if not paragraph_text or not mentions:
        return []

    sentences = split_into_sentences(paragraph_text)
    if not sentences:
        return []

    context_texts = [normalize_for_match(str(context)) for _, _, context in mentions if normalize_for_match(str(context))]
    context_token_sets = [set(re.findall(r"[a-z0-9]+", ctx.lower())) for ctx in context_texts if ctx]
    prior_work_pattern = re.compile(
        r"\b(prior work|previous work|related work|baseline|compare|comparison|compared|method|methods|algorithm|algorithms|study|studies|approach|approaches)\b",
        flags=re.IGNORECASE,
    )

    selected: List[str] = []
    for sentence in sentences:
        sentence_norm = normalize_for_match(sentence)
        if not sentence_norm:
            continue

        contains_citation_marker = bool(re.search(CITATION_BLOCK_PATTERN, sentence_norm))
        sentence_tokens = set(re.findall(r"[a-z0-9]+", sentence_norm.lower()))
        overlaps_context = False
        if sentence_tokens:
            for context_tokens in context_token_sets:
                if not context_tokens:
                    continue
                overlap = len(sentence_tokens & context_tokens)
                if overlap >= 4 or overlap >= max(2, int(0.4 * min(len(sentence_tokens), len(context_tokens)))):
                    overlaps_context = True
                    break

        if contains_citation_marker or overlaps_context:
            selected.append(sentence_norm)
            continue

        if prior_work_pattern.search(sentence_norm) and any(
            sentence_norm in ctx or ctx in sentence_norm for ctx in context_texts
        ):
            selected.append(sentence_norm)

    return [sentence for sentence in dict.fromkeys(selected) if sentence]


def build_citation_focus_text(
    paragraph_text: str,
    mentions: List[Tuple[str, str, str]],
    limit: int = 900,
) -> str:
    focus_sentences = extract_citation_focus_sentences(paragraph_text, mentions)
    if not focus_sentences:
        return ""
    focus = " ".join(focus_sentences)
    return focus[:limit]


def citation_focus_ratio(paragraph_text: str, mentions: List[Tuple[str, str, str]]) -> float:
    sentences = split_into_sentences(paragraph_text)
    if not sentences:
        return 0.0
    focus_sentences = extract_citation_focus_sentences(paragraph_text, mentions)
    if not focus_sentences:
        return 0.0
    return min(1.0, len(focus_sentences) / max(1, len(sentences)))


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
    snippets = {
        name: format_scored_segment_excerpt(name, item_to_content[name], limit=use_limit)
        for name in items
    }
    parent_content = sanitize_section_scoring_text(flatten_content_with_names(item_to_content, limit=None))
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
            think=False,
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
    validation_mode: str = "strict",
) -> Dict[str, float]:
    items = list(item_to_content.keys())
    use_limit = None if snippet_limit <= 0 else snippet_limit
    snippets = {
        name: format_scored_segment_excerpt(name, item_to_content[name], limit=use_limit)
        for name in items
    }
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
    parent_content = sanitize_section_scoring_text(
        flatten_content_with_names(item_to_content, limit=parent_limit)
    )

    system_prompt = SECTION_DIRECT_SYSTEM_PROMPT
    base_prompt = SECTION_DIRECT_USER_PROMPT_TEMPLATE.format(
        parent_name=parent_name,
        parent_score=total_score,
        parent_content=parent_content,
        items=json.dumps(item_payload, indent=2),
    )
    prompt = base_prompt
    alias_to_id = {item_id_to_name[item_id]: item_id for item_id in item_ids}

    for attempt in range(max(1, max_retries)):
        response = client.chat(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            options={"temperature": temperature},
            think=False,
        )
        raw_response = response.message.content if getattr(response, "message", None) else ""
        append_debug_log(
            debug_log_path,
            (
                f"[{log_tag}] parent={parent_name} sample={sample_idx} "
                f"attempt={attempt + 1}/{max(1, max_retries)}\n{raw_response}\n"
            ),
        )
        parsed_full = parse_score_map_from_response(
            raw_response,
            item_ids,
            allow_percentage=True,
            alias_to_id=alias_to_id,
        )
        parsed_scores = {
            item_id: max(0.0, safe_float(value, 0.0))
            for item_id, value in parsed_full.items()
        }

        rejection_reason = validate_allocation_distribution(
            parsed_scores,
            item_ids,
            mode=validation_mode,
        )
        if rejection_reason is None:
            parsed_scores = {
                item_id_to_name[item_id]: parsed_scores[item_id]
                for item_id in item_ids
            }
            return normalize_distribution(parsed_scores, total_score)

        if set(parsed_scores.keys()) == set(item_ids):
            append_debug_log(
                debug_log_path,
                (
                    f"[{log_tag}_reject] parent={parent_name} sample={sample_idx} "
                    f"attempt={attempt + 1}/{max(1, max_retries)} reason={rejection_reason}"
                ),
            )

        prompt = (
            f"{base_prompt}\n\n"
            "Your previous answer was incomplete or invalid.\n"
            "You must provide exactly one percentage for every child.\n"
            "Any readable one-line-per-item list is fine.\n"
            "Separators such as :, -, =, or -> are all acceptable.\n"
            "If you do not want to use ids, just output one line per child in the same order as the child list.\n"
            "Percentages must sum to 100.\n"
            "Do not omit any child. Do not add extra lines."
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
        sample_items = list(items)
        ordered_item_to_content = {item: item_to_content[item] for item in sample_items}
        append_debug_log(
            debug_log_path,
            f"[{log_tag}_item_order] parent={parent_name} sample={s + 1}/{sample_count} order={sample_items}",
        )
        try:
            sample_distributions.append(
                direct_allocate_scores(
                    client=client,
                    item_to_content=ordered_item_to_content,
                    total_score=total_score,
                    parent_name=parent_name,
                    model=model,
                    temperature=sample_temperature(temperature, s),
                    max_retries=max(1, max_retries),
                    debug_log_path=debug_log_path,
                    sample_idx=s + 1,
                    snippet_limit=snippet_limit,
                    log_tag=log_tag,
                    validation_mode="relaxed" if log_tag == "all_together_paragraphs" else "strict",
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

            try:
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

                return apply_minimum_positive_floor(final_scores, total_score, min_fraction=0.005)
            except ValueError:
                pass

        if len(items) == 2:
            append_debug_log(
                debug_log_path,
                f"[{log_tag}_pairwise_retry] parent={parent_name} reason=no_complete_samples",
            )
            try:
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
            except ValueError:
                pass

        append_debug_log(
            debug_log_path,
            f"[{log_tag}_length_fallback] parent={parent_name} reason=all_retries_exhausted items={items}",
        )
        lengths = {
            item: max(1, len(flatten_content_to_text(item_to_content[item], limit=None).split()))
            for item in items
        }
        total_len = max(1, sum(lengths.values()))
        return {item: total_score * (lengths[item] / total_len) for item in items}

    averaged = {item: 0.0 for item in items}
    for dist in sample_distributions:
        for item, value in dist.items():
            averaged[item] += value
    averaged = {item: value / len(sample_distributions) for item, value in averaged.items()}
    return apply_minimum_positive_floor(averaged, total_score, min_fraction=0.005)


def direct_only_allocate_scores(
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
    validation_mode: str = "strict",
) -> Dict[str, float]:
    items = list(item_to_content.keys())
    if not items:
        return {}
    if len(items) == 1:
        return {items[0]: total_score}

    sample_count = max(1, n_samples)
    sample_distributions: List[Dict[str, float]] = []
    for s in range(sample_count):
        sample_items = list(items)
        ordered_item_to_content = {item: item_to_content[item] for item in sample_items}
        append_debug_log(
            debug_log_path,
            f"[{log_tag}_item_order] parent={parent_name} sample={s + 1}/{sample_count} order={sample_items}",
        )
        try:
            sample_distributions.append(
                direct_allocate_scores(
                    client=client,
                    item_to_content=ordered_item_to_content,
                    total_score=total_score,
                    parent_name=parent_name,
                    model=model,
                    temperature=sample_temperature(temperature, s),
                    max_retries=max(1, max_retries),
                    debug_log_path=debug_log_path,
                    sample_idx=s + 1,
                    snippet_limit=snippet_limit,
                    log_tag=log_tag,
                    validation_mode=validation_mode,
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
        raise ValueError(
            f"Model failed to produce any complete direct-allocation sample for parent '{parent_name}'."
        )

    averaged = {item: 0.0 for item in items}
    for dist in sample_distributions:
        for item, value in dist.items():
            averaged[item] += value
    averaged = {item: value / len(sample_distributions) for item, value in averaged.items()}
    return apply_minimum_positive_floor(averaged, total_score, min_fraction=0.005)


def heuristic_top_level_scores(
    content_dict: Dict[str, Any],
    total_score: float,
) -> Dict[str, float]:
    if not content_dict:
        return {}

    raw_scores: Dict[str, float] = {}
    for section_name, section_content in content_dict.items():
        name_norm = normalized_key(section_name)
        base = 1.0

        if "introduction" in name_norm:
            base = 0.75
        elif any(token in name_norm for token in ("relatedwork", "priorwork", "background", "survey")):
            base = 0.80
        elif any(token in name_norm for token in ("conclusion", "futurework", "discussion")):
            base = 0.85
        elif any(token in name_norm for token in ("experiment", "results", "evaluation", "analysis")):
            base = 1.30
        elif any(token in name_norm for token in ("framework", "method", "algorithm", "solution")):
            base = 1.40
        elif any(token in name_norm for token in ("model", "problem", "definition", "preliminar")):
            base = 1.15

        structure_bonus = 1.0
        if isinstance(section_content, dict) and section_content:
            structure_bonus += min(0.35, 0.08 * len(section_content))

        text_blob = flatten_content_to_text(section_content, limit=4000)
        length_bonus = min(0.25, len(text_blob) / 12000.0) if text_blob else 0.0
        raw_scores[section_name] = base * (structure_bonus + length_bonus)

    return enforce_top_level_constraints(
        apply_minimum_positive_floor(raw_scores, total_score, min_fraction=0.01),
        total=total_score,
    )


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
    alias_to_id = {citation_id_to_name[citation_id]: citation_id for citation_id in citation_ids}

    for attempt in range(max(1, max_retries)):
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": CITATION_SPLIT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": temperature},
            think=False,
        )
        raw_response = response.message.content if getattr(response, "message", None) else ""
        append_debug_log(
            debug_log_path,
            (
                f"[citation_split] paragraph_id={paragraph_id} sample={sample_idx} "
                f"attempt={attempt + 1}/{max(1, max_retries)}\n{raw_response}\n"
            ),
        )
        parsed_full = parse_score_map_from_response(
            raw_response,
            citation_ids,
            allow_percentage=True,
            alias_to_id=alias_to_id,
        )
        parsed_scores = {
            citation_id: max(0.0, safe_float(value, 0.0))
            for citation_id, value in parsed_full.items()
        }

        if set(parsed_scores.keys()) == set(citation_ids) and any(v > 0.0 for v in parsed_scores.values()):
            parsed_scores = {
                citation_id_to_name[citation_id]: parsed_scores[citation_id]
                for citation_id in citation_ids
            }
            return normalize_distribution(parsed_scores, total_score)

        prompt = (
            f"{base_prompt}\n\n"
            "Your previous answer was incomplete or invalid.\n"
            "You must provide exactly one percentage for every citation.\n"
            f"Required counters: {', '.join(citation_ids)}\n"
            "Any readable one-line-per-item list is fine.\n"
            "Separators such as :, -, =, or -> are all acceptable.\n"
            "Percentages must sum to 100.\n"
            "Do not omit any citation. Do not add extra lines."
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
        try:
            sample_distributions.append(
                direct_allocate_citation_scores(
                    client=client,
                    paragraph_id=paragraph_id,
                    paragraph_text=paragraph_text,
                    citation_to_context=citation_to_context,
                    total_score=total_score,
                    model=model,
                    temperature=sample_temperature(temperature, s),
                    max_retries=max(1, max_retries),
                    debug_log_path=debug_log_path,
                    sample_idx=s + 1,
                )
            )
        except ValueError as exc:
            append_debug_log(
                debug_log_path,
                (
                    f"[citation_split_sample_skip] paragraph_id={paragraph_id} "
                    f"sample={s + 1}/{sample_count} reason={exc}"
                ),
            )

    if not sample_distributions:
        raise ValueError(
            f"Model failed to produce any complete citation-score sample for paragraph '{paragraph_id}'."
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
    allow_direct_fallback: bool = True,
) -> Dict[str, float]:
    items = list(item_to_content.keys())
    if not items:
        return {}
    if len(items) == 1:
        return {items[0]: total_score}

    snippets = {
        name: format_scored_segment_excerpt(name, item_to_content[name], limit=None)
        for name in items
    }
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
                temperature=sample_temperature(temperature, s),
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
                temperature=sample_temperature(temperature, s),
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
                    + ("Using direct allocation fallback." if allow_direct_fallback else "Keeping pairwise fallback.")
                )
            else:
                print(
                    f"[WARN] Pairwise outputs lacked variation for '{parent_name}' (sample {s + 1}). "
                    + ("Using direct allocation fallback." if allow_direct_fallback else "Keeping pairwise fallback.")
                )
            if allow_direct_fallback:
                sample_distributions.append(
                    direct_allocate_scores(
                        client=client,
                        item_to_content=item_to_content,
                        total_score=total_score,
                        parent_name=parent_name,
                        model=model,
                        temperature=sample_temperature(temperature, s),
                        max_retries=retry_count,
                        debug_log_path=debug_log_path,
                        sample_idx=s + 1,
                        log_tag="fallback",
                    )
                )
            else:
                sample_distributions.append(
                    apply_minimum_positive_floor(raw_scores, total_score, min_fraction=0.005)
                )
        else:
            sample_distributions.append(
                apply_minimum_positive_floor(raw_scores, total_score, min_fraction=0.005)
            )

    averaged = {item: 0.0 for item in items}
    for dist in sample_distributions:
        for item, value in dist.items():
            averaged[item] += value
    averaged = {item: value / sample_count for item, value in averaged.items()}
    return apply_minimum_positive_floor(averaged, total_score, min_fraction=0.005)


def enforce_top_level_constraints(scores: Dict[str, float], total: float) -> Dict[str, float]:
    constrained = dict(scores)
    if not constrained:
        return {}

    def norm(name: str) -> str:
        return normalized_key(name)

    support_sections: List[str] = []
    core_sections: List[str] = []
    preferred_core_sections: List[str] = []

    for section_name in constrained:
        name_norm = norm(section_name)
        is_support = any(
            token in name_norm
            for token in ("introduction", "relatedwork", "priorwork", "conclusion", "futurework", "background")
        )
        if is_support:
            support_sections.append(section_name)
        else:
            core_sections.append(section_name)
        if any(
            token in name_norm
            for token in ("framework", "algorithm", "experiment", "results", "method", "solution")
        ):
            preferred_core_sections.append(section_name)

    anchor_sections = preferred_core_sections or core_sections
    if not anchor_sections:
        return apply_minimum_positive_floor(constrained, total, min_fraction=0.005)

    core_anchor = max(constrained[name] for name in anchor_sections)
    caps: Dict[str, float] = {}
    for section_name in support_sections:
        name_norm = norm(section_name)
        if "introduction" in name_norm:
            caps[section_name] = max(0.12, 0.85 * core_anchor)
        elif "relatedwork" in name_norm or "priorwork" in name_norm or "background" in name_norm:
            caps[section_name] = max(0.10, 0.75 * core_anchor)
        elif "conclusion" in name_norm or "futurework" in name_norm:
            caps[section_name] = max(0.08, 0.60 * core_anchor)

    excess = 0.0
    for section_name, cap in caps.items():
        current = constrained.get(section_name, 0.0)
        if current > cap:
            excess += current - cap
            constrained[section_name] = cap

    recipients = preferred_core_sections or core_sections
    if excess > 0.0 and recipients:
        recipient_weights = {
            name: max(1e-9, constrained.get(name, 0.0))
            for name in recipients
        }
        redistributed = normalize_distribution(recipient_weights, excess)
        for name, value in redistributed.items():
            constrained[name] = constrained.get(name, 0.0) + value

    return apply_minimum_positive_floor(constrained, total, min_fraction=0.005)


def split_paragraph_channel_scores(
    client: Client,
    section_name: str,
    paragraph_items: Dict[str, str],
    paragraph_total_scores: Dict[str, float],
    mention_buckets: Dict[str, List[Tuple[str, str, str]]],
    model: str,
    n_samples: int,
    temperature: float,
    max_retries: int,
    snippet_limit: int,
    debug_log_path: str,
    strict: bool = False,
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
            "citation_focus_text": build_citation_focus_text(
                paragraph_items[paragraph_id],
                mention_buckets.get(paragraph_id, []),
            ),
        }
        for paragraph_id in paragraph_ids
    }
    _sys_prompt  = PARAGRAPH_CHANNEL_SPLIT_STRICT_SYSTEM_PROMPT if strict else PARAGRAPH_CHANNEL_SPLIT_SYSTEM_PROMPT
    _tmpl        = PARAGRAPH_CHANNEL_SPLIT_STRICT_USER_PROMPT_TEMPLATE if strict else PARAGRAPH_CHANNEL_SPLIT_USER_PROMPT_TEMPLATE
    user_prompt  = _tmpl.format(
        section_name=section_name,
        paragraphs_json=json.dumps(payload, indent=2)
    )

    technical_samples: List[Dict[str, float]] = []
    citation_samples: List[Dict[str, float]] = []

    for sample_idx in range(sample_count):
        current_temp = sample_temperature(temperature, sample_idx)
        parsed_pairs: Dict[str, Tuple[float, float]] = {}

        for attempt in range(max(1, max_retries)):
            try:
                response = client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": _sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    options={"temperature": current_temp},
                    think=False,
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
            focus_ratio = citation_focus_ratio(paragraph_items[paragraph_id], mentions)

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

                if is_citation_heavy_section_name(section_name):
                    min_citation_ratio = 0.0
                    if focus_ratio >= 0.60:
                        min_citation_ratio = 0.60
                    elif focus_ratio >= 0.40:
                        min_citation_ratio = 0.50
                    elif focus_ratio >= 0.25:
                        min_citation_ratio = 0.35

                    if min_citation_ratio > 0.0:
                        min_citation_value = total_val * min_citation_ratio
                        if c_val < min_citation_value:
                            c_val = min_citation_value
                            t_val = max(0.0, total_val - c_val)

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
        mentions = mention_buckets.get(paragraph_id, [])
        has_citations = bool(mentions)
        focus_ratio = citation_focus_ratio(paragraph_items[paragraph_id], mentions)

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

            if is_citation_heavy_section_name(section_name):
                min_citation_ratio = 0.0
                if focus_ratio >= 0.60:
                    min_citation_ratio = 0.60
                elif focus_ratio >= 0.40:
                    min_citation_ratio = 0.50
                elif focus_ratio >= 0.25:
                    min_citation_ratio = 0.35

                if min_citation_ratio > 0.0:
                    min_citation_value = total_val * min_citation_ratio
                    if avg_c < min_citation_value:
                        avg_c = min_citation_value
                        avg_t = max(0.0, total_val - avg_c)

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

    numeric_inner_pattern = r"\s*\d+\s*(?:[-,;–]\s*\d+\s*)*"
    is_numeric_container = bool(
        left_delim
        and right_delim
        and re.fullmatch(numeric_inner_pattern, inner)
    )

    if is_numeric_container:
        raw_parts = re.split(r"\s*[;,]\s*", inner)
    elif ";" in inner:
        all_semicolon_parts = inner.split(";")
        raw_parts = [p for p in all_semicolon_parts if re.search(r"\d{4}", p)]
        if not raw_parts:
            raw_parts = all_semicolon_parts
    elif left_delim == "[" and right_delim == "]" and "," in inner:
        raw_parts = inner.split(",")
    else:
        raw_parts = [inner]

    parts = [re.sub(r"\s+", " ", p).strip() for p in raw_parts if p.strip()]
    if not parts:
        return [block]

    if is_numeric_container:
        normalized_parts: List[str] = []
        for part in parts:
            cleaned = part.strip()
            if not cleaned:
                continue

            single_match = re.fullmatch(r"(\d+)", cleaned)
            if single_match:
                if int(single_match.group(1)) <= 0:
                    continue
                normalized_parts.append(f"{left_delim}{int(single_match.group(1))}{right_delim}")
                continue

            range_match = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", cleaned)
            if range_match:
                start_num = int(range_match.group(1))
                end_num = int(range_match.group(2))
                if start_num <= 0 or end_num <= 0:
                    continue
                if start_num > end_num:
                    start_num, end_num = end_num, start_num
                normalized_parts.extend(
                    f"{left_delim}{value}{right_delim}"
                    for value in range(start_num, end_num + 1)
                )
                continue

            # If the numeric citation part does not look like a positive citation id or range,
            # drop it instead of letting OCR/PDF debris become a fake citation such as [0] or (0).
            continue

        return normalized_parts

    if left_delim and right_delim:
        return [canonicalize_citation_key(f"{left_delim}{p}{right_delim}") for p in parts]
    return [canonicalize_citation_key(part) for part in parts]


def assign_importance_scores(
    content_dict: Dict[str, Any],
    citations_dict: Dict[str, Any],
    model: str = "llama3.2",
    host: str = "localhost:11434",
    n_samples: int = 3,
    temperature: float = 0.2,
    sample_temperature_jitter: float = DEFAULT_SAMPLE_TEMPERATURE_JITTER,
    seed: Optional[int] = None,
    max_retries: int = 3,
    paragraph_direct_max_tokens: int = 0,
    paragraph_compressed_snippet_limit: int = 180,
    debug_log_path: str = "",
    paper_id: str = DEFAULT_PAPER_ID,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Assign hierarchical section, paragraph, and citation scores with:
    - level-wise all-children-together scoring (parent sees all child nodes in one prompt),
    - repeated LLM sampling + averaging,
    - strict normalization at each tree level,
    - paragraph-level technical and citation channel decomposition.
    """
    if Client is None:
        raise ImportError("The 'ollama' Python package is required for LLM-based scoring.")
    global _SAMPLE_TEMPERATURE_JITTER, _SAMPLE_TEMPERATURE_RNG
    _SAMPLE_TEMPERATURE_JITTER = max(0.0, sample_temperature_jitter)
    _SAMPLE_TEMPERATURE_RNG = random.Random(seed) if seed is not None else random
    client = Client(host=host)
    citation_scores: Dict[str, Any] = {}
    section_scores: Dict[str, Any] = {}
    paragraph_scores: List[Dict[str, Any]] = []
    paragraph_citation_scores: List[Dict[str, Any]] = []

    try:
        top_level_scores = direct_only_allocate_scores(
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
            validation_mode="strict",
        )
    except ValueError as strict_exc:
        try:
            top_level_scores = direct_only_allocate_scores(
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
                log_tag="all_together_top_relaxed",
                validation_mode="relaxed",
            )
        except ValueError as relaxed_exc:
            append_debug_log(
                debug_log_path,
                (
                    "[all_together_top_heuristic_fallback] parent=Whole Paper "
                    f"strict_reason={strict_exc} relaxed_reason={relaxed_exc}"
                ),
            )
            top_level_scores = heuristic_top_level_scores(content_dict, total_score=1.0)
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
        paragraph_total = apply_minimum_positive_floor(paragraph_total, section_score, min_fraction=0.005)

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
            section_name=section_name,
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
        fallback_paragraph_allocations: Dict[str, Dict[str, float]] = {}
        total_mentions = sum(len(mentions) for mentions in mention_buckets.values())
        if total_mentions > 0 and sum(max(0.0, value) for value in paragraph_citation.values()) <= 0.0:
            append_debug_log(
                debug_log_path,
                (
                    f"[citation_leaf_length_fallback] section={' > '.join(section_path)} "
                    f"reason=zero_final_citation_score mentions={total_mentions} section_score={section_score:.6f}"
                ),
            )
            (
                paragraph_total,
                paragraph_technical,
                paragraph_citation,
                fallback_paragraph_allocations,
            ) = allocate_leaf_citation_scores_length_fallback(
                paragraph_items=paragraph_items,
                mention_buckets=mention_buckets,
                section_score=section_score,
            )

        for paragraph_name in paragraph_names:
            meta = paragraph_meta[paragraph_name]
            internal_paragraph_id = meta["internal_paragraph_id"]
            mentions = mention_buckets.get(paragraph_name, [])
            paragraph_t = max(0.0, paragraph_technical.get(paragraph_name, 0.0))
            paragraph_c = max(0.0, paragraph_citation.get(paragraph_name, 0.0))

            if paragraph_c <= 0.0 and mentions:
                _t_strict, _c_strict = split_paragraph_channel_scores(
                    client=client,
                    section_name=section_name,
                    paragraph_items={paragraph_name: paragraph_items[paragraph_name]},
                    paragraph_total_scores={paragraph_name: paragraph_total.get(paragraph_name, 0.0)},
                    mention_buckets={paragraph_name: mentions},
                    model=model,
                    n_samples=max(1, n_samples),
                    temperature=temperature,
                    max_retries=max(1, max_retries),
                    snippet_limit=snippet_limit,
                    debug_log_path=debug_log_path,
                    strict=True,
                )
                paragraph_t = max(0.0, _t_strict.get(paragraph_name, paragraph_t))
                paragraph_c = max(0.0, _c_strict.get(paragraph_name, 0.0))

                if paragraph_c <= 0.0:
                    total_val = max(0.0, safe_float(paragraph_total.get(paragraph_name), 0.0))
                    word_count = max(1, len(meta["text"].split()))
                    density = len(mentions) / word_count
                    fraction = max(0.05, min(0.30, density * 10))
                    paragraph_c = total_val * fraction
                    paragraph_t = max(0.0, total_val - paragraph_c)

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

            if paragraph_name in fallback_paragraph_allocations and fallback_paragraph_allocations[paragraph_name]:
                citation_split = fallback_paragraph_allocations[paragraph_name]
            else:
                citation_contexts: Dict[str, List[str]] = {}
                for citation, _, context in mentions:
                    citation_contexts.setdefault(citation, []).append(normalize_for_match(str(context)))

                citation_to_context: Dict[str, str] = {}
                for citation, contexts in citation_contexts.items():
                    unique_contexts = [ctx for ctx in dict.fromkeys(contexts) if ctx]
                    context_blob = " ".join(unique_contexts)[:700]
                    citation_to_context[citation] = context_blob if context_blob else meta["text"][:700]

                try:
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
                except ValueError as exc:
                    append_debug_log(
                        debug_log_path,
                        (
                            f"[citation_split_unresolved] paragraph_id={internal_paragraph_id} "
                            f"reason={exc} fallback=length_based"
                        ),
                    )
                    citation_split = allocate_paragraph_citation_scores_length_fallback(
                        mentions=mentions,
                        total_score=paragraph_c,
                    )
                    if not citation_split:
                        continue
            split_total = sum(citation_split.values())
            assert_close(split_total, paragraph_c, f"citation split for {internal_paragraph_id}")

            for citation, citation_value in citation_split.items():
                paragraph_citation_scores.append(
                    {
                        "section_path": list(section_path),
                        "paragraph_index": meta["paragraph_index"],
                        "paragraph": meta["text"],
                        "citation": citation,
                        "citation_score": citation_value,
                    }
                )

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

    return citation_scores, section_scores, paragraph_scores, paragraph_citation_scores


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
    parser.add_argument(
        "--model-tag",
        default="",
        help="Optional suffix to append to output/debug/prompt filenames so runs are distinguishable by model.",
    )
    parser.add_argument("--host", default="localhost:11434", help="Ollama host")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of LLM samples to average")
    parser.add_argument("--temperature", type=float, default=0.2, help="Base sampling temperature")
    parser.add_argument(
        "--sample-temperature-jitter",
        type=float,
        default=DEFAULT_SAMPLE_TEMPERATURE_JITTER,
        help="Maximum per-sample temperature increase added on top of --temperature.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible per-sample temperature jitter.",
    )
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
    model_tag = args.model_tag.strip() or sanitize_model_tag(args.model)

    debug_log_path = Path(args.debug_log)
    if model_tag:
        debug_log_path = debug_log_path.with_name(f"{debug_log_path.stem}_{model_tag}{debug_log_path.suffix}")

    prompts_output_path = Path(args.prompts_output) if args.prompts_output else None
    if prompts_output_path and model_tag:
        prompts_output_path = prompts_output_path.with_name(
            f"{prompts_output_path.stem}_{model_tag}{prompts_output_path.suffix}"
        )

    append_run_separator(str(debug_log_path), args)

    text = read_pdf_text(args.pdf)
    sections = DEFAULT_SECTIONS
    if args.sections_file and args.sections_var:
        sections = load_sections_from_file(args.sections_file, args.sections_var)
    citations, content = extract_citations_by_section(text, sections)
    (
        citation_importance,
        section_importance,
        paragraph_importance,
        paragraph_citation_importance,
    ) = assign_importance_scores(
        content_dict=content,
        citations_dict=citations,
        model=args.model,
        host=args.host,
        n_samples=max(1, args.n_samples),
        temperature=max(0.0, args.temperature),
        sample_temperature_jitter=max(0.0, args.sample_temperature_jitter),
        seed=args.seed,
        max_retries=max(1, args.max_retries),
        paragraph_direct_max_tokens=max(0, args.paragraph_direct_max_tokens),
        paragraph_compressed_snippet_limit=max(60, args.paragraph_compressed_snippet_limit),
        debug_log_path=str(debug_log_path),
        paper_id=args.paper_id,
    )

    output1_prefix = f"{args.output1}_{model_tag}" if model_tag else args.output1
    output2_prefix = f"{args.output2}_{model_tag}" if model_tag else args.output2
    paragraph_prefix_base = args.output3 if args.output3 else args.output2
    paragraph_prefix = f"{paragraph_prefix_base}_{model_tag}" if model_tag else paragraph_prefix_base

    citation_path = f"{output1_prefix}_citation_scores.json"
    section_path = f"{output2_prefix}_section_scores.json"
    paragraph_path = f"{paragraph_prefix}_paragraph_scores.json"
    paragraph_citation_path = f"{paragraph_prefix}_paragraph_citation_scores.json"

    with open(citation_path, "w", encoding="utf-8") as f:
        json.dump(citation_importance, f, indent=2)

    with open(section_path, "w", encoding="utf-8") as f:
        json.dump(section_importance, f, indent=2)

    with open(paragraph_path, "w", encoding="utf-8") as f:
        json.dump(paragraph_importance, f, indent=2)

    with open(paragraph_citation_path, "w", encoding="utf-8") as f:
        json.dump(paragraph_citation_importance, f, indent=2)

    if prompts_output_path:
        with open(prompts_output_path, "w", encoding="utf-8") as f:
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
