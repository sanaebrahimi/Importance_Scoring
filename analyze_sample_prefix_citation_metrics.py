import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from citation_resolver import CitationResolver
from evaluate_human_section_scores import (
    get_top_k_model_citations,
    load_json,
    reciprocal_rank,
    resolve_human_top_k_citations,
)


VALID_DEBUG_LOGS = {
    "llama3_2": "debug_llama3_2.log",
    "gemma2_2b": "debug_gemma2_2b.log",
    "phi3_medium_run2": "debug_phi3_medium_run2.log",
    "qwen3_4b": "debug_qwen3_4b.log",
    "qwen2_5_3b": "debug_qwen2_5_3b.log",
}


TAG_LINE_RE = re.compile(r"^\[[A-Za-z_]+")
PARAGRAPH_SCORING_RE = re.compile(
    r"^\[paragraph_scoring\] parent=(?P<parent>.+?) method=.* paragraphs=(?P<count>\d+)\b"
)
ITEM_ORDER_RE = re.compile(
    r"^\[(?P<tag>[A-Za-z_]+)_item_order\] parent=(?P<parent>.+?) sample=(?P<sample>\d+)/(?P<total>\d+) order=(?P<order>\[.*\])$"
)
SCORE_BLOCK_RE = re.compile(
    r"^\[(?P<tag>all_together_paragraphs(?:_[A-Za-z_]+)?)\] parent=(?P<parent>.+?) sample=(?P<sample>\d+) attempt=(?P<attempt>\d+)/(?P<max_attempt>\d+)$"
)
CHANNEL_BLOCK_RE = re.compile(
    r"^\[paragraph_channel_split\] sample=(?P<sample>\d+)/(?P<total>\d+) attempt=(?P<attempt>\d+)/(?P<max_attempt>\d+) temperature=(?P<temp>[-+0-9.eE]+)$"
)
CITATION_BLOCK_RE = re.compile(
    r"^\[citation_split\] paragraph_id=(?P<paragraph_id>.+?) sample=(?P<sample>\d+) attempt=(?P<attempt>\d+)/(?P<max_attempt>\d+)$"
)
CITATION_SKIP_RE = re.compile(
    r"^\[citation_split_sample_skip\] paragraph_id=(?P<paragraph_id>.+?) sample=(?P<sample>\d+)/(?P<total>\d+) reason="
)
RUN_START_RE = re.compile(r"^\[run_start\].* n_samples=(?P<n_samples>\d+)\b")


def normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (key or "").lower())


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_distribution(raw_scores: Dict[str, float], total: float) -> Dict[str, float]:
    cleaned = {key: max(0.0, safe_float(value, 0.0)) for key, value in raw_scores.items()}
    denom = sum(cleaned.values())
    if denom <= 0.0:
        if not cleaned:
            return {}
        uniform = total / len(cleaned)
        return {key: uniform for key in cleaned}
    normalized = {key: (value / denom) * total for key, value in cleaned.items()}
    residual = total - sum(normalized.values())
    best_key = max(normalized, key=normalized.get)
    normalized[best_key] += residual
    return normalized


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text


def parse_json_loose(text: str) -> Optional[Any]:
    stripped = strip_code_fences(text).strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except Exception:
        pass
    try:
        return ast.literal_eval(stripped)
    except Exception:
        pass
    return None


def coerce_non_negative_number(value: Any, allow_percentage: bool = False) -> Optional[float]:
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().rstrip(",")
        if not text:
            return None
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if not match:
            return None
        number = safe_float(match.group(0), float("nan"))
    if number != number:  # NaN
        return None
    if allow_percentage and number > 1.0:
        return max(0.0, number / 100.0)
    return max(0.0, number)


def parse_plaintext_score_lines(text: str, allow_percentage: bool = False) -> List[Tuple[str, float]]:
    pairs: List[Tuple[str, float]] = []
    for raw_line in strip_code_fences(text).splitlines():
        line = raw_line.strip().strip(",")
        if not line:
            continue
        line = re.sub(r"^[-*•]\s*", "", line)
        match = re.match(
            r"^(?P<label>[^:=>\-]{1,120}|[A-Za-z0-9 _().,\[\]&;:+/?-]{1,120})\s*(?::|=|->|-)\s*(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\s*%?)\s*$",
            line,
        )
        if not match:
            continue
        value = coerce_non_negative_number(match.group("value"), allow_percentage=allow_percentage)
        if value is None:
            continue
        pairs.append((match.group("label").strip(), value))
    return pairs


def parse_ordered_score_values(text: str, allow_percentage: bool = False) -> List[float]:
    values: List[float] = []
    for raw_line in strip_code_fences(text).splitlines():
        line = raw_line.strip().strip(",")
        if not line:
            continue
        line = re.sub(r"^[-*•]\s*", "", line)
        match = re.match(
            r"^(?:\d+[.)]?\s*[:=-]?\s*)?(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\s*%?)\s*$",
            line,
        )
        if not match:
            match = re.match(
                r"^\d+[.)]?\s*(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\s*%?)\s*$",
                line,
            )
        if not match:
            continue
        value = coerce_non_negative_number(match.group("value"), allow_percentage=allow_percentage)
        if value is None:
            continue
        values.append(value)
    return values


def parse_score_map_from_response(
    response_text: str,
    expected_ids: List[str],
    allow_percentage: bool = False,
    alias_to_id: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    expected_set = set(expected_ids)
    norm_to_id = {normalized_key(item_id): item_id for item_id in expected_ids}
    if alias_to_id:
        for alias, target_id in alias_to_id.items():
            if target_id in expected_set:
                norm_to_id[normalized_key(alias)] = target_id

    def resolve_target_id(key: Any) -> Optional[str]:
        key_str = str(key).strip()
        if key_str in expected_set:
            return key_str
        return norm_to_id.get(normalized_key(key_str))

    plaintext_pairs = parse_plaintext_score_lines(response_text, allow_percentage=allow_percentage)
    ordered_values = parse_ordered_score_values(response_text, allow_percentage=allow_percentage)
    if len(ordered_values) == len(expected_ids):
        return {expected_ids[idx]: ordered_values[idx] for idx in range(len(expected_ids))}

    parsed_scores: Dict[str, float] = {}
    for key, value in plaintext_pairs:
        target_id = resolve_target_id(key)
        if target_id is not None:
            parsed_scores[target_id] = value
    if len(parsed_scores) == len(expected_ids):
        return parsed_scores

    payload = parse_json_loose(response_text)
    if isinstance(payload, dict):
        for key, value in payload.items():
            target_id = resolve_target_id(key)
            if target_id is None:
                continue
            if isinstance(value, dict):
                for field in ("score", "value", "weight", "allocation", "credit"):
                    if field in value:
                        num = coerce_non_negative_number(value[field], allow_percentage=allow_percentage)
                        if num is not None:
                            parsed_scores[target_id] = num
                            break
            else:
                num = coerce_non_negative_number(value, allow_percentage=allow_percentage)
                if num is not None:
                    parsed_scores[target_id] = num
    elif isinstance(payload, list) and len(payload) == len(expected_ids):
        parsed_from_list: Dict[str, float] = {}
        for idx, entry in enumerate(payload):
            num = coerce_non_negative_number(entry, allow_percentage=allow_percentage)
            if num is not None:
                parsed_from_list[expected_ids[idx]] = num
        if len(parsed_from_list) == len(expected_ids):
            return parsed_from_list

    return parsed_scores


def parse_paragraph_channel_split_response(
    response_text: str,
    paragraph_ids: List[str],
    alias_to_id: Optional[Dict[str, str]] = None,
) -> Dict[str, Tuple[float, float]]:
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

    def coerce_pair(a: Any, b: Any) -> Optional[Tuple[float, float]]:
        first = coerce_non_negative_number(a, allow_percentage=True)
        second = coerce_non_negative_number(b, allow_percentage=True)
        if first is None or second is None:
            return None
        return first, second

    pairs: Dict[str, Tuple[float, float]] = {}
    payload = parse_json_loose(response_text)
    if isinstance(payload, dict):
        for key, value in payload.items():
            paragraph_id = resolve_id(key)
            if paragraph_id is None:
                continue
            pair: Optional[Tuple[float, float]] = None
            if isinstance(value, dict):
                pair = coerce_pair(
                    value.get("technical", value.get("technical_score", value.get("tech", value.get("t")))),
                    value.get("citation", value.get("citation_score", value.get("cite", value.get("c")))),
                )
            elif isinstance(value, list) and len(value) >= 2:
                pair = coerce_pair(value[0], value[1])
            elif isinstance(value, str):
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
                if len(nums) >= 2:
                    pair = coerce_pair(nums[0], nums[1])
            if pair is not None:
                pairs[paragraph_id] = pair
    if len(pairs) == len(paragraph_ids):
        return pairs

    for raw_line in strip_code_fences(response_text).splitlines():
        line = raw_line.strip().strip(",")
        if not line:
            continue
        line = re.sub(r"^[-*•]\s*", "", line)
        label_match = re.match(r"^(?P<label>[^:=]+)\s*[:=]\s*(?P<body>.+)$", line)
        if not label_match:
            continue
        paragraph_id = resolve_id(label_match.group("label"))
        if paragraph_id is None:
            continue
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", label_match.group("body"))
        if len(nums) >= 2:
            pair = coerce_pair(nums[0], nums[1])
            if pair is not None:
                pairs[paragraph_id] = pair
    return pairs


def read_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def parse_log_blocks(lines: Sequence[str]) -> List[Tuple[str, List[str]]]:
    blocks: List[Tuple[str, List[str]]] = []
    current_header: Optional[str] = None
    current_body: List[str] = []

    for line in lines:
        if TAG_LINE_RE.match(line):
            if current_header is not None:
                blocks.append((current_header, current_body))
            current_header = line
            current_body = []
        else:
            if current_header is not None:
                current_body.append(line)

    if current_header is not None:
        blocks.append((current_header, current_body))
    return blocks


def section_path_to_key(section_path: Sequence[str]) -> str:
    return " > ".join(section_path)


def paragraph_internal_id(section_path: Sequence[str], paragraph_index: int) -> str:
    return f"{section_path_to_key(section_path)}::p{paragraph_index}"


def build_leaf_contexts(
    paragraph_scores: List[dict],
    paragraph_citation_scores: List[dict],
) -> List[dict]:
    grouped_scores: Dict[Tuple[str, ...], List[dict]] = {}
    ordered_paths: List[Tuple[str, ...]] = []
    for row in paragraph_scores:
        path = tuple(row["section_path"])
        if path not in grouped_scores:
            grouped_scores[path] = []
            ordered_paths.append(path)
        grouped_scores[path].append(row)

    citations_by_paragraph: Dict[Tuple[Tuple[str, ...], int], List[str]] = defaultdict(list)
    for row in paragraph_citation_scores:
        path = tuple(row["section_path"])
        key = (path, int(row["paragraph_index"]))
        citations_by_paragraph[key].append(row["citation"])

    contexts: List[dict] = []
    for path in ordered_paths:
        rows = sorted(grouped_scores[path], key=lambda item: int(item["paragraph_index"]))
        paragraph_names = [f"Paragraph {int(row['paragraph_index'])}" for row in rows]
        paragraph_totals = {
            f"Paragraph {int(row['paragraph_index'])}": float(row["technical_score"]) + float(row["citation_score"])
            for row in rows
        }
        paragraph_technical_totals = {
            f"Paragraph {int(row['paragraph_index'])}": float(row["technical_score"])
            for row in rows
        }
        paragraph_citation_totals = {
            f"Paragraph {int(row['paragraph_index'])}": float(row["citation_score"])
            for row in rows
        }
        paragraph_has_citations = {
            f"Paragraph {int(row['paragraph_index'])}": bool(
                citations_by_paragraph.get((path, int(row["paragraph_index"])))
            )
            for row in rows
        }
        paragraph_id_to_name = {
            paragraph_internal_id(path, int(row["paragraph_index"])): f"Paragraph {int(row['paragraph_index'])}"
            for row in rows
        }
        paragraph_citations = {
            f"Paragraph {int(row['paragraph_index'])}": list(
                citations_by_paragraph.get((path, int(row["paragraph_index"])), [])
            )
            for row in rows
        }

        contexts.append(
            {
                "section_path": list(path),
                "section_key": section_path_to_key(path),
                "leaf_name": path[-1],
                "paragraph_count": len(rows),
                "leaf_total": sum(paragraph_totals.values()),
                "paragraph_names": paragraph_names,
                "paragraph_totals": paragraph_totals,
                "paragraph_technical_totals": paragraph_technical_totals,
                "paragraph_citation_totals": paragraph_citation_totals,
                "paragraph_has_citations": paragraph_has_citations,
                "paragraph_id_to_name": paragraph_id_to_name,
                "paragraph_citations": paragraph_citations,
            }
        )
    return contexts


def choose_leaf_context(
    contexts: List[dict],
    usage_count: Dict[Tuple[str, int], int],
    leaf_name: str,
    paragraph_count: int,
) -> Optional[dict]:
    for idx, ctx in enumerate(contexts):
        if usage_count[(ctx["section_key"], idx)]:
            continue
        if ctx["leaf_name"] == leaf_name and ctx["paragraph_count"] == paragraph_count:
            usage_count[(ctx["section_key"], idx)] = 1
            return ctx
    for idx, ctx in enumerate(contexts):
        if usage_count[(ctx["section_key"], idx)]:
            continue
        if ctx["leaf_name"] == leaf_name:
            usage_count[(ctx["section_key"], idx)] = 1
            return ctx
    return None


def paragraph_alias_map(paragraph_names: Sequence[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for idx, paragraph_name in enumerate(paragraph_names, start=1):
        aliases[paragraph_name] = str(idx)
        aliases[f"p{idx}"] = str(idx)
        aliases[f"paragraph {idx}"] = str(idx)
        aliases[f"paragraph{idx}"] = str(idx)
        aliases[str(idx)] = str(idx)
    return aliases


def citation_alias_map(citations: Sequence[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for idx, citation in enumerate(citations, start=1):
        aliases[citation] = str(idx)
        aliases[str(idx)] = str(idx)
    return aliases


def parse_order(raw: str) -> List[str]:
    try:
        value = ast.literal_eval(raw)
        if isinstance(value, list):
            return [str(item) for item in value]
    except Exception:
        pass
    return []


def replay_debug_log(
    debug_log_path: Path,
    paragraph_scores: List[dict],
    paragraph_citation_scores: List[dict],
) -> dict:
    lines = read_lines(debug_log_path)
    blocks = parse_log_blocks(lines)

    n_samples = 0
    for line in lines:
        match = RUN_START_RE.match(line)
        if match:
            n_samples = int(match.group("n_samples"))
            break
    if n_samples <= 0:
        n_samples = 5

    leaf_contexts = build_leaf_contexts(paragraph_scores, paragraph_citation_scores)
    leaf_usage: Dict[Tuple[str, int], int] = defaultdict(int)
    current_leaf: Optional[dict] = None

    item_orders: Dict[Tuple[str, str, int], List[str]] = {}
    paragraph_score_samples: Dict[str, Dict[int, Dict[int, Dict[str, float]]]] = defaultdict(lambda: defaultdict(dict))
    channel_samples: Dict[str, Dict[int, Dict[int, Dict[str, Dict[str, float]]]]] = defaultdict(lambda: defaultdict(dict))
    citation_samples: Dict[str, Dict[str, Dict[int, Dict[int, Dict[str, float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    citation_sample_skips: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))

    for header, body_lines in blocks:
        paragraph_match = PARAGRAPH_SCORING_RE.match(header)
        if paragraph_match:
            parent = paragraph_match.group("parent")
            leaf_name = parent[: -len("::paragraphs")] if parent.endswith("::paragraphs") else parent
            paragraph_count = int(paragraph_match.group("count"))
            current_leaf = choose_leaf_context(leaf_contexts, leaf_usage, leaf_name, paragraph_count)
            continue

        item_order_match = ITEM_ORDER_RE.match(header)
        if item_order_match:
            tag = item_order_match.group("tag")
            parent = item_order_match.group("parent")
            sample_idx = int(item_order_match.group("sample"))
            order = parse_order(item_order_match.group("order"))
            item_orders[(tag, parent, sample_idx)] = order
            continue

        paragraph_score_match = SCORE_BLOCK_RE.match(header)
        if paragraph_score_match and current_leaf is not None:
            tag = paragraph_score_match.group("tag")
            parent = paragraph_score_match.group("parent")
            sample_idx = int(paragraph_score_match.group("sample"))
            attempt_idx = int(paragraph_score_match.group("attempt"))
            leaf_key = current_leaf["section_key"]
            if attempt_idx in paragraph_score_samples[leaf_key][sample_idx]:
                continue

            paragraph_names = current_leaf["paragraph_names"]
            item_ids = [str(i + 1) for i in range(len(paragraph_names))]
            alias_to_id = paragraph_alias_map(paragraph_names)
            parsed = parse_score_map_from_response(
                "\n".join(body_lines),
                item_ids,
                allow_percentage=True,
                alias_to_id=alias_to_id,
            )
            if set(parsed.keys()) != set(item_ids):
                continue
            ordered = item_orders.get((tag, parent, sample_idx), paragraph_names)
            if len(ordered) != len(item_ids):
                continue
            item_id_to_name = {str(idx + 1): ordered[idx] for idx in range(len(ordered))}
            mapped = {item_id_to_name[item_id]: float(parsed[item_id]) for item_id in item_ids}
            normalized = normalize_distribution(mapped, current_leaf["leaf_total"])
            paragraph_score_samples[leaf_key][sample_idx][attempt_idx] = normalized
            continue

        channel_match = CHANNEL_BLOCK_RE.match(header)
        if channel_match and current_leaf is not None:
            sample_idx = int(channel_match.group("sample"))
            attempt_idx = int(channel_match.group("attempt"))
            leaf_key = current_leaf["section_key"]
            if attempt_idx in channel_samples[leaf_key][sample_idx]:
                continue

            paragraph_names = current_leaf["paragraph_names"]
            alias_to_id = paragraph_alias_map(paragraph_names)
            parsed_pairs = parse_paragraph_channel_split_response(
                "\n".join(body_lines),
                paragraph_names,
                alias_to_id=alias_to_id,
            )
            if len(parsed_pairs) != len(paragraph_names):
                continue

            sample_data: Dict[str, Dict[str, float]] = {}
            for paragraph_name in paragraph_names:
                total_val = current_leaf["paragraph_totals"][paragraph_name]
                has_citations = current_leaf["paragraph_has_citations"][paragraph_name]
                t_raw, c_raw = parsed_pairs[paragraph_name]
                t_raw = max(0.0, float(t_raw))
                c_raw = max(0.0, float(c_raw))
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
                sample_data[paragraph_name] = {"technical": t_val, "citation": c_val}
            channel_samples[leaf_key][sample_idx][attempt_idx] = sample_data
            continue

        citation_skip_match = CITATION_SKIP_RE.match(header)
        if citation_skip_match and current_leaf is not None:
            paragraph_id = citation_skip_match.group("paragraph_id")
            sample_idx = int(citation_skip_match.group("sample"))
            leaf_key = current_leaf["section_key"]
            paragraph_name = current_leaf["paragraph_id_to_name"].get(paragraph_id)
            if paragraph_name is not None:
                citation_sample_skips[leaf_key][paragraph_name].add(sample_idx)
            continue

        citation_match = CITATION_BLOCK_RE.match(header)
        if citation_match and current_leaf is not None:
            paragraph_id = citation_match.group("paragraph_id")
            sample_idx = int(citation_match.group("sample"))
            attempt_idx = int(citation_match.group("attempt"))
            leaf_key = current_leaf["section_key"]
            paragraph_name = current_leaf["paragraph_id_to_name"].get(paragraph_id)
            if paragraph_name is None:
                continue
            if attempt_idx in citation_samples[leaf_key][paragraph_name][sample_idx]:
                continue

            citations = current_leaf["paragraph_citations"][paragraph_name]
            if not citations:
                continue
            item_ids = [str(i + 1) for i in range(len(citations))]
            alias_to_id = citation_alias_map(citations)
            parsed = parse_score_map_from_response(
                "\n".join(body_lines),
                item_ids,
                allow_percentage=True,
                alias_to_id=alias_to_id,
            )
            if set(parsed.keys()) != set(item_ids):
                continue
            item_id_to_name = {str(idx + 1): citations[idx] for idx in range(len(citations))}
            mapped = {item_id_to_name[item_id]: float(parsed[item_id]) for item_id in item_ids}
            normalized = normalize_distribution(mapped, current_leaf["paragraph_citation_totals"][paragraph_name])
            citation_samples[leaf_key][paragraph_name][sample_idx][attempt_idx] = normalized
            continue

    return {
        "n_samples": n_samples,
        "leaf_contexts": leaf_contexts,
        "paragraph_score_samples": paragraph_score_samples,
        "channel_samples": channel_samples,
        "citation_samples": citation_samples,
        "citation_sample_skips": citation_sample_skips,
    }


def select_attempt_distribution(
    attempt_map: Dict[int, Any],
    attempt_budget: Optional[int],
) -> Optional[Any]:
    if not attempt_map:
        return None
    for attempt_idx in sorted(attempt_map):
        if attempt_budget is None or attempt_idx <= attempt_budget:
            return attempt_map[attempt_idx]
    return None


def prefix_average_distributions(
    sample_map: Dict[int, Dict[int, Dict[str, float]]],
    n_prefix: int,
    total_score: float,
    paragraph_names: Sequence[str],
    average_by_full_prefix: bool,
    attempt_budget: Optional[int] = None,
) -> Dict[str, float]:
    accum = {name: 0.0 for name in paragraph_names}
    present = 0
    for sample_idx in range(1, n_prefix + 1):
        dist = select_attempt_distribution(sample_map.get(sample_idx, {}), attempt_budget)
        if dist is None:
            continue
        present += 1
        for name in paragraph_names:
            accum[name] += float(dist.get(name, 0.0))
    denom = n_prefix if average_by_full_prefix else max(1, present)
    averaged = {name: accum[name] / denom for name in paragraph_names}
    if sum(averaged.values()) > 0:
        averaged = normalize_distribution(averaged, total_score)
    return averaged


def prefix_average_channels(
    sample_map: Dict[int, Dict[int, Dict[str, Dict[str, float]]]],
    n_prefix: int,
    paragraph_names: Sequence[str],
    attempt_budget: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    accum_t = {name: 0.0 for name in paragraph_names}
    accum_c = {name: 0.0 for name in paragraph_names}
    for sample_idx in range(1, n_prefix + 1):
        dist = select_attempt_distribution(sample_map.get(sample_idx, {}), attempt_budget)
        if dist is None:
            continue
        for name in paragraph_names:
            entry = dist.get(name, {})
            accum_t[name] += float(entry.get("technical", 0.0))
            accum_c[name] += float(entry.get("citation", 0.0))
    denom = max(1, n_prefix)
    return {
        name: {
            "technical": accum_t[name] / denom,
            "citation": accum_c[name] / denom,
        }
        for name in paragraph_names
    }


def prefix_average_citations(
    sample_map: Dict[int, Dict[int, Dict[str, float]]],
    n_prefix: int,
    citations: Sequence[str],
    total_score: float,
    attempt_budget: Optional[int] = None,
) -> Dict[str, float]:
    accum = {citation: 0.0 for citation in citations}
    for sample_idx in range(1, n_prefix + 1):
        dist = select_attempt_distribution(sample_map.get(sample_idx, {}), attempt_budget)
        if dist is None:
            continue
        for citation in citations:
            accum[citation] += float(dist.get(citation, 0.0))
    denom = max(1, n_prefix)
    averaged = {citation: accum[citation] / denom for citation in citations}
    if sum(averaged.values()) > 0:
        averaged = normalize_distribution(averaged, total_score)
    return averaged


def build_prefix_citation_scores(
    replay_data: dict,
    n_prefix: int,
    attempt_budget: Optional[int] = None,
) -> Tuple[Dict[str, dict], Dict[str, List[dict]], Dict[str, List[dict]]]:
    citation_scores: Dict[str, dict] = {}
    paragraph_scores: List[dict] = []
    paragraph_citation_scores: List[dict] = []

    for leaf_ctx in replay_data["leaf_contexts"]:
        leaf_key = leaf_ctx["section_key"]
        paragraph_names = leaf_ctx["paragraph_names"]
        channels = prefix_average_channels(
            replay_data["channel_samples"].get(leaf_key, {}),
            n_prefix,
            paragraph_names,
            attempt_budget=attempt_budget,
        )
        paragraph_totals = prefix_average_distributions(
            replay_data["paragraph_score_samples"].get(leaf_key, {}),
            n_prefix,
            leaf_ctx["leaf_total"],
            paragraph_names,
            average_by_full_prefix=False,
            attempt_budget=attempt_budget,
        )

        for paragraph_name in paragraph_names:
            paragraph_index = int(paragraph_name.split()[-1])
            base_total = paragraph_totals.get(paragraph_name, leaf_ctx["paragraph_totals"][paragraph_name])
            channel = channels.get(paragraph_name, {})
            t_val = float(channel.get("technical", leaf_ctx["paragraph_technical_totals"][paragraph_name]))
            c_val = float(channel.get("citation", leaf_ctx["paragraph_citation_totals"][paragraph_name]))
            channel_sum = max(0.0, t_val) + max(0.0, c_val)
            if not leaf_ctx["paragraph_has_citations"][paragraph_name]:
                t_val = base_total
                c_val = 0.0
            elif channel_sum > 0.0:
                t_val = base_total * max(0.0, t_val) / channel_sum
                c_val = base_total * max(0.0, c_val) / channel_sum
            else:
                t_val = base_total
                c_val = 0.0

            paragraph_scores.append(
                {
                    "section_path": list(leaf_ctx["section_path"]),
                    "paragraph_index": paragraph_index,
                    "technical_score": t_val,
                    "citation_score": c_val,
                }
            )

            citations = leaf_ctx["paragraph_citations"][paragraph_name]
            if not citations or c_val <= 0.0:
                continue

            citation_dist = prefix_average_citations(
                replay_data["citation_samples"].get(leaf_key, {}).get(paragraph_name, {}),
                n_prefix,
                citations,
                c_val,
                attempt_budget=attempt_budget,
            )
            for citation, value in citation_dist.items():
                paragraph_citation_scores.append(
                    {
                        "section_path": list(leaf_ctx["section_path"]),
                        "paragraph_index": paragraph_index,
                        "citation": citation,
                        "citation_score": value,
                    }
                )
                citation_scores.setdefault(citation, {"citation_score": 0.0})
                citation_scores[citation]["citation_score"] += value

    return citation_scores, paragraph_scores, paragraph_citation_scores


def compute_wtech_top_k(
    paragraph_scores: List[dict],
    paragraph_citation_scores: List[dict],
    k: int,
) -> List[Tuple[str, float]]:
    tech_by_paragraph = {
        (tuple(row["section_path"]), int(row["paragraph_index"])): float(row["technical_score"])
        for row in paragraph_scores
    }
    totals: Dict[str, float] = defaultdict(float)
    for row in paragraph_citation_scores:
        key = (tuple(row["section_path"]), int(row["paragraph_index"]))
        tech = tech_by_paragraph.get(key, 0.0)
        totals[row["citation"]] += float(row["citation_score"]) * tech
    ranked = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    return ranked[: max(1, k)]


def citation_report_from_ranked(
    paper_id: str,
    important_papers: List[dict],
    ranked_full: List[Tuple[str, float]],
    resolver: CitationResolver,
    k: int,
) -> dict:
    ranked_top_k = ranked_full[:k]
    ranked_top_k_keys = [key for key, _ in ranked_top_k]
    human_ranked = sorted(important_papers, key=lambda item: item.get("rank", 10**9))[:k]
    resolved_human_keys = resolve_human_top_k_citations(
        paper_id=paper_id,
        human_items=human_ranked,
        model_ranked_full=ranked_full,
        resolver=resolver,
    )
    resolved_human_set = {key for key in resolved_human_keys if key is not None}
    overlap_count = len(resolved_human_set.intersection(set(ranked_top_k_keys)))
    denom = max(1, len(human_ranked))
    return {
        "overlap_at_k": overlap_count,
        "recall_at_k": overlap_count / denom,
        "hit_at_k": 1.0 if overlap_count > 0 else 0.0,
        "mrr_human_rank1_in_model_top_k": reciprocal_rank(
            resolved_human_keys[0] if resolved_human_keys else None,
            ranked_top_k_keys,
        ),
    }


def mean_metric(rows: Sequence[dict], field: str) -> Optional[float]:
    if not rows:
        return None
    return sum(float(row[field]) for row in rows) / len(rows)


def analyze_model(
    model_tag: str,
    annotations: dict,
    results_root: Path,
    papers_dir: Path,
    top_k: int,
    resolver: CitationResolver,
) -> dict:
    debug_name = VALID_DEBUG_LOGS[model_tag]
    per_paper: Dict[str, dict] = {}
    sample_sizes: set = set()
    skipped: List[dict] = []

    for paper_id, payload in annotations["papers"].items():
        paper_dir = results_root / paper_id
        debug_path = paper_dir / debug_name
        paragraph_scores_path = paper_dir / f"{paper_id}_{model_tag}_paragraph_scores.json"
        paragraph_citation_scores_path = paper_dir / f"{paper_id}_{model_tag}_paragraph_citation_scores.json"
        citation_scores_path = paper_dir / f"{paper_id}_{model_tag}_citation_scores.json"
        if not payload.get("important_papers"):
            continue
        if not debug_path.exists() or not paragraph_scores_path.exists() or not paragraph_citation_scores_path.exists() or not citation_scores_path.exists():
            skipped.append({"paper_id": paper_id, "reason": "missing_required_files"})
            continue

        replay = replay_debug_log(
            debug_path,
            load_json(paragraph_scores_path),
            load_json(paragraph_citation_scores_path),
        )
        n_samples = int(replay["n_samples"])
        sample_sizes.update(range(1, n_samples + 1))

        paper_metrics_by_prefix: Dict[int, dict] = {}
        for prefix in range(1, n_samples + 1):
            citation_scores, paragraph_scores, paragraph_citation_scores = build_prefix_citation_scores(replay, prefix)
            ranked_w = get_top_k_model_citations(
                citation_scores,
                max(top_k, len(citation_scores)),
                paper_id=paper_id,
                resolver=resolver,
            )
            ranked_wtech = compute_wtech_top_k(
                paragraph_scores=paragraph_scores,
                paragraph_citation_scores=paragraph_citation_scores,
                k=max(top_k, 20),
            )
            metrics_w = citation_report_from_ranked(
                paper_id=paper_id,
                important_papers=payload["important_papers"],
                ranked_full=ranked_w,
                resolver=resolver,
                k=top_k,
            )
            metrics_wtech = citation_report_from_ranked(
                paper_id=paper_id,
                important_papers=payload["important_papers"],
                ranked_full=ranked_wtech,
                resolver=resolver,
                k=top_k,
            )
            paper_metrics_by_prefix[prefix] = {
                "w": metrics_w,
                "w_tech": metrics_wtech,
            }

        per_paper[paper_id] = {
            "n_samples": n_samples,
            "prefix_metrics": paper_metrics_by_prefix,
        }

    aggregate: Dict[int, dict] = {}
    for prefix in sorted(sample_sizes):
        rows_w: List[dict] = []
        rows_wtech: List[dict] = []
        for paper_payload in per_paper.values():
            prefix_metrics = paper_payload["prefix_metrics"].get(prefix)
            if not prefix_metrics:
                continue
            rows_w.append(prefix_metrics["w"])
            rows_wtech.append(prefix_metrics["w_tech"])
        aggregate[prefix] = {
            "papers_evaluated": len(rows_w),
            "w": {
                "mean_overlap_at_k": mean_metric(rows_w, "overlap_at_k"),
                "mean_recall_at_k": mean_metric(rows_w, "recall_at_k"),
                "mean_hit_at_k": mean_metric(rows_w, "hit_at_k"),
                "mean_mrr_human_rank1_in_model_top_k": mean_metric(rows_w, "mrr_human_rank1_in_model_top_k"),
            },
            "w_tech": {
                "mean_overlap_at_k": mean_metric(rows_wtech, "overlap_at_k"),
                "mean_recall_at_k": mean_metric(rows_wtech, "recall_at_k"),
                "mean_hit_at_k": mean_metric(rows_wtech, "hit_at_k"),
                "mean_mrr_human_rank1_in_model_top_k": mean_metric(rows_wtech, "mrr_human_rank1_in_model_top_k"),
            },
        }

    return {
        "model_tag": model_tag,
        "debug_log_name": debug_name,
        "aggregate_by_sample_size": aggregate,
        "per_paper": per_paper,
        "skipped": skipped,
        "notes": [
            "This analysis is a log-replay approximation based on saved debug logs.",
            "Lower-level samples were replayed from the logged paragraph allocation, paragraph channel split, and citation split stages.",
            "Because lower-level stages in the original run were executed after parent scores had already been averaged over all samples, the sample-size curves should be interpreted as replay-based estimates rather than exact fresh reruns at k=1..n.",
        ],
    }


def analyze_model_attempts(
    model_tag: str,
    annotations: dict,
    results_root: Path,
    papers_dir: Path,
    top_k: int,
    resolver: CitationResolver,
    max_attempt_budget: int = 5,
) -> dict:
    debug_name = VALID_DEBUG_LOGS[model_tag]
    per_paper: Dict[str, dict] = {}
    skipped: List[dict] = []

    for paper_id, payload in annotations["papers"].items():
        paper_dir = results_root / paper_id
        debug_path = paper_dir / debug_name
        paragraph_scores_path = paper_dir / f"{paper_id}_{model_tag}_paragraph_scores.json"
        paragraph_citation_scores_path = paper_dir / f"{paper_id}_{model_tag}_paragraph_citation_scores.json"
        citation_scores_path = paper_dir / f"{paper_id}_{model_tag}_citation_scores.json"
        if not payload.get("important_papers"):
            continue
        if not debug_path.exists() or not paragraph_scores_path.exists() or not paragraph_citation_scores_path.exists() or not citation_scores_path.exists():
            skipped.append({"paper_id": paper_id, "reason": "missing_required_files"})
            continue

        replay = replay_debug_log(
            debug_path,
            load_json(paragraph_scores_path),
            load_json(paragraph_citation_scores_path),
        )
        n_samples = int(replay["n_samples"])

        paper_metrics_by_budget: Dict[int, dict] = {}
        for attempt_budget in range(1, max_attempt_budget + 1):
            citation_scores, paragraph_scores, paragraph_citation_scores = build_prefix_citation_scores(
                replay,
                n_prefix=n_samples,
                attempt_budget=attempt_budget,
            )
            ranked_w = get_top_k_model_citations(
                citation_scores,
                max(top_k, len(citation_scores)),
                paper_id=paper_id,
                resolver=resolver,
            )
            metrics_w = citation_report_from_ranked(
                paper_id=paper_id,
                important_papers=payload["important_papers"],
                ranked_full=ranked_w,
                resolver=resolver,
                k=top_k,
            )
            paper_metrics_by_budget[attempt_budget] = {
                "w": metrics_w,
            }
        per_paper[paper_id] = {
            "n_samples": n_samples,
            "attempt_budget_metrics": paper_metrics_by_budget,
        }

    aggregate: Dict[int, dict] = {}
    for attempt_budget in range(1, max_attempt_budget + 1):
        rows_w: List[dict] = []
        for paper_payload in per_paper.values():
            budget_metrics = paper_payload["attempt_budget_metrics"].get(attempt_budget)
            if not budget_metrics:
                continue
            rows_w.append(budget_metrics["w"])
        aggregate[attempt_budget] = {
            "papers_evaluated": len(rows_w),
            "w": {
                "mean_overlap_at_k": mean_metric(rows_w, "overlap_at_k"),
                "mean_recall_at_k": mean_metric(rows_w, "recall_at_k"),
                "mean_hit_at_k": mean_metric(rows_w, "hit_at_k"),
                "mean_mrr_human_rank1_in_model_top_k": mean_metric(rows_w, "mrr_human_rank1_in_model_top_k"),
            },
        }

    return {
        "model_tag": model_tag,
        "debug_log_name": debug_name,
        "aggregate_by_attempt_budget": aggregate,
        "per_paper": per_paper,
        "skipped": skipped,
        "notes": [
            "This analysis is a log-replay approximation based on saved debug logs.",
            "Attempt budget means that, within each sample, only valid outputs from attempts up to the specified limit are allowed.",
            "The final score for a paper is still averaged across the full sample count; only retry allowance changes.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay sample-level citation scoring from debug logs and analyze top-k citation metrics as sample size increases."
    )
    parser.add_argument(
        "--annotations-file",
        default="human_expert_annotations.json",
        help="Annotation JSON to use as reference for the citation metrics.",
    )
    parser.add_argument(
        "--results-root",
        default="paper_results",
        help="Root directory containing per-paper result folders.",
    )
    parser.add_argument(
        "--papers-dir",
        default="papers",
        help="Directory containing the source PDFs for citation resolution.",
    )
    parser.add_argument(
        "--model-tags",
        nargs="*",
        default=list(VALID_DEBUG_LOGS.keys()),
        help="Model tags to analyze. Defaults to the valid debug log set.",
    )
    parser.add_argument(
        "--citation-top-k",
        type=int,
        default=4,
        help="Top-k citations to compare.",
    )
    parser.add_argument(
        "--analyze-by",
        choices=["samples", "attempts"],
        default="samples",
        help="Whether to analyze sample-size prefixes or attempt-budget prefixes.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save the full analysis JSON.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    papers_dir = Path(args.papers_dir)
    annotations = load_json(Path(args.annotations_file))

    resolver = CitationResolver()
    resolver.parse_all(results_root, papers_dir)

    selected_model_tags = [tag for tag in args.model_tags if tag in VALID_DEBUG_LOGS]
    summary = {
        "annotations_file": args.annotations_file,
        "citation_top_k": args.citation_top_k,
        "analyze_by": args.analyze_by,
        "models": {},
    }

    for model_tag in selected_model_tags:
        if args.analyze_by == "attempts":
            model_summary = analyze_model_attempts(
                model_tag=model_tag,
                annotations=annotations,
                results_root=results_root,
                papers_dir=papers_dir,
                top_k=max(1, args.citation_top_k),
                resolver=resolver,
            )
        else:
            model_summary = analyze_model(
                model_tag=model_tag,
                annotations=annotations,
                results_root=results_root,
                papers_dir=papers_dir,
                top_k=max(1, args.citation_top_k),
                resolver=resolver,
            )
        summary["models"][model_tag] = model_summary

        print()
        print(f"Model: {model_tag}")
        if args.analyze_by == "attempts":
            for attempt_budget, payload in model_summary["aggregate_by_attempt_budget"].items():
                print(
                    f"  attempt_budget={attempt_budget} papers={payload['papers_evaluated']} "
                    f"W(overlap={payload['w']['mean_overlap_at_k']}, recall={payload['w']['mean_recall_at_k']}, "
                    f"hit={payload['w']['mean_hit_at_k']}, mrr={payload['w']['mean_mrr_human_rank1_in_model_top_k']})"
                )
        else:
            for sample_size, payload in model_summary["aggregate_by_sample_size"].items():
                print(
                    f"  sample_size={sample_size} papers={payload['papers_evaluated']} "
                    f"W(overlap={payload['w']['mean_overlap_at_k']}, recall={payload['w']['mean_recall_at_k']}, "
                    f"hit={payload['w']['mean_hit_at_k']}, mrr={payload['w']['mean_mrr_human_rank1_in_model_top_k']}) "
                    f"W_tech(overlap={payload['w_tech']['mean_overlap_at_k']}, recall={payload['w_tech']['mean_recall_at_k']}, "
                    f"hit={payload['w_tech']['mean_hit_at_k']}, mrr={payload['w_tech']['mean_mrr_human_rank1_in_model_top_k']})"
                )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
