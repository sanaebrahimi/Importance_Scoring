"""
Patch citation scores for paragraphs that have detectable citations in their text
but were recorded with citation_score=0 in *_promptv2_paragraph_scores.json.

For each such paragraph:
  1. Re-runs the LLM channel-split (technical vs citation) using the paragraph's
     existing total_score.
  2. If the new citation_c > 0, runs allocate_citation_scores_for_paragraph.
  3. Updates only three files per paper:
       *_{model_tag}_paragraph_scores.json
       *_{model_tag}_paragraph_citation_scores.json
       *_{model_tag}_citation_scores.json

Also adds a "has_citation" boolean key to every paragraph entry in
*_{model_tag}_paragraph_scores.json.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Minimal citation regex kept at module level so --dry-run works without PyPDF2.
_NUMERIC_BRACKET = r"\[(?:\s*\d+\s*(?:[-,;–]\s*\d+\s*)*)\]"
_NUMERIC_PAREN   = r"\((?:\s*\d+\s*(?:[-,;–]\s*\d+\s*)*)\)"
_AUTHOR_YEAR     = r"\([A-Z][^)]*\d{4}[a-z]?\)"
_NARRATIVE_AY    = r"[A-Z][a-zA-Z\-']+(?:\s+[A-Z][a-zA-Z\-']+)*\s+et\s+al\.\s*\(\d{4}[a-z]?\)"
_CITATION_RE     = re.compile(
    rf"(?:{_NARRATIVE_AY})|(?:{_AUTHOR_YEAR})|(?:{_NUMERIC_BRACKET})|(?:{_NUMERIC_PAREN})"
)


def _split_block_simple(block: str) -> List[str]:
    """Minimal split for --dry-run when importance_score is unavailable."""
    inner = block.strip("[]() ")
    parts = re.split(r"[,;–\-]", inner)
    delim, end = ("[", "]") if block.startswith("[") else ("(", ")")
    result = [f"{delim}{p.strip()}{end}" for p in parts if p.strip().isdigit() and int(p.strip()) > 0]
    return result or [block]


def _detect_citations_in_text(text: str, split_fn=None) -> List[str]:
    """Return deduplicated list of individual citation keys found in paragraph text."""
    if split_fn is None:
        split_fn = _split_block_simple
    keys: List[str] = []
    for block in _CITATION_RE.findall(re.sub(r"\s+", " ", text)):
        keys.extend(split_fn(block))
    return list(dict.fromkeys(keys))


DEFAULT_PAPERS_DIR   = Path("papers")
DEFAULT_SECTIONS_FILE = Path("papers_section_titles.txt")
DEFAULT_RESULTS_ROOT  = Path("paper_results")


# ---------------------------------------------------------------------------
# Helpers used during the actual patch (require importance_score imports)
# ---------------------------------------------------------------------------

def _navigate_content(content: Any, section_path: List[str]) -> Optional[str]:
    node = content
    for part in section_path:
        if not isinstance(node, dict):
            return None
        node = node.get(part)
        if node is None:
            return None
    return node if isinstance(node, str) else None


def _navigate_citations(citations: Any, section_path: List[str]) -> Dict[str, Any]:
    node = citations
    for part in section_path:
        if not isinstance(node, dict):
            return {}
        node = node.get(part, {})
    return node if isinstance(node, dict) else {}


def _build_mention_buckets(
    paragraph_items: Dict[str, str],
    section_citations: Dict[str, Any],
    paragraph_total: Dict[str, float],
    normalize_for_match,
    split_citation_block,
) -> Dict[str, List[Tuple[str, str, str]]]:
    paragraph_norms = {name: normalize_for_match(t) for name, t in paragraph_items.items()}
    paragraph_tokens = {
        name: set(re.findall(r"[a-z0-9]+", paragraph_norms[name].lower()))
        for name in paragraph_items
    }
    mention_buckets: Dict[str, List[Tuple[str, str, str]]] = {name: [] for name in paragraph_items}

    for citation_block, context_value in section_citations.items():
        contexts = context_value if isinstance(context_value, list) else [context_value]
        for context in contexts:
            context_str = str(context)
            context_norm = normalize_for_match(context_str)
            target_paragraph: Optional[str] = None

            if context_norm:
                for name, pnorm in paragraph_norms.items():
                    if context_norm in pnorm:
                        target_paragraph = name
                        break

            if target_paragraph is None and context_norm:
                ctx_tokens = set(re.findall(r"[a-z0-9]+", context_norm.lower()))
                if ctx_tokens:
                    best_name, best_overlap = None, -1
                    for name, p_tokens in paragraph_tokens.items():
                        overlap = len(ctx_tokens & p_tokens)
                        if overlap > best_overlap:
                            best_overlap, best_name = overlap, name
                    if best_name is not None and best_overlap > 0:
                        target_paragraph = best_name

            if target_paragraph is None:
                target_paragraph = max(paragraph_items, key=lambda n: paragraph_total.get(n, 0.0))

            for cit in split_citation_block(citation_block):
                mention_buckets[target_paragraph].append((cit, citation_block, context_str))

    return mention_buckets


def sections_var_name(pdf_stem: str) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", pdf_stem).strip("_").upper()
    return f"{base}_SECTIONS"


# ---------------------------------------------------------------------------
# Core patch routine for one paper / model tag
# ---------------------------------------------------------------------------

def patch_paper(
    paper_id: str,
    paper_dir: Path,
    pdf_path: Path,
    sections_file: Path,
    model_tag: str,
    client: Any,
    model: str,
    n_samples: int,
    temperature: float,
    max_retries: int,
    snippet_limit: int,
    paragraph_direct_max_tokens: int,
    debug_log_path: Path,
    is_funcs: Dict[str, Any],
) -> bool:
    ps_path = paper_dir / f"{paper_id}_{model_tag}_paragraph_scores.json"
    pc_path = paper_dir / f"{paper_id}_{model_tag}_paragraph_citation_scores.json"
    cs_path = paper_dir / f"{paper_id}_{model_tag}_citation_scores.json"

    for p in (ps_path, pc_path, cs_path):
        if not p.exists():
            print(f"  [skip] missing {p.name}")
            return False

    # Unpack reused functions from importance_score
    normalize_for_match          = is_funcs["normalize_for_match"]
    split_citation_block         = is_funcs["split_citation_block"]
    safe_float                   = is_funcs["safe_float"]
    read_pdf_text                = is_funcs["read_pdf_text"]
    load_sections_from_file      = is_funcs["load_sections_from_file"]
    extract_citations_by_section = is_funcs["extract_citations_by_section"]
    split_text_into_paragraphs   = is_funcs["split_text_into_paragraphs"]
    split_paragraph_channel_scores         = is_funcs["split_paragraph_channel_scores"]
    allocate_citation_scores_for_paragraph = is_funcs["allocate_citation_scores_for_paragraph"]
    estimate_direct_allocation_tokens      = is_funcs["estimate_direct_allocation_tokens"]

    detect_cits = lambda t: _detect_citations_in_text(t, split_fn=split_citation_block)

    paragraphs: List[Dict[str, Any]]  = json.loads(ps_path.read_text())
    cit_entries: List[Dict[str, Any]] = json.loads(pc_path.read_text())
    citation_scores: Dict[str, Any]   = json.loads(cs_path.read_text())

    # ---- Step 1: Add has_citation to every paragraph ----
    text = read_pdf_text(str(pdf_path))
    sections_var = sections_var_name(pdf_path.stem)
    sections = load_sections_from_file(str(sections_file), sections_var)
    citations_by_section, content_by_section = extract_citations_by_section(text, sections)

    para_changed = False
    for para in paragraphs:
        has_cit = bool(detect_cits(para.get("paragraph", "")))
        if para.get("has_citation") != has_cit:
            para["has_citation"] = has_cit
            para_changed = True
        else:
            para.setdefault("has_citation", has_cit)

    # ---- Step 2: Collect paragraphs needing patching ----
    needs_patch: Dict[Tuple, List[Dict]] = defaultdict(list)
    for para in paragraphs:
        sp = tuple(para["section_path"])
        pi = para["paragraph_index"]
        if para.get("has_citation") and para.get("citation_score", 0.0) <= 0.0:
            needs_patch[(sp, pi)].append(para)

    if not needs_patch:
        if para_changed:
            ps_path.write_text(json.dumps(paragraphs, indent=2))
            print("  [updated] has_citation keys only (no score patches needed)")
        else:
            print("  [ok] nothing to patch")
        return True

    by_section: Dict[Tuple, List[int]] = defaultdict(list)
    for sp, pi in needs_patch:
        by_section[sp].append(pi)

    print(f"  [patch] {len(needs_patch)} paragraph(s) across {len(by_section)} section(s)")

    # Index: (section_path_tuple, para_index) → positions in `paragraphs` list
    para_index_map: Dict[Tuple, List[int]] = defaultdict(list)
    for i, para in enumerate(paragraphs):
        para_index_map[(tuple(para["section_path"]), para["paragraph_index"])].append(i)

    for section_path_tuple, patch_para_indices in by_section.items():
        section_path = list(section_path_tuple)

        section_content = _navigate_content(content_by_section, section_path)
        if not section_content:
            print(f"    [warn] could not find content for {section_path}")
            continue
        section_cit = _navigate_citations(citations_by_section, section_path)

        raw_paragraphs = split_text_into_paragraphs(section_content)
        if not raw_paragraphs:
            print(f"    [warn] no paragraphs extracted for {section_path}")
            continue

        paragraph_items: Dict[str, str] = {
            f"Paragraph {i + 1}": p for i, p in enumerate(raw_paragraphs)
        }

        # Collect total_scores for this section from the existing paragraph_scores
        para_total_by_idx: Dict[int, float] = {}
        for para in paragraphs:
            if tuple(para["section_path"]) != section_path_tuple:
                continue
            pi = para["paragraph_index"]
            if pi not in para_total_by_idx:
                para_total_by_idx[pi] = max(
                    0.0,
                    safe_float(para.get("technical_score", 0.0), 0.0)
                    + safe_float(para.get("citation_score", 0.0), 0.0),
                )

        paragraph_total: Dict[str, float] = {
            name: para_total_by_idx.get(int(name.split()[1]), 0.0)
            for name in paragraph_items
        }

        mention_buckets = _build_mention_buckets(
            paragraph_items, section_cit, paragraph_total,
            normalize_for_match, split_citation_block,
        )

        patch_names = {
            f"Paragraph {pi}" for pi in patch_para_indices
            if f"Paragraph {pi}" in paragraph_items
        }
        if not patch_names:
            print(f"    [warn] para indices {patch_para_indices} out of range for {section_path} "
                  f"(section has {len(paragraph_items)} paragraphs)")
            continue

        section_name = section_path[-1]
        parent_name  = f"{section_name}::paragraphs"
        est_tokens   = estimate_direct_allocation_tokens(parent_name, paragraph_items, snippet_limit=0)
        threshold    = paragraph_direct_max_tokens if paragraph_direct_max_tokens > 0 else 7000
        eff_snip     = max(60, snippet_limit) if est_tokens > threshold else 0

        paragraph_technical, paragraph_citation = split_paragraph_channel_scores(
            client=client,
            section_name=section_name,
            paragraph_items=paragraph_items,
            paragraph_total_scores=paragraph_total,
            mention_buckets=mention_buckets,
            model=model,
            n_samples=n_samples,
            temperature=temperature,
            max_retries=max_retries,
            snippet_limit=eff_snip,
            debug_log_path=str(debug_log_path),
        )

        for para_name in patch_names:
            para_idx = int(para_name.split()[1])
            new_c = max(0.0, paragraph_citation.get(para_name, 0.0))
            new_t = max(0.0, paragraph_technical.get(para_name, 0.0))

            if new_c <= 0.0:
                print(f"    [zero] {section_path} para={para_idx} still 0 after re-run")
                continue

            para_text    = needs_patch[(section_path_tuple, para_idx)][0].get("paragraph", "")
            detected_cits = detect_cits(para_text)
            if not detected_cits:
                continue

            # Build citation_to_context: prefer mention_buckets, fall back to paragraph text
            cit_ctx: Dict[str, List[str]] = {}
            for cit, _, ctx in mention_buckets.get(para_name, []):
                cit_ctx.setdefault(cit, []).append(normalize_for_match(ctx))
            for cit in detected_cits:
                if cit not in cit_ctx:
                    cit_ctx[cit] = [normalize_for_match(para_text)[:700]]

            citation_to_context: Dict[str, str] = {
                cit: " ".join(dict.fromkeys(ctxs))[:700] or para_text[:700]
                for cit, ctxs in cit_ctx.items()
                if cit in detected_cits
            }

            paragraph_id = f"{paper_id}::{' > '.join(section_path)}::p{para_idx}"
            citation_split = allocate_citation_scores_for_paragraph(
                client=client,
                paragraph_id=paragraph_id,
                paragraph_text=para_text,
                citation_to_context=citation_to_context,
                total_score=new_c,
                model=model,
                n_samples=n_samples,
                temperature=temperature,
                max_retries=max_retries,
                debug_log_path=str(debug_log_path),
            )

            # Update paragraph_scores entries
            for list_idx in para_index_map.get((section_path_tuple, para_idx), []):
                paragraphs[list_idx]["citation_score"] = new_c
                paragraphs[list_idx]["technical_score"] = new_t
                paragraphs[list_idx]["has_citation"] = True
            para_changed = True

            # Append to paragraph_citation_scores
            for cit, cit_val in citation_split.items():
                cit_entries.append({
                    "section_path": section_path,
                    "paragraph_index": para_idx,
                    "paragraph": para_text,
                    "citation": cit,
                    "citation_score": cit_val,
                })
                citation_scores.setdefault(cit, {"citation_score": 0.0})
                citation_scores[cit]["citation_score"] += cit_val

            print(
                f"    [patched] {section_path} para={para_idx}: "
                f"cit_c={new_c:.4f} citations={list(citation_split.keys())}"
            )

    ps_path.write_text(json.dumps(paragraphs, indent=2))
    pc_path.write_text(json.dumps(cit_entries, indent=2))
    cs_path.write_text(json.dumps(citation_scores, indent=2))
    print(f"  [saved] {ps_path.name}, {pc_path.name}, {cs_path.name}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch citation scores for paragraphs with undetected citations."
    )
    parser.add_argument("--model",     required=True, help="Ollama model name.")
    parser.add_argument("--model-tag", required=True, help="Model tag in output filenames.")
    parser.add_argument("--host",      default="http://localhost:11435")
    parser.add_argument("--n-samples", type=int,   default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sample-temperature-jitter", type=float, default=0.03)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--paragraph-direct-max-tokens",       type=int, default=1200)
    parser.add_argument("--paragraph-compressed-snippet-limit", type=int, default=8000)
    parser.add_argument("--papers-dir",    default=str(DEFAULT_PAPERS_DIR))
    parser.add_argument("--sections-file", default=str(DEFAULT_SECTIONS_FILE))
    parser.add_argument("--results-root",  default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--papers", nargs="+", default=[], metavar="STEM")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show how many paragraphs would be patched, without running the LLM.")
    args = parser.parse_args()

    papers_dir    = Path(args.papers_dir)
    sections_file = Path(args.sections_file)
    results_root  = Path(args.results_root)
    paper_filter  = set(args.papers)
    model_tag     = args.model_tag

    # --dry-run: only needs JSON + regex, no heavy imports
    if args.dry_run:
        total_paras = 0
        for pdf_path in sorted(papers_dir.glob("*.pdf")):
            if paper_filter and pdf_path.stem not in paper_filter:
                continue
            ps_path = results_root / pdf_path.stem / f"{pdf_path.stem}_{model_tag}_paragraph_scores.json"
            if not ps_path.exists():
                continue
            paras = json.loads(ps_path.read_text())
            needs = sum(
                1 for p in paras
                if _detect_citations_in_text(p.get("paragraph", ""))
                and p.get("citation_score", 0.0) <= 0.0
            )
            total_paras += needs
            print(f"[paper] {pdf_path.stem}: {needs} paragraph(s) would be patched")
        print(f"\n[dry-run total] {total_paras} paragraph(s) across all papers")
        return

    # Full run: need importance_score (PyPDF2) + ollama
    try:
        import importance_score as _is
    except ImportError as e:
        raise SystemExit(f"Cannot import importance_score: {e}\nInstall missing packages (e.g. PyPDF2).") from e

    if _is.Client is None:  # type: ignore[attr-defined]
        raise SystemExit("ollama Python package not installed.")

    from ollama import Client  # noqa: PLC0415

    # Mirror the global RNG setup from importance_score.py
    _is._SAMPLE_TEMPERATURE_JITTER = max(0.0, args.sample_temperature_jitter)
    _is._SAMPLE_TEMPERATURE_RNG = random.Random(args.seed) if args.seed is not None else random

    client = Client(host=args.host)

    is_funcs = {
        "normalize_for_match":           _is.normalize_for_match,
        "split_citation_block":          _is.split_citation_block,
        "safe_float":                    _is.safe_float,
        "read_pdf_text":                 _is.read_pdf_text,
        "load_sections_from_file":       _is.load_sections_from_file,
        "extract_citations_by_section":  _is.extract_citations_by_section,
        "split_text_into_paragraphs":    _is.split_text_into_paragraphs,
        "split_paragraph_channel_scores":         _is.split_paragraph_channel_scores,
        "allocate_citation_scores_for_paragraph": _is.allocate_citation_scores_for_paragraph,
        "estimate_direct_allocation_tokens":      _is.estimate_direct_allocation_tokens,
    }

    n_papers = n_patched = 0
    for pdf_path in sorted(papers_dir.glob("*.pdf")):
        if paper_filter and pdf_path.stem not in paper_filter:
            continue
        paper_id  = pdf_path.stem
        paper_dir = results_root / paper_id
        ps_path   = paper_dir / f"{paper_id}_{model_tag}_paragraph_scores.json"
        if not ps_path.exists():
            continue

        n_papers += 1
        print(f"[paper] {paper_id}")
        debug_log = paper_dir / f"debug_patch_{model_tag}.log"
        ok = patch_paper(
            paper_id=paper_id,
            paper_dir=paper_dir,
            pdf_path=pdf_path,
            sections_file=sections_file,
            model_tag=model_tag,
            client=client,
            model=args.model,
            n_samples=max(1, args.n_samples),
            temperature=max(0.0, args.temperature),
            max_retries=max(1, args.max_retries),
            snippet_limit=max(60, args.paragraph_compressed_snippet_limit),
            paragraph_direct_max_tokens=max(0, args.paragraph_direct_max_tokens),
            debug_log_path=debug_log,
            is_funcs=is_funcs,
        )
        if ok:
            n_patched += 1

    print(f"\n[done] processed {n_papers} papers, patched {n_patched}")


if __name__ == "__main__":
    main()
