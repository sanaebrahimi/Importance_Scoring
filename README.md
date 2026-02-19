# Importance Scoring Pipeline

This script reads a research paper PDF, extracts text by section/subsection, and assigns importance scores to:

- sections/subsections (hierarchical, normalized)
- citations (aggregated across the whole paper)

It uses Ollama (`llama3.2` by default) for pairwise section comparisons.

## What It Produces

Running `importance_score.py` writes two JSON files:

- `<output1>_citation_scores.json`
- `<output2>_section_scores.json`

Citation scores are **global**. If the same citation appears in multiple places, its final score is the sum of all mentions.

## Quick Start

From the folder containing `importance_score.py`:

```bash
python importance_score.py \
  --output1 results/run1 \
  --output2 results/run1 \
  --pdf adv_res_paper.pdf \
  --model llama3.2 \
  --host localhost:11434 \
  --n-samples 3 \
  --temperature 0.2 \
  --max-retries 3 \
  --debug-log pairwise_debug.log
```

If `results/` does not exist:

```bash
mkdir -p results
```

## Dependencies

- Python 3.9+
- `PyPDF2`
- `ollama` Python package
- Ollama server running and model available (default: `llama3.2`)

Example install:

```bash
pip install PyPDF2 ollama
```

## How the Code Is Organized

### 1. Input and section schema

- `DEFAULT_PDF_PATH`: default PDF file path
- `DEFAULT_SECTIONS`: nested section tree used to split paper content

This section tree controls the hierarchy used for both section scoring and citation attribution.

### 2. PDF reading and extraction

- `read_pdf_text(pdf_path)`: reads full text from PDF
- `extract_citations_by_section(text, sections)`: splits content by section tree and finds citation blocks

Citation regex targets forms like `(Author, 2020)` and keeps nearby sentence context.

### 3. Parsing and normalization helpers

- `parse_json_response(...)`: safely parses JSON, including fenced code blocks
- `safe_float(...)`: numeric conversion guard
- `normalize_distribution(...)`: enforces sum constraints exactly by normalization + residual fix
- `flatten_content_to_text(...)`: reduces nested section content to short snippets for prompts

### 4. Pairwise scoring logic

- `extract_probability_from_response(...)`: extracts probability from model output (`p`, `probability`, etc.)
- `query_pair_probability(...)`: asks model for one pair:
  - returns `p = P(item A > item B)` in `[0, 1]`
  - retries on invalid output
  - writes raw model text to debug log
- `pairwise_allocate_scores(...)`:
  - runs all pairwise comparisons for siblings at a tree node
  - repeats across `n_samples`
  - averages samples
  - normalizes to parent total score

### 5. Fallback when pairwise fails

- `direct_allocate_scores_fallback(...)`:
  - asks model for direct raw scores per item (JSON map)
  - normalizes to parent total
  - if model still fails, uses deterministic heuristic fallback based on snippet length and citation count

This avoids collapsing to equal scores when pairwise responses are bad.

### 6. Constraints and citation scoring

- `enforce_top_level_constraints(...)`: enforces one domain rule: `Introduction <= Experiment Results`
- `split_citation_block(...)`: splits grouped citations like `(A, 2020; B, 2021)`
- `assign_importance_scores(...)`:
  - builds hierarchical section scores
  - scores citations at leaf nodes
  - aggregates citation totals globally across sections/subsections

### 7. Entry point

- `main()` handles CLI arguments, runs pipeline, writes outputs, prints top citations and section hierarchy.

## Understanding `pairwise_debug.log`

The debug log stores raw model replies used in scoring. It is append-only.

There are two record types:

### `[pair]` entries

Example header:

```text
[pair] parent=Whole Paper sample=1 attempt=1/3 key=Introduction|Experiment Results
```

Meaning:

- `parent`: current tree node being scored
- `sample`: which averaging pass (`--n-samples`)
- `attempt`: retry count (`--max-retries`)
- `key`: compared pair (`item_a|item_b`)

The next lines are the model’s raw output for that pair.

### `[fallback]` entries

Example header:

```text
[fallback] parent=Whole Paper sample=2 attempt=1/3
```

Meaning: pairwise failed for all pairs in that sample, so the script requested direct raw item scores.

## How to Read the Log for Issues

If scores look suspiciously equal, check:

1. Are `[pair]` responses valid JSON with a probability?
2. Are many attempts used before success (`attempt=3/3` repeatedly)?
3. Are there many `[fallback]` blocks?

If yes, reduce prompt complexity or increase retries/samples:

```bash
python importance_score.py ... --n-samples 5 --max-retries 5
```

## Common Problems

### `FileNotFoundError` on output path

Create the parent directory (`mkdir -p results`) or use simple prefixes (`--output1 run1`).

### Equal section scores

Usually indicates invalid model outputs for pairwise prompts. Inspect `pairwise_debug.log` first.

### Missing dependencies

Install required packages in your active environment:

```bash
pip install PyPDF2 ollama
```

