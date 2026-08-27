# Normalized Reference Scoring

This repository scores scientific papers at three levels:

- top-level sections and subsections
- paragraphs
- cited references inside paragraphs

It also includes baseline methods, evaluation scripts, and citation-graph tooling for corpus-level analysis.

## What The Pipeline Produces

For each paper, the main pipeline writes normalized scores that sum to `1.0` at the paper level:

- `*_section_scores.json`: hierarchical section and subsection scores
- `*_paragraph_scores.json`: paragraph-level technical and citation-derived scores
- `*_citation_scores.json`: total contribution score assigned to each cited reference
- `*_paragraph_citation_scores.json`: per-paragraph citation allocations

These outputs can then be:

- compared against human or API annotations
- used to build weighted citation graphs
- used to generate plots and downstream corpus-level analyses

## Repository Layout

- `papers/`: input PDFs
- `papers_section_titles.txt`: section/subsection trees for each paper
- `paper_results/`: per-paper scoring outputs
- `importance_score.py`: main hierarchical scoring pipeline for one paper
- `run_importance_scores_all_papers.py`: batch runner for all papers
- `baselines/run_baseline.py`: run one baseline on one paper
- `baselines/run_baseline_all_papers.py`: batch runner for baselines
- `extract_sections.py`: extract section trees from a single PDF
- `evaluate_human_section_scores.py`: compare model outputs against annotations
- `run_knowledge_graph.py`: build citation mappings and corpus-level graph metrics
- `visualize_graph.py`: export an interactive HTML citation graph

## Requirements

- Python 3.10+
- PDF files placed in `papers/`
- a section tree for each paper in `papers_section_titles.txt`
- for local LLM scoring: an Ollama server with the model already pulled
- for API baselines: valid API keys in environment variables

## Installation

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

### 2. Install Python dependencies

Minimal dependencies for the core pipeline:

```bash
pip install pypdf PyPDF2 pymupdf ollama numpy matplotlib networkx python-louvain
```

Optional dependencies for API baselines:

```bash
pip install openai anthropic boto3
```

Notes:

- `pymupdf` is used by `extract_sections.py` and is the recommended extractor for section headings.
- `pypdf`/`PyPDF2` are used to read paper text for the scoring pipeline.
- `openai` is needed only for OpenAI-compatible API baselines.
- `anthropic` is needed only for the Anthropic full-paper baseline.
- `boto3` is optional and only needed if you want to experiment with Bedrock outside the current scripts.

### 3. If you use local models through Ollama

Start Ollama and make sure your model is available.

Example:

```bash
ollama serve
ollama pull qwen3:1.7b
```

If Ollama is running on another port or host, pass it with `--host`.

## Inputs The Repo Expects

### PDFs

Place PDF papers in `papers/`.

### Section tree file

Each paper needs a variable in `papers_section_titles.txt` named from the PDF stem.

Example:

- `papers/AFD.pdf` requires `AFD_SECTIONS`
- `papers/fairRQ.pdf` requires `FAIRRQ_SECTIONS`

The easiest way to create these is with `extract_sections.py`.

## Quick Start

### 1. Extract a section tree for one new paper

```bash
python3 extract_sections.py papers/your_paper.pdf
```

To append it directly to `papers_section_titles.txt`:

```bash
python3 extract_sections.py papers/your_paper.pdf --append papers_section_titles.txt
```

What it does:

- first tries embedded PDF bookmarks
- then tries font-aware heading extraction with `pymupdf`
- then falls back to numbered-heading detection

### 2. Run the main scoring pipeline on one paper

```bash
mkdir -p paper_results/your_paper

python3 importance_score.py \
  --pdf papers/your_paper.pdf \
  --pdf-text-backend pypdf \
  --paper-id your_paper \
  --sections-file papers_section_titles.txt \
  --sections-var YOUR_PAPER_SECTIONS \
  --output1 paper_results/your_paper/your_paper \
  --output2 paper_results/your_paper/your_paper \
  --output3 paper_results/your_paper/your_paper \
  --model 'qwen3:1.7b' \
  --model-tag qwen3_1_7b \
  --host http://localhost:11434 \
  --n-samples 5 \
  --temperature 0 \
  --sample-temperature-jitter 0.03 \
  --seed 42 \
  --max-retries 5 \
  --paragraph-direct-max-tokens 1200 \
  --paragraph-compressed-snippet-limit 8000 \
  --debug-log paper_results/your_paper/debug_qwen3_1_7b.log \
  --prompts-output paper_results/your_paper/prompts_qwen3_1_7b.json
```

This writes:

- `paper_results/your_paper/your_paper_qwen3_1_7b_section_scores.json`
- `paper_results/your_paper/your_paper_qwen3_1_7b_paragraph_scores.json`
- `paper_results/your_paper/your_paper_qwen3_1_7b_citation_scores.json`
- `paper_results/your_paper/your_paper_qwen3_1_7b_paragraph_citation_scores.json`
- `paper_results/your_paper/debug_qwen3_1_7b.log`
- `paper_results/your_paper/prompts_qwen3_1_7b.json`

### 3. Run the main scoring pipeline on all papers

```bash
python3 run_importance_scores_all_papers.py \
  --papers-dir papers \
  --sections-file papers_section_titles.txt \
  --results-root paper_results \
  --pdf-text-backend pypdf \
  --model 'qwen3:1.7b' \
  --model-tag qwen3_1_7b \
  --host http://localhost:11434 \
  --n-samples 5 \
  --temperature 0 \
  --sample-temperature-jitter 0.03 \
  --seed 42 \
  --max-retries 5 \
  --paragraph-direct-max-tokens 1200 \
  --paragraph-compressed-snippet-limit 8000 \
  --skip-existing \
  --continue-on-error
```

Useful options:

- `--papers AFD fairRQ`: run only selected paper stems
- `--run-suffix run2`: write a rerun to a new tag such as `qwen3_1_7b_run2`
- `--dry-run`: print the generated commands without running them

### 4. Run a long batch job with `nohup`

```bash
nohup python3 run_importance_scores_all_papers.py \
  --papers-dir papers \
  --sections-file papers_section_titles.txt \
  --results-root paper_results \
  --pdf-text-backend pypdf \
  --model 'qwen3:1.7b' \
  --model-tag qwen3_1_7b \
  --host http://localhost:11434 \
  --n-samples 5 \
  --temperature 0 \
  --sample-temperature-jitter 0.03 \
  --seed 42 \
  --max-retries 5 \
  --paragraph-direct-max-tokens 1200 \
  --paragraph-compressed-snippet-limit 8000 \
  --skip-existing \
  --continue-on-error \
  > nohup_qwen3_1_7b.out 2>&1 &
```

## Baselines

The repository supports several baselines:

- `citation_frequency`
- `length_weighted_frequency`
- `technical_section_prior`
- `single_pass_llm`
- `openai_full_paper`
- `anthropic_full_paper`
- `single_shot_citation_api`

### Run one baseline on one paper

Example: citation-frequency baseline

```bash
python3 baselines/run_baseline.py \
  --baseline citation_frequency \
  --pdf papers/your_paper.pdf \
  --paper-id your_paper \
  --sections-file papers_section_titles.txt \
  --sections-var YOUR_PAPER_SECTIONS \
  --output1 paper_results/your_paper/your_paper \
  --output2 paper_results/your_paper/your_paper \
  --output3 paper_results/your_paper/your_paper \
  --model-tag citation_frequency
```

### Run one baseline on all papers

Example: length-weighted frequency

```bash
python3 baselines/run_baseline_all_papers.py \
  --baseline length_weighted_frequency \
  --papers-dir papers \
  --sections-file papers_section_titles.txt \
  --results-root paper_results \
  --model-tag length_weighted_frequency \
  --skip-existing \
  --continue-on-error
```

### OpenAI-compatible full-paper baseline

This sends the full extracted paper text to an OpenAI-compatible `chat/completions` endpoint and writes standard output files.

```bash
export OPENAI_API_KEY='your_key_here'

python3 baselines/run_baseline.py \
  --baseline openai_full_paper \
  --pdf papers/your_paper.pdf \
  --paper-id your_paper \
  --sections-file papers_section_titles.txt \
  --sections-var YOUR_PAPER_SECTIONS \
  --output1 paper_results/your_paper/your_paper \
  --output2 paper_results/your_paper/your_paper \
  --output3 paper_results/your_paper/your_paper \
  --model-tag openai_full_paper \
  --model 'gpt-oss-120b' \
  --api-key-env OPENAI_API_KEY \
  --api-endpoint https://api.openai.com/v1/chat/completions \
  --request-timeout 600 \
  --max-output-tokens 12000 \
  --api-response-format none \
  --debug-log paper_results/your_paper/debug_openai_full_paper.log
```

If you use another OpenAI-compatible endpoint, replace `--api-endpoint`.

### Anthropic full-paper baseline

```bash
export ANTHROPIC_API_KEY='your_key_here'

python3 baselines/run_baseline.py \
  --baseline anthropic_full_paper \
  --pdf papers/your_paper.pdf \
  --paper-id your_paper \
  --sections-file papers_section_titles.txt \
  --sections-var YOUR_PAPER_SECTIONS \
  --output1 paper_results/your_paper/your_paper \
  --output2 paper_results/your_paper/your_paper \
  --output3 paper_results/your_paper/your_paper \
  --model-tag anthropic_full_paper_claude_sonnet_4_6_direct \
  --model 'claude-sonnet-4-6' \
  --api-key-env ANTHROPIC_API_KEY \
  --request-timeout 600 \
  --max-output-tokens 12000 \
  --debug-log paper_results/your_paper/debug_anthropic_full_paper.log
```

### Single-shot citation API baseline

This baseline only asks the API for citation scores from the full paper text. It does not produce meaningful section or paragraph scoring.

Example with Anthropic:

```bash
export ANTHROPIC_API_KEY='your_key_here'

python3 baselines/run_baseline.py \
  --baseline single_shot_citation_api \
  --api-provider anthropic \
  --pdf papers/your_paper.pdf \
  --paper-id your_paper \
  --sections-file papers_section_titles.txt \
  --sections-var YOUR_PAPER_SECTIONS \
  --output1 paper_results/your_paper/your_paper \
  --output2 paper_results/your_paper/your_paper \
  --output3 paper_results/your_paper/your_paper \
  --model-tag single_shot_citation_anthropic \
  --model 'claude-sonnet-4-6' \
  --api-key-env ANTHROPIC_API_KEY \
  --request-timeout 600 \
  --max-output-tokens 12000 \
  --debug-log paper_results/your_paper/debug_single_shot_citation_anthropic.log
```

## Evaluation

### Compare one model tag against human annotations

```bash
python3 evaluate_human_section_scores.py \
  --annotations-file human_expert_annotations.json \
  --results-root paper_results \
  --model-tag qwen3_1_7b \
  --bootstrap-samples 5000 \
  --seed 7 \
  --enable-citation-eval \
  --citation-top-k 4 \
  --output-json results/human_eval_qwen3_1_7b.json
```

This computes section-level agreement and, when enabled, citation top-k agreement.

### Compare against another annotation file

Example with API-produced annotations stored in JSON:

```bash
python3 evaluate_human_section_scores.py \
  --annotations-file chatgpt_baseline_annotations.json \
  --results-root paper_results \
  --model-tag qwen3_1_7b \
  --bootstrap-samples 5000 \
  --seed 7 \
  --enable-citation-eval \
  --citation-top-k 4 \
  --output-json results/api_eval_qwen3_1_7b.json
```

## Citation Resolution And Corpus-Level Graph Analysis

The graph layer resolves raw citation keys such as `(Smith et al., 2024)` or `[17]` into shared paper identities and then computes corpus-level influence metrics.

### Build or reuse citation mappings

```bash
python3 run_knowledge_graph.py \
  --results paper_results \
  --papers papers \
  --model-tag qwen3_1_7b \
  --save-mappings citation_mappings.json
```

If mappings already exist:

```bash
python3 run_knowledge_graph.py \
  --results paper_results \
  --papers papers \
  --model-tag qwen3_1_7b \
  --load-mappings citation_mappings.json
```

Useful options:

- `--exclude-papers PaperA PaperB`
- `--top-k 20`
- `--influence-depth 3`
- `--influence-decay 0.85`

### Export an interactive HTML graph

```bash
python3 visualize_graph.py \
  --results paper_results \
  --papers papers \
  --model-tag qwen3_1_7b \
  --load-mappings citation_mappings.json \
  --top-k 40 \
  --min-weight 0.005 \
  --output graph.html
```

If you omit `--model-tag`, the graph exporter will include every discovered model tag and let you switch models in the browser.

## GitHub Pages Export

To rebuild the static site in `docs/`:

```bash
./build_github_pages_site.sh \
  --model-tag qwen3_1_7b \
  --load-mappings citation_mappings.json \
  --top-k 40 \
  --min-weight 0.005
```

This refreshes:

- `docs/index.html`
- `docs/lib/`
- `docs/.nojekyll`

## Output Naming Rules

For the main pipeline, if your paper id is `AFD` and your model tag is `qwen3_1_7b`, the outputs are named:

- `AFD_qwen3_1_7b_section_scores.json`
- `AFD_qwen3_1_7b_paragraph_scores.json`
- `AFD_qwen3_1_7b_citation_scores.json`
- `AFD_qwen3_1_7b_paragraph_citation_scores.json`

For reruns, `--run-suffix` appends a suffix to the effective model tag.

Example:

```bash
--model-tag qwen3_1_7b --run-suffix run2
```

produces files tagged with `qwen3_1_7b_run2`.

## Troubleshooting

### Missing section mapping

If you see an error like:

```text
Missing sections mapping for YourPaper.pdf. Expected variable 'YOURPAPER_SECTIONS'
```

run:

```bash
python3 extract_sections.py papers/YourPaper.pdf --append papers_section_titles.txt
```

### Ollama import error

Install the Python client:

```bash
pip install ollama
```

and make sure the Ollama server is running.

### PDF text extraction issues

The current scoring pipeline only supports:

- `--pdf-text-backend pypdf`

If a paper extracts poorly, inspect the raw PDF text first before running a long scoring job.

### Reusing outputs safely

Use:

- `--skip-existing` to avoid overwriting old results
- `--run-suffix` to write a clean rerun into new files

## Recommended First Run

If you just want to verify that the repo works end to end:

1. Put one PDF in `papers/`
2. Run `extract_sections.py` on it and append to `papers_section_titles.txt`
3. Run `importance_score.py` on that one paper with a local Ollama model
4. Inspect the four JSON outputs in `paper_results/<paper>/`
5. Run `run_knowledge_graph.py --save-mappings citation_mappings.json`
6. Run `visualize_graph.py --load-mappings citation_mappings.json --output graph.html`

That gives you a complete single-paper smoke test before you launch large experiments.
