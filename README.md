# Normalized Reference Scoring

This project scores the content of research papers at three levels:

- sections and subsections
- paragraphs
- citations inside paragraphs

The code reads a PDF, matches its content to a predefined section tree, and then uses an Ollama-hosted language model to distribute importance scores across the paper. Scores are normalized so the full paper totals `1.0`.

## Example

```bash
python3 visualize_graph.py \
  --model-tag llama3_2 \
  --load-mappings citation_mappings.json \
  --top-k 40 \
  --min-weight 0.005 \
  --output graph_focused.html
```
`--top-k 40` pulls in more external cited works so you see the shared foundation across papers. `--min-weight 0.005` cuts the noise — only citations that carry real importance weight survive, so the edges you see are genuine dependencies rather than passing mentions. Using `--load-mappings` skips re-parsing the PDFs so it runs in a few seconds.

## GitHub Pages

The repo is set up so the interactive graph can be published as a static site from `docs/`.

Rebuild the published site with:

```bash
./build_github_pages_site.sh \
  --model-tag llama3_2 \
  --load-mappings citation_mappings.json \
  --top-k 40 \
  --min-weight 0.005
```

This writes the graph to `docs/index.html`, copies the local JS/CSS assets into `docs/lib`, and refreshes `docs/.nojekyll`.

The published page is not a fixed snapshot anymore. It includes input boxes for
`top-k` and `min-weight`, so you can change those graph filters directly in the
browser without regenerating the site.

To publish it on `github.io`:

1. Commit and push the `docs/` folder plus any graph updates you want online.
2. In GitHub, open `Settings` -> `Pages`.
3. Set `Source` to `Deploy from a branch`.
4. Choose branch `main` and folder `/docs`.
5. Save. GitHub will publish the site at `https://<your-username>.github.io/<repo-name>/`.

## Main Files

- [importance_score.py](importance_score.py): runs the scoring pipeline for one paper and writes JSON outputs for sections, paragraphs, and citations.
- [run_importance_scores_all_papers.py](run_importance_scores_all_papers.py): runs `importance_score.py` for every PDF in `papers/`, creates one results folder per paper, and saves logs plus prompt snapshots.
- [extract_sections.py](extract_sections.py): extracts the section/subsection tree from a **single** PDF without an LLM and appends the result to `papers_section_titles.txt`. Uses font-size analysis via pymupdf.
- [extract_paper_sections.py](extract_paper_sections.py): batch version — extracts section titles from every PDF in `papers/` and rewrites `papers_section_titles.txt`.
- [papers_section_titles.txt](papers_section_titles.txt): per-paper section trees used by the scorer.

## Expected Inputs

- PDF papers in [papers](papers)
- a section mapping for each paper in [papers_section_titles.txt](papers_section_titles.txt)
- an Ollama model running locally or on a reachable host

The scorer supports both author-year citations like `(Smith et al., 2024)` and numeric citations like `[2]` or `[7, 13]`.

## Outputs

For each paper, the batch runner creates a folder under [paper_results](paper_results) containing:

- `<paper>_<model_tag>_section_scores.json`
- `<paper>_<model_tag>_paragraph_scores.json`
- `<paper>_<model_tag>_citation_scores.json`
- `<paper>_<model_tag>_paragraph_citation_scores.json`
- `debug_<model_tag>.log`
- `prompts_<model_tag>.json`

These four JSON files are also the inputs to the citation-graph and knowledge-discovery layer described below.

## Typical Workflow

1. Extract the section tree for each new paper (run once per paper, appends to `papers_section_titles.txt`):

```bash
python3 extract_sections.py papers/your_paper.pdf --append papers_section_titles.txt
```

The script detects sections automatically from the PDF — no LLM needed. It tries three strategies in order: embedded PDF bookmarks, font-size analysis, and numbered-heading patterns.

To preview the output without writing it:

```bash
python3 extract_sections.py papers/your_paper.pdf
```

2. (Optional) Regenerate all section trees at once using the older batch extractor:

```bash
python3 extract_paper_sections.py
```

3. Score all papers for one model:

```bash
python3 run_importance_scores_all_papers.py \
  --model llama3.2 \
  --model-tag llama3_2 \
  --host http://localhost:11434 \
  --n-samples 5 \
  --temperature 0 \
  --max-retries 5 \
  --paragraph-direct-max-tokens 1200 \
  --paragraph-compressed-snippet-limit 800 \
  --continue-on-error \
  --skip-existing
```

For long runs on a server:

```bash
nohup python3 run_importance_scores_all_papers.py \
  --model qwen3:4b \
  --model-tag qwen3_4b \
  --host http://localhost:11434 \
  --n-samples 5 \
  --temperature 0 \
  --max-retries 5 \
  --paragraph-direct-max-tokens 1200 \
  --paragraph-compressed-snippet-limit 800 \
  --continue-on-error \
  --skip-existing \
  > nohup_run_qwen3_4b.out 2>&1 &
```

4. Score one paper directly:

```bash
python3 importance_score.py \
  --pdf papers/your_paper.pdf \
  --paper-id your_paper \
  --sections-file papers_section_titles.txt \
  --sections-var YOUR_PAPER_SECTIONS \
  --output1 paper_results/your_paper/your_paper \
  --output2 paper_results/your_paper/your_paper \
  --output3 paper_results/your_paper/your_paper \
  --model llama3.2 \
  --model-tag llama3_2 \
  --host http://localhost:11434 \
  --n-samples 5 \
  --temperature 0 \
  --max-retries 5 \
  --paragraph-direct-max-tokens 1200 \
  --paragraph-compressed-snippet-limit 800 \
  --debug-log paper_results/your_paper/debug_llama3_2.log \
  --prompts-output paper_results/your_paper/prompts_llama3_2.json
```

5. Inspect one paper's JSON outputs from the terminal:

```bash
python3 -m json.tool paper_results/your_paper/your_paper_llama3_2_section_scores.json
python3 -m json.tool paper_results/your_paper/your_paper_llama3_2_paragraph_scores.json
python3 -m json.tool paper_results/your_paper/your_paper_llama3_2_citation_scores.json
python3 -m json.tool paper_results/your_paper/your_paper_llama3_2_paragraph_citation_scores.json
```

6. Run the citation resolver and knowledge-discovery framework:

```bash
python3 run_knowledge_graph.py \
  --model-tag llama3_2 \
  --save-mappings citation_mappings.json
```

7. Build an interactive graph for one model:

```bash
python3 visualize_graph.py \
  --model-tag llama3_2 \
  --load-mappings citation_mappings.json \
  --top-k 40 \
  --min-weight 0.005 \
  --output graph_focused.html
```

If you omit `--model-tag`, the HTML bundles every discovered model and lets you switch between them in the browser.

## Evaluation

### Human-annotation evaluation

To compare model outputs against [human_expert_annotations.json](human_expert_annotations.json):

```bash
python3 evaluate_human_section_scores.py \
  --annotations-file human_expert_annotations.json \
  --model-tag llama3_2 \
  --enable-citation-eval \
  --citation-top-k 4 \
  --output-json results/human_eval_llama3_2.json
```

The same pattern works for other model tags, for example:

```bash
python3 evaluate_human_section_scores.py \
  --annotations-file human_expert_annotations.json \
  --model-tag gemma2_2b_2 \
  --enable-citation-eval \
  --citation-top-k 4 \
  --output-json results/human_eval_gemma2_2b_2.json
```

```bash
python3 evaluate_human_section_scores.py \
  --annotations-file human_expert_annotations.json \
  --model-tag qwen3_4b \
  --enable-citation-eval \
  --citation-top-k 4 \
  --output-json results/human_eval_qwen3_4b.json
```

### ChatGPT baseline evaluation

To compare the same model outputs against [chatgpt_baseline_annotations.json](chatgpt_baseline_annotations.json):

```bash
python3 evaluate_human_section_scores.py \
  --annotations-file chatgpt_baseline_annotations.json \
  --model-tag llama3_2 \
  --enable-citation-eval \
  --citation-top-k 4 \
  --output-json results/chatgpt_baseline_eval_llama3_2.json
```

### Sample-size citation analysis

The repo also includes a replay-based analysis over the saved `debug_<model_tag>.log` files, measuring how citation metrics change as the number of averaged samples increases from `1` to `5`:

```bash
python3 analyze_sample_prefix_citation_metrics.py \
  --annotations-file human_expert_annotations.json \
  --model-tags llama3.2_1b llama3_2 gemma2_2b_2 gemma3_4b phi3_medium_run2 qwen3_4b qwen2_5_3b \
  --output-json results/sample_prefix_citation_metrics_all_models.json
```

Then plot the `W`-based curves with matplotlib:

```bash
python3 plot_sample_prefix_metrics.py \
  --input-json results/sample_prefix_citation_metrics_all_models.json \
  --score-key w \
  --output-dir results/plots
```

This produces:

- `sample_prefix_metrics_w_combined.pdf`
- `sample_prefix_metrics_w_overlapat4.pdf`
- `sample_prefix_metrics_w_recallat4.pdf`
- `sample_prefix_metrics_w_hitat4.pdf`
- `sample_prefix_metrics_w_mrr.pdf`

## Knowledge Discovery Layer

The system has two layers that build on top of `importance_score.py`'s existing JSON outputs.

### Layer 1 — CitationResolver (`citation_resolver.py`)

Purpose: turn raw citation strings into structured, cross-paper identities.

`importance_score.py` produces `*_citation_scores.json` files where each citation is just a string key such as `(Wu et al.,2023)` or `[7]`. These keys are paper-local and opaque: `[7]` in paper A and `[7]` in paper B refer to different works. The resolver fixes this.

What it does:

- Reads each paper's PDF and finds the `References` section.
- For numeric citation style (`[1]`, `[2]`, ...), splits reference entries on `[N] ... [N+1]` boundaries. Resolution is **scoped per paper** so that `[2]` in paper A and `[2]` in paper B are resolved independently and never cross-contaminate.
- For author-year style (`(Hong et al., 2023)`), searches for the expected last name, anchors on the year, verifies that the matched name is the first author rather than a co-author in the middle of the entry, and checks that the first year in the extracted window matches the expected year.
- Parses each reference entry into authors, title, and year. Three reference formats are handled:
  - **NeurIPS / ACL style** — `Authors. Year. Title. Venue.` — year appears immediately after the author block.
  - **ACM style** — `Authors. Title. Venue, Year.` — year appears at the end, preceded by `,`.
  - **IEEE style** — `Authors, "Title," Venue, Year.` — title is enclosed in double quotation marks. The year may be embedded in a conference date (e.g. `23–24 Feb 2018`) rather than preceded by `,`; the resolver detects this via the quoted title and switches to quote-based splitting, avoiding false splits on venue abbreviations such as `no.` or `vol.`.
- Assigns a canonical identifier such as `hong_2023`, so two papers that cite the same resolved work share the same identifier.

### Layer 2 — CitationGraph + Analyzers (`citation_graph_framework.py`)

This layer is built on the four JSON outputs produced for each paper:

| JSON file | What it stores |
| --- | --- |
| `*_citation_scores.json` | `{citation_str: score}` — total importance assigned from this paper to each cited work |
| `*_paragraph_scores.json` | Per-paragraph `technical_score`, `citation_score`, and `section_path` |
| `*_paragraph_citation_scores.json` | Per-paragraph, per-citation allocations with `section_path`, `paragraph_index`, `citation`, and `citation_score` |
| `*_section_scores.json` | Hierarchical section tree with `total_score` and `citation_score` |

The citation graph uses `W(p,q)` as the citation-derived importance that paper `p` assigns to cited work `q`. The resolver mappings convert `q` from a raw citation string into a canonical paper identity, which creates cross-paper edges.

Section-level weights `W_s(p,q)` are built from the stored paragraph-level citation allocations by summing the `citation_score` values assigned to `q` over paragraphs that belong to section `s`.

Technical citation weights `W_tech(p,q)` are built from the same paragraph-level citation allocations, but each paragraph-to-citation contribution is weighted by the `technical_score` of the paragraph in which that citation appears:

`W_tech(p,q) = \sum_{r \in p,\ q \in r} w_r(p,q) \cdot technical_score(r)`

where `w_r(p,q)` is the paragraph-local citation allocation to cited work `q` in paragraph `r`.

## Score Definitions

The framework uses the following core scores and derived quantities:

| Symbol | Meaning |
| --- | --- |
| `technical_score(r)` | Technical contribution assigned to paragraph `r`. Higher values mean the paragraph contributes more of the paper's original technical content. |
| `citation_score(r)` | Citation-derived contribution assigned to paragraph `r`. Higher values mean the paragraph's importance comes more from how it uses prior work. |
| `total_score(s)` | Total normalized importance of section or subsection `s`. |
| `citation_score(s)` | Citation-derived part of the importance assigned to section or subsection `s`. |
| `W(p,q)` | Total citation importance that citing paper `p` assigns to cited work `q`. |
| `W_s(p,q)` | Citation importance that paper `p` assigns to cited work `q` from section `s` only. |
| `W_tech(p,q)` | Technical citation importance that paper `p` assigns to cited work `q`, computed by weighting each paragraph-local citation allocation by the `technical_score` of the paragraph where the citation appears. |
| `IPR(q)` | Global importance of cited work `q` in the weighted citation graph, computed with PageRank-style propagation over `W(p,q)`. |
| `Inf(q -> p)` | Multi-hop influence of cited work `q` on paper `p` through the weighted citation graph. |
| `Q` | Weighted modularity score used to evaluate community structure in the citation graph. |
| `TInf(q,t)` | Total incoming citation importance received by cited work `q` up to time `t`. |
| `Orig(p)` | Originality of paper `p`, defined as the sum of `technical_score` over all paragraphs in `p`. |
| `rho(p,q)` | Technical-versus-rhetorical role ratio for the citation from `p` to `q`. Larger values indicate that the citation is used more as a real technical dependency than as context or support. |
| `Found(q)` | Importance received by cited work `q` from technical or foundational parts of papers such as methods, proofs, and results. |
| `Periph(q)` | Importance received by cited work `q` from support-oriented parts of papers such as introduction, background, or related work. |
| `Gap(C,t)` | Research-gap score for a paper set or cluster `C` at time `t`, combining originality with downstream technical adoption. |

## Analyzers

Each analyzer reads from the graph and computes one quantity from the framework:

| Analyzer | What it computes | Key input |
| --- | --- | --- |
| `PageRankAnalyzer` | `IPR(q)` — global importance of each paper via weighted random walk | Edge weights `W(p,q)` |
| `InfluenceAnalyzer` | `Inf(q -> p)` — how much paper `q` influenced paper `p` across up to `K` hops | Edge weights and decay factor λ |
| `CommunityDetector` | Partition of papers into clusters by citation similarity | Weighted modularity `Q` |
| `TemporalAnalyzer` | `TInf(q,t)` — incoming importance up to year `t` | Publication year per paper |
| `OriginalityAnalyzer` | `Orig(p)` — sum of `technical_score` across all paragraphs | Paragraph scores |
| `SectionCitationAnalyzer` | `W_s(p,q)` and `rho(p,q)` — technical vs rhetorical citation role | Paragraph text and section tree |
| `FoundationalWorkAnalyzer` | `Found(q)` vs `Periph(q)` — cited in methods/results or only in support sections | Section classification |
| `ResearchGapDetector` | `Gap(C,t)` — original work not yet adopted by others | Originality plus adoption weights |

## How to Interpret the Results

- **PageRank / importance propagation.** A high `IPR(q)` means the cited work is linked from high-importance parts of many papers. Unlike standard citation counts, a citation appearing once in a crucial methods section can outweigh several low-value introductory mentions.

- **Originality.** A high `Orig(p)` means the paper's value comes mostly from its own technical content rather than from synthesizing prior work. A lower value indicates that much of the paper's importance is citation-derived.

- **Foundational vs peripheral use.** If `Found(q) >> Periph(q)`, then the cited work is functioning as a real technical dependency. If `Periph(q) >> Found(q)`, the work is mostly used for context, motivation, or related work.

- **Technical citation weight.** A high `W_tech(p,q)` means paper `p` relies on cited work `q` inside paragraphs that your framework judges to be technically important. This helps separate deep technical dependence from lighter background mention.

- **Per-citation role ratio.** The ratio `rho(p,q)` gives the same intuition at the paper-citation level: large values indicate technical dependence, while small values indicate rhetorical or contextual use.

- **Influence propagation.** A high total influence `sum_p Inf(q -> p)` suggests that many papers' core arguments depend directly or indirectly on work `q`. These are strong candidates for seminal works in the corpus.

- **Community detection.** Communities become meaningful once the resolver maps raw citations to shared canonical works. Papers that lean on the same foundational literature will cluster together.

- **Temporal influence.** `TInf(q,t)` requires publication years. Once provided, it can reveal whether the importance of a cited work is rising, stable, or declining over time.

- **Research gap score.** Computed as:

  
$Gap(C) = \sum_{p \in C} Orig(p)  −  \sum_{q \notin C, p \in C} W_{tech}(q \rightarrow p)$


  The first term is the total originality of the cluster — how much new technical content the papers contribute. The second term is the total technical citation weight with which papers *outside* the cluster cite papers *inside* the cluster, where each paragraph-local citation allocation is weighted by the `technical_score` of the citing paragraph. This measures how much those ideas have already been adopted as real technical dependencies.

  | Situation | Gap score |
  |---|---|
  | High originality, rarely cited in methods by others | Large positive — underexplored |
  | High originality, heavily cited in methods by others | Near zero or negative |
  | Low originality, rarely cited | Near zero — nothing to gap |

  A large positive value signals that a paper (or cluster) produced original technical work that the rest of the corpus has not yet built on. Note that the score is corpus-relative: it reflects adoption within the papers you scored, not globally.

- **Corpus-relativity caveat.** All scores — gap, originality, PageRank, influence — are computed over the papers present in `paper_results/`. A small or domain-skewed corpus will produce scores that reflect that local view.

## Terminal Commands to See Results

### View one paper's raw scoring outputs

```bash
python3 -m json.tool paper_results/your_paper/your_paper_llama3_2_section_scores.json
python3 -m json.tool paper_results/your_paper/your_paper_llama3_2_paragraph_scores.json
python3 -m json.tool paper_results/your_paper/your_paper_llama3_2_citation_scores.json
python3 -m json.tool paper_results/your_paper/your_paper_llama3_2_paragraph_citation_scores.json
```


## Notes

- `run_importance_scores_all_papers.py` currently defaults to `5` samples and `5` retries per sample.
- Large sections may be retried with compression and smaller paragraph batches when the model struggles to return a complete allocation.
- `extract_paper_sections.py` skips `Abstract` and stops at `References`.
