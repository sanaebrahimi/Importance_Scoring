# Normalized Reference Scoring

This project scores the content of research papers at three levels:

- sections and subsections
- paragraphs
- citations inside paragraphs

The code reads a PDF, matches its content to a predefined section tree, and then uses an Ollama-hosted language model to distribute importance scores across the paper. Scores are normalized so the full paper totals `1.0`.

## Example

```bash
python3 visualize_graph.py \
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
./build_github_pages_site.sh
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

- `<paper>_section_scores.json`
- `<paper>_paragraph_scores.json`
- `<paper>_citation_scores.json`
- `debug.log`
- `prompts.json`

These three JSON files are also the inputs to the citation-graph and knowledge-discovery layer described below.

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

3. Score all papers:

```bash
python3 run_importance_scores_all_papers.py --continue-on-error
```

3. Score one paper directly:

```bash
python3 importance_score.py \
  --pdf papers/your_paper.pdf \
  --paper-id your_paper \
  --sections-file papers_section_titles.txt \
  --sections-var YOUR_PAPER_SECTIONS \
  --output1 results/your_paper \
  --output2 results/your_paper \
  --output3 results/your_paper
```

4. Inspect one paper's JSON outputs from the terminal:

```bash
python3 -m json.tool paper_results/your_paper/your_paper_section_scores.json
python3 -m json.tool paper_results/your_paper/your_paper_paragraph_scores.json
python3 -m json.tool paper_results/your_paper/your_paper_citation_scores.json
```

5. Run the citation resolver and knowledge-discovery framework:

```bash
python3 run_knowledge_graph.py --save-mappings citation_mappings.json
```

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

This layer is built on the three JSON outputs produced for each paper:

| JSON file | What it stores |
| --- | --- |
| `*_citation_scores.json` | `{citation_str: score}` — total importance assigned from this paper to each cited work |
| `*_paragraph_scores.json` | Per-paragraph `technical_score`, `citation_score`, and `section_path` |
| `*_section_scores.json` | Hierarchical section tree with `total_score` and `citation_score` |

The citation graph uses `W(p,q)` as the citation-derived importance that paper `p` assigns to cited work `q`. The resolver mappings convert `q` from a raw citation string into a canonical paper identity, which creates cross-paper edges.

Section-level weights `W_s(p,q)` are approximated by scanning paragraph texts with citation regexes, splitting compound citation blocks into individual citations, and distributing each paragraph's `citation_score` proportionally across the citations that occur in that paragraph. This approximation is needed because the per-paragraph, per-citation allocation is not stored directly in the JSON outputs.

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

- **Per-citation role ratio.** The ratio `rho(p,q)` gives the same intuition at the paper-citation level: large values indicate technical dependence, while small values indicate rhetorical or contextual use.

- **Influence propagation.** A high total influence `sum_p Inf(q -> p)` suggests that many papers' core arguments depend directly or indirectly on work `q`. These are strong candidates for seminal works in the corpus.

- **Community detection.** Communities become meaningful once the resolver maps raw citations to shared canonical works. Papers that lean on the same foundational literature will cluster together.

- **Temporal influence.** `TInf(q,t)` requires publication years. Once provided, it can reveal whether the importance of a cited work is rising, stable, or declining over time.

- **Research gap score.** Computed as:

  
$Gap(C) = \sum_{p \in C} Orig(p)  −  \sum_{q \notin C, p \in C} W_{tech}(q \rightarrow p)$


  The first term is the total originality of the cluster — how much new technical content the papers contribute. The second term is the total weight with which papers *outside* the cluster cite papers *inside* the cluster from **technical sections** (methods, experiments), measuring how much those ideas have already been adopted as real dependencies.

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
python3 -m json.tool paper_results/your_paper/your_paper_section_scores.json
python3 -m json.tool paper_results/your_paper/your_paper_paragraph_scores.json
python3 -m json.tool paper_results/your_paper/your_paper_citation_scores.json
```


## Notes

- `run_importance_scores_all_papers.py` currently defaults to `5` samples and `5` retries per sample.
- Large sections may be retried with compression and smaller paragraph batches when the model struggles to return a complete allocation.
- `extract_paper_sections.py` skips `Abstract` and stops at `References`.
