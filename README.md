# Importance Scoring

This project scores the content of research papers at three levels:

- sections and subsections
- paragraphs
- citations inside paragraphs

The code reads a PDF, matches its content to a predefined section tree, and then uses an Ollama-hosted language model to distribute importance scores across the paper. Scores are normalized so the full paper totals `1.0`.

## Main Files

- [importance_score.py](importance_score.py): runs the scoring pipeline for one paper and writes JSON outputs for sections, paragraphs, and citations.
- [run_importance_scores_all_papers.py](run_importance_scores_all_papers.py): runs `importance_score.py` for every PDF in `papers/`, creates one results folder per paper, and saves logs plus prompt snapshots.
- [extract_paper_sections.py](extract_paper_sections.py): extracts section/subsection titles from each PDF and writes them into `papers_section_titles.txt`.
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

## Typical Workflow

1. Generate or refresh section trees:

```bash
python3 extract_paper_sections.py
```

2. Score all papers:

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

## Notes

- `run_importance_scores_all_papers.py` currently defaults to `5` samples and `5` retries per sample.
- Large sections may be retried with compression and smaller paragraph batches when the model struggles to return a complete allocation.
- `extract_paper_sections.py` skips `Abstract` and stops at `References`.
