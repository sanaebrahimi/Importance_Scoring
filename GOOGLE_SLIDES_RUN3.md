# Slide 1 - Importance Scoring Update (Run3)

- Project: section/citation importance scoring for an academic paper
- Model pipeline updated to improve robustness and reduce equal-score failures
- This deck summarizes what changed in code and what Run3 produced


---

# Slide 2 - What Changed in the Code

- Replaced fragile “all-pairs in one JSON” scoring with per-pair model calls.
- Added retries per pair (`--max-retries`) and stronger parsing for numeric probabilities.
- Kept multi-sample averaging (`--n-samples`) to stabilize final scores.
- Added direct-allocation fallback when all pairwise calls fail.
- Added deterministic heuristic fallback if model fallback is also invalid.
- Continued enforcing tree normalization:
  - top-level sections sum to 1.0
  - subsection siblings sum to parent score
- Citation scoring now aggregates repeated citations globally across sections/subsections.


---

# Slide 3 - Pairwise + Fallback (Simple)

- Think of pairwise as a mini tournament between sections.
- Example: compare `Introduction` vs `Experiment Results` and ask: which is more important?
- The model returns a number between 0 and 1.
- A higher number means item A wins more of that comparison.
- We repeat this for every pair, add up wins, then normalize so the total is exactly 1.0.
- Fallback means backup plan when the model answer is broken or missing.
- Fallback 1: for one failed pair, use `0.5` (tie).
- Fallback 2: if all pairs fail in a sample, ask for direct section scores instead.
- Fallback 3: if that also fails, use a simple local heuristic so scoring still completes.


---

# Slide 4 - Run3 Top-Level Section Scores

- Scores are normalized (sum = 1.0000).
- Rank order from Run3:
- Limitations: 0.2440
- Conclusion: 0.2048
- Experiment Results: 0.1786
- Learning Credibility Scores On-The-Fly: 0.1429
- CrS-Aware Aggregation: 0.1226
- Team of Agents: 0.0714
- System Overview: 0.0357
- Introduction: 0.0000


---

# Slide 5 - Run3: Experiment Results Breakdown

- Parent section: `Experiment Results`
- Subsection scores:
- Insights from Experimental Observations: 0.1190
- Collaboration Setup: 0.0595
- Experiments Setting: 0.0000

- Inside `Insights from Experimental Observations`:
- Adversary Proportion: 0.0595
- Topology and Link Density: 0.0397
- Judge Alters the Outcome: 0.0198
- Reasoning vs Multi-Choice Tasks: 0.0000


---

# Slide 6 - Run3 Citation Results (Aggregated)

- Total unique citations scored: 56 
- Total citation score mass: 0.6405
- Top citations by aggregated score:
- (Shapley, 1951): 0.1429 (mentions=1)
- (Ebrahimi et al.,2024): 0.1226 (mentions=1)
- (Ebrahimi et al., 2024): 0.0695 (mentions=3)
- (Liu et al., 2023): 0.0169 (mentions=3)
- (Amayuelas et al., 2024): 0.0159 (mentions=4)
- (Liang et al., 2023a): 0.0159 (mentions=3)
- (Zhang et al., 2024a): 0.0109 (mentions=2)
- (Li et al., 2024a): 0.0100 (mentions=3)
- (Bhat et al., 2023): 0.0100 (mentions=2)
- (Li et al., 2024b): 0.0100 (mentions=2)


---

# Slide 7 - Debugging and Traceability

- `pairwise_debug.log` now records raw model responses.
- Entry types:
  - `[pair] parent=... sample=... attempt=... key=A|B`
    - raw response for one pairwise comparison
  - `[fallback] parent=... sample=... attempt=...`
    - raw response for direct allocation fallback
- This makes it easy to inspect malformed outputs and retry behavior.


---

# Slide 8 - Notes and Next Improvements

- Run3 still shows extreme values (for example, Introduction = 0.0000).
- There are near-duplicate citation keys due to formatting variants
  (e.g., `(Ebrahimi et al.,2024)` vs `(Ebrahimi et al., 2024)`).
- Next steps:
  - normalize citation keys before aggregation
  - add score floor constraints for top-level sections
  - tune pair prompts with paper-specific few-shot examples
