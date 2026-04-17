"""
Formal Framework for Knowledge Discovery via Importance-Weighted Citation Graphs

Implements all definitions from the framework paper, consuming the JSON outputs
produced by importance_scoring.py:
  - *_section_scores.json   → hierarchical section importance tree
  - *_paragraph_scores.json → per-paragraph technical/citation scores
  - *_citation_scores.json  → per-citation aggregated importance scores

Quick start
-----------
    from citation_graph_framework import KnowledgeDiscoveryFramework

    fw = KnowledgeDiscoveryFramework.from_results_dir(
        "paper_results/",
        citation_mappings={"(Wu et al.,2023)": "some_paper_dir"},
        pub_dates={"adv_res_paper": 2024},
    )

    print(fw.graph)
    print(fw.pagerank.top_k(5))
    print(fw.originality.rank_by_originality())
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Section-type classification
# ---------------------------------------------------------------------------

_TECHNICAL_KEYWORDS = frozenset({
    "method", "methods", "approach", "algorithm", "implementation",
    "experiment", "experiments", "evaluation", "result", "results",
    "proof", "derivation", "formulation", "model", "architecture",
    "training", "dataset", "analysis", "ablation", "technical",
    "proposed", "framework", "system", "design",
})

_RHETORICAL_KEYWORDS = frozenset({
    "introduction", "intro", "related", "background", "conclusion",
    "discussion", "future", "abstract", "overview", "motivation",
    "acknowledgement", "acknowledgment", "limitations", "limitation",
})


def classify_section(name: str) -> str:
    """Return 'technical', 'rhetorical', or 'neutral' for a section name."""
    lower = name.lower()
    for kw in _TECHNICAL_KEYWORDS:
        if kw in lower:
            return "technical"
    for kw in _RHETORICAL_KEYWORDS:
        if kw in lower:
            return "rhetorical"
    return "neutral"


# ---------------------------------------------------------------------------
# Citation helpers (same patterns as importance_scoring.py)
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(
    r"\([A-Z][^)]*\d{4}[a-z]?\)"
    r"|\[(?:\s*\d+\s*(?:[-,;–]\s*\d+\s*)*)\]"
)


def _split_citation_block(block: str) -> List[str]:
    """Split a compound block into individual citation strings.

    Mirrors split_citation_block in importance_scoring.py so that
    '(Wu et al., 2023; Hong et al., 2023)' → ['(Wu et al., 2023)', '(Hong et al., 2023)'].
    """
    b = block.strip()
    if b.startswith("(") and b.endswith(")"):
        ld, rd, inner = "(", ")", b[1:-1]
    elif b.startswith("[") and b.endswith("]"):
        ld, rd, inner = "[", "]", b[1:-1]
    else:
        return [b]

    if ";" in inner:
        parts = inner.split(";")
    elif ld == "[" and "," in inner:
        parts = inner.split(",")
    else:
        parts = [inner]

    cleaned = [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]
    return [f"{ld}{p}{rd}" for p in cleaned] if cleaned else [b]


def _norm_cit(cit: str) -> str:
    """Strip all whitespace for robust citation-key matching."""
    return re.sub(r"\s+", "", cit)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ParagraphScore:
    """Single paragraph with its importance decomposition (Definition 1)."""

    section_path: List[str]
    paragraph_index: int
    paragraph: str
    technical_score: float   # S_tech(p_i)
    citation_score: float    # S_cite(p_i)

    def top_level_section(self) -> Optional[str]:
        return self.section_path[0] if self.section_path else None

    def citations_in_paragraph(self) -> List[str]:
        """Return individual citation strings found in the paragraph text.

        Compound blocks like '(A, 2023; B, 2024)' are split so each citation
        appears as its own entry; the caller's proportional formula then
        assigns each an equal share of the paragraph's citation_score.
        """
        result: List[str] = []
        for block in _CITATION_RE.findall(self.paragraph):
            result.extend(_split_citation_block(block))
        return result

    def unique_citations(self) -> Set[str]:
        return set(self.citations_in_paragraph())


@dataclass
class SectionScore:
    """Node in the hierarchical importance tree T_p = (V_p, E_p)."""

    name: str
    total_score: float
    citation_score: float
    subsections: Dict[str, "SectionScore"] = field(default_factory=dict)

    @property
    def technical_score(self) -> float:
        return self.total_score - self.citation_score

    @property
    def section_type(self) -> str:
        return classify_section(self.name)

    def all_flat(self) -> List["SectionScore"]:
        """Return this node and all descendants."""
        result: List[SectionScore] = [self]
        for sub in self.subsections.values():
            result.extend(sub.all_flat())
        return result


# ---------------------------------------------------------------------------
# Paper
# ---------------------------------------------------------------------------

class Paper:
    """
    A paper p ∈ P with its full hierarchical importance score tree T_p.

    Loads data from the three JSON files produced by importance_scoring.py.
    """

    def __init__(
        self,
        paper_id: str,
        results_dir: Path,
        pub_date: Optional[int] = None,
    ) -> None:
        self.paper_id = paper_id
        self.results_dir = Path(results_dir)
        self.pub_date = pub_date          # publication year for temporal analysis
        self._section_scores: Optional[Dict[str, SectionScore]] = None
        self._paragraph_scores: Optional[List[ParagraphScore]] = None
        self._citation_scores: Optional[Dict[str, float]] = None

    # --- loading ---

    def load(self) -> "Paper":
        """Load all score data from the results directory."""
        name = self.results_dir.name
        self._load_sections(self.results_dir / f"{name}_section_scores.json")
        self._load_paragraphs(self.results_dir / f"{name}_paragraph_scores.json")
        self._load_citations(self.results_dir / f"{name}_citation_scores.json")
        return self

    def _load_sections(self, path: Path) -> None:
        self._section_scores = {}
        if path.exists():
            with open(path) as f:
                self._section_scores = _parse_section_tree(json.load(f))

    def _load_paragraphs(self, path: Path) -> None:
        self._paragraph_scores = []
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            self._paragraph_scores = [
                ParagraphScore(
                    section_path=p["section_path"],
                    paragraph_index=p["paragraph_index"],
                    paragraph=p["paragraph"],
                    technical_score=p["technical_score"],
                    citation_score=p["citation_score"],
                )
                for p in raw
            ]

    def _load_citations(self, path: Path) -> None:
        self._citation_scores = {}
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            self._citation_scores = {k: v["citation_score"] for k, v in raw.items()}

    # --- properties ---

    @property
    def section_scores(self) -> Dict[str, SectionScore]:
        if self._section_scores is None:
            raise ValueError(f"{self.paper_id}: call load() first")
        return self._section_scores

    @property
    def paragraph_scores(self) -> List[ParagraphScore]:
        if self._paragraph_scores is None:
            raise ValueError(f"{self.paper_id}: call load() first")
        return self._paragraph_scores

    @property
    def citation_scores(self) -> Dict[str, float]:
        if self._citation_scores is None:
            raise ValueError(f"{self.paper_id}: call load() first")
        return self._citation_scores

    # --- derived quantities ---

    def originality_score(self) -> float:
        """Orig(p) = Σ S_tech(p_i)  (Definition 4.3)"""
        return sum(para.technical_score for para in self.paragraph_scores)

    def total_citation_importance(self) -> float:
        """Σ S_cite(p_i) over all paragraphs."""
        return sum(para.citation_score for para in self.paragraph_scores)

    def section_citation_weight(self, citation_key: str, section_name: str) -> float:
        """
        W_s(p, q): importance-weight of citation q in section s.  (Definition 4.1)

        Approximated by distributing each paragraph's citation_score
        proportionally across citations present in that paragraph, then
        summing over paragraphs that belong to the requested section.
        Whitespace-normalised matching handles spacing differences between
        the raw citation key and what the regex extracts from text.
        """
        norm_key = _norm_cit(citation_key)
        total = 0.0
        for para in self.paragraph_scores:
            if para.top_level_section() != section_name:
                continue
            cites = para.citations_in_paragraph()
            if not cites:
                continue
            norm_cites = [_norm_cit(c) for c in cites]
            count = norm_cites.count(norm_key)
            if count == 0:
                continue
            total += para.citation_score * count / len(cites)
        return total

    def citation_role_vector(self, citation_key: str) -> Dict[str, float]:
        """r⃗(p, q): maps each top-level section to W_s(p, q).  (Definition 4.2)"""
        return {
            s: self.section_citation_weight(citation_key, s)
            for s in self.section_scores
        }

    def __repr__(self) -> str:
        return f"Paper(id={self.paper_id!r}, pub_date={self.pub_date})"


def _parse_section_tree(raw: dict) -> Dict[str, SectionScore]:
    result: Dict[str, SectionScore] = {}
    for name, data in raw.items():
        result[name] = SectionScore(
            name=name,
            total_score=data["total_score"],
            citation_score=data["citation_score"],
            subsections=_parse_section_tree(data.get("subsections", {})),
        )
    return result


# ---------------------------------------------------------------------------
# Citation Graph  (Definition 2.1)
# ---------------------------------------------------------------------------

class CitationGraph:
    """
    Importance-Weighted Citation Graph  G = (P, E, W).

    Nodes  : paper IDs (corpus papers) and citation strings (external refs)
    Edges  : (p, q) iff paper p cites q
    Weights: W(p, q) = Σ_{p_i ∈ P(p)} S_i(q) · 1[q ∈ C(p_i)]
             = paper p's total citation_score allocated to q
             (read directly from *_citation_scores.json)
    """

    def __init__(self) -> None:
        self.papers: Dict[str, Paper] = {}
        self._citation_map: Dict[str, str] = {}          # citation_str → paper_id
        self._weights: Dict[Tuple[str, str], float] = defaultdict(float)
        self._out_weights: Dict[str, float] = defaultdict(float)
        self._in_weights: Dict[str, float] = defaultdict(float)

    # --- construction ---

    def add_paper(self, paper: Paper) -> "CitationGraph":
        self.papers[paper.paper_id] = paper
        return self

    def add_citation_mapping(self, citation_str: str, paper_id: str) -> "CitationGraph":
        """Resolve a raw citation string to a corpus paper_id."""
        self._citation_map[citation_str] = paper_id
        return self

    def add_citation_mappings(self, mappings: Dict[str, str]) -> "CitationGraph":
        self._citation_map.update(mappings)
        return self

    @classmethod
    def from_results_dir(
        cls,
        results_dir: str | Path,
        citation_mappings: Optional[Dict[str, str]] = None,
        pub_dates: Optional[Dict[str, int]] = None,
    ) -> "CitationGraph":
        """
        Load every paper subdirectory under results_dir and build the graph.

        Args:
            results_dir    : directory containing one sub-folder per paper.
            citation_mappings: maps raw citation strings → paper_ids (for
                               cross-paper edges between corpus papers).
            pub_dates      : maps paper_id → publication year.
        """
        path = Path(results_dir)
        graph = cls()
        for paper_dir in sorted(path.iterdir()):
            if not paper_dir.is_dir() or paper_dir.name.startswith("."):
                continue
            paper_id = paper_dir.name
            paper = Paper(paper_id, paper_dir, pub_date=(pub_dates or {}).get(paper_id))
            try:
                paper.load()
                graph.add_paper(paper)
            except Exception as exc:
                print(f"[CitationGraph] Warning: could not load {paper_id}: {exc}")
        if citation_mappings:
            graph.add_citation_mappings(citation_mappings)
        graph.build()
        return graph

    def build(self) -> "CitationGraph":
        """Compute all edge weights from loaded paper citation scores."""
        self._weights.clear()
        self._out_weights.clear()
        self._in_weights.clear()
        for paper_id, paper in self.papers.items():
            for cit_str, score in paper.citation_scores.items():
                target = self._citation_map.get(cit_str, cit_str)
                self._weights[(paper_id, target)] += score
                self._out_weights[paper_id] += score
                self._in_weights[target] += score
        return self

    # --- accessors ---

    def weight(self, src: str, tgt: str) -> float:
        """W(p, q)."""
        return self._weights.get((src, tgt), 0.0)

    def out_weight(self, paper_id: str) -> float:
        """W_out(p) = Σ_q W(p, q)."""
        return self._out_weights.get(paper_id, 0.0)

    def in_weight(self, node_id: str) -> float:
        """W_in(q) = Σ_p W(p, q)."""
        return self._in_weights.get(node_id, 0.0)

    def total_weight(self) -> float:
        """m = Σ_{(p,q) ∈ E} W(p, q)."""
        return sum(self._weights.values())

    def edges(self) -> Dict[Tuple[str, str], float]:
        return dict(self._weights)

    def successors(self, paper_id: str) -> Dict[str, float]:
        """Papers cited by paper_id → weight."""
        return {t: w for (s, t), w in self._weights.items() if s == paper_id}

    def predecessors(self, node_id: str) -> Dict[str, float]:
        """Papers that cite node_id → weight."""
        return {s: w for (s, t), w in self._weights.items() if t == node_id}

    def all_nodes(self) -> Set[str]:
        nodes: Set[str] = set(self.papers)
        for s, t in self._weights:
            nodes.add(s)
            nodes.add(t)
        return nodes

    def corpus_nodes(self) -> Set[str]:
        return set(self.papers)

    def _citation_strings_for(self, paper: Paper, cited_key: str) -> Set[str]:
        """
        Find citation strings in *paper* that resolve to cited_key.
        Works whether cited_key is a raw citation string or a paper_id.
        """
        return {
            cit for cit in paper.citation_scores
            if self._citation_map.get(cit, cit) == cited_key
        }

    def __repr__(self) -> str:
        return (
            f"CitationGraph(corpus={len(self.papers)}, "
            f"nodes={len(self.all_nodes())}, edges={len(self._weights)})"
        )


# ---------------------------------------------------------------------------
# PageRank  (Definition 3.1)
# ---------------------------------------------------------------------------

class PageRankAnalyzer:
    """
    Importance-Weighted PageRank.

    IPR(q) = (1-d)/|P| + d · Σ_{p:(p,q)∈E} [W(p,q)/W_out(p)] · IPR(p)
    """

    def __init__(self, graph: CitationGraph) -> None:
        self.graph = graph

    def compute(
        self,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> Dict[str, float]:
        """Return {node_id: IPR_score} after power iteration."""
        nodes = sorted(self.graph.all_nodes())
        n = len(nodes)
        if n == 0:
            return {}

        ipr = {nd: 1.0 / n for nd in nodes}
        teleport = (1.0 - damping) / n

        for _ in range(max_iter):
            new_ipr: Dict[str, float] = {nd: teleport for nd in nodes}

            for (src, tgt), w in self.graph.edges().items():
                w_out = self.graph.out_weight(src)
                if w_out > 0:
                    new_ipr.setdefault(tgt, teleport)
                    new_ipr[tgt] += damping * (w / w_out) * ipr[src]

            # dangling-node redistribution
            dangling = sum(ipr[nd] for nd in nodes if self.graph.out_weight(nd) == 0.0)
            if dangling > 0:
                share = damping * dangling / n
                for nd in nodes:
                    new_ipr[nd] += share

            diff = sum(abs(new_ipr.get(nd, 0.0) - ipr[nd]) for nd in nodes)
            ipr = new_ipr
            if diff < tol:
                break

        total = sum(ipr.values()) or 1.0
        return {k: v / total for k, v in ipr.items()}

    def top_k(self, k: int = 10, **kwargs) -> List[Tuple[str, float]]:
        scores = self.compute(**kwargs)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


# ---------------------------------------------------------------------------
# Influence Propagation  (Definition 3.2)
# ---------------------------------------------------------------------------

class InfluenceAnalyzer:
    """
    Multi-hop influence propagation.

    Inf_1(q→p)   = W(p, q)
    Inf_k(q→p)   = Σ_r W(p,r) · Inf_{k-1}(q→r)
    Inf(q→p)     = Σ_{k=1}^K λ^{k-1} · Inf_k(q→p)
    """

    def __init__(self, graph: CitationGraph) -> None:
        self.graph = graph

    def influence_from(
        self,
        source: str,
        max_depth: int = 3,
        decay: float = 0.5,
    ) -> Dict[str, float]:
        """Compute Inf(source→p) for all nodes p. Returns {node_id: influence}."""
        nodes = self.graph.all_nodes()
        # depth-1 influence
        inf_k = {p: self.graph.weight(p, source) for p in nodes}
        total = dict(inf_k)

        for k in range(2, max_depth + 1):
            new_inf_k: Dict[str, float] = defaultdict(float)
            for p in nodes:
                for r, w_pr in self.graph.successors(p).items():
                    new_inf_k[p] += w_pr * inf_k.get(r, 0.0)
            lam = decay ** (k - 1)
            for p, v in new_inf_k.items():
                total[p] = total.get(p, 0.0) + lam * v
            inf_k = dict(new_inf_k)

        return total

    def seminal_works(
        self,
        top_k: int = 10,
        restrict_to_corpus: bool = False,
        **kwargs,
    ) -> List[Tuple[str, float]]:
        """Rank by Σ_p Inf(q→p) — papers with broadest downstream influence."""
        targets = (
            self.graph.corpus_nodes() if restrict_to_corpus
            else self.graph.all_nodes()
        )
        totals: Dict[str, float] = defaultdict(float)
        for source in targets:
            for p, v in self.influence_from(source, **kwargs).items():
                if p in targets:
                    totals[source] += v
        return sorted(totals.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Community Detection  (Definition 3.3)
# ---------------------------------------------------------------------------

class CommunityDetector:
    """
    Citation community detection via weighted modularity maximisation.

    Q = (1/2m) Σ_{p,q} [W(p,q) − W_out(p)·W_in(q)/(2m)] δ(C(p),C(q))

    Uses the Louvain algorithm when python-louvain + networkx are available;
    falls back to a greedy pair-merge approach otherwise.
    """

    def __init__(self, graph: CitationGraph) -> None:
        self.graph = graph

    def detect(self, restrict_to_corpus: bool = True) -> Dict[str, int]:
        """Return {paper_id: community_id}."""
        try:
            return self._louvain(restrict_to_corpus)
        except ImportError:
            return self._greedy(restrict_to_corpus)

    def _louvain(self, restrict_to_corpus: bool) -> Dict[str, int]:
        import community as community_louvain
        import networkx as nx

        G = self._nx_graph(restrict_to_corpus)
        return community_louvain.best_partition(G, weight="weight")

    def _greedy(self, restrict_to_corpus: bool) -> Dict[str, int]:
        nodes = list(
            self.graph.corpus_nodes() if restrict_to_corpus
            else self.graph.all_nodes()
        )
        comm = {n: i for i, n in enumerate(nodes)}
        m = self.graph.total_weight()
        if m == 0:
            return comm

        improved = True
        while improved:
            improved = False
            best_gain, best_pair = 0.0, None
            for i, p in enumerate(nodes):
                for q in nodes[i + 1:]:
                    if comm[p] == comm[q]:
                        continue
                    gain = self._delta_q(p, q, m)
                    if gain > best_gain:
                        best_gain, best_pair = gain, (p, q)
            if best_pair:
                p, q = best_pair
                old = comm[q]
                new = comm[p]
                for nd in nodes:
                    if comm[nd] == old:
                        comm[nd] = new
                improved = True

        unique = sorted(set(comm.values()))
        remap = {c: i for i, c in enumerate(unique)}
        return {n: remap[c] for n, c in comm.items()}

    def _delta_q(self, p: str, q: str, m: float) -> float:
        w_pq = self.graph.weight(p, q) + self.graph.weight(q, p)
        penalty = self.graph.out_weight(p) * self.graph.in_weight(q) / (4 * m * m)
        return w_pq / (2 * m) - penalty

    def modularity(self, partition: Dict[str, int]) -> float:
        """Compute Q for an arbitrary partition."""
        m = self.graph.total_weight()
        if m == 0:
            return 0.0
        q = 0.0
        for (p, r), w in self.graph.edges().items():
            if p in partition and r in partition and partition[p] == partition[r]:
                q += w - self.graph.out_weight(p) * self.graph.in_weight(r) / (2 * m)
        return q / (2 * m)

    def _nx_graph(self, restrict_to_corpus: bool):
        import networkx as nx

        G = nx.DiGraph()
        nodes = (
            self.graph.corpus_nodes() if restrict_to_corpus
            else self.graph.all_nodes()
        )
        G.add_nodes_from(nodes)
        for (src, tgt), w in self.graph.edges().items():
            if src in nodes and tgt in nodes:
                G.add_edge(src, tgt, weight=w)
        return G


# ---------------------------------------------------------------------------
# Temporal Analysis  (Definition 3.4)
# ---------------------------------------------------------------------------

class TemporalAnalyzer:
    """
    TInf(q, t) = Σ_{p ∈ P_t} W(p, q)

    Requires papers to have pub_date set (pass pub_dates to from_results_dir).
    """

    def __init__(self, graph: CitationGraph) -> None:
        self.graph = graph

    def temporal_influence(self, node_id: str, up_to_year: int) -> float:
        """TInf(q, t): total incoming weight from papers published ≤ up_to_year."""
        total = 0.0
        for src_id, src_paper in self.graph.papers.items():
            if src_paper.pub_date is not None and src_paper.pub_date <= up_to_year:
                total += self.graph.weight(src_id, node_id)
        return total

    def influence_trajectory(
        self,
        node_id: str,
        years: Optional[List[int]] = None,
    ) -> Dict[int, float]:
        """Return {year: TInf(node_id, year)} for each year."""
        if years is None:
            years = sorted(
                {p.pub_date for p in self.graph.papers.values() if p.pub_date is not None}
            )
        return {yr: self.temporal_influence(node_id, yr) for yr in years}

    def classify_trajectory(self, node_id: str) -> str:
        """'rising', 'declining', 'stable', or 'insufficient_data'."""
        traj = self.influence_trajectory(node_id)
        if len(traj) < 2:
            return "insufficient_data"
        vals = [traj[y] for y in sorted(traj)]
        avg_delta = (vals[-1] - vals[0]) / (len(vals) - 1)
        if avg_delta > 0:
            return "rising"
        if avg_delta < 0:
            return "declining"
        return "stable"


# ---------------------------------------------------------------------------
# Section-level Citation Analysis  (Definitions 4.1, 4.2, 4.4)
# ---------------------------------------------------------------------------

class SectionCitationAnalyzer:
    """Section-level citation profiles and technical/rhetorical classification."""

    def __init__(self, graph: CitationGraph) -> None:
        self.graph = graph

    def section_weight(
        self, citing_id: str, cited_key: str, section_name: str
    ) -> float:
        """W_s(p, q)  (Definition 4.1)."""
        paper = self.graph.papers.get(citing_id)
        if paper is None:
            return 0.0
        # find the citation string(s) that correspond to cited_key
        cit_strings = self.graph._citation_strings_for(paper, cited_key) or {cited_key}
        return sum(paper.section_citation_weight(cs, section_name) for cs in cit_strings)

    def role_vector(self, citing_id: str, cited_key: str) -> Dict[str, float]:
        """r⃗(p, q): section → W_s(p, q).  (Definition 4.2)"""
        paper = self.graph.papers.get(citing_id)
        if paper is None:
            return {}
        cit_strings = self.graph._citation_strings_for(paper, cited_key) or {cited_key}
        result: Dict[str, float] = {}
        for sec in paper.section_scores:
            result[sec] = sum(
                paper.section_citation_weight(cs, sec) for cs in cit_strings
            )
        return result

    def classify_citation(
        self,
        citing_id: str,
        cited_key: str,
        tech_sections: Optional[Set[str]] = None,
        rhet_sections: Optional[Set[str]] = None,
    ) -> Tuple[float, str]:
        """
        ρ(p,q) = Σ_{s∈S_tech} W_s / Σ_{s∈S_rhet} W_s  (Definition 4.4)

        Returns (rho, label) where label ∈ {'technical','rhetorical','balanced','unclassified'}.
        """
        paper = self.graph.papers.get(citing_id)
        if paper is None:
            return 0.0, "unclassified"

        if tech_sections is None:
            tech_sections = {
                s for s, sec in paper.section_scores.items()
                if sec.section_type == "technical"
            }
        if rhet_sections is None:
            rhet_sections = {
                s for s, sec in paper.section_scores.items()
                if sec.section_type == "rhetorical"
            }

        tech_w = sum(self.section_weight(citing_id, cited_key, s) for s in tech_sections)
        rhet_w = sum(self.section_weight(citing_id, cited_key, s) for s in rhet_sections)

        if tech_w == 0 and rhet_w == 0:
            return 0.0, "unclassified"
        if rhet_w == 0:
            return float("inf"), "technical"
        rho = tech_w / rhet_w
        if rho > 2.0:
            return rho, "technical"
        if rho < 0.5:
            return rho, "rhetorical"
        return rho, "balanced"


# ---------------------------------------------------------------------------
# Originality  (Definition 4.3)
# ---------------------------------------------------------------------------

class OriginalityAnalyzer:
    """Orig(p) = Σ_i S_tech(p_i) ∈ [0, 1]."""

    def __init__(self, graph: CitationGraph) -> None:
        self.graph = graph

    def originality(self, paper_id: str) -> float:
        paper = self.graph.papers.get(paper_id)
        return paper.originality_score() if paper else 0.0

    def compute_all(self) -> Dict[str, float]:
        return {pid: self.originality(pid) for pid in self.graph.papers}

    def rank_by_originality(self) -> List[Tuple[str, float]]:
        return sorted(self.compute_all().items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Foundational / Peripheral  (Definition 5.2)
# ---------------------------------------------------------------------------

class FoundationalWorkAnalyzer:
    """
    Found(q)  = Σ_{p∈P} Σ_{s∈S_tech} W_s(p, q)
    Periph(q) = Σ_{p∈P} Σ_{s∈S_rhet} W_s(p, q)
    """

    def __init__(self, graph: CitationGraph) -> None:
        self.graph = graph

    def foundational_score(self, cited_key: str) -> float:
        total = 0.0
        for paper in self.graph.papers.values():
            cit_strings = self.graph._citation_strings_for(paper, cited_key) or set()
            for cs in cit_strings:
                for sec_name, sec in paper.section_scores.items():
                    if sec.section_type == "technical":
                        total += paper.section_citation_weight(cs, sec_name)
        return total

    def peripheral_score(self, cited_key: str) -> float:
        total = 0.0
        for paper in self.graph.papers.values():
            cit_strings = self.graph._citation_strings_for(paper, cited_key) or set()
            for cs in cit_strings:
                for sec_name, sec in paper.section_scores.items():
                    if sec.section_type == "rhetorical":
                        total += paper.section_citation_weight(cs, sec_name)
        return total

    def classify(self, cited_key: str, threshold: float = 2.0) -> str:
        """'foundational', 'peripheral', 'balanced', or 'unclassified'."""
        found = self.foundational_score(cited_key)
        periph = self.peripheral_score(cited_key)
        if found == 0 and periph == 0:
            return "unclassified"
        if periph == 0:
            return "foundational"
        ratio = found / periph
        if ratio > threshold:
            return "foundational"
        if ratio < 1.0 / threshold:
            return "peripheral"
        return "balanced"

    def compute_all(self) -> Dict[str, Dict]:
        """Scores and classification for every cited key across the corpus."""
        all_keys: Set[str] = set()
        for paper in self.graph.papers.values():
            for cit in paper.citation_scores:
                all_keys.add(self.graph._citation_map.get(cit, cit))
        return {
            key: {
                "foundational_score": self.foundational_score(key),
                "peripheral_score": self.peripheral_score(key),
                "classification": self.classify(key),
            }
            for key in all_keys
        }


# ---------------------------------------------------------------------------
# Research Gap Detection  (Definition 5.3)
# ---------------------------------------------------------------------------

class ResearchGapDetector:
    """
    Gap(C, t) = Σ_{p∈C_t} Orig(p)
              − Σ_{q∈P\C_t} Σ_{p∈C_t} W_{S_tech}(q, p)

    High Gap → technically original cluster not yet built upon by outside work.
    """

    def __init__(self, graph: CitationGraph) -> None:
        self.graph = graph
        self._orig = OriginalityAnalyzer(graph)

    def _tech_weight_toward_cluster(
        self,
        citing_id: str,
        cluster_set: Set[str],
    ) -> float:
        """W_{S_tech}(q, p) summed over p ∈ cluster."""
        paper = self.graph.papers.get(citing_id)
        if paper is None:
            return 0.0
        total = 0.0
        for cit_str in paper.citation_scores:
            resolved = self.graph._citation_map.get(cit_str, cit_str)
            if resolved not in cluster_set:
                continue
            for sec_name, sec in paper.section_scores.items():
                if sec.section_type == "technical":
                    total += paper.section_citation_weight(cit_str, sec_name)
        return total

    def gap_score(
        self,
        cluster: List[str],
        up_to_year: Optional[int] = None,
    ) -> float:
        """
        Args:
            cluster     : list of paper_ids forming cluster C.
            up_to_year  : restrict to papers published ≤ this year.
        """
        cluster_set = set(cluster)

        cluster_t = [
            p for p in cluster_set
            if p in self.graph.papers
            and (
                up_to_year is None
                or self.graph.papers[p].pub_date is None
                or self.graph.papers[p].pub_date <= up_to_year
            )
        ]
        cluster_t_set = set(cluster_t)

        orig_sum = sum(self._orig.originality(p) for p in cluster_t)

        outside = [p for p in self.graph.papers if p not in cluster_set]
        adoption_sum = sum(
            self._tech_weight_toward_cluster(q, cluster_t_set) for q in outside
        )

        return orig_sum - adoption_sum


# ---------------------------------------------------------------------------
# Concept-Level Citation Graph  (Definition 5.1)
# ---------------------------------------------------------------------------

ConceptExtractor = Callable[[str], List[str]]


class ConceptCitationGraph:
    """
    G_K = (K ∪ P, E_K, W_K)

    W_K(k, q) = Σ_{p∈P} Σ_{p_i: k∈p_i, q∈C(p_i)} S_i(q)

    Provide a ConceptExtractor to map paragraph text → [concept, ...].
    The default extractor uses simple heuristics (capitalised phrases and
    hyphenated terms); replace with an NLP pipeline for better results.
    """

    def __init__(
        self,
        graph: CitationGraph,
        concept_extractor: Optional[ConceptExtractor] = None,
    ) -> None:
        self.graph = graph
        self.concept_extractor = concept_extractor or _default_concept_extractor
        self._weights: Dict[Tuple[str, str], float] = defaultdict(float)

    def build(self) -> "ConceptCitationGraph":
        self._weights.clear()
        for paper in self.graph.papers.values():
            for para in paper.paragraph_scores:
                concepts = self.concept_extractor(para.paragraph)
                citations = para.citations_in_paragraph()
                if not concepts or not citations:
                    continue
                score_per_cit = para.citation_score / len(citations)
                for cit in citations:
                    for concept in set(concepts):
                        self._weights[(concept, cit)] += score_per_cit
        return self

    def weight(self, concept: str, citation: str) -> float:
        return self._weights.get((concept, citation), 0.0)

    def citations_for_concept(self, concept: str) -> Dict[str, float]:
        result = {c: w for (k, c), w in self._weights.items() if k == concept}
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def concepts_for_citation(self, citation: str) -> Dict[str, float]:
        result = {k: w for (k, c), w in self._weights.items() if c == citation}
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def top_concepts(self, k: int = 20) -> List[Tuple[str, float]]:
        totals: Dict[str, float] = defaultdict(float)
        for (concept, _), w in self._weights.items():
            totals[concept] += w
        return sorted(totals.items(), key=lambda x: x[1], reverse=True)[:k]


def _default_concept_extractor(text: str) -> List[str]:
    caps = re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+", text)
    hyphenated = re.findall(r"\b[a-z]+(?:-[a-z]+)+\b", text)
    return caps + hyphenated


# ---------------------------------------------------------------------------
# Main Framework
# ---------------------------------------------------------------------------

class KnowledgeDiscoveryFramework:
    """
    Unified entry point for all analyses in the knowledge-discovery framework.

    Sections 3–5 of the framework paper are accessible via sub-analyzers:
      .pagerank         → PageRankAnalyzer       (§3.1)
      .influence        → InfluenceAnalyzer       (§3.2)
      .communities      → CommunityDetector       (§3.3)
      .temporal         → TemporalAnalyzer         (§3.4)
      .section_citations→ SectionCitationAnalyzer  (§4.1–4.2, 4.4)
      .originality      → OriginalityAnalyzer      (§4.3)
      .foundational     → FoundationalWorkAnalyzer (§5.2)
      .research_gaps    → ResearchGapDetector      (§5.3)
      .concept_graph    → ConceptCitationGraph      (§5.1, after build_concept_graph())
    """

    def __init__(self, graph: CitationGraph) -> None:
        self.graph = graph
        self.pagerank = PageRankAnalyzer(graph)
        self.influence = InfluenceAnalyzer(graph)
        self.communities = CommunityDetector(graph)
        self.temporal = TemporalAnalyzer(graph)
        self.section_citations = SectionCitationAnalyzer(graph)
        self.originality = OriginalityAnalyzer(graph)
        self.foundational = FoundationalWorkAnalyzer(graph)
        self.research_gaps = ResearchGapDetector(graph)
        self._concept_graph: Optional[ConceptCitationGraph] = None

    @classmethod
    def from_results_dir(
        cls,
        results_dir: str | Path,
        citation_mappings: Optional[Dict[str, str]] = None,
        pub_dates: Optional[Dict[str, int]] = None,
    ) -> "KnowledgeDiscoveryFramework":
        """Build the framework from a directory of importance-scoring results."""
        graph = CitationGraph.from_results_dir(results_dir, citation_mappings, pub_dates)
        return cls(graph)

    def build_concept_graph(
        self,
        concept_extractor: Optional[ConceptExtractor] = None,
    ) -> "KnowledgeDiscoveryFramework":
        """Construct the concept-level citation graph (§5.1)."""
        self._concept_graph = ConceptCitationGraph(self.graph, concept_extractor)
        self._concept_graph.build()
        return self

    @property
    def concept_graph(self) -> ConceptCitationGraph:
        if self._concept_graph is None:
            raise ValueError("Call build_concept_graph() first.")
        return self._concept_graph

    def summary(self) -> Dict:
        """High-level corpus and graph statistics."""
        papers = self.graph.papers
        orig_scores = {pid: p.originality_score() for pid, p in papers.items()}
        avg_orig = sum(orig_scores.values()) / len(papers) if papers else 0.0
        return {
            "corpus_papers": len(papers),
            "total_nodes": len(self.graph.all_nodes()),
            "total_edges": len(self.graph.edges()),
            "total_weight": round(self.graph.total_weight(), 6),
            "avg_originality": round(avg_orig, 6),
            "papers": {
                pid: {
                    "originality": round(orig_scores[pid], 6),
                    "citation_importance": round(p.total_citation_importance(), 6),
                    "unique_citations": len(p.citation_scores),
                }
                for pid, p in papers.items()
            },
        }

    def __repr__(self) -> str:
        return f"KnowledgeDiscoveryFramework(graph={self.graph!r})"
