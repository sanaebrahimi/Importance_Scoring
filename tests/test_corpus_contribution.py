import unittest
from pathlib import Path

from citation_graph_framework import (
    CitationGraph,
    CorpusContributionAnalyzer,
    Paper,
    ParagraphCitationScore,
    ParagraphScore,
)


class DummyPaper:
    def __init__(self, paper_id: str, originality: float, citation_scores: dict[str, float]) -> None:
        self.paper_id = paper_id
        self._originality = originality
        self.citation_scores = citation_scores

    def originality_score(self) -> float:
        return self._originality


class CorpusContributionTests(unittest.TestCase):
    def test_normalized_influence_uses_propagated_mass(self) -> None:
        graph = CitationGraph()
        graph.add_paper(DummyPaper("A", 2.0, {"citB": 0.5}))
        graph.add_paper(DummyPaper("B", 3.0, {"citC": 0.2}))
        graph.add_paper(DummyPaper("C", 5.0, {}))
        graph.add_citation_mappings(
            {
                "A": {"citB": "B"},
                "B": {"citC": "C"},
                "C": {},
            }
        )
        graph.build()

        analyzer = CorpusContributionAnalyzer(graph)

        sigma = analyzer.compute()
        propagated = analyzer.propagated_influence_mass()
        normalized = analyzer.normalized_influence()

        self.assertAlmostEqual(sigma["A"], 2.0, places=8)
        self.assertAlmostEqual(sigma["B"], 4.5, places=8)
        self.assertAlmostEqual(sigma["C"], 6.5, places=8)

        self.assertAlmostEqual(propagated["A"], 0.0, places=8)
        self.assertAlmostEqual(propagated["B"], 1.5, places=8)
        self.assertAlmostEqual(propagated["C"], 1.5, places=8)

        self.assertAlmostEqual(normalized["A"], 0.0, places=8)
        self.assertAlmostEqual(normalized["B"], 0.5, places=8)
        self.assertAlmostEqual(normalized["C"], 0.5, places=8)

        # Backward-compatible alias should stay numerically identical.
        self.assertEqual(propagated, analyzer.external_contribution())

    def test_source_weighted_contribution_uses_ancestor_technical_scores(self) -> None:
        graph = CitationGraph()
        graph.add_paper(DummyPaper("A", 2.0, {"citB": 0.5}))
        graph.add_paper(DummyPaper("B", 3.0, {"citC": 0.2}))
        graph.add_paper(DummyPaper("C", 5.0, {}))
        graph.add_citation_mappings(
            {
                "A": {"citB": "B"},
                "B": {"citC": "C"},
                "C": {},
            }
        )
        graph.build()

        analyzer = CorpusContributionAnalyzer(graph)

        sigma_src = analyzer.compute_source_weighted()
        propagated_src = analyzer.source_weighted_propagated_mass()
        normalized_src = analyzer.normalized_source_weighted_influence()

        self.assertAlmostEqual(sigma_src["A"], 2.0, places=8)
        self.assertAlmostEqual(sigma_src["B"], 4.0, places=8)  # 3 + 0.5 * 2
        self.assertAlmostEqual(sigma_src["C"], 5.8, places=8)  # 5 + 0.2 * 4

        self.assertAlmostEqual(propagated_src["A"], 0.0, places=8)
        self.assertAlmostEqual(propagated_src["B"], 1.0, places=8)
        self.assertAlmostEqual(propagated_src["C"], 0.8, places=8)

        self.assertAlmostEqual(normalized_src["A"], 0.0, places=8)
        self.assertAlmostEqual(normalized_src["B"], 1.0 / 1.8, places=8)
        self.assertAlmostEqual(normalized_src["C"], 0.8 / 1.8, places=8)

    def test_graph_uses_paragraph_fallback_for_missing_internal_citation(self) -> None:
        paper_a = Paper("A", Path("."))
        paper_a._section_scores = {}
        paper_a._paragraph_scores = [
            ParagraphScore(
                section_path=["Intro"],
                paragraph_index=1,
                paragraph="Alpha cites [1] and [2].",
                technical_score=0.3,
                citation_score=0.6,
            )
        ]
        paper_a._paragraph_citation_scores = [
            ParagraphCitationScore(
                section_path=["Intro"],
                paragraph_index=1,
                paragraph="Alpha cites [1] and [2].",
                citation="[1]",
                citation_score=0.3,
            )
        ]
        paper_a._has_paragraph_citation_scores_file = True
        paper_a._citation_scores = {"[1]": 0.3}

        paper_b = Paper("B", Path("."))
        paper_b._section_scores = {}
        paper_b._paragraph_scores = []
        paper_b._paragraph_citation_scores = []
        paper_b._has_paragraph_citation_scores_file = False
        paper_b._citation_scores = {}

        graph = CitationGraph()
        graph.add_paper(paper_a)
        graph.add_paper(paper_b)
        graph.add_citation_mappings({"A": {"[1]": "B", "[2]": "B"}, "B": {}})
        graph.build()

        self.assertAlmostEqual(graph.weight("A", "B"), 0.3, places=8)

        audit = graph.internal_citation_audit()
        status_by_key = {item.citation_key: item.status for item in audit if item.source_paper == "A"}
        self.assertEqual(status_by_key["[1]"], "paper_score")
        self.assertEqual(status_by_key["[2]"], "ignored_duplicate_missing")

    def test_graph_uses_paragraph_fallback_when_no_explicit_internal_score_exists(self) -> None:
        paper_a = Paper("A", Path("."))
        paper_a._section_scores = {}
        paper_a._paragraph_scores = [
            ParagraphScore(
                section_path=["Intro"],
                paragraph_index=1,
                paragraph="Alpha cites [2] and [3].",
                technical_score=0.2,
                citation_score=0.8,
            )
        ]
        paper_a._paragraph_citation_scores = []
        paper_a._has_paragraph_citation_scores_file = True
        paper_a._citation_scores = {}

        paper_b = Paper("B", Path("."))
        paper_b._section_scores = {}
        paper_b._paragraph_scores = []
        paper_b._paragraph_citation_scores = []
        paper_b._has_paragraph_citation_scores_file = False
        paper_b._citation_scores = {}

        graph = CitationGraph()
        graph.add_paper(paper_a)
        graph.add_paper(paper_b)
        graph.add_citation_mappings({"A": {"[2]": "B"}, "B": {}})
        graph.build()

        self.assertAlmostEqual(graph.weight("A", "B"), 0.4, places=8)
        audit = graph.internal_citation_audit()
        self.assertEqual(audit[0].status, "paragraph_fallback")
        self.assertAlmostEqual(audit[0].fallback_score, 0.4, places=8)


if __name__ == "__main__":
    unittest.main()
