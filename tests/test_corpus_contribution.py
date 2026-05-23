import unittest

from citation_graph_framework import CitationGraph, CorpusContributionAnalyzer


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


if __name__ == "__main__":
    unittest.main()
