import unittest

from citation_graph_framework import CitationGraph, CommunityDetector, ParagraphScore


class _DummyPaper:
    def __init__(self, paper_id: str, paragraphs):
        self.paper_id = paper_id
        self._paragraph_scores = paragraphs

    @property
    def paragraph_scores(self):
        return self._paragraph_scores


class CommunityBaselineTests(unittest.TestCase):
    def _graph(self) -> CitationGraph:
        graph = CitationGraph()
        graph.papers = {
            "A": _DummyPaper(
                "A",
                [
                    ParagraphScore(["Intro"], 0, "See [1] and [1].", 0.0, 0.0),
                    ParagraphScore(["Method"], 1, "Also compare with [2].", 0.0, 0.0),
                ],
            ),
            "B": _DummyPaper(
                "B",
                [
                    ParagraphScore(["Intro"], 0, "Following [3].", 0.0, 0.0),
                ],
            ),
            "C": _DummyPaper("C", []),
        }
        graph._citation_map = {
            "A": {
                "[1]": "B",
                "[2]": "C",
            },
            "B": {
                "[3]": "A",
            },
            "C": {},
        }
        return graph

    def test_mention_count_baseline_sums_counts_across_directions(self) -> None:
        detector = CommunityDetector(self._graph())
        graph = detector._baseline_nx_graph("mention_count", restrict_to_corpus=True)

        self.assertTrue(graph.has_edge("A", "B"))
        self.assertTrue(graph.has_edge("A", "C"))
        self.assertEqual(graph["A"]["B"]["weight"], 3.0)
        self.assertEqual(graph["A"]["C"]["weight"], 1.0)

    def test_binary_baseline_collapses_to_unit_edges(self) -> None:
        detector = CommunityDetector(self._graph())
        graph = detector._baseline_nx_graph("binary", restrict_to_corpus=True)

        self.assertEqual(graph["A"]["B"]["weight"], 1.0)
        self.assertEqual(graph["A"]["C"]["weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
