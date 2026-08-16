import unittest

from importance_score import repair_partial_citation_scores


class CitationSplitRepairTests(unittest.TestCase):
    def test_repair_partial_citation_scores_fills_missing_ids(self) -> None:
        repaired = repair_partial_citation_scores(
            parsed_scores={"1": 60.0, "2": 30.0},
            citation_ids=["1", "2", "3"],
            citation_id_to_name={"1": "[12]", "2": "[18]", "3": "[27]"},
            total_score=0.12,
            citation_base_weights={"[12]": 1.0, "[18]": 1.0, "[27]": 2.0},
        )

        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertEqual(set(repaired.keys()), {"[12]", "[18]", "[27]"})
        self.assertAlmostEqual(sum(repaired.values()), 0.12)
        self.assertGreater(repaired["[27]"], 0.0)

    def test_repair_partial_citation_scores_rejects_too_sparse_output(self) -> None:
        repaired = repair_partial_citation_scores(
            parsed_scores={"1": 100.0},
            citation_ids=["1", "2", "3", "4", "5"],
            citation_id_to_name={
                "1": "[1]",
                "2": "[2]",
                "3": "[3]",
                "4": "[4]",
                "5": "[5]",
            },
            total_score=0.2,
            citation_base_weights={"[1]": 1.0, "[2]": 1.0, "[3]": 1.0, "[4]": 1.0, "[5]": 1.0},
        )

        self.assertIsNone(repaired)


if __name__ == "__main__":
    unittest.main()
