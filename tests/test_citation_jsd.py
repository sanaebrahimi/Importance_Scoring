from __future__ import annotations

import unittest

from evaluate_citation_jsd import (
    aggregate_scores_by_target,
    jensen_shannon_divergence,
    normalize_citation_only_vector,
)


class StubResolver:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def target_id_for(self, citation_key: str, paper_id: str) -> str | None:
        return self.mapping.get(citation_key)

    def resolve(self, citation_key: str, paper_id: str = "") -> None:
        return None


class CitationJsdTests(unittest.TestCase):
    def test_normalize_citation_only_vector_rescales_by_raw_sum(self) -> None:
        vector, assigned_mass = normalize_citation_only_vector([0.2, 0.3])
        self.assertAlmostEqual(assigned_mass, 0.5, places=8)
        self.assertEqual(len(vector), 2)
        self.assertAlmostEqual(vector[0], 0.4, places=8)
        self.assertAlmostEqual(vector[1], 0.6, places=8)
        self.assertAlmostEqual(sum(vector), 1.0, places=8)

    def test_jsd_is_zero_for_identical_vectors(self) -> None:
        value = jensen_shannon_divergence([0.1, 0.2, 0.7], [0.1, 0.2, 0.7])
        self.assertAlmostEqual(value, 0.0, places=12)

    def test_aggregate_scores_by_target_merges_aliases(self) -> None:
        resolver = StubResolver(
            {
                "[1]": "smith_2020",
                "(Smith et al., 2020)": "smith_2020",
            }
        )
        grouped = aggregate_scores_by_target(
            paper_id="demo",
            score_map={
                "[1]": 0.2,
                "(Smith et al., 2020)": 0.3,
                "[2]": 0.1,
            },
            resolver=resolver,
        )
        self.assertAlmostEqual(grouped["smith_2020"]["score"], 0.5, places=8)
        self.assertIn("2", next(key for key in grouped if key != "smith_2020"))

    def test_jsd_reflects_sparse_vs_dense_difference(self) -> None:
        dense, _ = normalize_citation_only_vector([0.6, 0.4])
        sparse, _ = normalize_citation_only_vector([0.05, 0.0])
        value = jensen_shannon_divergence(dense, sparse)
        self.assertGreater(value, 0.0)
        self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
