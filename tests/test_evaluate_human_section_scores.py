import math
import unittest

from evaluate_human_section_scores import (
    aggregate_reference_in_model_top_k_report,
    citation_reference_in_model_top_k_report,
    jensen_shannon_divergence,
    kl_divergence,
    metric_bundle,
)


class EvaluateHumanSectionScoresTests(unittest.TestCase):
    def test_kl_divergence_is_zero_for_identical_distributions(self) -> None:
        self.assertAlmostEqual(kl_divergence([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) or 0.0, 0.0, places=12)

    def test_jensen_shannon_divergence_is_zero_for_identical_distributions(self) -> None:
        self.assertAlmostEqual(
            jensen_shannon_divergence([0.1, 0.9], [0.1, 0.9]) or 0.0,
            0.0,
            places=12,
        )

    def test_jensen_shannon_divergence_is_symmetric_and_finite_with_zeros(self) -> None:
        left = jensen_shannon_divergence([1.0, 0.0], [0.0, 1.0])
        right = jensen_shannon_divergence([0.0, 1.0], [1.0, 0.0])
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertAlmostEqual(left or 0.0, right or 0.0, places=12)
        self.assertTrue(math.isfinite(left or 0.0))

    def test_metric_bundle_includes_requested_distribution_metrics(self) -> None:
        metrics = metric_bundle([0.6, 0.4], [0.5, 0.5])
        self.assertIn("kl_divergence", metrics)
        self.assertIn("jensen_shannon_divergence", metrics)
        self.assertIn("spearman", metrics)
        self.assertIn("kendall_tau_b", metrics)

    def test_human_top_4_in_model_top_10_report_counts_recovered_citations(self) -> None:
        human_items = [
            {"rank": 1, "citation": "A"},
            {"rank": 2, "citation": "B"},
            {"rank": 3, "citation": "C"},
            {"rank": 4, "citation": "D"},
        ]
        citation_json = {
            "A": 1.0,
            "X": 0.95,
            "B": 0.9,
            "Y": 0.85,
            "Z": 0.8,
            "C": 0.75,
            "W": 0.7,
            "V": 0.65,
            "U": 0.6,
            "T": 0.55,
            "D": 0.5,
        }
        report = citation_reference_in_model_top_k_report(
            paper_id="demo",
            human_items=human_items,
            citation_json=citation_json,
            resolver=None,
            reference_k=4,
            model_k=10,
        )
        self.assertEqual(report["metrics"]["overlap_count"], 3)
        self.assertAlmostEqual(report["metrics"]["recall"], 0.75)
        self.assertAlmostEqual(report["metrics"]["hit_any"], 1.0)
        self.assertAlmostEqual(report["metrics"]["hit_all"], 0.0)
        self.assertAlmostEqual(report["metrics"]["mrr_human_rank1_in_model_top_k"], 1.0)

    def test_aggregate_human_top_4_in_model_top_10_report(self) -> None:
        aggregate = aggregate_reference_in_model_top_k_report(
            [
                {
                    "reference_k": 4,
                    "model_k": 10,
                    "metrics": {
                        "overlap_count": 4,
                        "recall": 1.0,
                        "hit_any": 1.0,
                        "hit_all": 1.0,
                        "mrr_human_rank1_in_model_top_k": 1.0,
                    },
                },
                {
                    "reference_k": 4,
                    "model_k": 10,
                    "metrics": {
                        "overlap_count": 2,
                        "recall": 0.5,
                        "hit_any": 1.0,
                        "hit_all": 0.0,
                        "mrr_human_rank1_in_model_top_k": 0.5,
                    },
                },
            ],
            n_bootstrap=200,
            seed=7,
        )
        self.assertEqual(aggregate["reference_k"], 4)
        self.assertEqual(aggregate["model_k"], 10)
        self.assertAlmostEqual(aggregate["mean_overlap_count"] or 0.0, 3.0)
        self.assertAlmostEqual(aggregate["mean_recall"] or 0.0, 0.75)
        self.assertAlmostEqual(aggregate["mean_hit_all"] or 0.0, 0.5)
        self.assertIsNotNone(aggregate["bootstrap_ci_recall"])


if __name__ == "__main__":
    unittest.main()
