import unittest
from collections import defaultdict
from pathlib import Path

from citation_resolver import (
    CitationResolver,
    ReferenceEntry,
    _extract_references_block,
    _extract_pdf_title,
    _looks_like_low_quality_title,
    _normalize_title_key,
    _parse_entry,
    _stable_external_id_from_title,
)


class CitationResolverMatchingTests(unittest.TestCase):
    def _resolver_with_titles(self, mapping: dict[str, str]) -> CitationResolver:
        resolver = CitationResolver()
        resolver._corpus_titles = dict(mapping)
        resolver._title_to_paper_ids = defaultdict(list)
        for paper_id, title in mapping.items():
            resolver._title_to_paper_ids[_normalize_title_key(title)].append(paper_id)
        return resolver

    def test_canonical_target_matches_clean_title_to_corpus_title_with_author_suffix(self) -> None:
        resolver = self._resolver_with_titles(
            {
                "AttentionAllYouNeed": "Attention Is All You Need Ashish Vaswani Google Brain",
            }
        )
        entry = ReferenceEntry(
            title="Attention is all you need.",
            authors=["Ashish Vaswani"],
            first_author_last="vaswani",
            year=2017,
            numeric_key="[1]",
            raw_text="Ashish Vaswani et al. 2017. Attention is all you need.",
            canonical_id="vaswani_2017",
        )

        self.assertEqual(resolver._canonical_target(entry), "AttentionAllYouNeed")

    def test_canonical_target_uses_raw_text_when_parsed_title_is_broken(self) -> None:
        resolver = self._resolver_with_titles(
            {
                "AttentionAllYouNeed": "Attention Is All You Need",
            }
        )
        entry = ReferenceEntry(
            title="2017.",
            authors=["Ashish Vaswani"],
            first_author_last="vaswani",
            year=2017,
            numeric_key="[1]",
            raw_text=(
                "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, "
                "Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. "
                "Attention is all you need. In Advances in Neural Information Processing Systems."
            ),
            canonical_id="vaswani_2017",
        )

        self.assertEqual(resolver._canonical_target(entry), "AttentionAllYouNeed")

    def test_canonical_target_prefers_exact_match_over_longer_containment_match(self) -> None:
        resolver = self._resolver_with_titles(
            {
                "LSTM": "Long Short-Term Memory",
                "LSM-Memory": "Long Short-Term Memory Networks for Machine Reading",
            }
        )
        entry = ReferenceEntry(
            title="Long short-term memory.",
            authors=["Sepp Hochreiter"],
            first_author_last="hochreiter",
            year=1997,
            numeric_key="[1]",
            raw_text="Sepp Hochreiter and Jurgen Schmidhuber. 1997. Long short-term memory.",
            canonical_id="hochreiter_1997",
        )

        self.assertEqual(resolver._canonical_target(entry), "LSTM")

    def test_canonical_target_does_not_collapse_short_title_onto_longer_different_title(self) -> None:
        resolver = self._resolver_with_titles(
            {
                "LSM-Memory": "Long Short-Term Memory Networks for Machine Reading",
            }
        )
        entry = ReferenceEntry(
            title="Long short-term memory.",
            authors=["Sepp Hochreiter"],
            first_author_last="hochreiter",
            year=1997,
            numeric_key="[1]",
            raw_text="Sepp Hochreiter and Jurgen Schmidhuber. 1997. Long short-term memory.",
            canonical_id="hochreiter_1997",
        )

        self.assertEqual(
            resolver._canonical_target(entry),
            "ref_longshorttermmemory_1997",
        )

    def test_canonical_target_does_not_use_raw_text_spillover_for_good_titles(self) -> None:
        resolver = self._resolver_with_titles(
            {
                "Gen-Sequences-RNN": "Generating Sequences With Recurrent Neural Networks",
            }
        )
        entry = ReferenceEntry(
            title="A Theoretically Grounded Application of Dropout in Recurrent Neural Networks.",
            authors=["Yarin Gal"],
            first_author_last="gal",
            year=2015,
            numeric_key="(Gal, 2015)",
            raw_text=(
                "Yarin Gal. 2015. A Theoretically Grounded Application of Dropout in Recurrent "
                "Neural Networks. arXiv preprint arXiv:1512.05287. Alex Graves. 2013. "
                "Generating sequences with recurrent neural networks."
            ),
            canonical_id="gal_2015",
        )

        self.assertEqual(
            resolver._canonical_target(entry),
            "ref_atheoreticallygroundedapplicationofdropoutinrecurrentneuralnetworks_2015",
        )

    def test_canonical_target_uses_stable_title_based_id_for_external_work(self) -> None:
        resolver = self._resolver_with_titles({})
        entry = ReferenceEntry(
            title="Attention is all you need.",
            authors=["Ashish Vaswani"],
            first_author_last="vaswani",
            year=2017,
            numeric_key="[1]",
            raw_text="Ashish Vaswani et al. 2017. Attention is all you need.",
            canonical_id="vaswani_2017",
            stable_external_id=_stable_external_id_from_title("Attention is all you need.", 2017),
        )

        self.assertEqual(
            resolver._canonical_target(entry),
            "ref_attentionisallyouneed_2017",
        )

    def test_extract_pdf_title_does_not_absorb_author_line(self) -> None:
        pdf_path = Path("papers/Case 1/LSM-Memory.pdf")
        if not pdf_path.exists():
            self.skipTest("LSM-Memory.pdf fixture is not present")

        self.assertEqual(
            _extract_pdf_title(pdf_path),
            "Long Short-Term Memory-Networks for Machine Reading",
        )

    def test_parse_entry_keeps_title_before_year_when_reference_has_tail_spillover(self) -> None:
        raw = (
            "Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang, and Jingbo Zhu. "
            "Fast and accurate shift-reduce constituent parsing. "
            "In Proceedings of the 51st Annual Meeting of the ACL (Volume 1: Long Papers) , "
            "pages 434–443. ACL, August 2013. "
            "12 Attention Visualizations Input-Input Layer5 It is in this spirit that a majority "
            "of American governments have passed new laws since 2009 making the registration or "
            "voting process more difficult ."
        )

        entry = _parse_entry(raw, numeric_key="[40]")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(
            entry.title,
            "Fast and accurate shift-reduce constituent parsing.",
        )
        self.assertEqual(entry.canonical_id, "zhu_2013")

    def test_low_quality_title_heuristic_flags_year_only_and_noise(self) -> None:
        self.assertTrue(_looks_like_low_quality_title("2017."))
        self.assertTrue(_looks_like_low_quality_title("a"))
        self.assertTrue(_looks_like_low_quality_title("12 Attention Visualizations Input-Input Layer5"))
        self.assertFalse(_looks_like_low_quality_title("Attention is all you need."))

    def test_extract_references_block_starts_after_references_heading(self) -> None:
        text = (
            "Introduction\n"
            "Some body text.\n\n"
            "References\n"
            "[1] First reference.\n"
            "[2] Second reference.\n"
        )

        self.assertEqual(
            _extract_references_block(text),
            "[1] First reference.\n[2] Second reference.",
        )

    def test_extract_references_block_accepts_numbered_bibliography_heading(self) -> None:
        text = (
            "5 Conclusion\n"
            "Closing text.\n\n"
            "6. Bibliography\n"
            "Smith, J. 2020. Example title.\n"
        )

        self.assertEqual(
            _extract_references_block(text),
            "Smith, J. 2020. Example title.",
        )


if __name__ == "__main__":
    unittest.main()
