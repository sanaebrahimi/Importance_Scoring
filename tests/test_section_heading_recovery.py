import sys
import types
import unittest
from pathlib import Path


ollama = types.ModuleType("ollama")


class DummyClient:
    pass


ollama.Client = DummyClient
sys.modules.setdefault("ollama", ollama)

from importance_score import (  # noqa: E402
    SECTION_LEAD_IN_NODE,
    detect_dominant_citation_style,
    extract_citations_by_section,
    find_heading_line_offsets_global,
    heading_matches_expected_title,
    load_sections_from_file,
    read_pdf_text,
)


ROOT = Path(__file__).resolve().parents[1]
SECTIONS_FILE = str(ROOT / "papers_section_titles.txt")
PAPERS_DIR = ROOT / "papers"


def extract_content(pdf_name: str, variable_name: str):
    sections = load_sections_from_file(SECTIONS_FILE, variable_name)
    text = read_pdf_text(str(PAPERS_DIR / pdf_name))
    _, content = extract_citations_by_section(text, sections)
    return content


def extract_citation_tree(pdf_name: str, variable_name: str):
    sections = load_sections_from_file(SECTIONS_FILE, variable_name)
    text = read_pdf_text(str(PAPERS_DIR / pdf_name))
    citations, _ = extract_citations_by_section(text, sections)
    return citations


def flatten_citation_keys(citation_tree):
    keys = []
    stack = [citation_tree]
    while stack:
        current = stack.pop()
        for key, value in current.items():
            if isinstance(value, dict):
                stack.append(value)
            else:
                keys.append(key)
    return keys


class SectionHeadingRecoveryTests(unittest.TestCase):
    def test_heading_matching_handles_unicode_and_question_marks(self) -> None:
        self.assertTrue(heading_matches_expected_title("d=1", "3.1𝑑=1"))
        self.assertTrue(
            heading_matches_expected_title(
                "Why Shapley?",
                "3.1 Why Shapley?",
            )
        )
        self.assertTrue(
            heading_matches_expected_title(
                "Hierarchical -net Navigation Graph (HENN)",
                "Hierarchical 𝜀-net Navigation Graph (HENN)",
            )
        )
        self.assertTrue(
            heading_matches_expected_title(
                "E More on Experiments",
                "More on Experiments",
            )
        )

    def test_find_heading_offsets_recovers_reported_headings(self) -> None:
        henn_text = read_pdf_text(str(PAPERS_DIR / "HENN_ICLR_26-1.pdf"))
        self.assertNotEqual(
            find_heading_line_offsets_global(
                henn_text,
                "Hierarchical -net Navigation Graph (HENN)",
            ),
            (-1, -1),
        )
        self.assertNotEqual(
            find_heading_line_offsets_global(henn_text, "E More on Experiments"),
            (-1, -1),
        )

        ranked_text = read_pdf_text(str(PAPERS_DIR / "ranked-ret.pdf"))
        self.assertNotEqual(
            find_heading_line_offsets_global(
                ranked_text,
                "EpsRange: -sampling in Higher Dimensions",
            ),
            (-1, -1),
        )

        shapley_text = read_pdf_text(str(PAPERS_DIR / "shapley.pdf"))
        self.assertNotEqual(
            find_heading_line_offsets_global(shapley_text, "Why Shapley?"),
            (-1, -1),
        )

    def test_extracts_fixed_reported_subsections(self) -> None:
        fcm = extract_content("FCM_SIGMOD_CRV.pdf", "FCM_SIGMOD_CRV_SECTIONS")
        self.assertIn("d=1", fcm["Fair-Count-Min using Group-Aware Semi-Uniform Hashing"])
        self.assertIn("d=1", fcm["Price of Fairness Analysis"])

        henn = extract_content("HENN_ICLR_26-1.pdf", "HENN_ICLR_26_1_SECTIONS")
        self.assertIn("Hierarchical -net Navigation Graph (HENN)", henn)
        self.assertIn("E More on Experiments", henn)

        ranked = extract_content("ranked-ret.pdf", "RANKED_RET_SECTIONS")
        self.assertIn("EpsRange: -sampling in Higher Dimensions", ranked["-sampling Approach"])

        shapley = extract_content("shapley.pdf", "SHAPLEY_SECTIONS")
        self.assertIn("Why Shapley?", shapley["SHAPLEY VALUE BASED SOLUTION"])
        self.assertIn(SECTION_LEAD_IN_NODE, shapley["PRELIMINARIES"])
        self.assertIn(SECTION_LEAD_IN_NODE, shapley["PRELIMINARIES"]["Problem definition"])

        fairrq = extract_content("fairRQ.pdf", "FAIRRQ_SECTIONS")
        self.assertIn("Prepossessing", fairrq["SINGLE-PREDICATE RANGE QUERIES"])

    def test_efficient_mm_algo_uses_author_year_citations_and_repairs_wrapped_names(self) -> None:
        text = read_pdf_text(str(PAPERS_DIR / "Efficient-MM-Algo.pdf"))
        self.assertEqual(detect_dominant_citation_style(text), "author_year")

        citations = extract_citation_tree("Efficient-MM-Algo.pdf", "EFFICIENT_MM_ALGO_SECTIONS")
        citation_keys = flatten_citation_keys(citations)

        self.assertNotIn("(1)", citation_keys)
        self.assertIn(
            "(Dai et al., 2021; Wu et al., 2016; McK-instry et al., 2018; Zhu et al., 2016; Courbariaux et al., 2015; Hubara et al., 2016)",
            citation_keys,
        )

    def test_narrative_author_year_citations_are_not_downgraded_to_year_tokens(self) -> None:
        algorithmic_text = read_pdf_text(str(PAPERS_DIR / "AlgorithmicCollectiveAction.pdf"))
        self.assertEqual(detect_dominant_citation_style(algorithmic_text), "author_year")

        algorithmic_citations = extract_citation_tree(
            "AlgorithmicCollectiveAction.pdf",
            "ALGORITHMICCOLLECTIVEACTION_SECTIONS",
        )
        algorithmic_keys = flatten_citation_keys(algorithmic_citations)
        self.assertIn("(Vincent et al., 2021)", algorithmic_keys)
        self.assertIn("(Wood et al., 2019)", algorithmic_keys)
        self.assertNotIn("(2019)", algorithmic_keys)
        self.assertNotIn("(2021)", algorithmic_keys)
        self.assertNotIn("Vincent et al. (2021)", algorithmic_keys)
        self.assertNotIn("Wood et al. (2019)", algorithmic_keys)

        infini_text = read_pdf_text(str(PAPERS_DIR / "INFINI-GRAM-MINI.pdf"))
        self.assertEqual(detect_dominant_citation_style(infini_text), "author_year")

        infini_citations = extract_citation_tree("INFINI-GRAM-MINI.pdf", "INFINI_GRAM_MINI_SECTIONS")
        infini_keys = flatten_citation_keys(infini_citations)
        self.assertIn("(Elazar et al., 2024)", infini_keys)
        self.assertIn("(Gog et al., 2014)", infini_keys)
        self.assertNotIn("(2024)", infini_keys)
        self.assertNotIn("(2014)", infini_keys)
        self.assertNotIn("Elazar et al. (2024)", infini_keys)
        self.assertNotIn("Gog et al. (2014)", infini_keys)
        self.assertNotIn("Liu et al. (2024)", infini_keys)


if __name__ == "__main__":
    unittest.main()
