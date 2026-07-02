import sys
import types
import unittest


ollama = types.ModuleType("ollama")


class DummyClient:
    pass


ollama.Client = DummyClient
sys.modules.setdefault("ollama", ollama)

from baselines.models import (  # noqa: E402
    OpenAIFullPaperBaseline,
    aggregate_citation_scores_from_paragraphs,
    apply_scaled_section_citation_scores,
    annotate_section_citation_scores,
    build_raw_section_tree_from_flat_node_scores,
    build_leaf_paragraph_inventory,
    canonicalize_inventory_citation,
    collect_citation_inventory,
    enforce_section_score_conservation,
    extract_text_fallback_paragraph_payloads,
    normalize_citation_scores,
    normalize_openai_paragraph_citation_scores,
    normalize_openai_paragraph_scores,
    normalize_section_scores_to_schema,
    parse_contribution_tree_text_response,
    render_authoritative_hierarchy,
    scale_section_citation_scores,
    total_section_citation_score,
)


class OpenAIFullPaperBaselineHelpersTests(unittest.TestCase):
    def test_normalize_section_scores_to_schema_fills_missing_siblings_and_preserves_parent_budget(self) -> None:
        schema = {
            "Introduction": {},
            "Method": {
                "Setup": {},
                "Main Algorithm": {},
            },
            "Experiments": {},
        }
        raw_scores = {
            "Introduction": {"total_score": 5},
            "Method": {
                "total_score": 25,
                "subsections": {
                    "Main Algorithm": {"total_score": 9},
                },
            },
        }

        normalized = normalize_section_scores_to_schema(schema, raw_scores, total_score=1.0)

        top_level_total = sum(node["total_score"] for node in normalized.values())
        self.assertAlmostEqual(top_level_total, 1.0, places=8)
        self.assertGreater(normalized["Method"]["total_score"], normalized["Introduction"]["total_score"])
        self.assertGreater(normalized["Method"]["total_score"], normalized["Experiments"]["total_score"])

        method_children = normalized["Method"]["subsections"]
        child_total = sum(node["total_score"] for node in method_children.values())
        self.assertAlmostEqual(child_total, normalized["Method"]["total_score"], places=8)
        self.assertGreater(
            method_children["Main Algorithm"]["total_score"],
            method_children["Setup"]["total_score"],
        )

    def test_normalize_citation_scores_accepts_nested_payloads_and_backfills_missing_entries(self) -> None:
        citation_inventory = {
            "(Smith et al., 2020)": {"mention_count": 3, "sections": ["Method"]},
            "(Lee and Chen, 2021)": {"mention_count": 1, "sections": ["Experiments"]},
            "(Garcia, 2019)": {"mention_count": 2, "sections": ["Introduction"]},
        }
        raw_scores = {
            "(Smith et al., 2020)": {"citation_score": 4},
            "(Garcia, 2019)": {"score": 1},
        }

        normalized = normalize_citation_scores(citation_inventory, raw_scores)

        self.assertEqual(set(normalized.keys()), set(citation_inventory.keys()))
        total = sum(item["citation_score"] for item in normalized.values())
        self.assertAlmostEqual(total, 1.0, places=8)
        self.assertGreater(
            normalized["(Smith et al., 2020)"]["citation_score"],
            normalized["(Garcia, 2019)"]["citation_score"],
        )
        self.assertEqual(normalized["(Lee and Chen, 2021)"]["citation_score"], 0.0)

    def test_normalize_citation_scores_merges_numeric_spacing_variants(self) -> None:
        citation_inventory = {
            "[23]": {"mention_count": 3, "sections": ["Method"]},
            "[17]": {"mention_count": 2, "sections": ["Method"]},
            "[21]": {"mention_count": 1, "sections": ["Experiments"]},
        }
        raw_scores = [
            {"citation": "[ 23]", "citation_score": 0.15},
            {"citation": "[23]", "citation_score": 0.10},
            {"citation": "[ 17 ]", "citation_score": 0.20},
            {"citation": "[21]", "citation_score": 0.05},
        ]

        normalized = normalize_citation_scores(citation_inventory, raw_scores)

        self.assertEqual(canonicalize_inventory_citation("[ 23 ]"), "[23]")
        self.assertAlmostEqual(sum(item["citation_score"] for item in normalized.values()), 1.0, places=8)
        self.assertGreater(normalized["[23]"]["citation_score"], normalized["[21]"]["citation_score"])
        self.assertGreater(normalized["[17]"]["citation_score"], normalized["[21]"]["citation_score"])

    def test_collect_citation_inventory_splits_grouped_numeric_blocks(self) -> None:
        citation_tree = {
            "Method": {
                "[10,27,35, 37]": ["ctx1", "ctx2"],
                "[23]": ["ctx3"],
            }
        }
        inventory = collect_citation_inventory(citation_tree)
        self.assertEqual(set(inventory.keys()), {"[10]", "[27]", "[35]", "[37]", "[23]"})
        self.assertEqual(inventory["[10]"]["mention_count"], 2)
        self.assertEqual(inventory["[27]"]["mention_count"], 2)
        self.assertEqual(inventory["[35]"]["mention_count"], 2)
        self.assertEqual(inventory["[37]"]["mention_count"], 2)
        self.assertEqual(inventory["[23]"]["mention_count"], 1)

    def test_normalize_citation_scores_splits_grouped_numeric_response_keys(self) -> None:
        citation_inventory = {
            "[10]": {"mention_count": 1, "sections": ["Method"]},
            "[27]": {"mention_count": 1, "sections": ["Method"]},
            "[35]": {"mention_count": 1, "sections": ["Method"]},
            "[37]": {"mention_count": 1, "sections": ["Method"]},
        }
        raw_scores = [{"citation": "[10,27,35, 37]", "citation_score": 1.0}]
        normalized = normalize_citation_scores(citation_inventory, raw_scores)
        for citation in citation_inventory:
            self.assertAlmostEqual(normalized[citation]["citation_score"], 0.25, places=8)

    def test_full_paper_prompt_uses_authoritative_hierarchy_text(self) -> None:
        baseline = OpenAIFullPaperBaseline(pdf_path="dummy.pdf")
        prompt = baseline._build_prompt(
            paper_id="demo",
            full_paper_text="A short paper.",
            section_schema={"Intro": {}, "Method": {}},
            content_dict={"Intro": "alpha", "Method": "beta"},
            citation_inventory={
                "(Smith et al., 2020)": {"mention_count": 2, "sections": ["Method"]},
                "(Lee et al., 2021)": {"mention_count": 1, "sections": ["Intro"]},
            },
            paragraph_inventory=[
                {
                    "section_path": ["Method"],
                    "paragraphs": [
                        {
                            "paragraph_id": "demo::Method::p1",
                            "section_path": ["Method"],
                            "paragraph_index": 1,
                            "text": "Paragraph text.",
                            "has_citations": True,
                            "citation_focus_text": "Citation sentence.",
                            "citations": [
                                {"citation": "(Smith et al., 2020)", "mention_count": 2, "context": "ctx"}
                            ],
                        }
                    ],
                }
            ],
        )
        self.assertIn("Assign contribution scores to the provided hierarchy using the paper text.", prompt)
        self.assertIn("The hierarchy provided below is authoritative.", prompt)
        self.assertIn("Your task is to score the provided hierarchy, not reconstruct the paper structure.", prompt)
        self.assertIn("Score every node in the hierarchy.", prompt)
        self.assertIn("Do not add, remove, rename, merge, split, or reorder nodes.", prompt)
        self.assertIn("Do not invent sections, subsections, paragraphs, appendices, or aggregate nodes.", prompt)
        self.assertIn("Paragraphs are the lowest document-level nodes.", prompt)
        self.assertIn(
            "A paragraph containing citations is not a leaf node. Citations discussed within that paragraph are treated as child citation nodes.",
            prompt,
        )
        self.assertIn("Contribution scores must be assigned recursively through the hierarchy.", prompt)
        self.assertIn("Do not skip hierarchy levels.", prompt)
        self.assertIn("Do not distribute a parent's score directly to grandchildren or deeper descendants.", prompt)
        self.assertIn("Exact normalization will be performed after output.", prompt)
        self.assertIn("Citation contribution is computed bottom-up from citation-containing paragraphs.", prompt)
        self.assertIn("[1, 10]", prompt)
        self.assertIn("The sum of all citation contribution scores should approximately equal the root paper Citation Score.", prompt)
        self.assertIn("Output Format", prompt)
        self.assertIn("Node Title: Total Score | Citation Score", prompt)
        self.assertIn("* Total Score is the node's contribution score.", prompt)
        self.assertIn("Citation Contributions", prompt)
        self.assertIn("Citation Identifier: Score", prompt)
        self.assertIn("Paper Text\n\nA short paper.\n\nHierarchy To Score\n\nPaper", prompt)
        self.assertIn("  - Intro", prompt)
        self.assertIn("  - Method", prompt)
        self.assertIn("    - Paragraph 1", prompt)
        self.assertIn("      - (Smith et al., 2020)", prompt)
        self.assertNotIn("Construct a Contribution Tree for the following paper.", prompt)

    def test_render_authoritative_hierarchy_includes_paragraphs_and_citations(self) -> None:
        hierarchy = render_authoritative_hierarchy(
            section_schema={
                "Intro": {},
                "Method": {
                    "Setup": {},
                },
            },
            paragraph_inventory=[
                {
                    "section_path": ["Intro"],
                    "paragraphs": [
                        {
                            "paragraph_index": 1,
                            "citations": [],
                        }
                    ],
                },
                {
                    "section_path": ["Method", "Setup"],
                    "paragraphs": [
                        {
                            "paragraph_index": 1,
                            "citations": [
                                {"citation": "[12]"},
                                {"citation": "[27]"},
                            ],
                        }
                    ],
                },
            ],
        )
        self.assertEqual(
            hierarchy,
            "Paper\n"
            "  - Intro\n"
            "    - Paragraph 1\n"
            "  - Method\n"
            "    - Setup\n"
            "      - Paragraph 1\n"
            "        - [12]\n"
            "        - [27]",
        )

    def test_full_paper_system_prompt_uses_requested_text(self) -> None:
        self.assertEqual(
            OpenAIFullPaperBaseline.SYSTEM_PROMPT,
            "You are a scientific contribution-scoring assistant.\n\n"
            "Follow the contribution-scoring framework exactly as described by the user.\n\n"
            "Perform careful hierarchical reasoning over the document structure.\n\n"
            "Use citation identifiers exactly as they appear in the paper.\n\n"
            "Return only the requested scores and format. Do not provide explanations, reasoning traces, commentary, or summaries unless explicitly requested.",
        )

    def test_leaf_paragraph_inventory_splits_grouped_numeric_blocks_per_paragraph(self) -> None:
        inventory = build_leaf_paragraph_inventory(
            content_tree={"Method": "Alpha [10,27,35,37] beta."},
            citation_tree={"Method": {"[10,27,35,37]": ["Alpha [10,27,35,37] beta."]}},
            paper_id="demo",
        )
        citations = inventory[0]["paragraphs"][0]["citations"]
        self.assertEqual(
            {entry["citation"] for entry in citations},
            {"[10]", "[27]", "[35]", "[37]"},
        )

    def test_openai_paragraph_and_citation_outputs_are_normalized_to_section_budget(self) -> None:
        section_scores = normalize_section_scores_to_schema({"Method": {}}, {"Method": {"total_score": 1.0}}, total_score=1.0)
        paragraph_inventory = [
            {
                "section_path": ["Method"],
                "paragraphs": [
                    {
                        "paragraph_id": "demo::Method::p1",
                        "section_path": ["Method"],
                        "paragraph_index": 1,
                        "text": "Para 1",
                        "has_citations": True,
                        "citation_focus_text": "ctx",
                        "citations": [
                            {"citation": "[10]", "mention_count": 2, "context": "ctx"},
                            {"citation": "[27]", "mention_count": 1, "context": "ctx"},
                        ],
                    },
                    {
                        "paragraph_id": "demo::Method::p2",
                        "section_path": ["Method"],
                        "paragraph_index": 2,
                        "text": "Para 2",
                        "has_citations": False,
                        "citation_focus_text": "",
                        "citations": [],
                    },
                ],
            }
        ]
        paragraph_scores = normalize_openai_paragraph_scores(
            paragraph_inventory=paragraph_inventory,
            raw_paragraph_scores=[
                {"paragraph_id": "demo::Method::p1", "technical_score": 2.0, "citation_score": 1.0},
                {"paragraph_id": "demo::Method::p2", "technical_score": 3.0, "citation_score": 0.0},
            ],
            normalized_section_scores=section_scores,
        )
        self.assertAlmostEqual(
            sum(item["technical_score"] + item["citation_score"] for item in paragraph_scores),
            1.0,
            places=8,
        )
        paragraph_citation_scores = normalize_openai_paragraph_citation_scores(
            paragraph_inventory=paragraph_inventory,
            paragraph_scores=paragraph_scores,
            raw_paragraph_citation_scores=[
                {"paragraph_id": "demo::Method::p1", "citation": "[10,27]", "citation_score": 1.0}
            ],
        )
        flat = aggregate_citation_scores_from_paragraphs(paragraph_citation_scores)
        self.assertAlmostEqual(
            sum(item["citation_score"] for item in flat.values()),
            next(item["citation_score"] for item in paragraph_scores if item["paragraph_id"] == "demo::Method::p1"),
            places=8,
        )
        annotate_section_citation_scores(section_scores, paragraph_scores)
        self.assertAlmostEqual(section_scores["Method"]["citation_score"], sum(item["citation_score"] for item in flat.values()), places=8)

    def test_parse_contribution_tree_text_response_extracts_nodes_and_citations(self) -> None:
        response = """
Paper: 1.0 | Citation: 0.4
Introduction: 0.2 | Citation: 0.05
Method: 0.8 | Citation: 0.35
Main Algorithm: 0.6 | Citation: 0.25

Citation Contributions

[10,27]: 0.2
[35]: 0.2
"""
        node_scores, citation_scores, root_citation = parse_contribution_tree_text_response(response)
        self.assertAlmostEqual(root_citation, 0.4, places=8)
        self.assertAlmostEqual(node_scores["Method"]["total_score"], 0.8, places=8)
        self.assertAlmostEqual(node_scores["Main Algorithm"]["citation_score"], 0.25, places=8)
        self.assertAlmostEqual(citation_scores["[10,27]"]["citation_score"], 0.2, places=8)
        self.assertAlmostEqual(citation_scores["[35]"]["citation_score"], 0.2, places=8)

    def test_parse_contribution_tree_text_response_extracts_markdown_nodes_and_table_rows(self) -> None:
        response = """
**Contribution Tree**

- **Paper (root) – 1.000 | 0.315**
  - **Introduction – 0.150 | 0.045**
  - **Data Model & Problem Definition – 0.120 | 0.036**

**Citation Contributions**

| Citation Identifier | Score |
|---------------------|-------|
| [21] | 0.030 |
| [5]  | 0.015 |
"""
        node_scores, citation_scores, root_citation = parse_contribution_tree_text_response(response)
        self.assertAlmostEqual(root_citation, 0.315, places=8)
        self.assertAlmostEqual(node_scores["Introduction"]["total_score"], 0.150, places=8)
        self.assertAlmostEqual(node_scores["Data Model & Problem Definition"]["citation_score"], 0.036, places=8)
        self.assertAlmostEqual(citation_scores["[21]"]["citation_score"], 0.030, places=8)
        self.assertAlmostEqual(citation_scores["[5]"]["citation_score"], 0.015, places=8)

    def test_extract_text_fallback_paragraph_payloads_recovers_paragraphs_and_citations(self) -> None:
        section_schema = {
            "Introduction": {},
            "Method": {
                "Setup": {},
            },
        }
        paragraph_inventory = [
            {
                "section_path": ["Introduction"],
                "paragraphs": [
                    {
                        "paragraph_id": "demo::Introduction::p1",
                        "paragraph_index": 1,
                        "text": "Intro paragraph 1",
                        "has_citations": True,
                        "citations": [{"citation": "[10]"}, {"citation": "[27]"}],
                    },
                    {
                        "paragraph_id": "demo::Introduction::p2",
                        "paragraph_index": 2,
                        "text": "Intro paragraph 2",
                        "has_citations": False,
                        "citations": [],
                    },
                ],
            },
            {
                "section_path": ["Method", "Setup"],
                "paragraphs": [
                    {
                        "paragraph_id": "demo::Method > Setup::p1",
                        "paragraph_index": 1,
                        "text": "Setup paragraph 1",
                        "has_citations": True,
                        "citations": [{"citation": "[35]"}],
                    }
                ],
            },
        ]
        response = """
Paper: 1.000 | 0.200
Introduction: 0.400 | 0.100
- Paragraph 1: 0.200 | 0.100
  - [10]: 0.060
  - [27]: 0.040
- Paragraph 2: 0.200 | 0.000
Method: 0.600 | 0.100
- Setup: 0.600 | 0.100
  - Paragraph 1: 0.600 | 0.100
    - [35]: 0.100

Citation Contributions
[10]: 0.060
[27]: 0.040
[35]: 0.100
"""
        paragraph_scores, paragraph_citation_scores = extract_text_fallback_paragraph_payloads(
            response,
            section_schema=section_schema,
            paragraph_inventory=paragraph_inventory,
        )

        self.assertEqual(len(paragraph_scores), 3)
        self.assertEqual(len(paragraph_citation_scores), 3)
        self.assertEqual(paragraph_scores[0]["paragraph_id"], "demo::Introduction::p1")
        self.assertAlmostEqual(paragraph_scores[0]["technical_score"], 0.1, places=8)
        self.assertAlmostEqual(paragraph_scores[0]["citation_score"], 0.1, places=8)
        self.assertEqual(paragraph_scores[2]["section_path"], ["Method", "Setup"])
        self.assertEqual(
            [item["citation"] for item in paragraph_citation_scores],
            ["[10]", "[27]", "[35]"],
        )

    def test_title_lookup_article_stripping_matches_minor_heading_variants(self) -> None:
        schema = {"B Proofs of theorems": {}}
        raw_tree = build_raw_section_tree_from_flat_node_scores(
            schema,
            {"B Proofs of the theorems": {"total_score": 0.4, "citation_score": 0.0}},
        )
        self.assertAlmostEqual(raw_tree["B Proofs of theorems"]["total_score"], 0.4, places=8)

    def test_plain_text_section_and_citation_scores_align_to_schema(self) -> None:
        flat_node_scores = {
            "Introduction": {"total_score": 0.2, "citation_score": 0.05},
            "Method": {"total_score": 0.8, "citation_score": 0.35},
            "Setup": {"total_score": 0.2, "citation_score": 0.10},
            "Main Algorithm": {"total_score": 0.6, "citation_score": 0.25},
        }
        schema = {
            "Introduction": {},
            "Method": {
                "Setup": {},
                "Main Algorithm": {},
            },
        }
        raw_tree = build_raw_section_tree_from_flat_node_scores(schema, flat_node_scores)
        normalized = normalize_section_scores_to_schema(schema, raw_tree, total_score=1.0)
        apply_scaled_section_citation_scores(normalized, raw_tree)

        self.assertAlmostEqual(normalized["Introduction"]["total_score"], 0.2, places=8)
        self.assertAlmostEqual(normalized["Method"]["total_score"], 0.8, places=8)
        self.assertAlmostEqual(normalized["Method"]["citation_score"], 0.35, places=8)
        self.assertAlmostEqual(
            normalized["Method"]["subsections"]["Setup"]["citation_score"]
            + normalized["Method"]["subsections"]["Main Algorithm"]["citation_score"],
            normalized["Method"]["citation_score"],
            places=8,
        )

    def test_plain_text_section_normalization_does_not_invent_uniform_subsections_when_disabled(self) -> None:
        schema = {
            "Method": {
                "Setup": {},
                "Main Algorithm": {},
            },
        }
        raw_tree = build_raw_section_tree_from_flat_node_scores(
            schema,
            {"Method": {"total_score": 1.0, "citation_score": 0.4}},
        )
        normalized = normalize_section_scores_to_schema(
            schema,
            raw_tree,
            total_score=1.0,
            fill_missing_uniform=False,
        )
        self.assertAlmostEqual(normalized["Method"]["total_score"], 1.0, places=8)
        self.assertEqual(normalized["Method"]["subsections"]["Setup"]["total_score"], 0.0)
        self.assertEqual(normalized["Method"]["subsections"]["Main Algorithm"]["total_score"], 0.0)

    def test_enforce_section_score_conservation_rescales_children_to_parent(self) -> None:
        section_tree = {
            "Introduction": {"total_score": 0.2, "citation_score": 0.0, "subsections": {}},
            "Method": {
                "total_score": 0.8,
                "citation_score": 0.0,
                "subsections": {
                    "Setup": {"total_score": 0.1, "citation_score": 0.0, "subsections": {}},
                    "Main Algorithm": {"total_score": 0.1, "citation_score": 0.0, "subsections": {}},
                },
            },
        }

        total = enforce_section_score_conservation(section_tree, expected_total=1.0)

        self.assertAlmostEqual(total, 1.0, places=8)
        self.assertAlmostEqual(
            section_tree["Method"]["subsections"]["Setup"]["total_score"]
            + section_tree["Method"]["subsections"]["Main Algorithm"]["total_score"],
            section_tree["Method"]["total_score"],
            places=8,
        )
        self.assertAlmostEqual(section_tree["Method"]["subsections"]["Setup"]["total_score"], 0.4, places=8)
        self.assertAlmostEqual(section_tree["Method"]["subsections"]["Main Algorithm"]["total_score"], 0.4, places=8)

    def test_scale_section_citation_scores_matches_root_target(self) -> None:
        section_tree = {
            "Introduction": {"total_score": 0.4, "citation_score": 0.2, "subsections": {}},
            "Method": {"total_score": 0.6, "citation_score": 0.1, "subsections": {}},
        }
        new_total = scale_section_citation_scores(section_tree, 0.24)
        self.assertAlmostEqual(new_total, 0.24, places=8)
        self.assertAlmostEqual(section_tree["Introduction"]["citation_score"], 0.16, places=8)
        self.assertAlmostEqual(section_tree["Method"]["citation_score"], 0.08, places=8)

    def test_total_section_citation_score_uses_bottom_up_child_totals(self) -> None:
        section_tree = {
            "Introduction": {"total_score": 0.2, "citation_score": 0.05, "subsections": {}},
            "Method": {
                "total_score": 0.8,
                "citation_score": 0.35,
                "subsections": {
                    "Setup": {"total_score": 0.2, "citation_score": 0.10, "subsections": {}},
                    "Main Algorithm": {"total_score": 0.6, "citation_score": 0.25, "subsections": {}},
                },
            },
        }
        self.assertAlmostEqual(total_section_citation_score(section_tree), 0.40, places=8)

    def test_plain_text_citation_scores_scale_to_requested_total_without_fallback(self) -> None:
        citation_inventory = {
            "[10]": {"mention_count": 1, "sections": ["Method"]},
            "[27]": {"mention_count": 1, "sections": ["Method"]},
            "[35]": {"mention_count": 1, "sections": ["Method"]},
            "[99]": {"mention_count": 1, "sections": ["Method"]},
        }
        normalized = normalize_citation_scores(
            citation_inventory,
            {"[10,27]": {"citation_score": 0.2}, "[35]": {"citation_score": 0.2}},
            total_score=0.4,
            fallback_to_mentions=False,
        )
        self.assertAlmostEqual(sum(item["citation_score"] for item in normalized.values()), 0.4, places=8)
        self.assertAlmostEqual(normalized["[10]"]["citation_score"], 0.1, places=8)
        self.assertAlmostEqual(normalized["[27]"]["citation_score"], 0.1, places=8)
        self.assertAlmostEqual(normalized["[35]"]["citation_score"], 0.2, places=8)
        self.assertEqual(normalized["[99]"]["citation_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
