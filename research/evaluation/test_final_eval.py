from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scentai import orchestrator
from research.evaluation import build_final_eval_v1
from research.evaluation import final_eval_runtime


def fake_result(query: str, conversation_context=None):
    second_turn = bool(conversation_context)
    perfume_id = 2 if second_turn else 1
    label = "Second Choice by Test House" if second_turn else "First Choice by Test House"
    action = "more_options" if second_turn else "new_request"
    candidate = {
        "perfume_id": perfume_id,
        "name": label.split(" by ")[0],
        "brand": "Test House",
        "label": label,
        "accords_csv": "citrus, aromatic",
        "notes_csv": "bergamot",
        "gender": "unisex",
    }
    return {
        "query": query,
        "route": "llm_grounded",
        "response_language": "en",
        "answer": f"1. {label}\nThis is a clean fragrance for comfortable daily wear.",
        "plan": {
            "intent": "recommendation",
            "requested_count": 1,
            "conversation_action": action,
            "required_terms": [],
            "excluded_terms": [],
        },
        "candidates": [candidate],
        "validation": {
            "pass": True,
            "reasons": [],
            "numbered_recommendations": [label],
            "mentioned_candidates": [label],
            "performance_calibration_violations": [],
        },
        "generation_attempts": 1,
        "timings": {"total_seconds": 1.0},
    }


class FakePipeline:
    def __init__(self):
        self.calls = 0

    def run(self, query, *, conversation_context=None):
        self.calls += 1
        return fake_result(query, conversation_context)


class FinalEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        final_eval_runtime.response_language_matches = orchestrator.response_language_matches
        final_eval_runtime.candidate_has_term = orchestrator.candidate_has_term
        final_eval_runtime.ScentAISession = orchestrator.ScentAISession

    @unittest.skipUnless(
        Path("scentai_catalog.sqlite3").is_file(),
        "external catalog artifact is required for label validation",
    )
    def test_fixed_set_schema_and_catalog_labels(self):
        cases = build_final_eval_v1.all_cases()
        manifest = build_final_eval_v1.validate_cases(cases, Path("scentai_catalog.sqlite3"))
        self.assertEqual(manifest["case_count"], 120)
        self.assertEqual(manifest["category_counts"], build_final_eval_v1.CATEGORY_TARGETS)

    def test_scorer_rejects_excluded_candidate(self):
        case = build_final_eval_v1.recommendation_cases()[0]
        result = fake_result(case["query"])
        result["plan"]["requested_count"] = 3
        result["plan"]["excluded_terms"] = ["vanilla"]
        candidate = result["candidates"][0]
        candidate["accords_csv"] = "vanilla, citrus"
        result["validation"]["numbered_recommendations"] = [candidate["label"]] * 3
        score = final_eval_runtime.score_final_case(case, result)
        self.assertIn("excluded_terms_absent", score["failures"])

    def test_trait_aliases_accept_specific_spicy_taxonomy(self):
        self.assertTrue(
            final_eval_runtime.plan_contains_traits(["spicy"], ["warm spicy"])
        )

    def test_trait_aliases_cover_filter_families(self):
        self.assertTrue(final_eval_runtime.plan_contains_traits(["leather"], ["leathery"]))
        self.assertTrue(final_eval_runtime.plan_contains_traits(["musk"], ["musky"]))
        self.assertTrue(final_eval_runtime.plan_contains_traits(["oud"], ["agarwood"]))

    def test_scorer_rejects_unrequested_hard_exclusion(self):
        case = {
            "expected": {
                "response_language": "en",
                "intent_in": ["recommendation"],
                "requested_count": 1,
                "excluded_terms": ["tobacco"],
            }
        }
        result = fake_result("Exclude tobacco and show another choice.")
        result["plan"]["excluded_terms"] = ["tobacco", "show another choice"]
        score = final_eval_runtime.score_final_case(case, result)
        self.assertIn("no_unexpected_excluded_terms", score["failures"])
        self.assertEqual(score["details"]["unexpected_excluded_terms"], ["show another choice"])

    def test_follow_up_action_variants_are_semantically_compatible(self):
        case = {
            "expected": {
                "response_language": "en",
                "intent_in": ["recommendation"],
                "requested_count": 1,
                "conversation_action": "refine_previous",
            }
        }
        result = fake_result("Give me one new unisex option.", conversation_context={"prior": True})
        score = final_eval_runtime.score_final_case(case, result)
        self.assertTrue(score["checks"]["conversation_action"])

    def test_runner_resume_restores_conversation_without_regeneration(self):
        cases = [
            {
                "id": "resume_001",
                "version": "final_eval_v1",
                "category": "conversation",
                "language": "en",
                "query": "Recommend one fragrance.",
                "tags": [],
                "session_id": "resume_session",
                "turn": 1,
                "expected": {
                    "response_language": "en",
                    "intent_in": ["recommendation"],
                    "requested_count": 1,
                    "conversation_action": "new_request",
                },
            },
            {
                "id": "resume_002",
                "version": "final_eval_v1",
                "category": "conversation",
                "language": "en",
                "query": "Give me one different option.",
                "tags": [],
                "session_id": "resume_session",
                "turn": 2,
                "expected": {
                    "response_language": "en",
                    "intent_in": ["recommendation"],
                    "requested_count": 1,
                    "conversation_action": "more_options",
                    "no_repeat_previous": True,
                },
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "outputs_path": root / "outputs.jsonl",
                "summary_path": root / "summary.json",
                "human_review_path": root / "human.csv",
                "metadata_path": root / "metadata.json",
            }
            first_pipeline = FakePipeline()
            summary = final_eval_runtime.run_final_evaluation(
                first_pipeline,
                cases,
                resume=True,
                **paths,
            )
            self.assertEqual(first_pipeline.calls, 2)
            self.assertEqual(summary["pass_count"], 2)
            output_rows = final_eval_runtime.read_jsonl(paths["outputs_path"])
            self.assertEqual(len(output_rows), 2)
            self.assertTrue(output_rows[1]["score"]["checks"]["conversation_no_repeat"])

            resumed_pipeline = FakePipeline()
            resumed_summary = final_eval_runtime.run_final_evaluation(
                resumed_pipeline,
                cases,
                resume=True,
                **paths,
            )
            self.assertEqual(resumed_pipeline.calls, 0)
            self.assertEqual(resumed_summary["pass_count"], 2)
            self.assertEqual(len(final_eval_runtime.read_jsonl(paths["outputs_path"])), 2)


if __name__ == "__main__":
    unittest.main()
