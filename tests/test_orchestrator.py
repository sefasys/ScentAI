from __future__ import annotations

import unittest

from scentai.orchestrator import (
    ADVISOR_ANSWER_PROMPT,
    LEGACY_ANSWER_PROMPT,
    ScentAISession,
    ScentAIOrchestrator,
    explicit_brand_context,
    infer_discovery_mode,
    inherit_conversation_plan,
    normalize_plan,
    semantic_search_query,
    validate_answer,
)


PRADA = {
    "perfume_id": 1,
    "name": "Prada L'Homme",
    "brand": "Prada",
    "label": "Prada L'Homme by Prada",
    "gender": "male",
    "rating": 4.3,
    "popularity": 12000,
    "accords_csv": "iris, powdery, clean, woody",
    "notes_csv": "iris, neroli",
    "seasons_csv": "spring, summer, autumn",
    "time_profile_csv": "day",
    "longevity": 3.5,
    "sillage": 2.2,
}

VERSACE = {
    "perfume_id": 2,
    "name": "Versace Pour Homme",
    "brand": "Versace",
    "label": "Versace Pour Homme by Versace",
    "gender": "male",
    "rating": 4.27,
    "popularity": 21000,
    "accords_csv": "citrus, fresh, aromatic",
    "notes_csv": "lemon, neroli",
    "seasons_csv": "spring, summer",
    "time_profile_csv": "day",
    "longevity": 3.1,
    "sillage": 2.0,
}


class FakeVLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def chat(self, model, messages, *, max_tokens, json_mode=False):
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens, "json_mode": json_mode})
        if not self.outputs:
            raise AssertionError("Unexpected model call")
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output, {"elapsed_seconds": 0.01, "usage": {"total_tokens": 10}}


class FakeRetrieval:
    def __init__(self):
        self.search_payloads = []
        self.similar_payloads = []
        self.resolved = {
            "Prada L'Homme": PRADA,
            "Versace Pour Homme": VERSACE,
            "Aventus": {**VERSACE, "perfume_id": 3, "name": "Aventus", "brand": "Creed", "label": "Aventus by Creed"},
        }

    def health(self):
        return {"status": "ok"}

    def search(self, payload):
        self.search_payloads.append(payload)
        excluded = {int(value) for value in payload.get("exclude_ids", [])}
        results = [item for item in (PRADA, VERSACE) if item["perfume_id"] not in excluded]
        return {"route": "semantic", "elapsed_seconds": 0.2, "result_count": len(results), "results": results}

    def resolve(self, hint):
        return self.resolved.get(hint)

    def similar(self, payload):
        self.similar_payloads.append(payload)
        return {
            "route": "community_similarity",
            "elapsed_seconds": 0.02,
            "source": self.resolved["Aventus"],
            "result_count": 2,
            "results": [PRADA, VERSACE],
        }


def orchestrator(outputs):
    return ScentAIOrchestrator(
        FakeVLLM(outputs),
        FakeRetrieval(),
        planner_model="base",
        answer_model="scentai",
    )


class CleanOrchestratorTests(unittest.TestCase):
    def test_generic_brand_guard_rejects_style_collision(self):
        raw = {
            "intent": "recommendation",
            "requested_brand": {"value": "Clean", "evidence": "clean"},
            "excluded_terms": [{"value": "vanilla", "evidence": "vanilla"}],
        }
        plan = normalize_plan(raw, "I need a clean office scent without vanilla")
        self.assertNotIn("requested_brand", plan)
        self.assertEqual(plan["excluded_terms"], ["vanilla"])

    def test_semantic_query_removes_requested_output_count(self):
        query = "I need a clean office scent without vanilla. Recommend exactly 3."
        cleaned = semantic_search_query(query, {"requested_count": 3})
        self.assertEqual(cleaned, "I need a clean office scent without vanilla.")
        self.assertNotIn("3", cleaned)

    def test_planner_semantic_query_separates_vibe_from_required_traits(self):
        raw = {
            "intent": "recommendation",
            "semantic_query": "romantic date night fragrance",
            "required_terms": [
                {"value": "vanilla", "evidence": "Vanilya"},
                {"value": "warm spicy", "evidence": "baharatlı"},
            ],
        }
        query = "Bana date akşamı için bir parfüm öner. Vanilya ve baharatlı olsun."
        plan = normalize_plan(raw, query)
        self.assertEqual(semantic_search_query(query, plan), "romantic date night fragrance")
        self.assertNotIn("vanilla", plan["semantic_query"])

    def test_turkish_singular_request_and_trait_aliases_are_normalized(self):
        raw = {
            "intent": "recommendation",
            "required_terms": [
                {"value": "vanilya", "evidence": "Vanilya"},
                {"value": "baharatlı", "evidence": "baharatlı"},
            ],
        }
        query = "Bana date akşamı için bir parfüm öner. Vanilya ve baharatlı olsun."
        plan = normalize_plan(raw, query)
        self.assertEqual(plan["requested_count"], 1)
        self.assertEqual(plan["required_terms"], ["vanilla", "spicy"])

    def test_explicit_brand_guard_accepts_house_request(self):
        self.assertTrue(explicit_brand_context("Show me fragrances from Versace", "Versace", "Versace"))
        raw = {
            "intent": "recommendation",
            "requested_brand": {"value": "Versace", "evidence": "from Versace"},
        }
        self.assertEqual(normalize_plan(raw, "Show me fragrances from Versace")["requested_brand"], "Versace")

    def test_explicit_popularity_mode_is_recovered_when_planner_omits_field(self):
        plan = normalize_plan(
            {"intent": "recommendation", "confidence": 1.0},
            "Recommend exactly 3 popular fragrances that must have vanilla.",
        )
        self.assertEqual(plan["discovery_mode"], "mainstream")
        self.assertEqual(infer_discovery_mode("Bana az bilinen niş parfümler öner"), "niche")
        self.assertIsNone(infer_discovery_mode("Bana vanilyalı bir parfüm öner"))

    def test_recommendation_runs_planner_search_and_grounded_answer(self):
        pipeline = orchestrator([
            '{"intent":"recommendation","confidence":0.96,"wanted_terms":[{"value":"clean","evidence":"clean"}],"excluded_terms":[{"value":"vanilla","evidence":"vanilla"}],"requested_count":{"value":2,"evidence":"two"}}',
            "1. **Prada L'Homme by Prada**\nA polished iris-led office scent.\n\n2. **Versace Pour Homme by Versace**\nBrighter and more casual for warm days.",
        ])
        result = pipeline.run("Give me two clean office scents without vanilla")
        self.assertEqual(result["route"], "llm_grounded")
        self.assertTrue(result["validation"]["pass"])
        self.assertEqual(result["plan"]["requested_count"], 2)
        self.assertEqual(pipeline.retrieval.search_payloads[0]["exclude_terms"], ["vanilla"])
        self.assertTrue(pipeline.vllm.calls[0]["json_mode"])

    def test_similarity_uses_community_endpoint(self):
        pipeline = orchestrator([
            '{"intent":"similarity","confidence":0.99,"perfumes":[{"value":"Aventus","evidence":"Aventus"}]}',
            "1. Prada L'Homme by Prada\nA cleaner, more powdery direction.\n2. Versace Pour Homme by Versace\nA brighter citrus direction.",
        ])
        result = pipeline.run("Show me perfumes similar to Aventus")
        self.assertEqual(result["route"], "llm_grounded_similarity")
        self.assertEqual(pipeline.retrieval.similar_payloads[0]["hint"], "Aventus")
        self.assertEqual(result["reference"]["label"], "Aventus by Creed")

    def test_invalid_first_answer_is_retried(self):
        pipeline = orchestrator([
            '{"intent":"recommendation","confidence":0.9}',
            "1. Imaginary Perfume by Fictional House\nPerfect for everything.",
            "1. Prada L'Homme by Prada\nA restrained powdery office profile.",
        ])
        result = pipeline.run("Recommend an office fragrance")
        self.assertEqual(result["generation_attempts"], 2)
        self.assertTrue(result["validation"]["pass"])
        self.assertEqual(result["generation_failures"][0]["reasons"], ["unsupported_numbered_perfume", "no_context_perfume_mentioned"])

    def test_exact_lookup_skips_answer_generation(self):
        pipeline = orchestrator([
            '{"intent":"exact_lookup","confidence":1,"perfumes":[{"value":"Prada L Homme","evidence":"Prada L Homme"}],"requested_fields":[{"value":"rating","evidence":"rating"}]}',
        ])
        pipeline.retrieval.resolved["Prada L Homme"] = PRADA
        result = pipeline.run("What is the exact rating of Prada L Homme?")
        self.assertEqual(result["route"], "deterministic_exact_lookup")
        self.assertIn("Rating: 4.3", result["answer"])
        self.assertEqual(len(pipeline.vllm.calls), 1)

    def test_comparison_requires_both_resolved_perfumes(self):
        pipeline = orchestrator([
            '{"intent":"comparison","confidence":1,"perfumes":[{"value":"Prada L Homme","evidence":"Prada L Homme"},{"value":"Versace Pour Homme","evidence":"Versace Pour Homme"}]}',
            "Prada L'Homme by Prada feels powdery and restrained for office wear. Versace Pour Homme by Versace feels brighter and more casual in summer.",
        ])
        pipeline.retrieval.resolved["Prada L Homme"] = PRADA
        result = pipeline.run("Compare Prada L Homme and Versace Pour Homme")
        self.assertEqual(result["route"], "llm_grounded_comparison")
        self.assertTrue(result["validation"]["pass"])
        self.assertEqual(len(result["candidates"]), 2)

    def test_unresolved_comparison_does_not_generate(self):
        pipeline = orchestrator([
            '{"intent":"comparison","confidence":1,"perfumes":[{"value":"Prada L Homme","evidence":"Prada L Homme"},{"value":"Unknown Scent","evidence":"Unknown Scent"}]}',
        ])
        pipeline.retrieval.resolved["Prada L Homme"] = PRADA
        result = pipeline.run("Compare Prada L Homme and Unknown Scent")
        self.assertEqual(result["route"], "comparison_unresolved")
        self.assertEqual(len(pipeline.vllm.calls), 1)

    def test_similarity_falls_back_to_semantic_search_without_graph_edges(self):
        pipeline = orchestrator([
            '{"intent":"similarity","confidence":1,"perfumes":[{"value":"Aventus","evidence":"Aventus"}]}',
            "1. Prada L'Homme by Prada\nA cleaner alternative.",
        ])
        pipeline.retrieval.similar = lambda payload: {
            "route": "community_similarity",
            "source": pipeline.retrieval.resolved["Aventus"],
            "results": [],
        }
        result = pipeline.run("Find something similar to Aventus")
        self.assertEqual(result["retrieval"]["route"], "semantic_similarity_fallback")
        self.assertIn("aventus", pipeline.retrieval.search_payloads[0]["exclude_terms"])

    def test_unsupported_intent_returns_without_retrieval(self):
        pipeline = orchestrator(['{"intent":"unsupported_price","confidence":0.98}'])
        result = pipeline.run("What is its current store price?")
        self.assertEqual(result["route"], "unsupported_price")
        self.assertEqual(result["generation_attempts"], 0)
        self.assertFalse(pipeline.retrieval.search_payloads)

    def test_two_invalid_planner_outputs_default_to_semantic_recommendation(self):
        pipeline = orchestrator([
            "not json",
            "still not json",
            "1. Prada L'Homme by Prada\nA restrained office option.",
        ])
        result = pipeline.run("Recommend an office fragrance")
        self.assertEqual(result["plan"]["intent"], "recommendation")
        self.assertTrue(result["timings"]["planner"]["defaulted_to_semantic_recommendation"])
        self.assertTrue(result["validation"]["pass"])

    def test_generation_errors_use_grounded_fallback(self):
        pipeline = orchestrator([
            '{"intent":"recommendation","confidence":0.9}',
            RuntimeError("temporary model failure"),
            RuntimeError("temporary model failure"),
        ])
        result = pipeline.run("Recommend an office fragrance")
        self.assertEqual(result["route"], "validated_template_fallback")
        self.assertEqual(result["generation_attempts"], 2)
        self.assertTrue(result["validation"]["pass"])

    def test_answer_prompt_and_model_can_be_overridden_for_ab_diagnosis(self):
        pipeline = orchestrator([
            '{"intent":"recommendation","confidence":0.9}',
            "1. Prada L'Homme by Prada\nA restrained office option.",
        ])
        result = pipeline.run(
            "Recommend an office fragrance",
            answer_prompt_override=LEGACY_ANSWER_PROMPT,
            answer_model_override="base-answer",
        )
        answer_call = pipeline.vllm.calls[-1]
        self.assertEqual(answer_call["model"], "base-answer")
        self.assertIn(LEGACY_ANSWER_PROMPT, answer_call["messages"][0]["content"])
        self.assertEqual(result["answer_prompt_mode"], "legacy")
        self.assertEqual(result["answer_model"], "base-answer")

    def test_advisor_prompt_is_the_default_answer_mode(self):
        pipeline = orchestrator([
            '{"intent":"recommendation","confidence":0.9}',
            "1. Prada L'Homme by Prada\nA restrained office option.",
        ])
        result = pipeline.run("Recommend an office fragrance")
        self.assertIn(ADVISOR_ANSWER_PROMPT, pipeline.vllm.calls[-1]["messages"][0]["content"])
        self.assertEqual(result["answer_prompt_mode"], "advisor")

    def test_repair_model_is_used_only_after_primary_validation_failure(self):
        pipeline = ScentAIOrchestrator(
            FakeVLLM([
                '{"intent":"recommendation","confidence":0.9}',
                "1. Imaginary Perfume by Fictional House\nInvented.",
                "1. Prada L'Homme by Prada\nA restrained office option.",
            ]),
            FakeRetrieval(),
            planner_model="base",
            answer_model="advisor-base",
            repair_answer_model="grounded-lora",
        )
        result = pipeline.run("Recommend an office fragrance")
        self.assertTrue(result["validation"]["pass"])
        self.assertEqual(
            [call["model"] for call in pipeline.vllm.calls],
            ["base", "advisor-base", "grounded-lora"],
        )
        self.assertEqual(
            [metric["model"] for metric in result["timings"]["generation"]],
            ["advisor-base", "grounded-lora"],
        )

    def test_retrieval_error_returns_explicit_failure(self):
        pipeline = orchestrator(['{"intent":"recommendation","confidence":0.9}'])
        pipeline.retrieval.search = lambda payload: (_ for _ in ()).throw(RuntimeError("service down"))
        result = pipeline.run("Recommend an office fragrance")
        self.assertEqual(result["route"], "retrieval_error")
        self.assertFalse(result["validation"]["pass"])

    def test_validator_detects_candidate_with_excluded_trait(self):
        plan = {"intent": "recommendation", "excluded_terms": ["vanilla"]}
        vanilla = {**PRADA, "accords_csv": "iris, vanilla"}
        report = validate_answer("1. Prada L'Homme by Prada", plan, [vanilla])
        self.assertFalse(report["pass"])
        self.assertIn("strict_filter_violation", report["reasons"])

    def test_numbered_parser_preserves_colon_inside_catalog_name(self):
        kyoto = {
            **PRADA,
            "perfume_id": 9,
            "name": "Series 3 Incense: Kyoto",
            "brand": "Comme des Garcons",
            "label": "Series 3 Incense: Kyoto by Comme des Garcons",
        }
        plan = {"intent": "recommendation", "requested_count": 1, "excluded_terms": []}
        report = validate_answer("1. Series 3 Incense: Kyoto by Comme des Garcons\nA smoky incense profile.", plan, [kyoto])
        self.assertTrue(report["pass"], report)

    def test_numbered_parser_prefers_longest_flanker_name_over_shared_prefix(self):
        pour_homme = {
            **VERSACE,
            "name": "Versace Pour Homme",
            "label": "Versace Pour Homme by Versace",
        }
        dylan_blue = {
            **VERSACE,
            "perfume_id": 3,
            "name": "Versace Pour Homme Dylan Blue",
            "label": "Versace Pour Homme Dylan Blue by Versace",
        }
        answer = (
            "1. Versace Pour Homme by Versace\nA bright profile.\n"
            "2. Versace Pour Homme Dylan Blue by Versace\nA denser profile."
        )
        plan = {"intent": "recommendation", "requested_count": 2, "excluded_terms": [], "required_terms": []}
        report = validate_answer(answer, plan, [pour_homme, dylan_blue])
        self.assertTrue(report["pass"], report)
        self.assertEqual(
            report["numbered_recommendations"],
            ["Versace Pour Homme by Versace", "Versace Pour Homme Dylan Blue by Versace"],
        )

    def test_turkish_mechanical_template_is_rejected(self):
        plan = {"intent": "recommendation", "requested_count": 1, "excluded_terms": [], "required_terms": []}
        answer = "1. Prada L'Homme by Prada\nBu parfümü buraya dahil etme sebebim iris akorları."
        report = validate_answer(answer, plan, [PRADA])
        self.assertFalse(report["pass"])
        self.assertIn("mechanical_template_language", report["reasons"])

    def test_followup_plan_inherits_constraints_and_excludes_previous_recommendations(self):
        current = {
            "intent": "recommendation",
            "confidence": 0.98,
            "wanted_terms": [],
            "required_terms": [],
            "excluded_terms": [],
            "requested_fields": [],
            "conversation_action": "more_options",
        }
        context = {
            "previous_plan": {
                "intent": "recommendation",
                "semantic_query": "warm vanilla evening fragrance",
                "wanted_terms": ["vanilla"],
                "required_terms": [],
                "excluded_terms": ["smoky"],
                "requested_count": 3,
            },
            "previous_recommendation_ids": [1, 2, 2],
        }
        merged = inherit_conversation_plan(current, context)
        self.assertEqual(merged["semantic_query"], "warm vanilla evening fragrance")
        self.assertEqual(merged["wanted_terms"], ["vanilla"])
        self.assertEqual(merged["excluded_terms"], ["smoky"])
        self.assertEqual(merged["exclude_candidate_ids"], [1, 2])

    def test_followup_negative_constraint_overrides_previous_positive_trait(self):
        current = {
            "intent": "recommendation",
            "wanted_terms": ["fresh"],
            "required_terms": [],
            "excluded_terms": ["vanilla"],
            "requested_fields": [],
            "conversation_action": "refine_previous",
        }
        context = {
            "previous_plan": {
                "intent": "recommendation",
                "semantic_query": "warm evening fragrance",
                "wanted_terms": ["vanilla"],
                "required_terms": ["vanilla"],
                "excluded_terms": [],
            },
            "previous_recommendation_ids": [1],
        }
        merged = inherit_conversation_plan(current, context)
        self.assertEqual(merged["wanted_terms"], ["fresh"])
        self.assertEqual(merged["required_terms"], [])
        self.assertEqual(merged["excluded_terms"], ["vanilla"])

    def test_stateful_session_requests_different_options_under_previous_constraints(self):
        pipeline = orchestrator([
            '{"intent":"recommendation","confidence":0.99,"semantic_query":"warm vanilla evening fragrance","wanted_terms":[{"value":"vanilla","evidence":"vanilla"}],"requested_count":{"value":1,"evidence":"one"}}',
            "1. Prada L'Homme by Prada\nA polished option.",
            '{"intent":"recommendation","confidence":0.99,"conversation_action":"more_options"}',
            "1. Versace Pour Homme by Versace\nA brighter alternative.",
        ])
        session = ScentAISession(pipeline)
        first = session.run("I tend to enjoy vanilla. Recommend one fragrance")
        second = session.run("I want another option")
        self.assertTrue(first["validation"]["pass"])
        self.assertTrue(second["validation"]["pass"])
        self.assertEqual(second["plan"]["wanted_terms"], ["vanilla"])
        self.assertEqual(pipeline.retrieval.search_payloads[1]["exclude_ids"], [1])
        self.assertEqual(second["candidates"][0]["perfume_id"], 2)


if __name__ == "__main__":
    unittest.main()
