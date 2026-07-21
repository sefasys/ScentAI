from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from research.runtime.catalog import RuntimeCatalog
from research.runtime.intent_router import render_comparison
from research.runtime.model_pipeline import (
    ScentAIModelPipeline,
    brand_plan_value,
    find_generic_catalog_phrases,
    find_misspelled_metric_terms,
    validate_runtime_answer,
    validated_plan_values,
)
from research.runtime.grounding_checker import score_case_result
from research.runtime.prompts import SYSTEM_PROMPT
from research.runtime.query_analyzer import analyze_query, entity_matches
from research.runtime.rag import (
    PerfumeCandidate,
    brand_dedup,
    confidence_adjusted_rating,
    explicit_metric_sort,
    has_accidental_brand_collision,
    reference_resolution_score,
    score_candidate,
    score_multi_similarity_candidate,
    score_similarity_candidate,
    violates_negative,
)
from research.runtime.scentai_pipeline import build_template_recommendation
from research.runtime.vllm_backend import (
    PLANNER_JSON_SCHEMA,
    VLLMHTTPMessageGenerator,
    answer_token_budget,
    is_comparison_messages,
    is_planner_messages,
    validate_adapter_config,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt is True
        assert tokenize is False
        return "<bos>" + "|".join(message["content"] for message in messages) + "<model>"

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": list(range(max(len(text.split()), 1)))}


class FakeHTTPResponse:
    ok = True
    status_code = 200
    text = ""

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class FakeHTTPSession:
    def __init__(self):
        self.requests = []

    def post(self, url, *, json, timeout):
        self.requests.append({"url": url, "json": json, "timeout": timeout})
        output = '{"intent":"recommendation","confidence":1.0}' if json["model"] == "base" else "Answer"
        return FakeHTTPResponse(
            {
                "choices": [{"text": output, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 7},
            }
        )


def candidate(
    perfume_id: int,
    name: str,
    brand: str,
    accords: str,
    *,
    notes: str = "",
    seasons: str = "",
    time_profile: str = "",
    gender: str = "unisex",
) -> PerfumeCandidate:
    metadata = {
        "perfume_id": perfume_id,
        "name": name,
        "brand": brand,
        "gender": gender,
        "accords_csv": accords,
        "notes_csv": notes,
        "seasons_csv": seasons,
        "time_profile_csv": time_profile,
        "rating": 4.1,
        "popularity": 1000,
    }
    return PerfumeCandidate(perfume_id, name, brand, "", metadata, 0.2, 0.8, 0.8, ("semantic",))


class FakeRetriever:
    def __init__(self, candidates):
        self.candidates = candidates

    def retrieve(self, query, *, top_k, fetch_k, analysis=None):
        analysis = analysis or analyze_query(query)
        kept = [item for item in self.candidates if not violates_negative(item.metadata, analysis)]
        return kept[:top_k], analysis

    def candidates_by_ids(self, perfume_ids):
        wanted = set(perfume_ids)
        return [item for item in self.candidates if item.perfume_id in wanted]


class FakeCatalog:
    def __init__(self, candidates):
        self.candidates = candidates

    def extract_mentions(self, query):
        names = [item.name for item in self.candidates if item.name.lower() in query.lower()]
        return tuple(names)

    def resolve(self, hint):
        match = next((item for item in self.candidates if item.name.lower() == str(hint).lower()), None)
        if match is None:
            return None
        return {"perfume_id": match.perfume_id, "name": match.name, "brand": match.brand}

    def compare(self, hints):
        rows = []
        missing = []
        for hint in hints:
            match = next((item for item in self.candidates if item.name.lower() == hint.lower()), None)
            if match is None:
                missing.append(hint)
            else:
                rows.append({"perfume_id": match.perfume_id, "name": match.name, "brand": match.brand})
        return rows, missing


CATALOG_PATH = Path(__file__).resolve().parents[1] / "scentai_catalog.sqlite3"


@unittest.skipUnless(
    CATALOG_PATH.is_file(),
    "external catalog artifact is required for runtime integration tests",
)
class RuntimePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime_catalog = RuntimeCatalog(CATALOG_PATH)

    @classmethod
    def tearDownClass(cls):
        cls.runtime_catalog.connection.close()

    def test_shortened_product_family_resolves_to_clear_canonical_member(self):
        shortened = self.runtime_catalog.resolve("Club de Nuit by Armaf")
        bare = self.runtime_catalog.resolve("Club de Nuit")
        explicit = self.runtime_catalog.resolve("Club de Nuit Bling by Armaf")
        self.assertEqual(shortened["name"], "Club de Nuit Intense Man")
        self.assertEqual(shortened["brand"], "Armaf")
        self.assertEqual(bare["name"], "Club de Nuit Intense Man")
        self.assertEqual(explicit["name"], "Club de Nuit Bling")

    def test_comparison_fallback_is_explanatory_and_uses_canonical_resolution(self):
        analysis = replace(
            analyze_query("compare them"),
            comparison_perfumes=("Club de Nuit by Armaf", "Team Five by Adidas"),
        )
        answer, rows = render_comparison(self.runtime_catalog, analysis)
        self.assertEqual(rows[0]["name"], "Club de Nuit Intense Man")
        self.assertIn("Character:", answer)
        self.assertIn("Wear:", answer)
        self.assertIn("Performance:", answer)
        self.assertIn("How to choose:", answer)
        self.assertNotIn("Database comparison:", answer)

    def test_dynamic_brand_exclusion_handles_typo(self):
        analysis = analyze_query("A spicy date scent. Exclude all kinds of Kurkdijan.")
        self.assertEqual(analysis.excluded_entities, ("kurkdijan",))
        self.assertTrue(entity_matches("kurkdijan", "Maison Francis Kurkdjian"))

    def test_turkish_brand_request_is_detected(self):
        analysis = analyze_query("Versace'den erkek parfümü öner")
        self.assertEqual(analysis.requested_brand, "versace")

    def test_dynamic_reference_perfume_is_detected(self):
        english = analyze_query("Recommend something similar to Bleu de Chanel but less smoky")
        turkish = analyze_query("Aventus'a benzeyen ama daha az smoky bir parfüm")
        ordinary = analyze_query("I would like a fresh perfume")
        self.assertEqual(english.reference_perfume, "bleu de chanel")
        self.assertEqual(turkish.reference_perfume, "aventus")
        self.assertEqual(english.reference_relation, "similar")
        self.assertIsNone(ordinary.reference_perfume)

    def test_dupe_requests_are_cross_brand_alternatives(self):
        analysis = analyze_query("A dupe for Dior Sauvage without ambroxan")
        self.assertEqual(analysis.reference_perfume, "dior sauvage")
        self.assertEqual(analysis.reference_relation, "alternative")

    def test_reference_resolution_prefers_exact_name(self):
        exact = candidate(1, "Aventus", "Creed", "fruity, woody")
        flanker = candidate(2, "Aventus Cologne", "Creed", "citrus, woody")
        self.assertGreater(reference_resolution_score("Aventus", exact), reference_resolution_score("Aventus", flanker))

    def test_structural_similarity_beats_unrelated_candidate(self):
        reference = candidate(
            1, "Reference", "Brand", "citrus, fruity, woody, fresh", notes="pineapple, bergamot, musk", seasons="summer", time_profile="day", gender="male"
        )
        close = candidate(
            2, "Close", "Other", "citrus, fruity, woody, aromatic", notes="pineapple, bergamot", seasons="summer", time_profile="day", gender="male"
        )
        distant = candidate(
            3, "Distant", "Other", "vanilla, sweet, amber", notes="caramel, tonka bean", seasons="winter", time_profile="night", gender="female"
        )
        analysis = analyze_query("something like Reference")
        self.assertGreater(
            score_similarity_candidate(close, reference, analysis).final_score,
            score_similarity_candidate(distant, reference, analysis).final_score,
        )

    def test_unknown_exclusion_can_filter_a_note(self):
        analysis = analyze_query("A dupe for Dior Sauvage without ambroxan")
        unsafe = candidate(1, "Unsafe", "Brand", "fresh spicy", notes="ambroxan, bergamot")
        self.assertTrue(violates_negative(unsafe.metadata, analysis))

    def test_operational_intents_are_parsed(self):
        comparison = analyze_query("Compare Aventus and Bleu de Chanel")
        counted = analyze_query("Recommend exactly two fresh perfumes")
        ranked = analyze_query("highest-rated Versace perfumes after 2020")
        contradiction = analyze_query("sweet vanilla perfume without sweet or vanilla")
        self.assertEqual(comparison.comparison_perfumes, ("aventus", "bleu de chanel"))
        self.assertEqual(counted.requested_count, 2)
        self.assertEqual(ranked.sort_by, "rating")
        self.assertEqual(ranked.year_min, 2021)
        self.assertEqual(contradiction.contradictions, ("sweet", "vanilla"))

    def test_multi_reference_and_collection_intents_are_parsed(self):
        bridge = analyze_query("Something between Aventus and Hacivat.")
        collection = analyze_query(
            "My collection: Aventus, Bleu de Chanel, Prada L'Homme. What is missing from my collection?"
        )
        self.assertEqual(bridge.reference_perfumes, ("aventus", "hacivat"))
        self.assertEqual(collection.owned_perfumes, ("aventus", "bleu de chanel", "prada l'homme"))
        self.assertTrue(collection.collection_gap_request)

    def test_multi_reference_scoring_rewards_a_balanced_bridge(self):
        first = candidate(1, "First", "Brand A", "citrus, fruity, woody")
        second = candidate(2, "Second", "Brand B", "woody, green, aromatic")
        balanced = candidate(3, "Balanced", "Brand C", "citrus, woody, green, aromatic")
        one_sided = candidate(4, "One Sided", "Brand D", "citrus, fruity, woody")
        analysis = analyze_query("Something between First and Second")
        self.assertGreater(
            score_multi_similarity_candidate(balanced, [first, second], analysis).final_score,
            score_multi_similarity_candidate(one_sided, [first, second], analysis).final_score,
        )

    def test_owned_perfume_is_filtered(self):
        analysis = analyze_query("I own Aventus. Recommend a fresh perfume")
        owned = candidate(1, "Aventus", "Creed", "fruity, woody")
        other = candidate(2, "Other", "Brand", "fresh, citrus")
        self.assertTrue(violates_negative(owned.metadata, analysis))
        self.assertFalse(violates_negative(other.metadata, analysis))

    def test_rating_sort_shrinks_tiny_vote_five_star_records(self):
        tiny = candidate(1, "Tiny", "Brand A", "woody")
        tiny.metadata.update(rating=5.0, popularity=1)
        established = candidate(2, "Established", "Brand B", "woody")
        established.metadata.update(rating=4.6, popularity=1000)
        ranked = explicit_metric_sort([tiny, established], "rating")
        self.assertEqual(ranked[0].name, "Established")
        self.assertLess(confidence_adjusted_rating(5.0, 1), confidence_adjusted_rating(4.6, 1000))

    def test_duplicate_product_labels_are_removed(self):
        first = candidate(1, "Acqua di Gio", "Giorgio Armani", "fresh, aquatic")
        duplicate = candidate(2, "Acqua di Gio", "Giorgio Armani", "fresh, aquatic")
        self.assertEqual(len(brand_dedup([first, duplicate], max_per_brand=5)), 1)

    def test_model_planner_handles_free_form_comparison_and_model_writes_answer(self):
        aventus = candidate(1, "Aventus", "Creed", "fruity, woody, smoky")
        hacivat = candidate(2, "Hacivat", "Nishane", "green, aromatic, woody")
        outputs = iter([
            '{"intent":"comparison","perfumes":["Aventus","Hacivat"]}',
            (
                "Aventus by Creed lists fruity, woody, and smoky accords, while Hacivat by Nishane "
                "lists green, aromatic, and woody accords. Choose according to which recorded profile "
                "better fits your intended use."
            ),
        ])

        def generator(messages, max_new_tokens):
            return next(outputs)

        items = [aventus, hacivat]
        pipeline = ScentAIModelPipeline(
            FakeRetriever(items),
            generator,
            catalog=FakeCatalog(items),
        )
        result = pipeline.run("Aventus mu Hacivat mı bana daha uygun?")
        self.assertEqual(result.route, "llm_grounded_comparison")
        self.assertEqual(result.analysis["comparison_perfumes"], ["Aventus", "Hacivat"])
        self.assertEqual(result.analysis["debug"]["comparison_detected_by"], "model_planner")
        self.assertTrue(result.validation["pass"])

    def test_model_planner_routes_a_natural_single_perfume_profile(self):
        aventus = candidate(
            1,
            "Aventus",
            "Creed",
            "musky, leather, citrus, fruity, smoky, woody",
            notes="pineapple, bergamot, birch",
            seasons="spring, summer, autumn",
            time_profile="day, night",
            gender="male",
        )
        aventus.metadata.update(longevity=3.37, sillage=2.32)
        outputs = iter([
            '{"intent":"perfume_profile","perfumes":[{"value":"Aventus","evidence":"Aventus"}]}',
            (
                "Aventus by Creed reads as a fruity-citrus profile with a darker smoky and woody edge, "
                "grounded in its listed pineapple, bergamot, and birch notes. Its Spring, Summer, and Autumn "
                "coverage plus Day and Night use makes it look highly versatile within that recorded range. "
                "The 3.37/5 longevity suggests solid staying power, while 2.32/4 sillage points to a moderate presence."
            ),
        ])

        def generator(messages, max_new_tokens):
            return next(outputs)

        pipeline = ScentAIModelPipeline(
            FakeRetriever([aventus]),
            generator,
            catalog=FakeCatalog([aventus]),
        )
        result = pipeline.run("Aventus nasıl bir parfüm; karakteri ve kullanım alanları nasıl?")
        self.assertEqual(result.route, "llm_grounded_perfume_profile")
        self.assertEqual(result.analysis["target_perfumes"], ["Aventus"])
        self.assertIn("highly versatile", result.answer)
        self.assertTrue(result.validation["pass"])

    def test_two_liked_perfumes_do_not_automatically_become_comparison(self):
        aventus = candidate(1, "Aventus", "Creed", "fruity, woody")
        hacivat = candidate(2, "Hacivat", "Nishane", "green, woody")
        safe = candidate(3, "Safe Choice", "Other", "fresh, citrus")
        outputs = iter([
            '{"intent":"other","perfumes":[]}',
            "Safe Choice by Other fits through its listed fresh and citrus accords.",
        ])

        def generator(messages, max_new_tokens):
            return next(outputs)

        items = [aventus, hacivat, safe]
        pipeline = ScentAIModelPipeline(FakeRetriever(items), generator, catalog=FakeCatalog(items), top_k=3)
        result = pipeline.run("I like Aventus and Hacivat; recommend a fresh alternative.")
        self.assertNotEqual(result.route, "llm_grounded_comparison")
        self.assertFalse(result.analysis["comparison_perfumes"])

    def test_model_planner_turns_semantic_language_into_grounded_constraints(self):
        first = candidate(1, "Air One", "Brand A", "fresh, citrus", seasons="summer", time_profile="day")
        second = candidate(2, "Air Two", "Brand B", "fresh, aquatic", seasons="summer", time_profile="day")
        sweet = candidate(3, "Syrup", "Brand C", "sweet, vanilla", seasons="winter", time_profile="night")
        plan = {
            "intent": "recommendation",
            "confidence": 0.94,
            "perfumes": [],
            "season": {"value": "summer", "evidence": "scorching afternoons"},
            "time_profile": {"value": "day", "evidence": "afternoons"},
            "wanted_accords": [{"value": "fresh", "evidence": "airy"}],
            "excluded_accords": [{"value": "sweet", "evidence": "not syrupy"}],
            "requested_count": {"value": 2, "evidence": "a pair"},
        }
        outputs = iter([
            __import__("json").dumps(plan),
            (
                "You are after an airy pair for hot daytime wear.\n"
                "1. Air One by Brand A\nIts fresh-citrus character reads bright and uncluttered, "
                "while the recorded summer/day profile keeps it focused on the heat.\n"
                "2. Air Two by Brand B\nIts fresh-aquatic character gives the request a cooler direction, "
                "and its summer/day profile supports the same practical role."
            ),
        ])

        def generator(messages, max_new_tokens):
            return next(outputs)

        items = [first, second, sweet]
        pipeline = ScentAIModelPipeline(FakeRetriever(items), generator, catalog=FakeCatalog(items), top_k=3)
        result = pipeline.run("Give me a pair that feels airy for scorching afternoons, but not syrupy.")
        self.assertEqual(result.analysis["model_intent"], "recommendation")
        self.assertEqual(result.analysis["season"], "summer")
        self.assertEqual(result.analysis["time_profile"], "day")
        self.assertEqual(result.analysis["requested_count"], 2)
        self.assertIn("fresh", result.analysis["wanted_accords"])
        self.assertIn("sweet", result.analysis["negative_accords"])
        self.assertNotIn("Syrup by Brand C", result.answer)
        self.assertTrue(result.validation["pass"])

    def test_model_planner_can_route_implicit_medical_safety_request(self):
        safe = candidate(1, "Example", "Brand", "fresh")

        def generator(messages, max_new_tokens):
            return '{"intent":"unsupported_medical","confidence":0.9}'

        pipeline = ScentAIModelPipeline(FakeRetriever([safe]), generator, catalog=FakeCatalog([safe]))
        result = pipeline.run("Would this be gentle enough for someone with sensitive breathing?")
        self.assertEqual(result.route, "unsupported_medical")
        self.assertEqual(result.generation_attempts, 0)

    def test_invalid_planner_json_is_repaired_once(self):
        safe = candidate(1, "Safe Choice", "Good Brand", "fresh, citrus")
        outputs = iter([
            "```json not-valid ```",
            '{"intent":"recommendation","confidence":0.88}',
            (
                "You are looking for a bright, uncomplicated profile.\n"
                "1. Safe Choice by Good Brand\n"
                "Its fresh-citrus character reads clean and easy to wear."
            ),
        ])

        def generator(messages, max_new_tokens):
            return next(outputs)

        pipeline = ScentAIModelPipeline(
            FakeRetriever([safe]),
            generator,
            catalog=FakeCatalog([safe]),
        )
        result = pipeline.run("Recommend a fresh scent")
        self.assertEqual(result.analysis["model_intent"], "recommendation")
        self.assertEqual(result.analysis["planner_confidence"], 0.88)
        self.assertTrue(result.validation["pass"])

    def test_planner_constraint_without_query_evidence_is_rejected(self):
        plan = {
            "excluded_notes": [
                {"value": "oud", "evidence": "heavy oud"},
                {"value": "vanilla", "evidence": "skip vanilla"},
            ]
        }
        values = validated_plan_values(plan, "excluded_notes", "Please skip vanilla in the recommendations.")
        self.assertEqual(values, ["vanilla"])

    def test_ambiguous_clean_word_is_not_silently_treated_as_brand(self):
        ambiguous = {"requested_brand": {"value": "Clean", "evidence": "clean"}}
        explicit = {"requested_brand": {"value": "Clean", "evidence": "from Clean"}}
        normal = {"requested_brand": {"value": "Versace", "evidence": "Versace"}}
        self.assertIsNone(brand_plan_value(ambiguous, "Recommend a clean office scent."))
        self.assertEqual(brand_plan_value(explicit, "Recommend something from Clean."), "Clean")
        self.assertEqual(brand_plan_value(normal, "Versace erkek parfümü öner."), "Versace")

    def test_ambiguous_scent_word_does_not_boost_same_named_brand(self):
        analysis = analyze_query("I need a clean office scent without vanilla.")
        clean_brand = candidate(1, "Fresh Laundry", "Clean", "fresh, musky, soapy")
        neutral_brand = candidate(2, "Paper Soap", "J-Scent", "fresh, musky, soapy")
        self.assertTrue(has_accidental_brand_collision(clean_brand, analysis))
        self.assertFalse(has_accidental_brand_collision(neutral_brand, analysis))
        self.assertLess(
            score_candidate(clean_brand, analysis).final_score,
            score_candidate(neutral_brand, analysis).final_score,
        )

    def test_explicit_clean_brand_request_keeps_normal_brand_scoring(self):
        analysis = replace(analyze_query("Show me perfumes from Clean."), requested_brand="Clean")
        clean_brand = candidate(1, "Fresh Laundry", "Clean", "fresh, musky, soapy")
        self.assertFalse(has_accidental_brand_collision(clean_brand, analysis))
        scored = score_candidate(clean_brand, analysis)
        self.assertNotIn("ambiguous_brand_collision=-0.14", scored.reasons)

    def test_collision_brand_is_capped_without_affecting_other_brands(self):
        items = [
            candidate(1, "One", "Clean", "fresh"),
            candidate(2, "Two", "Clean", "fresh"),
            candidate(3, "Three", "Other", "fresh"),
            candidate(4, "Four", "Other", "fresh"),
        ]
        result = brand_dedup(items, max_per_brand=2, per_brand_limits={"clean": 1})
        self.assertEqual([item.name for item in result], ["One", "Three", "Four"])

    def test_vllm_backend_routes_planner_and_comparison_budgets(self):
        planner = [{"role": "system", "content": "You are the intent planner for a grounded perfume assistant."}]
        comparison = [{"role": "system", "content": "You are ScentAI, a careful perfume comparison assistant."}]
        ordinary = [{"role": "system", "content": "You are ScentAI."}]
        self.assertTrue(is_planner_messages(planner))
        self.assertTrue(is_comparison_messages(comparison))
        self.assertFalse(is_planner_messages(ordinary))
        self.assertEqual(answer_token_budget(ordinary, 500), 280)
        self.assertEqual(answer_token_budget(comparison, 500), 380)
        self.assertEqual(answer_token_budget(comparison, 200), 200)

    def test_vllm_http_generator_separates_base_planner_and_lora_answer(self):
        session = FakeHTTPSession()
        generator = VLLMHTTPMessageGenerator(
            FakeTokenizer(),
            base_url="http://127.0.0.1:8000/",
            base_model_name="base",
            adapter_model_name="scentai",
            session=session,
        )
        planner = [
            {
                "role": "system",
                "content": "You are the intent planner for a grounded perfume assistant.",
            }
        ]
        answer = [{"role": "system", "content": "You are ScentAI."}]

        self.assertIn('"intent"', generator(planner, 500))
        self.assertEqual(generator(answer, 500), "Answer")

        planner_payload = session.requests[0]["json"]
        answer_payload = session.requests[1]["json"]
        self.assertEqual(planner_payload["model"], "base")
        self.assertIn("structured_outputs", planner_payload)
        self.assertEqual(answer_payload["model"], "scentai")
        self.assertNotIn("structured_outputs", answer_payload)
        self.assertEqual(answer_payload["max_tokens"], 280)
        self.assertTrue(all(isinstance(token_id, int) for token_id in answer_payload["prompt"]))
        self.assertFalse(generator.metrics[0].used_lora)
        self.assertTrue(generator.metrics[1].used_lora)

    def test_planner_schema_requires_grounded_core_fields(self):
        self.assertEqual(PLANNER_JSON_SCHEMA["required"], ["intent", "confidence"])
        self.assertFalse(PLANNER_JSON_SCHEMA["additionalProperties"])
        self.assertIn("comparison", PLANNER_JSON_SCHEMA["properties"]["intent"]["enum"])

    def test_vllm_adapter_preflight_accepts_current_adapter(self):
        adapter = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "scentai-gemma-4-12b-it-pilot-fastmodel-lora"
            / "best_lora_adapter"
        )
        if not adapter.exists():
            self.skipTest("Local pilot adapter is not present")
        config = validate_adapter_config(adapter, max_lora_rank=16)
        self.assertEqual(config["r"], 16)
        self.assertFalse(config.get("use_dora", False))

    def test_comparison_metric_misspelling_is_detected(self):
        self.assertEqual(find_misspelled_metric_terms("Its lorgonity is stronger."), ["lorgonity"])
        self.assertEqual(find_misspelled_metric_terms("Its longevity and sillage are recorded."), [])

    def test_general_prompt_requires_consultative_explanations(self):
        self.assertIn("warm, perceptive perfume consultant", SYSTEM_PROMPT)
        self.assertIn("character, why it fits, where or when it would work", SYSTEM_PROMPT)
        self.assertIn("do not call a perfume four-season unless all four seasons", SYSTEM_PROMPT)

    def test_mechanical_catalog_reason_is_rejected(self):
        answer = "1. Anelo by Pernoire\nWhy: I would include it for the tropical, citrus accords."
        self.assertEqual(find_generic_catalog_phrases(answer), ["why_i_would_include_it_for"])
        analysis = analyze_query("Recommend a tropical perfume")
        context = """[PERFUMES]
Anelo by Pernoire - unisex
Accords: tropical, citrus
[/PERFUMES]"""
        report = validate_runtime_answer(answer, context, analysis)
        self.assertFalse(report["pass"])
        self.assertEqual(report["generic_catalog_phrases"], ["why_i_would_include_it_for"])

    def test_best_pick_explanation_is_not_parsed_as_part_of_perfume_name(self):
        context = """[PERFUMES]
Safe Choice by Good Brand - unisex
Accords: fresh, citrus
[/PERFUMES]"""
        report = score_case_result(
            {
                "name": "best-pick-parser",
                "context": context,
                "answer": "Best pick: Safe Choice by Good Brand, because its fresh profile is the strongest fit.",
            }
        )
        self.assertTrue(report["pass"])
        self.assertEqual(report["unsupported_perfume_mentions"], [])

    def test_template_fallback_explains_character_wear_and_performance(self):
        anelo = candidate(
            1,
            "Anelo",
            "Pernoire",
            "tropical, fresh spicy, citrus",
            notes="pineapple, mandarin, ginger",
            seasons="spring, summer",
            time_profile="day, night",
        )
        anelo.metadata.update(longevity=3.96, sillage=2.77)
        answer = build_template_recommendation(
            "Recommend a tropical summer perfume",
            [anelo],
            analyze_query("Recommend a tropical summer perfume"),
        )
        self.assertIn("tropical, fresh spicy, and citrus-led character", answer)
        self.assertIn("Spring and Summer", answer)
        self.assertIn("3.96/5 longevity suggests strong recorded staying power", answer)
        self.assertIn("2.77/4 sillage suggests a noticeable presence", answer)
        self.assertFalse(find_generic_catalog_phrases(answer))

    def test_negative_terms_are_hard_filtered(self):
        analysis = analyze_query("Clean office scent without vanilla or smoky accords")
        unsafe = candidate(1, "Unsafe", "Brand", "vanilla, smoky, woody")
        safe = candidate(2, "Safe", "Brand", "fresh, citrus, clean")
        self.assertTrue(violates_negative(unsafe.metadata, analysis))
        self.assertFalse(violates_negative(safe.metadata, analysis))

    def test_invalid_llm_output_retries_then_falls_back(self):
        safe = candidate(2, "Safe Choice", "Good Brand", "fresh, citrus, clean")
        calls = []

        def bad_generator(messages, max_new_tokens):
            calls.append(messages)
            return "1. Invented Perfume by Imaginary Brand"

        pipeline = ScentAIModelPipeline(FakeRetriever([safe]), bad_generator, top_k=3)
        result = pipeline.run("Recommend a clean scent")
        self.assertEqual(result.route, "validated_template_fallback")
        self.assertEqual(result.generation_attempts, 2)
        self.assertEqual(len(result.generation_failures), 2)
        self.assertIn("Safe Choice by Good Brand", result.answer)
        self.assertTrue(result.validation["pass"])

    def test_excluded_entity_mention_fails_validation(self):
        safe = candidate(2, "Safe Choice", "Good Brand", "fresh, citrus, clean")
        analysis = analyze_query("Recommend something but avoid Tom Ford")
        context = """[PERFUMES]
Safe Choice by Good Brand - unisex
Accords: fresh, citrus, clean
[/PERFUMES]"""
        report = validate_runtime_answer("Tom Ford would normally fit, but choose Safe Choice by Good Brand.", context, analysis)
        self.assertFalse(report["pass"])
        self.assertEqual(report["excluded_entity_mentions"], ["tom ford"])


if __name__ == "__main__":
    unittest.main()
