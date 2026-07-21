from __future__ import annotations

import unittest

from scentai.orchestrator import (
    candidate_mentions,
    candidate_answer_sections,
    candidate_has_term,
    explicit_performance_labels,
    extract_numbered_recommendations,
    fallback_answer,
    normalize_plan,
    performance_claim_violations,
    unique_candidates,
    validate_answer,
)


def trait(value: str, evidence: str) -> dict[str, str]:
    return {"value": value, "evidence": evidence}


class PlannerNormalizationTests(unittest.TestCase):
    def test_explicit_english_trait_category_becomes_required(self) -> None:
        query = "Recommend exactly 3 coconut fragrances without vanilla."
        plan = normalize_plan(
            {
                "intent": "recommendation",
                "wanted_terms": [trait("coconut", "coconut")],
                "excluded_terms": [trait("vanilla", "vanilla")],
            },
            query,
        )
        self.assertEqual(plan["required_terms"], ["coconut"])
        self.assertEqual(plan["wanted_terms"], [])
        self.assertEqual(plan["excluded_terms"], ["vanilla"])

    def test_category_phrase_evidence_still_becomes_required(self) -> None:
        query = "Recommend exactly 3 rose fragrances that must not contain musk."
        plan = normalize_plan(
            {
                "intent": "recommendation",
                "wanted_terms": [trait("rose", "rose fragrances")],
                "excluded_terms": [trait("musk", "must not contain musk")],
            },
            query,
        )
        self.assertEqual(plan["required_terms"], ["rose"])
        self.assertEqual(plan["excluded_terms"], ["musk"])

    def test_trailing_hard_marker_applies_to_coordinated_traits(self) -> None:
        query = "Date akşamı için vanilya ve baharat mutlaka bulunan tam 3 parfüm öner."
        plan = normalize_plan(
            {
                "intent": "recommendation",
                "wanted_terms": [trait("vanilla", "vanilya")],
                "required_terms": [trait("warm spicy", "baharat mutlaka bulunan")],
            },
            query,
        )
        self.assertEqual(plan["required_terms"], ["warm spicy", "vanilla"])
        self.assertEqual(plan["wanted_terms"], [])

    def test_hard_marker_recovers_requirements_when_planner_omits_them(self) -> None:
        plan = normalize_plan(
            {"intent": "recommendation"},
            "Erkekler için oud ve deri mutlaka bulunan tam 3 parfüm öner.",
        )
        self.assertEqual(plan["required_terms"], ["oud", "leather"])

    def test_english_must_have_recovers_requirement_when_planner_omits_it(self) -> None:
        plan = normalize_plan(
            {"intent": "recommendation"},
            "Recommend exactly 3 winter fragrances that must have coffee.",
        )
        self.assertEqual(plan["required_terms"], ["coffee"])

    def test_mixed_language_not_trait_is_recovered_without_planner_help(self) -> None:
        plan = normalize_plan(
            {"intent": "recommendation", "semantic_query": "clean fragrance that is not sweet"},
            "clean ama sweet olmayan tam 3 bisey oner",
        )
        self.assertEqual(plan["excluded_terms"], ["sweet"])

    def test_productive_turkish_privative_suffix_is_recovered(self) -> None:
        plan = normalize_plan(
            {"intent": "recommendation"},
            "Ofis icin vanilyasiz tam 3 parfum oner.",
        )
        self.assertEqual(plan["excluded_terms"], ["vanilla"])

    def test_follow_up_command_is_not_mistaken_for_an_exclusion(self) -> None:
        plan = normalize_plan(
            {"intent": "recommendation"},
            "Keep the same request, but exclude tobacco and show three different choices.",
        )
        self.assertEqual(plan["excluded_terms"], ["tobacco"])

    def test_common_negative_grammar_is_recovered(self) -> None:
        cases = {
            "I want something less leathery.": ["leather"],
            "Recommend a vanilla-free fragrance.": ["vanilla"],
            "Anything except cinnamon.": ["cinnamon"],
            "Avoid anything with animalic notes.": ["animalic"],
            "Tütün içermeyen bir parfüm öner.": ["tobacco"],
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                plan = normalize_plan({"intent": "recommendation"}, query)
                self.assertEqual(plan["excluded_terms"], expected)

    def test_non_trait_negative_clause_is_not_a_catalog_exclusion(self) -> None:
        queries = [
            "Compare Aventus and Explorer without declaring a winner.",
            "I want freshness without compromising longevity.",
            "No preference on gender, recommend exactly 3.",
            "Recommend no more than 3 options.",
            "Without a doubt, recommend Aventus.",
        ]
        for query in queries:
            with self.subTest(query=query):
                plan = normalize_plan({"intent": "recommendation"}, query)
                self.assertEqual(plan["excluded_terms"], [])

    def test_multilingual_and_noisy_traits_are_canonicalized(self) -> None:
        cases = {
            "Prada L'Homme'a benzeyen ama pudralı olmayan tam 3 parfüm öner.": ["powdery"],
            "need exactly 3 ofice scents no vanila pls": ["vanilla"],
            "smth like santal 33 but less leathery exactly 3": ["leather"],
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                plan = normalize_plan({"intent": "recommendation"}, query)
                self.assertEqual(plan["excluded_terms"], expected)

    def test_additional_natural_negative_phrases_are_recovered(self) -> None:
        cases = {
            "I don't want vanilla.": ["vanilla"],
            "Nothing animalic, please.": ["animalic"],
            "Avoid all kinds of tobacco.": ["tobacco"],
            "I want a scent without a strong vanilla note.": ["vanilla"],
            "Gül barındırmayan bir parfüm öner.": ["rose"],
            "Tütün istemiyorum.": ["tobacco"],
            "Vanilya hariç herhangi bir şey öner.": ["vanilla"],
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                plan = normalize_plan({"intent": "recommendation"}, query)
                self.assertEqual(plan["excluded_terms"], expected)

    def test_explicit_olsun_recovers_required_trait(self) -> None:
        plan = normalize_plan(
            {"intent": "recommendation"},
            "Vanilya olsun, sıcak bir parfüm öner.",
        )
        self.assertEqual(plan["required_terms"], ["vanilla"])

    def test_multiword_turkish_requirement_and_taxonomy_noun(self) -> None:
        coconut = normalize_plan(
            {"intent": "recommendation"},
            "Hindistan cevizi mutlaka bulunan tropikal tam 3 parfüm öner.",
        )
        aromatic = normalize_plan(
            {"intent": "recommendation"},
            "Aromatik akor mutlaka bulunan tam 3 parfüm öner.",
        )
        self.assertEqual(coconut["required_terms"], ["coconut"])
        self.assertEqual(aromatic["required_terms"], ["aromatic"])

    def test_turkish_connectors_do_not_become_required_traits(self) -> None:
        green = normalize_plan(
            {"intent": "recommendation"},
            "İlkbahar için az bilinen, niş ve yeşil karakterli tam 3 parfüm öner.",
        )
        comparative = normalize_plan(
            {"intent": "similarity"},
            "Light Blue benzeri ama daha karakterli tam 3 seçenek ver.",
        )
        self.assertEqual(green["required_terms"], ["green"])
        self.assertEqual(comparative["required_terms"], [])

    def test_turkish_trait_adjective_becomes_required(self) -> None:
        query = "Bana popüler, vanilyalı tam 3 parfüm öner."
        plan = normalize_plan(
            {
                "intent": "recommendation",
                "wanted_terms": [trait("vanilla", "vanilyalı")],
            },
            query,
        )
        self.assertEqual(plan["required_terms"], ["vanilla"])

    def test_soft_preference_stays_wanted(self) -> None:
        query = "I would prefer a fresh fragrance for work."
        plan = normalize_plan(
            {
                "intent": "recommendation",
                "wanted_terms": [trait("fresh", "fresh")],
            },
            query,
        )
        self.assertEqual(plan["wanted_terms"], ["fresh"])
        self.assertEqual(plan["required_terms"], [])

    def test_social_outcome_claim_overrides_recommendation(self) -> None:
        query = "Which perfume is guaranteed to attract compliments on a date?"
        plan = normalize_plan({"intent": "recommendation"}, query)
        self.assertEqual(plan["intent"], "unsupported_social_claim")

    def test_explicit_audience_season_and_time_are_recovered(self) -> None:
        query = "Recommend exactly 3 spring fragrances for women for daytime wear."
        plan = normalize_plan({"intent": "recommendation"}, query)
        self.assertEqual(plan.get("gender"), "female")
        self.assertEqual(plan.get("season"), "spring")
        self.assertEqual(plan.get("time_profile"), "day")

    def test_date_night_is_recovered_when_planner_omits_time(self) -> None:
        plan = normalize_plan(
            {"intent": "recommendation"},
            "Recommend exactly 3 fruity sweet date fragrances with no vanilla.",
        )
        self.assertEqual(plan.get("time_profile"), "night")

    def test_coordinated_exclusion_list_recovers_every_unknown_item(self) -> None:
        plan = normalize_plan(
            {
                "intent": "recommendation",
                "excluded_terms": [trait("vanilla", "vanilla")],
            },
            "Recommend office fragrances with no vanilla, oud, or smoky accords.",
        )
        self.assertEqual(plan["excluded_terms"], ["vanilla", "oud", "smoky"])

    def test_candidate_deduplication_uses_id_and_normalized_label(self) -> None:
        candidates = [
            {"perfume_id": 1, "label": "Example by House"},
            {"perfume_id": 2, "label": "Example by House"},
            {"perfume_id": 3, "label": "Other by House"},
            {"perfume_id": 3, "label": "Other duplicate label"},
        ]
        self.assertEqual(
            [candidate["perfume_id"] for candidate in unique_candidates(candidates)],
            [1, 3],
        )

    def test_only_explicit_database_trait_is_kept_required(self) -> None:
        query = "Misk mutlaka bulunan, tene yakın ve temiz hissettiren tam 3 parfüm öner."
        plan = normalize_plan(
            {
                "intent": "recommendation",
                "required_terms": [
                    trait("musk", "Misk mutlaka bulunan"),
                    trait("skin scent", "tene yakın"),
                    trait("clean", "temiz hissettiren"),
                ],
            },
            query,
        )
        self.assertEqual(plan["required_terms"], ["musk"])
        self.assertEqual(plan["wanted_terms"], ["skin scent", "clean"])

    def test_style_descriptor_is_not_required_alongside_incense(self) -> None:
        query = "Tütsü mutlaka bulunan, sıra dışı ve niş tam 3 parfüm öner."
        plan = normalize_plan(
            {
                "intent": "recommendation",
                "required_terms": [
                    trait("incense", "Tütsü mutlaka bulunan"),
                    trait("unusual", "sıra dışı"),
                ],
            },
            query,
        )
        self.assertEqual(plan["required_terms"], ["incense"])
        self.assertEqual(plan["wanted_terms"], ["unusual"])

    def test_negative_trait_overrides_planner_positive_mistake(self) -> None:
        query = "Ofis için vanilyasız tam 3 temiz parfüm öner."
        plan = normalize_plan(
            {
                "intent": "recommendation",
                "required_terms": [
                    trait("vanilla", "vanilyasız"),
                    trait("clean", "temiz"),
                ],
                "excluded_terms": [trait("vanilla", "vanilyasız")],
            },
            query,
        )
        self.assertEqual(plan["required_terms"], [])
        self.assertEqual(plan["excluded_terms"], ["vanilla"])
        self.assertEqual(plan["wanted_terms"], ["clean"])

    def test_similarity_brand_scopes_reference_and_soft_amber_stays_wanted(self) -> None:
        query = "Give me exactly 3 alternatives to Grand Soir by Maison Francis Kurkdjian with a softer amber profile."
        plan = normalize_plan(
            {
                "intent": "alternative",
                "perfumes": [trait("Grand Soir", "Grand Soir")],
                "required_terms": [trait("soft amber", "softer amber profile")],
                "requested_brand": trait("Maison Francis Kurkdjian", "by Maison Francis Kurkdjian"),
            },
            query,
        )
        self.assertEqual(plan["required_terms"], [])
        self.assertEqual(plan["wanted_terms"], ["soft amber"])
        self.assertEqual(plan.get("reference_brand"), "Maison Francis Kurkdjian")
        self.assertNotIn("requested_brand", plan)

    def test_trait_families_are_enforced_for_candidates(self) -> None:
        musky = {"name": "Example", "brand": "House", "accords_csv": "musky, citrus", "notes_csv": ""}
        suede = {"name": "Other", "brand": "House", "accords_csv": "woody", "notes_csv": "suede"}
        self.assertTrue(candidate_has_term(musky, "musk"))
        self.assertTrue(candidate_has_term(suede, "leather"))

    def test_numbered_heading_accepts_safe_unique_catalog_suffix(self) -> None:
        candidates = [
            {
                "name": "Comme des Garcons Series 3 Incense: Kyoto",
                "brand": "Comme des Garcons",
                "label": "Comme des Garcons Series 3 Incense: Kyoto by Comme des Garcons",
            },
            {
                "name": "A*Men Pure Malt",
                "brand": "Mugler",
                "label": "A*Men Pure Malt by Mugler",
            },
        ]
        answer = "1. Kyoto by Comme des Garcons\n2. A*Men Pure Malt by Mugler"
        selected, unknown = extract_numbered_recommendations(answer, candidates)
        self.assertEqual(selected, [candidates[0]["label"], candidates[1]["label"]])
        self.assertEqual(unknown, [])


class PerformanceParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            {
                "perfume_id": 1825,
                "name": "Tobacco Vanille",
                "brand": "Tom Ford",
                "label": "Tobacco Vanille by Tom Ford",
                "longevity": 4.04,
                "sillage": 2.81,
            },
            {
                "perfume_id": 5059,
                "name": "Team Five",
                "brand": "Adidas",
                "label": "Team Five by Adidas",
                "longevity": 2.27,
                "sillage": 1.54,
            },
        ]

    def test_natural_comparison_paragraphs_are_attributed_to_candidates(self) -> None:
        answer = (
            "Tobacco Vanille by Tom Ford offers strong longevity and noticeable sillage.\n\n"
            "Team Five by Adidas has light longevity and restrained sillage.\n\n"
            "Tobacco Vanille by Tom Ford lasts longer than Team Five by Adidas."
        )
        sections = candidate_answer_sections(answer, self.candidates)
        self.assertEqual(
            explicit_performance_labels(sections["Tobacco Vanille by Tom Ford"], "longevity"),
            {"high"},
        )
        self.assertEqual(
            explicit_performance_labels(sections["Tobacco Vanille by Tom Ford"], "sillage"),
            {"high"},
        )
        self.assertEqual(
            explicit_performance_labels(sections["Team Five by Adidas"], "longevity"),
            {"low"},
        )
        self.assertEqual(
            explicit_performance_labels(sections["Team Five by Adidas"], "sillage"),
            {"low"},
        )
        self.assertEqual(performance_claim_violations(answer, self.candidates), [])

    def test_nested_perfume_name_does_not_count_as_two_mentions(self) -> None:
        candidates = [
            {"name": "Eros", "brand": "Versace", "label": "Eros by Versace"},
            {"name": "Eros Flame", "brand": "Versace", "label": "Eros Flame by Versace"},
        ]
        self.assertEqual(
            candidate_mentions("Eros Flame by Versace is warmer.", candidates),
            ["Eros Flame by Versace"],
        )
        self.assertEqual(
            candidate_mentions("Eros is fresher, while Eros Flame is warmer.", candidates),
            ["Eros by Versace", "Eros Flame by Versace"],
        )

    def test_comparison_fallback_is_calibrated_and_actionable(self) -> None:
        candidates = [
            {
                "name": "Eros",
                "brand": "Versace",
                "label": "Eros by Versace",
                "accords_csv": "vanilla, fresh spicy, fruity, green",
                "seasons_csv": "spring, summer",
                "time_profile_csv": "day, night",
                "longevity": 3.48,
                "sillage": 2.46,
            },
            {
                "name": "Eros Flame",
                "brand": "Versace",
                "label": "Eros Flame by Versace",
                "accords_csv": "vanilla, fresh spicy, sweet, warm spicy",
                "seasons_csv": "autumn, winter",
                "time_profile_csv": "night",
                "longevity": 3.68,
                "sillage": 2.50,
            },
        ]
        plan = {"intent": "comparison", "required_terms": [], "excluded_terms": []}
        answer = fallback_answer(plan, candidates, response_language="tr")
        self.assertIn("Pratik fark:", answer)
        self.assertIn("ilkbahar, yaz", answer)
        self.assertTrue(validate_answer(answer, plan, candidates, response_language="tr")["pass"])

    def test_turkish_particle_and_qualifier_are_parsed(self) -> None:
        answer = (
            "Turkish Leather by Pryn Parfum çok güçlü bir kalıcılığa sahip olduğu için "
            "uzun süre yerini korurken, yayılımı da güçlü bir seviyede seyrediyor."
        )
        self.assertEqual(explicit_performance_labels(answer, "longevity"), {"high"})
        self.assertEqual(explicit_performance_labels(answer, "sillage"), {"high"})

    def test_opposite_comparison_classes_do_not_leak_between_metrics(self) -> None:
        answer = "Its longevity is strong but its sillage is restrained."
        self.assertEqual(explicit_performance_labels(answer, "longevity"), {"high"})
        self.assertEqual(explicit_performance_labels(answer, "sillage"), {"low"})

    def test_moderate_compounds_remain_moderate(self) -> None:
        self.assertEqual(
            explicit_performance_labels("Its longevity is moderate-light.", "longevity"),
            {"moderate"},
        )

    def test_moderate_sillage_can_be_noticeable_without_becoming_high(self) -> None:
        answer = (
            "Its moderate sillage provides a balanced projection that is "
            "noticeable but not overwhelming."
        )
        self.assertEqual(explicit_performance_labels(answer, "sillage"), {"moderate"})

    def test_explicit_strong_sillage_still_conflicts_with_moderate(self) -> None:
        answer = "Its sillage is moderate at first, but later the sillage becomes strong."
        self.assertEqual(
            explicit_performance_labels(answer, "sillage"),
            {"moderate", "high"},
        )
        self.assertEqual(
            explicit_performance_labels("Its sillage is moderate-restrained.", "sillage"),
            {"moderate"},
        )

    def test_relative_comparative_is_not_an_absolute_class(self) -> None:
        self.assertEqual(
            explicit_performance_labels("It has a more noticeable sillage than the reference.", "sillage"),
            set(),
        )


if __name__ == "__main__":
    unittest.main()
