from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable

from research.runtime.catalog import RuntimeCatalog
from research.runtime.exact_lookup import is_exact_lookup_query, maybe_answer_exact_lookup, render_database_lookup_answer
from research.runtime.grounding_checker import score_case_result
from research.runtime.intent_router import (
    contradiction_answer,
    prepare_collection_gap,
    render_comparison,
    unsupported_intent_answer,
)
from research.runtime.prompts import build_generation_messages
from research.runtime.query_analyzer import (
    AMBIGUOUS_BRAND_TERMS,
    KNOWN_ACCORDS,
    KNOWN_NOTES,
    QueryAnalysis,
    VIBE_ALIASES,
    analyze_query,
    entity_matches,
    extract_owned_perfumes,
    normalize_match_text,
)
from research.runtime.rag import PerfumeCandidate, ScentRetriever, build_perfume_context
from research.runtime.scentai_pipeline import analysis_to_dict, build_template_recommendation, candidate_to_dict


MessageGenerator = Callable[[list[dict[str, str]], int], str]


PLANNER_SYSTEM_PROMPT = """You are the intent planner for a grounded perfume assistant.
Understand the user's meaning, not just keywords. Return one compact JSON object and nothing else.

Allowed intents: recommendation, similarity, alternative, preference_recommendation, comparison,
perfume_profile, exact_lookup, collection_gap, ranking, unsupported_price, unsupported_availability,
unsupported_medical, unsupported_social_claim, unsupported_layering.

Schema:
{
  "intent": "...",
  "confidence": 0.0,
  "perfumes": [{"value":"...","evidence":"exact user quote"}],
  "requested_brand": {"value":"...","evidence":"exact user quote"} or null,
  "gender": {"value":"male|female|unisex","evidence":"exact user quote"} or null,
  "season": {"value":"spring|summer|autumn|winter","evidence":"exact user quote"} or null,
  "time_profile": {"value":"day|night","evidence":"exact user quote"} or null,
  "wanted_accords": [], "wanted_notes": [], "excluded_accords": [], "excluded_notes": [],
  "excluded_entities": [], "owned_perfumes": [],
  "requested_count": null, "sort_by": null, "year_min": null, "year_max": null
}

Every populated constraint must be an object with `value` and an exact evidence quote from the user.
`requested_count`, `sort_by`, and year fields use the same object form when populated.
Omit null and empty fields from the output to keep the JSON compact.
Set `requested_brand` only when the user clearly refers to a brand. Words such as clean, fresh, date,
office, rose, musk, and similar scent/style terms are not brands unless the wording explicitly marks them as one.
Use comparison only for contrast/choice/difference requests. Use similarity only for literal smell/profile
similarity, alternative for a dupe or substitute, and preference_recommendation for recommendations based
on liked perfumes. Use exact_lookup only when the user asks for exact, verbatim, or database fields. Use
perfume_profile when the user asks what one perfume is like, wants its character/vibe, wear situations,
practical performance, versatility, or a natural explanation of its notes. Do not invent perfume names or
constraints. Omit fields the user did not imply."""


MODEL_UNSUPPORTED_ANSWERS = {
    "unsupported_price": "I do not have a current price source, so I cannot safely answer price or budget questions.",
    "unsupported_availability": "I do not have a live availability source, so I cannot verify stock or discontinued status.",
    "unsupported_medical": "I cannot guarantee medical or allergy safety from perfume database fields. Check the ingredient label and consult a qualified professional when needed.",
    "unsupported_social_claim": "The database does not record compliment outcomes, so I cannot rank perfumes by compliments.",
    "unsupported_layering": "The database does not contain verified layering guidance, so I cannot make a grounded layering claim.",
}

@dataclass
class ModelPipelineResult:
    query: str
    route: str
    answer: str
    analysis: dict[str, Any]
    candidates: list[dict[str, Any]]
    validation: dict[str, Any]
    generation_attempts: int
    generation_failures: list[dict[str, Any]]


class ScentAIModelPipeline:
    """End-to-end retrieval, hard filtering, generation, and output validation."""

    def __init__(
        self,
        retriever: ScentRetriever,
        generator: MessageGenerator,
        *,
        top_k: int = 10,
        fetch_k: int = 100,
        max_new_tokens: int = 300,
        retry_once: bool = True,
        catalog: RuntimeCatalog | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.max_new_tokens = max_new_tokens
        self.retry_once = retry_once
        self.catalog = catalog
        self._planner_cache: dict[str, dict[str, Any]] = {}

    def run(self, query: str, *, profile_text: str = "No preferences recorded yet.") -> ModelPipelineResult:
        analysis = analyze_query(query)
        profile_owned = extract_owned_perfumes(profile_text.lower())
        if profile_owned:
            owned = tuple(dict.fromkeys([*analysis.owned_perfumes, *profile_owned]))
            analysis = replace(
                analysis,
                owned_perfumes=owned,
                debug={**analysis.debug, "owned_perfumes": list(owned)},
            )
        analysis = self._model_assisted_analysis(query, analysis)

        model_unsupported = MODEL_UNSUPPORTED_ANSWERS.get(analysis.model_intent or "")
        if model_unsupported:
            return self._result(
                query,
                analysis.model_intent or "unsupported",
                model_unsupported,
                analysis,
                [],
                {},
                0,
                [],
            )
        unsupported = unsupported_intent_answer(query)
        if unsupported:
            unsupported_route, answer = unsupported
            return self._result(query, f"unsupported_{unsupported_route}", answer, analysis, [], {}, 0, [])

        contradiction = contradiction_answer(analysis)
        if contradiction:
            return self._result(query, "clarify_contradiction", contradiction, analysis, [], {}, 0, [])

        if analysis.comparison_perfumes:
            return self._run_model_comparison(query, analysis, profile_text)

        if analysis.model_intent == "perfume_profile" and analysis.target_perfumes:
            profile_result = self._run_model_profile(query, analysis, profile_text)
            if profile_result is not None:
                return profile_result

        if analysis.model_intent == "exact_lookup" and analysis.target_perfumes:
            exact_result = self._run_planned_exact_lookup(query, analysis)
            if exact_result is not None:
                return exact_result

        collection_route = False
        retrieval_query = query
        if analysis.collection_gap_request:
            if not self.catalog:
                answer = "The deterministic collection catalog is not available in this runtime."
                return self._result(query, "collection_unavailable", answer, analysis, [], {}, 0, [])
            prepared, note = prepare_collection_gap(self.catalog, analysis)
            if prepared is None:
                return self._result(query, "clarify_collection", note, analysis, [], {}, 0, [])
            analysis = prepared
            profile_text = profile_text + "\n" + note
            retrieval_query = f"{query}\nCollection gap target: {analysis.debug['collection_gap_target']}"
            collection_route = True

        requested_top_k = max(self.top_k, (analysis.requested_count or 0) + 3)
        candidates, analysis = self.retriever.retrieve(
            retrieval_query,
            top_k=requested_top_k,
            fetch_k=self.fetch_k,
            analysis=analysis,
        )
        context = build_perfume_context(candidates)

        exact_answer = self._exact_lookup(query, context, candidates)
        if exact_answer is not None:
            return self._result(query, "deterministic_exact_lookup", exact_answer, analysis, candidates, {}, 0, [])

        if not candidates:
            answer = "I could not find a safe grounded match after applying the requested filters."
            return self._result(query, "no_safe_match", answer, analysis, candidates, {}, 0, [])

        messages = build_generation_messages(query, candidates, profile_text=profile_text, analysis=analysis)
        answer = self.generator(messages, self.max_new_tokens).strip()
        validation = validate_runtime_answer(answer, context, analysis)
        attempts = 1
        generation_failures: list[dict[str, Any]] = []

        if not validation["pass"] and self.retry_once:
            generation_failures.append(compact_validation_failure(validation))
            retry_messages = build_retry_messages(messages, answer, validation)
            answer = self.generator(retry_messages, self.max_new_tokens).strip()
            validation = validate_runtime_answer(answer, context, analysis)
            attempts = 2

        if not validation["pass"]:
            generation_failures.append(compact_validation_failure(validation))
            answer = build_template_recommendation(query, candidates, analysis)
            validation = validate_runtime_answer(answer, context, analysis)
            route = "validated_template_fallback"
        else:
            if collection_route:
                route = "llm_grounded_collection_gap"
            elif analysis.resolved_references:
                route = "llm_grounded_multi_reference_similarity"
            elif analysis.resolved_reference:
                route = "llm_grounded_reference_similarity"
            else:
                route = "llm_grounded"

        return self._result(query, route, answer, analysis, candidates, validation, attempts, generation_failures)

    def _model_assisted_analysis(
        self,
        query: str,
        analysis: QueryAnalysis,
    ) -> QueryAnalysis:
        if not self.catalog:
            return analysis
        mentions = self.catalog.extract_mentions(query)
        planning_messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Detected catalog mentions: {list(mentions)}\nUser request: {query}",
            },
        ]
        cache_key = normalize_match_text(query)
        plan = self._planner_cache.get(cache_key)
        if plan is None:
            raw_plan = self.generator(planning_messages, 128).strip()
            plan = parse_planner_json(raw_plan)
            if plan.get("parse_error"):
                repair_messages = [
                    *planning_messages,
                    {"role": "assistant", "content": raw_plan},
                    {
                        "role": "user",
                        "content": (
                            "That response was not valid JSON. Return exactly one compact JSON object matching the schema. "
                            "Do not use Markdown fences or explanatory text."
                        ),
                    },
                ]
                repaired_raw_plan = self.generator(repair_messages, 128).strip()
                repaired_plan = parse_planner_json(repaired_raw_plan)
                if not repaired_plan.get("parse_error"):
                    plan = repaired_plan
                else:
                    plan["repair_parse_error"] = repaired_plan.get("parse_error")
            if len(self._planner_cache) >= 512:
                self._planner_cache.pop(next(iter(self._planner_cache)))
            self._planner_cache[cache_key] = plan
        intent = str(plan.get("intent") or "").strip().lower()
        allowed_intents = {
            "recommendation", "similarity", "alternative", "preference_recommendation", "comparison",
            "perfume_profile",
            "exact_lookup", "collection_gap", "ranking", *MODEL_UNSUPPORTED_ANSWERS,
        }
        if intent not in allowed_intents:
            intent = "recommendation"
        confidence = clamp_float(plan.get("confidence"))
        planned_perfumes = validated_plan_values(plan, "perfumes", query)
        resolved_perfumes = []
        for value in planned_perfumes:
            row = self.catalog.resolve(str(value))
            if row:
                label = str(row["name"])
                if label not in resolved_perfumes:
                    resolved_perfumes.append(label)
        if not resolved_perfumes:
            resolved_perfumes = list(mentions)

        requested_brand = brand_plan_value(plan, query)
        gender = enum_plan_value(plan, "gender", query, {"male", "female", "unisex"})
        season = enum_plan_value(plan, "season", query, {"spring", "summer", "autumn", "winter"})
        time_profile = enum_plan_value(plan, "time_profile", query, {"day", "night"})
        sort_by = enum_plan_value(
            plan,
            "sort_by",
            query,
            {"rating", "popularity", "year", "longevity", "sillage", "value_score"},
        )
        requested_count = int_plan_value(plan, "requested_count", query, minimum=1, maximum=10)
        year_min = int_plan_value(plan, "year_min", query, minimum=1700, maximum=2100)
        year_max = int_plan_value(plan, "year_max", query, minimum=1700, maximum=2100)
        wanted_accords = set(analysis.wanted_accords) | set(validated_plan_values(plan, "wanted_accords", query))
        wanted_notes = set(analysis.wanted_notes) | set(validated_plan_values(plan, "wanted_notes", query))
        negative_accords = set(analysis.negative_accords) | set(validated_plan_values(plan, "excluded_accords", query))
        negative_notes = set(analysis.negative_notes) | set(validated_plan_values(plan, "excluded_notes", query))
        excluded_entities = set(analysis.excluded_entities) | set(validated_plan_values(plan, "excluded_entities", query))
        owned = list(analysis.owned_perfumes)
        for value in validated_plan_values(plan, "owned_perfumes", query):
            row = self.catalog.resolve(str(value))
            label = f"{row['name']} by {row['brand']}" if row else str(value)
            if label not in owned:
                owned.append(label)

        comparison_perfumes = analysis.comparison_perfumes
        reference_perfume = analysis.reference_perfume
        reference_perfumes = analysis.reference_perfumes
        target_perfumes = analysis.target_perfumes
        reference_relation = analysis.reference_relation
        collection_gap_request = analysis.collection_gap_request or intent == "collection_gap"
        contradictions = set(analysis.contradictions)
        contradictions.update(wanted_accords & negative_accords)
        contradictions.update(wanted_notes & negative_notes)
        if intent == "comparison" and len(resolved_perfumes) >= 2:
            comparison_perfumes = tuple(resolved_perfumes[:2])
        elif intent in {"similarity", "alternative", "preference_recommendation"} and resolved_perfumes:
            reference_relation = "alternative" if intent == "alternative" else "similar"
            if len(resolved_perfumes) >= 2:
                reference_perfume = None
                reference_perfumes = tuple(resolved_perfumes[:2])
            else:
                reference_perfume = resolved_perfumes[0]
                reference_perfumes = ()
        elif intent in {"exact_lookup", "perfume_profile"} and resolved_perfumes:
            target_perfumes = tuple(resolved_perfumes[:1])

        return replace(
            analysis,
            model_intent=intent,
            planner_confidence=confidence,
            gender=gender or analysis.gender,
            season=season or analysis.season,
            time_profile=time_profile or analysis.time_profile,
            wanted_accords=tuple(sorted(wanted_accords - negative_accords)),
            wanted_notes=tuple(sorted(wanted_notes - negative_notes)),
            negative_accords=tuple(sorted(negative_accords)),
            negative_notes=tuple(sorted(negative_notes)),
            excluded_entities=tuple(sorted(excluded_entities)),
            requested_brand=str(requested_brand) if requested_brand else analysis.requested_brand,
            reference_perfume=reference_perfume,
            reference_perfumes=reference_perfumes,
            reference_relation=reference_relation,
            comparison_perfumes=comparison_perfumes,
            target_perfumes=target_perfumes,
            owned_perfumes=tuple(owned),
            collection_gap_request=collection_gap_request,
            contradictions=tuple(sorted(contradictions)),
            requested_count=requested_count or analysis.requested_count,
            sort_by=sort_by or analysis.sort_by,
            year_min=year_min or analysis.year_min,
            year_max=year_max or analysis.year_max,
            debug={
                **analysis.debug,
                "model_intent": intent,
                "planner_confidence": confidence,
                "comparison_perfumes": list(comparison_perfumes),
                "comparison_detected_by": "model_planner" if intent == "comparison" else analysis.debug.get("comparison_detected_by"),
                "model_planner": plan,
                "catalog_mentions": list(mentions),
            },
        )

    def _run_model_comparison(
        self,
        query: str,
        analysis: QueryAnalysis,
        profile_text: str,
    ) -> ModelPipelineResult:
        if not self.catalog:
            answer = "The comparison catalog is not available in this runtime."
            return self._result(query, "comparison_unavailable", answer, analysis, [], {}, 0, [])
        rows, missing = self.catalog.compare(list(analysis.comparison_perfumes))
        if missing or len(rows) != 2:
            names = missing or list(analysis.comparison_perfumes)
            answer = "I could not resolve two distinct comparison perfumes: " + ", ".join(names) + "."
            return self._result(query, "comparison_unresolved", answer, analysis, [], {}, 0, [])
        candidates = self.retriever.candidates_by_ids([int(row["perfume_id"]) for row in rows])
        if len(candidates) != 2:
            answer = "The two comparison records could not be loaded from the vector catalog."
            return self._result(query, "comparison_unavailable", answer, analysis, candidates, {}, 0, [])

        context = build_perfume_context(candidates)
        messages = build_generation_messages(query, candidates, profile_text=profile_text, analysis=analysis)
        answer = self.generator(messages, self.max_new_tokens).strip()
        validation = validate_runtime_answer(answer, context, analysis)
        attempts = 1
        failures: list[dict[str, Any]] = []
        if not validation["pass"] and self.retry_once:
            failures.append(compact_validation_failure(validation))
            answer = self.generator(build_retry_messages(messages, answer, validation), self.max_new_tokens).strip()
            validation = validate_runtime_answer(answer, context, analysis)
            attempts = 2
        if validation["pass"]:
            route = "llm_grounded_comparison"
        else:
            failures.append(compact_validation_failure(validation))
            fallback = render_comparison(self.catalog, analysis)
            assert fallback is not None
            answer, _ = fallback
            validation = validate_runtime_answer(answer, context, analysis)
            route = "validated_comparison_fallback"
        return self._result(query, route, answer, analysis, candidates, validation, attempts, failures)

    def _run_planned_exact_lookup(
        self,
        query: str,
        analysis: QueryAnalysis,
    ) -> ModelPipelineResult | None:
        if not self.catalog:
            return None
        row = self.catalog.resolve(analysis.target_perfumes[0])
        if not row:
            return None
        candidates = self.retriever.candidates_by_ids([int(row["perfume_id"])])
        if not candidates:
            return None
        answer = render_database_lookup_answer(build_perfume_context(candidates) + "\n\n" + query)
        return self._result(
            query,
            "model_planned_exact_lookup",
            answer,
            analysis,
            candidates,
            {},
            0,
            [],
        )

    def _run_model_profile(
        self,
        query: str,
        analysis: QueryAnalysis,
        profile_text: str,
    ) -> ModelPipelineResult | None:
        if not self.catalog:
            return None
        row = self.catalog.resolve(analysis.target_perfumes[0])
        if not row:
            return None
        candidates = self.retriever.candidates_by_ids([int(row["perfume_id"])])
        if not candidates:
            return None

        context = build_perfume_context(candidates)
        messages = build_generation_messages(query, candidates, profile_text=profile_text, analysis=analysis)
        answer = self.generator(messages, self.max_new_tokens).strip()
        validation = validate_runtime_answer(answer, context, analysis)
        attempts = 1
        failures: list[dict[str, Any]] = []
        if not validation["pass"] and self.retry_once:
            failures.append(compact_validation_failure(validation))
            answer = self.generator(build_retry_messages(messages, answer, validation), self.max_new_tokens).strip()
            validation = validate_runtime_answer(answer, context, analysis)
            attempts = 2
        if validation["pass"]:
            route = "llm_grounded_perfume_profile"
        else:
            failures.append(compact_validation_failure(validation))
            answer = build_template_recommendation(query, candidates, analysis)
            validation = validate_runtime_answer(answer, context, analysis)
            route = "validated_profile_fallback"
        return self._result(query, route, answer, analysis, candidates, validation, attempts, failures)

    @staticmethod
    def _exact_lookup(query: str, context: str, candidates: list[PerfumeCandidate]) -> str | None:
        answer = maybe_answer_exact_lookup(context + "\n\n" + query)
        if answer is None and is_exact_lookup_query(query) and candidates:
            answer = render_database_lookup_answer(build_perfume_context(candidates[:1]) + "\n\n" + query)
        return answer

    @staticmethod
    def _result(
        query: str,
        route: str,
        answer: str,
        analysis: QueryAnalysis,
        candidates: list[PerfumeCandidate],
        validation: dict[str, Any],
        attempts: int,
        generation_failures: list[dict[str, Any]],
    ) -> ModelPipelineResult:
        return ModelPipelineResult(
            query=query,
            route=route,
            answer=answer,
            analysis=analysis_to_dict(analysis),
            candidates=[candidate_to_dict(candidate) for candidate in candidates],
            validation=validation,
            generation_attempts=attempts,
            generation_failures=generation_failures,
        )


def validate_runtime_answer(answer: str, context: str, analysis: QueryAnalysis) -> dict[str, Any]:
    payload = {
        "name": "runtime",
        "context": context,
        "answer": answer,
        "excluded_terms": [*analysis.negative_accords, *analysis.negative_notes],
        "forbidden_perfumes": [],
    }
    report = score_case_result(payload)
    excluded_entity_mentions = [
        entity for entity in analysis.excluded_entities if entity_matches(entity, answer)
    ]
    if excluded_entity_mentions:
        report["excluded_entity_mentions"] = excluded_entity_mentions
        report["hard_fail_reasons"].append("Answer mentioned an excluded brand or perfume.")
        report["pass"] = False
    else:
        report["excluded_entity_mentions"] = []
    generic_phrases = find_generic_catalog_phrases(answer)
    report["generic_catalog_phrases"] = generic_phrases
    if generic_phrases:
        report["hard_fail_reasons"].append(
            "Answer used a mechanical catalog-rationale template instead of explaining character, wear context, and practical tradeoffs."
        )
        report["pass"] = False
    if analysis.requested_count is not None:
        actual_count = len(report.get("mentioned_context_perfumes", []))
        report["requested_count"] = analysis.requested_count
        report["actual_recommendation_count"] = actual_count
        if actual_count != analysis.requested_count:
            report["hard_fail_reasons"].append(
                f"Answer recommended {actual_count} perfumes; exactly {analysis.requested_count} were requested."
            )
            report["pass"] = False
    if analysis.comparison_perfumes:
        mentioned_count = len(report.get("mentioned_context_perfumes", []))
        report["comparison_perfume_count"] = mentioned_count
        if mentioned_count != 2:
            report["hard_fail_reasons"].append("A comparison answer must discuss both context perfumes.")
            report["pass"] = False
        misspelled_terms = find_misspelled_metric_terms(answer)
        report["misspelled_metric_terms"] = misspelled_terms
        if misspelled_terms:
            report["hard_fail_reasons"].append(
                "Comparison answer contains misspelled database terminology: " + ", ".join(misspelled_terms)
            )
            report["pass"] = False
    owned_mentions = [entity for entity in analysis.owned_perfumes if entity_matches(entity, answer)]
    if owned_mentions:
        report["owned_perfume_mentions"] = owned_mentions
        report["hard_fail_reasons"].append("Answer recommended or mentioned an already-owned perfume.")
        report["pass"] = False
    else:
        report["owned_perfume_mentions"] = []
    return report


def build_retry_messages(
    messages: list[dict[str, str]],
    invalid_answer: str,
    validation: dict[str, Any],
) -> list[dict[str, str]]:
    reasons = "; ".join(validation.get("hard_fail_reasons", [])) or "grounding validation failed"
    correction = (
        "Your previous answer was rejected by the deterministic validator. "
        f"Reasons: {reasons}. Produce a completely new answer. Mention only perfume names present in "
        "[PERFUMES], obey every strict filter, and use only fields printed on each exact card. "
        "Write like a perfume consultant: turn the card into character, suitable settings, practical performance, "
        "and a meaningful tradeoff. Do not use repetitive 'Why: It matches through' or 'I would include it for' templates."
    )
    return [
        *messages,
        {"role": "assistant", "content": invalid_answer},
        {"role": "user", "content": correction},
    ]


def compact_validation_failure(validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "hard_fail_reasons": list(validation.get("hard_fail_reasons", [])),
        "unsupported_perfume_mentions": list(validation.get("unsupported_perfume_mentions", [])),
        "forbidden_perfume_mentions": list(validation.get("forbidden_perfume_mentions", [])),
        "strict_filter_violations": list(validation.get("strict_filter_violations", [])),
        "excluded_entity_mentions": list(validation.get("excluded_entity_mentions", [])),
        "unsupported_note_claims": list(validation.get("unsupported_note_claims", [])),
        "generic_catalog_phrases": list(validation.get("generic_catalog_phrases", [])),
    }


def find_generic_catalog_phrases(answer: str) -> list[str]:
    patterns = {
        "why_it_matches_through": r"\bwhy\s*:\s*it matches through\b",
        "why_i_would_include_it_for": r"\bwhy\s*:\s*i would include it (?:for|because)\b",
        "why_it_fits_through": r"\bwhy\s*:\s*it fits (?:the brief )?through\b",
        "why_it_earns_a_place": r"\bwhy\s*:\s*it earns a place here with\b",
        "why_the_fit_comes_from": r"\bwhy\s*:\s*the fit comes from\b",
        "why_the_useful_part": r"\bwhy\s*:\s*the useful part here is\b",
        "why_listed_accords": r"\bwhy\s*:\s*listed accords include\b",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, answer, re.I)]


def result_as_dict(result: ModelPipelineResult) -> dict[str, Any]:
    return asdict(result)


def parse_planner_json(raw: str) -> dict[str, Any]:
    start = raw.find("{")
    if start < 0:
        return {"intent": "other", "perfumes": [], "parse_error": "no_json_object"}
    try:
        payload, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError:
        return {"intent": "other", "perfumes": [], "parse_error": "invalid_json"}
    if not isinstance(payload, dict):
        return {"intent": "other", "perfumes": [], "parse_error": "not_an_object"}
    return payload


def validated_plan_values(plan: dict[str, Any], key: str, query: str) -> list[Any]:
    raw = plan.get(key)
    items = raw if isinstance(raw, list) else ([] if raw is None else [raw])
    output = []
    query_norm = normalize_match_text(query)
    for item in items:
        if isinstance(item, dict):
            value = item.get("value")
            evidence = str(item.get("evidence") or "")
            if value is None or not evidence_supported(evidence, query_norm):
                continue
        else:
            value = item
            if value is None or normalize_match_text(str(value)) not in query_norm:
                continue
        output.append(value)
    return output


def first_plan_value(plan: dict[str, Any], key: str, query: str) -> Any | None:
    values = validated_plan_values(plan, key, query)
    return values[0] if values else None


def brand_plan_value(plan: dict[str, Any], query: str) -> str | None:
    value = first_plan_value(plan, "requested_brand", query)
    if value is None:
        return None
    brand = str(value).strip()
    normalized = normalize_match_text(brand)
    ambiguous = normalized in {normalize_match_text(term) for term in AMBIGUOUS_BRAND_TERMS}
    if ambiguous and not has_explicit_brand_context(query, brand):
        return None
    return brand


def has_explicit_brand_context(query: str, brand: str) -> bool:
    escaped = re.escape(brand)
    patterns = [
        rf"\b(?:brand|from|by|house of)\s+{escaped}\b",
        rf"\b{escaped}\s+(?:brand|markası|markasi)\b",
        rf"\b{escaped}['’]?(?:den|dan|ten|tan)\b",
    ]
    return any(re.search(pattern, query, re.I) for pattern in patterns)


def enum_plan_value(plan: dict[str, Any], key: str, query: str, allowed: set[str]) -> str | None:
    value = first_plan_value(plan, key, query)
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else None


def int_plan_value(
    plan: dict[str, Any],
    key: str,
    query: str,
    *,
    minimum: int,
    maximum: int,
) -> int | None:
    value = first_plan_value(plan, key, query)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if minimum <= parsed <= maximum else None


def evidence_supported(evidence: str, query_norm: str) -> bool:
    evidence_norm = normalize_match_text(evidence)
    return bool(evidence_norm and evidence_norm in query_norm)


def clamp_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return min(max(parsed, 0.0), 1.0)


def find_misspelled_metric_terms(answer: str) -> list[str]:
    canonical_terms = ("longevity", "sillage")
    tokens = {token.lower() for token in re.findall(r"[A-Za-z]{5,14}", answer)}
    misspelled = []
    for token in tokens:
        for canonical in canonical_terms:
            if token == canonical:
                continue
            if token[0] != canonical[0] or token[-2:] != canonical[-2:]:
                continue
            if abs(len(token) - len(canonical)) > 2:
                continue
            if levenshtein_distance(token, canonical) <= 3:
                misspelled.append(token)
                break
    return sorted(misspelled)


def levenshtein_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]
