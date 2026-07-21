from __future__ import annotations

import random
from collections import Counter
from typing import Any

from research.core.cards import build_perfume_context, parse_card_accords, parse_card_notes
from research.core.config import DatasetConfig, L5_DEFAULT_RATIOS, counts_from_ratios
from research.core.data import Perfume
from research.core.llm_query import QueryGenerator
from research.core.messages import L5_SYSTEM_PROMPT, build_messages
from research.core.preference import (
    UserPreferenceProfile,
    build_user_profile_context,
    perfume_label,
    profile_to_debug as preference_to_debug,
)
from research.core.semantic import (
    OCCASION_PROFILES,
    VIBE_PROFILES,
    SemanticProfile,
    find_semantic_matches,
    profile_to_debug,
    score_perfume,
)


ANSWER_LIMIT = 3
MIN_SCORE = 3.0

L5_SCENARIOS = {
    "fresh summer daytime": OCCASION_PROFILES["summer_day"],
    "polished office wear": OCCASION_PROFILES["office"],
    "warm winter evenings": OCCASION_PROFILES["winter_evening"],
    "date night without going too loud": OCCASION_PROFILES["date_night"],
    "clean everyday scent": VIBE_PROFILES["clean"],
    "quiet luxury": VIBE_PROFILES["expensive"],
    "cozy but controlled": VIBE_PROFILES["cozy"],
}


def generate_l5_records(
    perfumes: list[Perfume],
    config: DatasetConfig,
    query_generator: QueryGenerator | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(config.seed)
    counts = l5_category_counts(config)
    indexes = build_generation_indexes(perfumes)
    qgen = query_generator or QueryGenerator()
    schedule = build_l5_schedule(counts, rng)

    records: list[dict[str, Any]] = []
    produced = {category: 0 for category in counts}

    for index, category in enumerate(schedule, 1):
        record_rng = random.Random(record_seed(config.seed or 0, index))
        records.append(make_l5_record(category, perfumes, indexes, config, record_rng, qgen))
        produced[category] += 1

    rng.shuffle(records)
    return records, produced


def l5_category_counts(config: DatasetConfig) -> dict[str, int]:
    return config.category_counts or counts_from_ratios(config.total, L5_DEFAULT_RATIOS)


def build_l5_schedule(counts: dict[str, int], rng: random.Random) -> list[str]:
    schedule = [category for category, count in counts.items() for _ in range(count)]
    rng.shuffle(schedule)
    return schedule


def record_seed(base_seed: int, record_index: int) -> int:
    return (base_seed * 1_000_037) + record_index


def make_l5_record(
    category: str,
    perfumes: list[Perfume],
    indexes: dict[str, list[str]],
    config: DatasetConfig,
    rng: random.Random,
    qgen: QueryGenerator,
) -> dict[str, Any]:
    use_rag = rng.random() < config.rag_ratio
    question, answer, context_perfumes, meta = _make_sample(category, perfumes, indexes, use_rag, rng, qgen)
    user_profile_context = build_user_profile_context(meta["user_profile_obj"])
    perfume_context = build_perfume_context(context_perfumes) or ""
    context = user_profile_context + perfume_context
    debug_meta = {
        "level": "L5",
        "category": category,
        "rag": bool(perfume_context),
        "profile": profile_to_debug(meta["semantic_profile"]),
        "user_profile": preference_to_debug(meta["user_profile_obj"]),
        "answer_ids": [p.get("id") for p in meta["selected"]],
        "best_pick_id": meta["selected"][0].get("id"),
        "query_source": qgen.last_source,
        "conflict": meta.get("conflict", False),
        "profile_update": meta.get("profile_update", False),
    }
    return build_messages(
        question,
        answer,
        context,
        debug_meta,
        config.include_debug_meta,
        system_prompt=L5_SYSTEM_PROMPT,
    )


def build_generation_indexes(perfumes: list[Perfume]) -> dict[str, list[str]]:
    accord_counter: Counter[str] = Counter()
    note_counter: Counter[str] = Counter()
    for perfume in perfumes:
        meta = perfume["metadata"]
        for accord in meta.get("accords_list") or []:
            accord_counter[accord.lower()] += 1
        for note in meta.get("notes_list") or []:
            note_counter[note.lower()] += 1
    return {
        "accords": [item for item, _count in accord_counter.most_common(60)],
        "notes": [item for item, _count in note_counter.most_common(90)],
        "avoid_common": ["sweet", "vanilla", "rose", "oud", "smoky", "leather", "tobacco", "powdery", "coconut"],
    }


def _make_sample(
    category: str,
    perfumes: list[Perfume],
    indexes: dict[str, list[str]],
    use_rag: bool,
    rng: random.Random,
    qgen: QueryGenerator,
) -> tuple[str, str, list[Perfume], dict[str, Any]]:
    makers = {
        "empty_profile": make_empty_profile,
        "profile_likes": make_profile_likes,
        "profile_dislikes": make_profile_dislikes,
        "profile_likes_and_dislikes": make_profile_likes_and_dislikes,
        "avoid_previous_recommendations": make_avoid_previous_recommendations,
        "profile_query_conflict": make_profile_query_conflict,
        "profile_update_request": make_profile_update_request,
        "low_confidence_profile": make_low_confidence_profile,
    }
    if category not in makers:
        raise ValueError(f"Unknown L5 category: {category}")
    return makers[category](perfumes, indexes, use_rag, rng, qgen)


def make_empty_profile(perfumes, indexes, use_rag, rng, qgen):
    for _ in range(200):
        label, mapping = rng.choice(list(L5_SCENARIOS.items()))
        profile = UserPreferenceProfile(confidence="none")
        semantic = _semantic_from_profile(label, mapping, profile)
        sample = _sample_l5_answer(perfumes, semantic, profile, use_rag, rng)
        if sample:
            question = qgen.generate("empty_profile", semantic, rng)
            answer = format_l5_answer(sample[2], semantic, profile, "empty_profile", rng)
            return question, answer, sample[1], _meta(semantic, profile, sample[2], qgen)
    raise RuntimeError("Could not generate L5 empty_profile sample")


def make_profile_likes(perfumes, indexes, use_rag, rng, qgen):
    for _ in range(250):
        liked = _pick_reference_perfume(perfumes, rng)
        if not liked:
            continue
        label, mapping = rng.choice(list(L5_SCENARIOS.items()))
        profile = _profile_from_liked(liked, rng)
        semantic = _semantic_from_profile(label, mapping, profile)
        sample = _sample_l5_answer(perfumes, semantic, profile, use_rag, rng, exclude_ids={liked.get("id")})
        if sample:
            question = qgen.generate("profile_likes", semantic, rng)
            answer = format_l5_answer(sample[2], semantic, profile, "profile_likes", rng)
            return question, answer, sample[1], _meta(semantic, profile, sample[2], qgen)
    raise RuntimeError("Could not generate L5 profile_likes sample")


def make_profile_dislikes(perfumes, indexes, use_rag, rng, qgen):
    for _ in range(250):
        label, mapping = rng.choice(list(L5_SCENARIOS.items()))
        disliked_term = rng.choice(indexes["avoid_common"])
        profile = UserPreferenceProfile(
            disliked_notes=(disliked_term,),
            disliked_accords=(disliked_term,),
            confidence="high",
        )
        semantic = _semantic_from_profile(label, mapping, profile, hard_avoid=True)
        sample = _sample_l5_answer(perfumes, semantic, profile, use_rag, rng)
        if sample:
            question = qgen.generate("profile_dislikes", semantic, rng)
            answer = format_l5_answer(sample[2], semantic, profile, "profile_dislikes", rng)
            return question, answer, sample[1], _meta(semantic, profile, sample[2], qgen)
    raise RuntimeError("Could not generate L5 profile_dislikes sample")


def make_profile_likes_and_dislikes(perfumes, indexes, use_rag, rng, qgen):
    for _ in range(300):
        liked = _pick_reference_perfume(perfumes, rng)
        if not liked:
            continue
        label, mapping = rng.choice(list(L5_SCENARIOS.items()))
        avoid = _pick_absent_term(liked, indexes["avoid_common"], rng)
        profile = _profile_from_liked(
            liked,
            rng,
            disliked_notes=(avoid,),
            disliked_accords=(avoid,),
            confidence="high",
        )
        semantic = _semantic_from_profile(label, mapping, profile, hard_avoid=True)
        sample = _sample_l5_answer(perfumes, semantic, profile, use_rag, rng, exclude_ids={liked.get("id")})
        if sample:
            question = qgen.generate("profile_likes_and_dislikes", semantic, rng)
            answer = format_l5_answer(sample[2], semantic, profile, "profile_likes_and_dislikes", rng)
            return question, answer, sample[1], _meta(semantic, profile, sample[2], qgen)
    raise RuntimeError("Could not generate L5 profile_likes_and_dislikes sample")


def make_avoid_previous_recommendations(perfumes, indexes, use_rag, rng, qgen):
    for _ in range(300):
        label, mapping = rng.choice(list(L5_SCENARIOS.items()))
        previous = find_semantic_matches(perfumes, _semantic_from_profile(label, mapping, UserPreferenceProfile()), limit=3)
        if len(previous) < 2:
            continue
        profile = UserPreferenceProfile(previously_recommended=tuple(previous), confidence="medium")
        semantic = _semantic_from_profile(label, mapping, profile)
        sample = _sample_l5_answer(perfumes, semantic, profile, use_rag, rng, exclude_ids={p.get("id") for p in previous})
        if sample:
            question = qgen.generate("avoid_previous_recommendations", semantic, rng)
            answer = format_l5_answer(sample[2], semantic, profile, "avoid_previous_recommendations", rng)
            return question, answer, sample[1], _meta(semantic, profile, sample[2], qgen)
    raise RuntimeError("Could not generate L5 avoid_previous_recommendations sample")


def make_profile_query_conflict(perfumes, indexes, use_rag, rng, qgen):
    conflict_profiles = [
        ("cozy but controlled", VIBE_PROFILES["cozy"], "sweet"),
        ("soft rose evening scent", VIBE_PROFILES["elegant"], "rose"),
        ("dark winter scent", VIBE_PROFILES["dark"], "smoky"),
    ]
    for _ in range(300):
        label, mapping, disliked = rng.choice(conflict_profiles)
        profile = UserPreferenceProfile(
            disliked_notes=(disliked,),
            disliked_accords=(disliked,),
            confidence="high",
            notes=("Current request partially conflicts with stored dislikes.",),
        )
        semantic = _semantic_from_profile(label, mapping, profile, hard_avoid=True)
        sample = _sample_l5_answer(perfumes, semantic, profile, use_rag, rng)
        if sample:
            question = qgen.generate("profile_query_conflict", semantic, rng)
            answer = format_l5_answer(sample[2], semantic, profile, "profile_query_conflict", rng)
            return question, answer, sample[1], _meta(semantic, profile, sample[2], qgen, conflict=True)
    raise RuntimeError("Could not generate L5 profile_query_conflict sample")


def make_profile_update_request(perfumes, indexes, use_rag, rng, qgen):
    for _ in range(300):
        updated_like = rng.choice(["rose", "vanilla", "leather"])
        mapping = {
            "soft_accords": (updated_like, "soft spicy", "musky"),
            "soft_notes": (updated_like,),
        }
        label = f"soft {updated_like} scent"
        profile = UserPreferenceProfile(
            disliked_notes=(updated_like,),
            disliked_accords=(updated_like,),
            confidence="medium",
            notes=(f"User says their old dislike of {updated_like} may have changed.",),
            extra={"updated_preference": updated_like},
        )
        semantic = _semantic_from_profile(label, mapping, profile, ignore_dislikes=True)
        sample = _sample_l5_answer(perfumes, semantic, profile, use_rag, rng)
        if sample:
            question = qgen.generate("profile_update_request", semantic, rng)
            answer = format_l5_answer(sample[2], semantic, profile, "profile_update_request", rng)
            return question, answer, sample[1], _meta(semantic, profile, sample[2], qgen, profile_update=True)
    raise RuntimeError("Could not generate L5 profile_update_request sample")


def make_low_confidence_profile(perfumes, indexes, use_rag, rng, qgen):
    for _ in range(250):
        liked = _pick_reference_perfume(perfumes, rng)
        if not liked:
            continue
        label, mapping = rng.choice(list(L5_SCENARIOS.items()))
        profile = _profile_from_liked(liked, rng, confidence="low")
        semantic = _semantic_from_profile(label, mapping, profile, weak_profile=True)
        sample = _sample_l5_answer(perfumes, semantic, profile, use_rag, rng, exclude_ids={liked.get("id")})
        if sample:
            question = qgen.generate("low_confidence_profile", semantic, rng)
            answer = format_l5_answer(sample[2], semantic, profile, "low_confidence_profile", rng)
            return question, answer, sample[1], _meta(semantic, profile, sample[2], qgen)
    raise RuntimeError("Could not generate L5 low_confidence_profile sample")


def _sample_l5_answer(
    perfumes: list[Perfume],
    semantic: SemanticProfile,
    profile: UserPreferenceProfile,
    use_rag: bool,
    rng: random.Random,
    exclude_ids: set[Any] | None = None,
) -> tuple[str, list[Perfume], list[Perfume]] | None:
    exclude_ids = exclude_ids or set()
    matches = [
        p
        for p in find_semantic_matches(perfumes, semantic, limit=40, min_score=MIN_SCORE)
        if p.get("id") not in exclude_ids and p.get("id") not in {d.get("id") for d in profile.disliked_perfumes}
    ]
    if len(matches) < 2:
        return None
    selected = matches[:ANSWER_LIMIT]
    if not use_rag:
        return "", [], selected

    context = _build_l5_context(selected, perfumes, semantic, profile, rng, exclude_ids)
    final = _rank_pool([p for p in context if p.get("id") not in exclude_ids], semantic, ANSWER_LIMIT)
    if len(final) < 2:
        return None
    return "", context, final


def format_l5_answer(
    selected: list[Perfume],
    semantic: SemanticProfile,
    profile: UserPreferenceProfile,
    category: str,
    rng: random.Random,
) -> str:
    lines = [_answer_intro(semantic, profile, category, rng), ""]
    for index, perfume in enumerate(selected, 1):
        lines.append(f"{index}. {perfume['name']} by {perfume['brand']}")
        lines.append(f"Why: {_why_sentence(perfume, semantic, profile, category, rng)}")
        lines.append("")
    best = selected[0]
    lines.append(f"Best pick: {best['name']} by {best['brand']}")
    return "\n".join(lines).strip()


def _answer_intro(semantic: SemanticProfile, profile: UserPreferenceProfile, category: str, rng: random.Random) -> str:
    label = semantic.label.replace("_", " ")
    if category == "empty_profile":
        return rng.choice(
            [
                f"Your profile does not add much yet, so I would choose mainly for {label}.",
                f"Since there are no stable preferences recorded, I would keep the recommendation focused on {label}.",
            ]
        )
    if category == "profile_query_conflict":
        return rng.choice(
            [
                f"This partly conflicts with your stored dislikes, so I would choose safer options for {label}.",
                f"I would treat this as a compromise: respect the new request while keeping your dislikes in mind.",
            ]
        )
    if category == "profile_update_request":
        return rng.choice(
            [
                f"Since your current message updates an older preference, I would follow the new direction carefully.",
                f"I would treat your latest request as more important than the older dislike, but still keep the pick gentle.",
            ]
        )
    if category == "avoid_previous_recommendations":
        return f"I would avoid repeating previous recommendations and look for a fresh option for {label}."
    if category == "low_confidence_profile":
        return f"Your profile is still low-confidence, so I would use it lightly and prioritize {label}."
    if profile.disliked_notes or profile.disliked_accords:
        return f"Using your profile, I would aim for {label} while avoiding your known dislikes."
    return f"Using your profile, I would prioritize options that connect with your known taste for {label}."


def _why_sentence(
    perfume: Perfume,
    semantic: SemanticProfile,
    profile: UserPreferenceProfile,
    category: str,
    rng: random.Random,
) -> str:
    evidence = []
    accord_matches = _matched_terms(parse_card_accords(perfume.get("card_text", "")), semantic)
    note_matches = _matched_terms(parse_card_notes(perfume.get("card_text", "")), semantic, notes=True)
    if accord_matches:
        evidence.append(f"the {', '.join(accord_matches[:3])} accords")
    if note_matches:
        evidence.append(f"notes like {', '.join(note_matches[:3])}")

    meta = perfume["metadata"]
    season = _matched_any(meta.get("best_seasons") or [], semantic.hard_seasons)
    time = _matched_any(meta.get("time_profile") or [], semantic.hard_times)
    if season and time:
        evidence.append(f"{season}/{time} suitability")
    elif season:
        evidence.append(f"{season} suitability")
    elif time:
        evidence.append(f"a {time} wear profile")

    avoid = _avoidance_support(perfume, profile)
    if avoid:
        evidence.append(avoid)

    if not evidence:
        evidence.append("the overall fit between the profile, request, and perfume card")

    joined = _join_evidence(evidence[:3])
    if category == "empty_profile":
        return f"It fits the current request through {joined}."
    if category == "low_confidence_profile":
        return f"It is a cautious fit because of {joined}, without assuming too much about your taste."
    if category == "profile_query_conflict":
        return f"It is a safer compromise because of {joined}."
    if category == "profile_update_request":
        return f"It follows your updated direction through {joined}."
    if category == "avoid_previous_recommendations":
        return f"It gives you a new option through {joined}."
    return rng.choice(
        [
            f"It connects your profile to the request through {joined}.",
            f"It makes sense for your stored taste because of {joined}.",
            f"I would include it because {joined}.",
        ]
    )


def _build_l5_context(
    selected: list[Perfume],
    perfumes: list[Perfume],
    semantic: SemanticProfile,
    profile: UserPreferenceProfile,
    rng: random.Random,
    exclude_ids: set[Any],
    context_size: int = 12,
) -> list[Perfume]:
    context = list(selected)
    seen = {p.get("id") for p in context} | exclude_ids
    disliked_ids = {p.get("id") for p in profile.disliked_perfumes}
    seen |= disliked_ids

    for perfume in rng.sample(perfumes, min(1400, len(perfumes))):
        if perfume.get("id") in seen:
            continue
        if score_perfume(perfume, semantic) < MIN_SCORE:
            context.append(perfume)
            seen.add(perfume.get("id"))
        if len(context) >= 7:
            break
    if len(context) < context_size:
        fillers = [p for p in rng.sample(perfumes, min(700, len(perfumes))) if p.get("id") not in seen]
        context.extend(fillers[: context_size - len(context)])
    rng.shuffle(context)
    return context[:context_size]


def _rank_pool(pool: list[Perfume], semantic: SemanticProfile, limit: int) -> list[Perfume]:
    scored = [(score_perfume(perfume, semantic), perfume) for perfume in pool]
    scored.sort(key=lambda item: (item[0], item[1]["metadata"].get("popularity", 0) or 0), reverse=True)
    return [perfume for _score, perfume in scored[:limit]]


def _semantic_from_profile(
    label: str,
    mapping: dict[str, tuple[str, ...]],
    profile: UserPreferenceProfile,
    hard_avoid: bool = False,
    soft_avoid: bool = False,
    ignore_dislikes: bool = False,
    weak_profile: bool = False,
) -> SemanticProfile:
    liked_weight = 1 if weak_profile else 2
    liked_accords = profile.liked_accords[:liked_weight]
    liked_notes = profile.liked_notes[:liked_weight]
    avoid_accords = () if ignore_dislikes or not hard_avoid else profile.disliked_accords
    avoid_notes = () if ignore_dislikes or not hard_avoid else profile.disliked_notes
    soft_avoid_accords = profile.disliked_accords if soft_avoid or (not hard_avoid and not ignore_dislikes) else ()
    soft_avoid_notes = profile.disliked_notes if soft_avoid or (not hard_avoid and not ignore_dislikes) else ()
    return SemanticProfile(
        label=label,
        gender_any_of=profile.preferred_gender_any_of,
        hard_seasons=profile.preferred_seasons or mapping.get("hard_seasons", ()),
        hard_times=profile.preferred_times or mapping.get("hard_times", ()),
        wanted_accords=mapping.get("wanted_accords", ()),
        wanted_notes=mapping.get("wanted_notes", ()),
        soft_accords=mapping.get("soft_accords", ()) + liked_accords,
        soft_notes=mapping.get("soft_notes", ()) + liked_notes,
        avoid_accords=avoid_accords,
        avoid_notes=avoid_notes,
        soft_avoid_accords=mapping.get("soft_avoid_accords", ()) + soft_avoid_accords,
        soft_avoid_notes=soft_avoid_notes,
    )


def _profile_from_liked(
    liked: Perfume,
    rng: random.Random,
    disliked_notes: tuple[str, ...] = (),
    disliked_accords: tuple[str, ...] = (),
    confidence: str = "medium",
) -> UserPreferenceProfile:
    meta = liked["metadata"]
    liked_accords = tuple((meta.get("accords_list") or [])[:3])
    liked_notes = tuple((meta.get("notes_list") or [])[:3])
    gender = _gender_any_of_from_perfume(meta, rng)
    return UserPreferenceProfile(
        liked_perfumes=(liked,),
        liked_notes=liked_notes,
        liked_accords=liked_accords,
        disliked_notes=disliked_notes,
        disliked_accords=disliked_accords,
        preferred_gender_any_of=gender,
        confidence=confidence,
    )


def _pick_reference_perfume(perfumes: list[Perfume], rng: random.Random) -> Perfume | None:
    for _ in range(100):
        perfume = rng.choice(perfumes)
        meta = perfume["metadata"]
        if (meta.get("accords_list") or []) and (meta.get("notes_list") or []):
            return perfume
    return None


def _pick_absent_term(perfume: Perfume, terms: list[str], rng: random.Random) -> str:
    meta = perfume["metadata"]
    available = meta["_accords_set"] | meta["_notes_set"]
    candidates = [term for term in terms if term not in available]
    return rng.choice(candidates or terms)


def _gender_any_of_from_perfume(meta: dict[str, Any], rng: random.Random) -> tuple[str, ...]:
    gender = (meta.get("gender") or "").lower()
    if gender == "male":
        return ("male", "unisex")
    if gender == "female":
        return ("female", "unisex")
    if gender == "unisex" and rng.random() < 0.8:
        return ("unisex",)
    return ()


def _matched_terms(terms: list[str], semantic: SemanticProfile, notes: bool = False) -> list[str]:
    wanted = semantic.wanted_notes + semantic.soft_notes if notes else semantic.wanted_accords + semantic.soft_accords
    wanted_set = {term.lower() for term in wanted}
    matches = []
    seen = set()
    for term in terms:
        key = term.lower()
        if key in wanted_set and key not in seen:
            seen.add(key)
            matches.append(term)
    return matches


def _matched_any(available: list[str], wanted: tuple[str, ...]) -> str:
    wanted_set = {item.lower() for item in wanted}
    for item in available:
        if item.lower() in wanted_set:
            return item.lower()
    return ""


def _avoidance_support(perfume: Perfume, profile: UserPreferenceProfile) -> str:
    avoid_terms = _unique_terms(profile.disliked_accords + profile.disliked_notes)
    if not avoid_terms:
        return ""
    meta = perfume["metadata"]
    available = meta["_accords_set"] | meta["_notes_set"]
    if any(term.lower() in available for term in avoid_terms):
        return ""
    return f"no listed {avoid_terms[0]} note or accord"


def _unique_terms(items: tuple[str, ...]) -> list[str]:
    seen = set()
    unique = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _join_evidence(items: list[str]) -> str:
    if len(items) <= 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{items[0]}, {items[1]}, and {items[2]}"


def _meta(
    semantic: SemanticProfile,
    profile: UserPreferenceProfile,
    selected: list[Perfume],
    qgen: QueryGenerator,
    conflict: bool = False,
    profile_update: bool = False,
) -> dict[str, Any]:
    return {
        "semantic_profile": semantic,
        "user_profile_obj": profile,
        "selected": selected,
        "query_source": qgen.last_source,
        "conflict": conflict,
        "profile_update": profile_update,
    }
