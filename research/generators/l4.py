from __future__ import annotations

import random
from collections import Counter
from typing import Any

from research.core.cards import build_perfume_context, parse_card_accords, parse_card_notes
from research.core.config import DatasetConfig, L4_DEFAULT_RATIOS, counts_from_ratios
from research.core.data import Perfume
from research.core.llm_query import QueryGenerator
from research.core.messages import L4_SYSTEM_PROMPT, build_messages
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

STYLE_LABELS = {
    "quiet luxury": "expensive",
    "understated elegance": "elegant",
    "clean and polished": "clean",
    "dark academic": "dark",
    "soft cozy": "cozy",
    "bright and effortless": "bright",
}

UPGRADE_TARGETS = {
    "special winter evenings": "winter_evening",
    "polished office wear": "office",
    "elegant dinners": "date_night",
    "clean summer days": "summer_day",
}

OCCASION_LABELS = {
    "office": "polished office wear",
    "date_night": "date night without feeling overwhelming",
    "summer_day": "fresh summer daytime wear",
    "winter_evening": "warm winter evenings",
    "wedding": "wedding guest wear",
    "gym_after": "fresh post-gym daytime wear",
}

AVOID_TERMS = (
    "sweet",
    "vanilla",
    "rose",
    "oud",
    "smoky",
    "leather",
    "tobacco",
    "powdery",
    "coconut",
)


def generate_l4_records(
    perfumes: list[Perfume],
    config: DatasetConfig,
    query_generator: QueryGenerator | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(config.seed)
    counts = l4_category_counts(config)
    indexes = build_generation_indexes(perfumes)
    qgen = query_generator or QueryGenerator()
    schedule = build_l4_schedule(counts, rng)

    records: list[dict[str, Any]] = []
    produced = {category: 0 for category in counts}

    for index, category in enumerate(schedule, 1):
        record_rng = random.Random(record_seed(config.seed or 0, index))
        records.append(make_l4_record(category, perfumes, indexes, config, record_rng, qgen))
        produced[category] += 1

    rng.shuffle(records)
    return records, produced


def l4_category_counts(config: DatasetConfig) -> dict[str, int]:
    return config.category_counts or counts_from_ratios(config.total, L4_DEFAULT_RATIOS)


def build_l4_schedule(counts: dict[str, int], rng: random.Random) -> list[str]:
    schedule = [category for category, count in counts.items() for _ in range(count)]
    rng.shuffle(schedule)
    return schedule


def record_seed(base_seed: int, record_index: int) -> int:
    return (base_seed * 1_000_033) + record_index


def make_l4_record(
    category: str,
    perfumes: list[Perfume],
    indexes: dict[str, list[str]],
    config: DatasetConfig,
    rng: random.Random,
    qgen: QueryGenerator,
) -> dict[str, Any]:
    use_rag = rng.random() < config.rag_ratio
    question, answer, context_perfumes, meta = _make_sample(category, perfumes, indexes, use_rag, rng, qgen)
    context = build_perfume_context(context_perfumes)
    debug_meta = {"level": "L4", "category": category, "rag": bool(context), **meta}
    return build_messages(
        question,
        answer,
        context,
        debug_meta,
        config.include_debug_meta,
        system_prompt=L4_SYSTEM_PROMPT,
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
        "accords": [item for item, _count in accord_counter.most_common(50)],
        "notes": [item for item, _count in note_counter.most_common(80)],
        "avoid_common": list(AVOID_TERMS),
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
        "persona_upgrade": make_persona_upgrade,
        "occasion_with_tradeoffs": make_occasion_with_tradeoffs,
        "compare_and_decide": make_compare_and_decide,
        "avoidance_reasoning": make_avoidance_reasoning,
        "style_translation": make_style_translation,
        "collection_gap": make_collection_gap,
        "no_strong_match": make_no_strong_match,
    }
    if category not in makers:
        raise ValueError(f"Unknown L4 category: {category}")
    return makers[category](perfumes, indexes, use_rag, rng, qgen)


def make_persona_upgrade(perfumes, indexes, use_rag, rng, qgen):
    for _ in range(250):
        ref = rng.choice(perfumes)
        target_label, occasion_key = rng.choice(list(UPGRADE_TARGETS.items()))
        profile = _profile_from_mapping(
            target_label,
            OCCASION_PROFILES[occasion_key],
            gender_any_of=_gender_any_of_from_perfume(ref["metadata"], rng),
            reference_name=ref["name"],
            reference_brand=ref["brand"],
            extra={"reference_id": ref.get("id")},
        )
        sample = _sample_l4_answer(perfumes, profile, use_rag, rng, "persona_upgrade")
        if sample:
            question = qgen.generate("persona_upgrade", profile, rng)
            return question, sample[0], sample[1], _meta(profile, sample[2], qgen, best_pick=sample[2][0])
    raise RuntimeError("Could not generate L4 persona_upgrade sample")


def make_occasion_with_tradeoffs(perfumes, indexes, use_rag, rng, qgen):
    labels = {
        "summer office confidence": "summer_day",
        "wedding guest but not loud": "wedding",
        "date night that is not overwhelming": "date_night",
        "fresh after-gym but still adult": "gym_after",
        "winter evening with warmth and polish": "winter_evening",
    }
    for _ in range(250):
        label, occasion_key = rng.choice(list(labels.items()))
        profile = _profile_from_mapping(
            label,
            OCCASION_PROFILES[occasion_key],
            gender_any_of=_maybe_gender_any_of(rng),
        )
        sample = _sample_l4_answer(perfumes, profile, use_rag, rng, "occasion_with_tradeoffs")
        if sample:
            question = qgen.generate("occasion_with_tradeoffs", profile, rng)
            return question, sample[0], sample[1], _meta(profile, sample[2], qgen, best_pick=sample[2][0])
    raise RuntimeError("Could not generate L4 occasion_with_tradeoffs sample")


def make_compare_and_decide(perfumes, indexes, use_rag, rng, qgen):
    for _ in range(250):
        occasion_key = rng.choice(list(OCCASION_PROFILES))
        label = OCCASION_LABELS.get(occasion_key, occasion_key.replace("_", " "))
        profile = _profile_from_mapping(label, OCCASION_PROFILES[occasion_key], gender_any_of=_maybe_gender_any_of(rng))
        sample = _sample_l4_answer(perfumes, profile, True, rng, "compare_and_decide")
        if not sample or len(sample[2]) < 3:
            continue
        options = [f"{p['name']} by {p['brand']}" for p in sample[2]]
        profile = _copy_profile(profile, extra={**profile.extra, "options": tuple(options)})
        question = qgen.generate("compare_and_decide", profile, rng)
        answer = format_l4_answer(sample[2], profile, "compare_and_decide", rng, no_strong_match=False, compare=True)
        context = sample[1] if use_rag else []
        return question, answer, context, _meta(profile, sample[2], qgen, best_pick=sample[2][0])
    raise RuntimeError("Could not generate L4 compare_and_decide sample")


def make_avoidance_reasoning(perfumes, indexes, use_rag, rng, qgen):
    for _ in range(300):
        vibe = rng.choice(list(VIBE_PROFILES.keys()))
        avoid = rng.choice(indexes["avoid_common"])
        profile = _profile_from_mapping(
            vibe,
            VIBE_PROFILES[vibe],
            gender_any_of=_maybe_gender_any_of(rng),
            avoid_accords=(avoid,),
            avoid_notes=(avoid,),
        )
        sample = _sample_l4_answer(perfumes, profile, use_rag, rng, "avoidance_reasoning")
        if sample:
            question = qgen.generate("avoidance_reasoning", profile, rng)
            return question, sample[0], sample[1], _meta(profile, sample[2], qgen, best_pick=sample[2][0])
    raise RuntimeError("Could not generate L4 avoidance_reasoning sample")


def make_style_translation(perfumes, indexes, use_rag, rng, qgen):
    for _ in range(250):
        style_label, vibe = rng.choice(list(STYLE_LABELS.items()))
        profile = _profile_from_mapping(style_label, VIBE_PROFILES[vibe], gender_any_of=_maybe_gender_any_of(rng))
        sample = _sample_l4_answer(perfumes, profile, use_rag, rng, "style_translation")
        if sample:
            question = qgen.generate("style_translation", profile, rng)
            return question, sample[0], sample[1], _meta(profile, sample[2], qgen, best_pick=sample[2][0])
    raise RuntimeError("Could not generate L4 style_translation sample")


def make_collection_gap(perfumes, indexes, use_rag, rng, qgen):
    for _ in range(300):
        owned = rng.sample(perfumes, 3)
        label, occasion_key = rng.choice(list(UPGRADE_TARGETS.items()))
        profile = _profile_from_mapping(
            label,
            OCCASION_PROFILES[occasion_key],
            gender_any_of=_maybe_gender_any_of(rng),
            extra={
                "collection": tuple(f"{p['name']} by {p['brand']}" for p in owned),
                "collection_ids": tuple(p.get("id") for p in owned),
            },
        )
        sample = _sample_l4_answer(
            perfumes,
            profile,
            use_rag,
            rng,
            "collection_gap",
            exclude_ids={p.get("id") for p in owned},
        )
        if sample:
            question = qgen.generate("collection_gap", profile, rng)
            return question, sample[0], sample[1], _meta(profile, sample[2], qgen, best_pick=sample[2][0])
    raise RuntimeError("Could not generate L4 collection_gap sample")


def make_no_strong_match(perfumes, indexes, use_rag, rng, qgen):
    for _ in range(300):
        label = rng.choice(["fresh aquatic summer", "clean office scent without sweetness", "dark smoky winter", "soft elegant floral"])
        mapping = {
            "fresh aquatic summer": {
                "hard_seasons": ("summer",),
                "hard_times": ("day",),
                "soft_accords": ("aquatic", "fresh", "citrus"),
                "avoid_accords": ("sweet",),
            },
            "clean office scent without sweetness": {
                "hard_times": ("day",),
                "soft_accords": ("clean", "fresh", "musky"),
                "avoid_accords": ("sweet", "vanilla"),
            },
            "dark smoky winter": {
                "hard_seasons": ("winter",),
                "hard_times": ("night",),
                "soft_accords": ("smoky", "dark", "woody"),
            },
            "soft elegant floral": {
                "soft_accords": ("floral", "powdery", "musky"),
                "soft_notes": ("rose", "iris", "jasmine"),
                "avoid_accords": ("oud", "smoky"),
            },
        }[label]
        profile = _profile_from_mapping(label, mapping, gender_any_of=_maybe_gender_any_of(rng))
        context = _build_weak_context(perfumes, profile, rng)
        if len(context) < 6:
            continue
        closest = _rank_pool(context, profile, limit=2)
        if not closest:
            continue
        profile = _copy_profile(profile, extra={**profile.extra, "options": tuple(f"{p['name']} by {p['brand']}" for p in context[:4])})
        question = qgen.generate("no_strong_match", profile, rng)
        answer = format_l4_answer(closest, profile, "no_strong_match", rng, no_strong_match=True, compare=False)
        return question, answer, context, _meta(profile, closest, qgen, best_pick=closest[0])
    raise RuntimeError("Could not generate L4 no_strong_match sample")


def _sample_l4_answer(
    perfumes: list[Perfume],
    profile: SemanticProfile,
    use_rag: bool,
    rng: random.Random,
    category: str,
    exclude_ids: set[Any] | None = None,
) -> tuple[str, list[Perfume], list[Perfume]] | None:
    exclude_ids = exclude_ids or set()
    matches = [p for p in find_semantic_matches(perfumes, profile, limit=30, min_score=MIN_SCORE) if p.get("id") not in exclude_ids]
    if len(matches) < 2:
        return None

    selected = matches[:ANSWER_LIMIT]
    if not use_rag:
        return format_l4_answer(selected, profile, category, rng), [], selected

    context = _build_l4_context(selected, perfumes, profile, rng)
    final = _rank_pool([p for p in context if p.get("id") not in exclude_ids], profile, ANSWER_LIMIT)
    if len(final) < 2:
        return None
    return format_l4_answer(final, profile, category, rng), context, final


def format_l4_answer(
    selected: list[Perfume],
    profile: SemanticProfile,
    category: str,
    rng: random.Random,
    no_strong_match: bool = False,
    compare: bool = False,
) -> str:
    lines = [_answer_intro(selected, profile, category, rng, no_strong_match, compare), ""]
    for index, perfume in enumerate(selected, 1):
        lines.append(f"{index}. {perfume['name']} by {perfume['brand']}")
        lines.append(f"Why: {_why_sentence(perfume, profile, category, index, rng)}")
        lines.append("")

    best = selected[0]
    lines.append(f"Best pick: {best['name']} by {best['brand']}")
    return "\n".join(lines).strip()


def _answer_intro(
    selected: list[Perfume],
    profile: SemanticProfile,
    category: str,
    rng: random.Random,
    no_strong_match: bool,
    compare: bool,
) -> str:
    label = _public_label(profile.label)
    best = f"{selected[0]['name']} by {selected[0]['brand']}"
    avoid = _avoid_text(profile)
    reference = _reference_text(profile)

    if no_strong_match:
        return rng.choice(
            [
                "I do not see a perfect match in this set, so I would treat these as the closest compromises.",
                "The available options only partially fit the brief; these are the ones I would consider first.",
                "None of these fully nails the request, but two options come closer than the rest.",
            ]
        )
    if compare:
        return rng.choice(
            [
                f"Among these options, I would put {best} first for {label}.",
                f"If I had to choose one for {label}, {best} has the strongest fit.",
                f"For this scenario, I would rank them by how directly their notes, accords, and wear profile support {label}.",
            ]
        )
    if category == "persona_upgrade":
        return rng.choice(
            [
                f"Since {reference}, I would move toward options that keep the idea familiar but make it better suited to {label}.",
                f"I would use your current reference as a starting point, then push the recommendation toward {label}.",
                f"The safest upgrade path is to keep the style coherent while choosing something better suited to {label}.",
            ]
        )
    if category == "occasion_with_tradeoffs":
        return rng.choice(
            [
                f"For {label}, the balance is enough character to feel intentional without becoming hard to wear.",
                f"I would focus on perfumes that suit {label} while staying controlled and wearable.",
                f"This is a tradeoff brief, so I would prioritize options that match the setting before chasing intensity.",
            ]
        )
    if category == "avoidance_reasoning":
        return rng.choice(
            [
                f"Because you want to avoid {avoid}, I would keep the picks focused on safer scent directions.",
                f"The key is to satisfy the {label} mood while steering clear of {avoid}.",
                f"I would filter this through your dislike of {avoid}, then choose the most relevant matches.",
            ]
        )
    if category == "style_translation":
        return rng.choice(
            [
                f"For a {label} style, I would translate the mood into concrete notes, accords, and wear context.",
                f"I would keep this aesthetic grounded by choosing perfumes whose materials actually support a {label} impression.",
                f"The goal is not just a nice perfume, but one whose profile clearly reads as {label}.",
            ]
        )
    if category == "collection_gap":
        return rng.choice(
            [
                f"To fill that gap, I would add something that gives your collection a clearer role for {label}.",
                f"Your next bottle should do a job the current lineup does not fully cover: {label}.",
                f"I would choose the next addition by role first, then pick the strongest match for {label}.",
            ]
        )
    return f"I would prioritize perfumes whose notes, accords, and wear profile support {label}."


def _why_sentence(perfume: Perfume, profile: SemanticProfile, category: str, index: int, rng: random.Random) -> str:
    meta = perfume["metadata"]
    evidence = []

    accord_matches = _matched_terms(parse_card_accords(perfume.get("card_text", "")), profile)
    note_matches = _matched_terms(parse_card_notes(perfume.get("card_text", "")), profile, notes=True)
    if accord_matches:
        evidence.append(f"the {', '.join(accord_matches[:3])} accords")
    if note_matches:
        evidence.append(f"notes like {', '.join(note_matches[:3])}")

    season_match = _matched_any(meta.get("best_seasons") or [], profile.hard_seasons)
    time_match = _matched_any(meta.get("time_profile") or [], profile.hard_times)
    if season_match and time_match:
        evidence.append(f"{season_match}/{time_match} suitability")
    elif season_match:
        evidence.append(f"{season_match} suitability")
    elif time_match:
        evidence.append(f"a {time_match} wear profile")

    rating = meta.get("rating")
    popularity = meta.get("popularity") or 0
    if rating and rating >= 4.0 and popularity >= 50:
        evidence.append(f"a {rating:.2f}/5 rating with enough votes to be useful")

    avoid_clause = _avoidance_support(perfume, profile)
    if avoid_clause:
        evidence.append(avoid_clause)

    if category == "no_strong_match" and _only_avoidance_evidence(evidence):
        return f"This is only a compromise pick: it avoids the main conflict, but the scent profile is still a partial match."

    if not evidence:
        evidence.append("the overall match between its metadata and the request")

    joined = _join_evidence(evidence[:3])
    starters = {
        "compare_and_decide": [
            f"It ranks here because of {joined}.",
            f"I would place it here because of {joined}.",
            f"Its advantage is {joined}.",
        ],
        "no_strong_match": [
            f"It is not perfect, but it still offers {joined}.",
            f"This is a compromise pick because {joined}.",
            f"It gets closest by offering {joined}.",
        ],
        "avoidance_reasoning": [
            f"It works for the brief because {joined}.",
            f"It keeps the recommendation on track through {joined}.",
            f"The useful part here is {joined}.",
        ],
    }
    default = [
        f"It fits the brief through {joined}.",
        f"This makes sense here because of {joined}.",
        f"I would include it for {joined}.",
        f"The fit comes from {joined}.",
        f"I would shortlist it because of {joined}.",
        f"It earns a place here with {joined}.",
    ]
    return rng.choice(starters.get(category, default))


def _build_l4_context(
    selected: list[Perfume],
    perfumes: list[Perfume],
    profile: SemanticProfile,
    rng: random.Random,
    context_size: int = 12,
) -> list[Perfume]:
    context = list(selected)
    seen = {p.get("id") for p in context}

    hard_negatives = []
    for perfume in rng.sample(perfumes, min(1200, len(perfumes))):
        if perfume.get("id") in seen:
            continue
        score = score_perfume(perfume, profile)
        if score < MIN_SCORE:
            hard_negatives.append(perfume)
            seen.add(perfume.get("id"))
        if len(hard_negatives) >= 4:
            break
    context.extend(hard_negatives)

    needed = context_size - len(context)
    if needed > 0:
        fillers = [p for p in rng.sample(perfumes, min(600, len(perfumes))) if p.get("id") not in seen]
        context.extend(fillers[:needed])

    rng.shuffle(context)
    return context[:context_size]


def _build_weak_context(perfumes: list[Perfume], profile: SemanticProfile, rng: random.Random, context_size: int = 10) -> list[Perfume]:
    pool = []
    for perfume in rng.sample(perfumes, min(3000, len(perfumes))):
        score = score_perfume(perfume, profile)
        if 0 <= score < MIN_SCORE:
            pool.append(perfume)
        if len(pool) >= context_size:
            break
    rng.shuffle(pool)
    return pool[:context_size]


def _rank_pool(pool: list[Perfume], profile: SemanticProfile, limit: int) -> list[Perfume]:
    scored = [(score_perfume(perfume, profile), perfume) for perfume in pool]
    scored.sort(key=lambda item: (item[0], item[1]["metadata"].get("popularity", 0) or 0), reverse=True)
    return [perfume for _score, perfume in scored[:limit]]


def _profile_from_mapping(
    label: str,
    mapping: dict[str, tuple[str, ...]],
    gender_any_of: tuple[str, ...] = (),
    avoid_accords: tuple[str, ...] = (),
    avoid_notes: tuple[str, ...] = (),
    reference_name: str | None = None,
    reference_brand: str | None = None,
    extra: dict[str, Any] | None = None,
) -> SemanticProfile:
    return SemanticProfile(
        label=label,
        gender_any_of=gender_any_of,
        hard_seasons=mapping.get("hard_seasons", ()),
        hard_times=mapping.get("hard_times", ()),
        wanted_accords=mapping.get("wanted_accords", ()),
        wanted_notes=mapping.get("wanted_notes", ()),
        soft_accords=mapping.get("soft_accords", ()),
        soft_notes=mapping.get("soft_notes", ()),
        avoid_accords=avoid_accords + mapping.get("avoid_accords", ()),
        avoid_notes=avoid_notes + mapping.get("avoid_notes", ()),
        soft_avoid_accords=mapping.get("soft_avoid_accords", ()),
        soft_avoid_notes=mapping.get("soft_avoid_notes", ()),
        reference_name=reference_name,
        reference_brand=reference_brand,
        extra=extra or {},
    )


def _copy_profile(profile: SemanticProfile, extra: dict[str, Any]) -> SemanticProfile:
    return SemanticProfile(
        label=profile.label,
        gender_any_of=profile.gender_any_of,
        prefer_gender=profile.prefer_gender,
        hard_seasons=profile.hard_seasons,
        hard_times=profile.hard_times,
        wanted_accords=profile.wanted_accords,
        wanted_notes=profile.wanted_notes,
        soft_accords=profile.soft_accords,
        soft_notes=profile.soft_notes,
        avoid_accords=profile.avoid_accords,
        avoid_notes=profile.avoid_notes,
        soft_avoid_accords=profile.soft_avoid_accords,
        soft_avoid_notes=profile.soft_avoid_notes,
        reference_name=profile.reference_name,
        reference_brand=profile.reference_brand,
        extra=extra,
    )


def _maybe_gender_any_of(rng: random.Random) -> tuple[str, ...]:
    roll = rng.random()
    if roll < 0.25:
        return ("male", "unisex")
    if roll < 0.50:
        return ("female", "unisex")
    if roll < 0.60:
        return ("unisex",)
    return ()


def _gender_any_of_from_perfume(meta: dict[str, Any], rng: random.Random) -> tuple[str, ...]:
    gender = (meta.get("gender") or "").lower()
    if gender == "male":
        return ("male", "unisex")
    if gender == "female":
        return ("female", "unisex")
    if gender == "unisex" and rng.random() < 0.8:
        return ("unisex",)
    return _maybe_gender_any_of(rng)


def _matched_terms(terms: list[str], profile: SemanticProfile, notes: bool = False) -> list[str]:
    wanted = profile.wanted_notes + profile.soft_notes if notes else profile.wanted_accords + profile.soft_accords
    wanted_set = {term.lower() for term in wanted}
    return [term for term in terms if term.lower() in wanted_set]


def _matched_any(available: list[str], wanted: tuple[str, ...]) -> str:
    wanted_set = {item.lower() for item in wanted}
    for item in available:
        if item.lower() in wanted_set:
            return item.lower()
    return ""


def _public_label(label: str) -> str:
    return OCCASION_LABELS.get(label, label.replace("_", " "))


def _reference_text(profile: SemanticProfile) -> str:
    if profile.reference_name and profile.reference_brand:
        return f"you already have {profile.reference_name} by {profile.reference_brand} as a reference"
    return "you already have a clear reference point"


def _avoid_text(profile: SemanticProfile) -> str:
    terms = _unique_terms(profile.avoid_accords + profile.avoid_notes)
    if not terms:
        return "the materials you dislike"
    return _join_terms(terms[:3])


def _avoidance_support(perfume: Perfume, profile: SemanticProfile) -> str:
    avoid_terms = _unique_terms(profile.avoid_accords + profile.avoid_notes)
    if not avoid_terms:
        return ""
    meta = perfume["metadata"]
    available = {term.lower() for term in (meta.get("accords_list") or [])}
    available |= {term.lower() for term in (meta.get("notes_list") or [])}
    if any(term.lower() in available for term in avoid_terms):
        return ""
    return f"no listed {avoid_terms[0]} note or accord"


def _only_avoidance_evidence(evidence: list[str]) -> bool:
    return bool(evidence) and all(item.startswith("no listed ") for item in evidence)


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


def _join_terms(items: list[str]) -> str:
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return f"{items[0]}, {items[1]}, or {items[2]}"


def _join_evidence(items: list[str]) -> str:
    if len(items) <= 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{items[0]}, {items[1]}, and {items[2]}"


def _meta(profile: SemanticProfile, selected: list[Perfume], qgen: QueryGenerator, best_pick: Perfume) -> dict[str, Any]:
    return {
        "profile": profile_to_debug(profile),
        "answer_ids": [p.get("id") for p in selected],
        "best_pick_id": best_pick.get("id"),
        "query_source": qgen.last_source,
    }
