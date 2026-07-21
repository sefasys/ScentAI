from __future__ import annotations

import random
from collections import Counter
from typing import Any

from research.core.cards import build_perfume_context
from research.core.config import DatasetConfig, L3_DEFAULT_RATIOS, counts_from_ratios
from research.core.data import Perfume, build_id_index
from research.core.llm_query import QueryGenerator
from research.core.messages import L3_SYSTEM_PROMPT, build_messages
from research.core.semantic import (
    CONTRADICTION_PROFILES,
    OCCASION_PROFILES,
    VIBE_PROFILES,
    SemanticProfile,
    find_semantic_matches,
    profile_to_debug,
    score_perfume,
)


ANSWER_LIMIT = 3
MIN_SCORE = 3.0


def generate_l3_records(
    perfumes: list[Perfume],
    config: DatasetConfig,
    query_generator: QueryGenerator | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(config.seed)
    counts = l3_category_counts(config)
    indexes = build_generation_indexes(perfumes)
    id_index = build_id_index(perfumes)
    qgen = query_generator or QueryGenerator()

    schedule = build_l3_schedule(counts, rng)

    records: list[dict[str, Any]] = []
    produced = {category: 0 for category in counts}

    for index, category in enumerate(schedule, 1):
        record_rng = random.Random(record_seed(config.seed, index))
        records.append(make_l3_record(category, perfumes, indexes, id_index, config, record_rng, qgen))
        produced[category] += 1

    rng.shuffle(records)
    return records, produced


def l3_category_counts(config: DatasetConfig) -> dict[str, int]:
    return config.category_counts or counts_from_ratios(config.total, L3_DEFAULT_RATIOS)


def build_l3_schedule(counts: dict[str, int], rng: random.Random) -> list[str]:
    schedule = [category for category, count in counts.items() for _ in range(count)]
    rng.shuffle(schedule)
    return schedule


def record_seed(base_seed: int, record_index: int) -> int:
    return (base_seed * 1_000_003) + record_index


def make_l3_record(
    category: str,
    perfumes: list[Perfume],
    indexes: dict[str, list[str]],
    id_index: dict[str, Perfume],
    config: DatasetConfig,
    rng: random.Random,
    qgen: QueryGenerator,
) -> dict[str, Any]:
    use_rag = rng.random() < config.rag_ratio
    question, answer, context_perfumes, meta = _make_sample(
        category,
        perfumes,
        indexes,
        id_index,
        use_rag,
        rng,
        qgen,
    )
    context = build_perfume_context(context_perfumes)
    debug_meta = {"level": "L3", "category": category, "rag": bool(context), **meta}
    return build_messages(
        question,
        answer,
        context,
        debug_meta,
        config.include_debug_meta,
        system_prompt=L3_SYSTEM_PROMPT,
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
        "avoid_common": [
            "tobacco",
            "leather",
            "oud",
            "patchouli",
            "rose",
            "vanilla",
            "coconut",
            "powdery",
            "smoky",
            "animalic",
            "sweet",
        ],
    }


def _make_sample(
    category: str,
    perfumes: list[Perfume],
    indexes: dict[str, list[str]],
    id_index: dict[str, Perfume],
    use_rag: bool,
    rng: random.Random,
    qgen: QueryGenerator,
) -> tuple[str, str, list[Perfume], dict[str, Any]]:
    makers = {
        "casual_vibe": make_casual_vibe,
        "occasion": make_occasion,
        "likes_dislikes": make_likes_dislikes,
        "negative_preference": make_negative_preference,
        "reference_similarity": make_reference_similarity,
        "messy_query": make_messy_query,
        "conceptual_contradiction": make_conceptual_contradiction,
    }
    if category not in makers:
        raise ValueError(f"Unknown L3 category: {category}")
    return makers[category](perfumes, indexes, id_index, use_rag, rng, qgen)


def make_casual_vibe(perfumes, indexes, id_index, use_rag, rng, qgen):
    for _ in range(200):
        vibe = rng.choice(list(VIBE_PROFILES.keys()))
        gender = _maybe_gender_any_of(rng)
        profile = _profile_from_mapping(vibe, VIBE_PROFILES[vibe], gender_any_of=gender)
        sample = _sample_semantic_answer(perfumes, profile, use_rag, rng)
        if sample:
            question = qgen.generate("casual_vibe", profile, rng)
            return question, sample[0], sample[1], _meta(profile, sample[2], qgen)
    raise RuntimeError("Could not generate L3 casual_vibe sample")


def make_occasion(perfumes, indexes, id_index, use_rag, rng, qgen):
    for _ in range(200):
        occasion = rng.choice(list(OCCASION_PROFILES.keys()))
        gender = _maybe_gender_any_of(rng)
        profile = _profile_from_mapping(occasion, OCCASION_PROFILES[occasion], gender_any_of=gender)
        sample = _sample_semantic_answer(perfumes, profile, use_rag, rng)
        if sample:
            question = qgen.generate("occasion", profile, rng)
            return question, sample[0], sample[1], _meta(profile, sample[2], qgen)
    raise RuntimeError("Could not generate L3 occasion sample")


def make_likes_dislikes(perfumes, indexes, id_index, use_rag, rng, qgen):
    for _ in range(300):
        ref = rng.choice(perfumes)
        meta = ref["metadata"]
        accords = meta.get("accords_list") or []
        notes = meta.get("notes_list") or []
        if not accords and not notes:
            continue

        liked_accords = tuple(a.lower() for a in accords[:2])
        liked_notes = tuple(n.lower() for n in notes[:2])
        avoid = _pick_absent_terms(meta, indexes["avoid_common"], rng, count=1)
        if not avoid:
            continue

        profile = SemanticProfile(
            label="likes and dislikes",
            gender_any_of=_gender_any_of_from_perfume(meta, rng),
            wanted_accords=liked_accords[:1],
            wanted_notes=liked_notes[:1],
            soft_accords=liked_accords[1:],
            soft_notes=liked_notes[1:],
            avoid_accords=tuple(avoid),
            extra={"reference_id": ref.get("id")},
        )
        sample = _sample_semantic_answer(perfumes, profile, use_rag, rng)
        if sample:
            question = qgen.generate("likes_dislikes", profile, rng)
            return question, sample[0], sample[1], _meta(profile, sample[2], qgen)
    raise RuntimeError("Could not generate L3 likes_dislikes sample")


def make_negative_preference(perfumes, indexes, id_index, use_rag, rng, qgen):
    for _ in range(300):
        vibe = rng.choice(list(VIBE_PROFILES.keys()))
        mapping = VIBE_PROFILES[vibe]
        avoid = rng.choice(indexes["avoid_common"])
        profile = _profile_from_mapping(
            vibe,
            mapping,
            gender_any_of=_maybe_gender_any_of(rng),
            avoid_accords=(avoid,),
        )
        sample = _sample_semantic_answer(perfumes, profile, use_rag, rng)
        if sample:
            question = qgen.generate("negative_preference", profile, rng)
            return question, sample[0], sample[1], _meta(profile, sample[2], qgen)
    raise RuntimeError("Could not generate L3 negative_preference sample")


def make_reference_similarity(perfumes, indexes, id_index, use_rag, rng, qgen):
    for _ in range(300):
        ref = rng.choice(perfumes)
        similar_items = ref.get("similar", {}).get("reminds_me_of", [])
        similar = [id_index[item["id"]] for item in similar_items[:8] if item.get("id") in id_index]

        meta = ref["metadata"]
        accords = tuple(a.lower() for a in (meta.get("accords_list") or [])[:3])
        notes = tuple(n.lower() for n in (meta.get("notes_list") or [])[:2])
        if not accords and not similar:
            continue

        profile = SemanticProfile(
            label="similar to a known perfume",
            wanted_accords=accords[:2],
            soft_accords=accords[2:],
            soft_notes=notes,
            reference_name=ref["name"],
            reference_brand=ref["brand"],
            extra={"reference_id": ref.get("id")},
        )

        if similar:
            selected = _rank_pool(similar, profile, limit=ANSWER_LIMIT)
            if len(selected) < 1:
                selected = similar[:ANSWER_LIMIT]
            context = _build_l3_context(selected, perfumes, profile, rng) if use_rag else []
            final = _rank_pool(context, profile, ANSWER_LIMIT) if use_rag else selected[:ANSWER_LIMIT]
            if not final:
                continue
            question = qgen.generate("reference_similarity", profile, rng)
            return question, format_l3_answer(final), context, _meta(profile, final, qgen)

        sample = _sample_semantic_answer(perfumes, profile, use_rag, rng)
        if sample:
            question = qgen.generate("reference_similarity", profile, rng)
            return question, sample[0], sample[1], _meta(profile, sample[2], qgen)
    raise RuntimeError("Could not generate L3 reference_similarity sample")


def make_messy_query(perfumes, indexes, id_index, use_rag, rng, qgen):
    for _ in range(300):
        ref = rng.choice(perfumes)
        meta = ref["metadata"]
        accords = tuple(a.lower() for a in (meta.get("accords_list") or [])[:3])
        if len(accords) < 2:
            continue
        profile = SemanticProfile(
            label=" ".join(accords[:2]),
            gender_any_of=_gender_any_of_from_perfume(meta, rng),
            wanted_accords=accords[:1],
            soft_accords=accords[1:],
            extra={"reference_id": ref.get("id")},
        )
        sample = _sample_semantic_answer(perfumes, profile, use_rag, rng)
        if sample:
            question = qgen.generate("messy_query", profile, rng, messy=True)
            return question, sample[0], sample[1], _meta(profile, sample[2], qgen)
    raise RuntimeError("Could not generate L3 messy_query sample")


def make_conceptual_contradiction(perfumes, indexes, id_index, use_rag, rng, qgen):
    for _ in range(200):
        concept = rng.choice(list(CONTRADICTION_PROFILES.keys()))
        profile = _profile_from_mapping(
            concept,
            CONTRADICTION_PROFILES[concept],
            gender_any_of=_maybe_gender_any_of(rng),
        )
        sample = _sample_semantic_answer(perfumes, profile, use_rag, rng)
        if sample:
            question = qgen.generate("conceptual_contradiction", profile, rng)
            return question, sample[0], sample[1], _meta(profile, sample[2], qgen)
    raise RuntimeError("Could not generate L3 conceptual_contradiction sample")


def _sample_semantic_answer(
    perfumes: list[Perfume],
    profile: SemanticProfile,
    use_rag: bool,
    rng: random.Random,
) -> tuple[str, list[Perfume], list[Perfume]] | None:
    matches = find_semantic_matches(perfumes, profile, limit=20, min_score=MIN_SCORE)
    if not matches:
        return None

    selected = matches[:ANSWER_LIMIT]
    if not use_rag:
        return format_l3_answer(selected), [], selected

    context = _build_l3_context(selected, perfumes, profile, rng)
    final = _rank_pool(context, profile, ANSWER_LIMIT)
    if not final:
        return None
    return format_l3_answer(final), context, final


def _build_l3_context(
    selected: list[Perfume],
    perfumes: list[Perfume],
    profile: SemanticProfile,
    rng: random.Random,
    context_size: int = 10,
) -> list[Perfume]:
    context = list(selected)
    seen = {id(p) for p in context}

    hard_negatives = []
    for perfume in rng.sample(perfumes, min(1000, len(perfumes))):
        if id(perfume) in seen:
            continue
        score = score_perfume(perfume, profile)
        if score < MIN_SCORE:
            hard_negatives.append(perfume)
            seen.add(id(perfume))
        if len(hard_negatives) >= 4:
            break
    context.extend(hard_negatives)

    needed = context_size - len(context)
    if needed > 0:
        fillers = [p for p in rng.sample(perfumes, min(500, len(perfumes))) if id(p) not in seen]
        context.extend(fillers[:needed])

    rng.shuffle(context)
    return context[:context_size]


def _rank_pool(pool: list[Perfume], profile: SemanticProfile, limit: int) -> list[Perfume]:
    scored = [(score_perfume(p, profile), p) for p in pool]
    scored = [(score, perfume) for score, perfume in scored if score >= MIN_SCORE]
    scored.sort(key=lambda item: (item[0], item[1]["metadata"].get("popularity", 0)), reverse=True)
    return [perfume for _score, perfume in scored[:limit]]


def format_l3_answer(perfumes: list[Perfume]) -> str:
    return "\n".join(f"{idx}. {p['name']} by {p['brand']}" for idx, p in enumerate(perfumes, 1))


def _profile_from_mapping(
    label: str,
    mapping: dict[str, tuple[str, ...]],
    gender_any_of: tuple[str, ...] = (),
    avoid_accords: tuple[str, ...] = (),
) -> SemanticProfile:
    return SemanticProfile(
        label=label,
        gender_any_of=gender_any_of,
        prefer_gender=_preferred_gender(gender_any_of),
        hard_seasons=mapping.get("hard_seasons", ()),
        hard_times=mapping.get("hard_times", ()),
        soft_accords=mapping.get("soft_accords", ()),
        soft_notes=mapping.get("soft_notes", ()),
        avoid_accords=avoid_accords,
        soft_avoid_accords=mapping.get("soft_avoid_accords", ()),
        soft_avoid_notes=mapping.get("soft_avoid_notes", ()),
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


def _gender_any_of_from_perfume(meta: dict, rng: random.Random) -> tuple[str, ...]:
    gender = (meta.get("gender") or "").lower()
    if rng.random() > 0.65:
        return ()
    if gender == "male":
        return ("male", "unisex")
    if gender == "female":
        return ("female", "unisex")
    if gender == "unisex":
        return ("unisex",)
    return ()


def _preferred_gender(gender_any_of: tuple[str, ...]) -> str | None:
    genders = set(gender_any_of)
    if genders == {"male", "unisex"}:
        return "male"
    if genders == {"female", "unisex"}:
        return "female"
    if genders == {"unisex"}:
        return "unisex"
    return None


def _pick_absent_terms(meta: dict, candidates: list[str], rng: random.Random, count: int) -> list[str]:
    available = meta["_accords_set"] | meta["_notes_set"]
    absent = [term for term in candidates if term.lower() not in available]
    if not absent:
        return []
    return rng.sample(absent, min(count, len(absent)))


def _meta(profile: SemanticProfile, answer_perfumes: list[Perfume], qgen: QueryGenerator | None = None) -> dict[str, Any]:
    return {
        "profile": profile_to_debug(profile),
        "answer_ids": [p.get("id") for p in answer_perfumes if p.get("id")],
        "query_source": qgen.last_source if qgen else "unknown",
    }
