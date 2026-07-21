from __future__ import annotations

import heapq
import random
import re
from collections import Counter
from typing import Any

from research.core.cards import build_perfume_context
from research.core.config import DatasetConfig, L2_DEFAULT_RATIOS, counts_from_ratios
from research.core.data import Perfume
from research.core.filtering import Criteria, sort_value
from research.core.formatting import fmt_list, fmt_rating, fmt_val, title_items
from research.core.messages import L2_SYSTEM_PROMPT, build_messages


SINGLE_FILTER_TEMPLATES = [
    "List perfumes with {accord} accords.",
    "Show fragrances containing {note}.",
    "Find perfumes suitable for {season}.",
    "List {gender} perfumes.",
    "Show perfumes for {time} use.",
]

MULTI_FILTER_TEMPLATES = [
    "List {gender} perfumes for {season} with {accord} accords.",
    "Find {gender} {season} perfumes containing {note}.",
    "Show {gender} fragrances for {time} use with {accord} accords.",
    "List {season} perfumes with {accord} accords and {note} notes.",
]

NEGATIVE_FILTER_TEMPLATES = [
    "List {gender} perfumes for {season} with {accord} accords but without {neg}.",
    "Find {accord} perfumes containing {note}, excluding {neg}.",
    "Show {season} fragrances with {accord} accords that do not contain {neg}.",
]

NUMERIC_FILTER_TEMPLATES = [
    "Show perfumes rated above {min_rating}.",
    "List fragrances with at least {min_votes} votes.",
    "Find {accord} perfumes rated above {min_rating} with at least {min_votes} votes.",
    "Show {gender} perfumes with longevity above {min_longevity}.",
    "List perfumes with sillage above {min_sillage}.",
]

RANKING_TEMPLATES = {
    "rating": [
        "Show the highest rated {accord} perfumes.",
        "List the top rated {gender} fragrances.",
    ],
    "popularity": [
        "Show the most popular {accord} perfumes by vote count.",
        "List the most voted {season} fragrances.",
    ],
    "longevity": [
        "Show the longest lasting {accord} perfumes.",
        "List {gender} perfumes with the highest longevity.",
    ],
    "sillage": [
        "Show {accord} perfumes with the strongest sillage.",
        "List perfumes with the highest sillage for {season}.",
    ],
}

COMPARISON_TEMPLATES = {
    "rating": "Which perfume has a higher rating, {name1} by {brand1} or {name2} by {brand2}?",
    "popularity": "Which is more popular, {name1} by {brand1} or {name2} by {brand2}?",
    "longevity": "Which has better longevity, {name1} by {brand1} or {name2} by {brand2}?",
    "sillage": "Which has stronger sillage, {name1} by {brand1} or {name2} by {brand2}?",
    "accords": "Compare the accord profiles of {name1} by {brand1} and {name2} by {brand2}.",
    "seasons": "Compare the seasonal suitability of {name1} by {brand1} and {name2} by {brand2}.",
}

INCLUSIVE_GENDER_TEMPLATES = [
    "List perfumes for men, including unisex options, with {accord} accords.",
    "Show fragrances for women, including unisex options, suitable for {season}.",
    "Find male and unisex perfumes containing {note}.",
    "Find female and unisex perfumes with {accord} accords.",
]

NO_MATCH_TEXT = "No matching perfumes found."


def generate_l2_records(perfumes: list[Perfume], config: DatasetConfig) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(config.seed)
    counts = config.category_counts or counts_from_ratios(config.total, L2_DEFAULT_RATIOS)
    indexes = build_generation_indexes(perfumes)
    schedule = [category for category, count in counts.items() for _ in range(count)]
    rng.shuffle(schedule)

    records: list[dict[str, Any]] = []
    produced = {category: 0 for category in counts}

    for category in schedule:
        use_rag = rng.random() < config.rag_ratio
        question, answer, context_perfumes, meta = _make_sample(category, perfumes, indexes, use_rag, rng)
        context = build_perfume_context(context_perfumes)
        debug_meta = {"level": "L2", "category": category, "rag": bool(context), **meta}
        records.append(
            build_messages(
                question,
                answer,
                context,
                debug_meta,
                config.include_debug_meta,
                system_prompt=L2_SYSTEM_PROMPT,
            )
        )
        produced[category] += 1

    rng.shuffle(records)
    return records, produced


def build_generation_indexes(perfumes: list[Perfume]) -> dict[str, Any]:
    accord_counter: Counter[str] = Counter()
    note_counter: Counter[str] = Counter()
    by_gender: dict[str, list[Perfume]] = {}
    by_season: dict[str, list[Perfume]] = {}
    by_time: dict[str, list[Perfume]] = {}
    by_accord: dict[str, list[Perfume]] = {}
    by_note: dict[str, list[Perfume]] = {}

    for perfume in perfumes:
        meta = perfume["metadata"]
        gender = str(meta.get("gender") or "").lower()
        if gender:
            by_gender.setdefault(gender, []).append(perfume)
        for season in meta.get("best_seasons") or []:
            by_season.setdefault(season.lower(), []).append(perfume)
        for time in meta.get("time_profile") or []:
            by_time.setdefault(time.lower(), []).append(perfume)
        for accord in meta.get("accords_list") or []:
            accord_low = accord.lower()
            accord_counter[accord_low] += 1
            by_accord.setdefault(accord_low, []).append(perfume)
        for note in meta.get("notes_list") or []:
            note_low = note.lower()
            note_counter[note_low] += 1
            by_note.setdefault(note_low, []).append(perfume)

    return {
        "accords": [item for item, _count in accord_counter.most_common(40)],
        "notes": [item for item, _count in note_counter.most_common(60)],
        "genders": ["male", "female", "unisex"],
        "seasons": ["spring", "summer", "autumn", "winter"],
        "times": ["day", "night"],
        "_by_gender": by_gender,
        "_by_season": by_season,
        "_by_time": by_time,
        "_by_accord": by_accord,
        "_by_note": by_note,
    }


def _make_sample(
    category: str,
    perfumes: list[Perfume],
    indexes: dict[str, Any],
    use_rag: bool,
    rng: random.Random,
) -> tuple[str, str, list[Perfume], dict[str, Any]]:
    makers = {
        "single_filter": make_single_filter,
        "multi_filter": make_multi_filter,
        "negative_filter": make_negative_filter,
        "numeric_filter": make_numeric_filter,
        "ranking": make_ranking,
        "comparison": make_comparison,
        "no_match": make_no_match,
        "inclusive_gender_explicit": make_inclusive_gender_explicit,
    }
    if category not in makers:
        raise ValueError(f"Unknown L2 category: {category}")
    return makers[category](perfumes, indexes, use_rag, rng)


def make_single_filter(perfumes: list[Perfume], indexes: dict[str, Any], use_rag: bool, rng: random.Random):
    for _ in range(200):
        variant = rng.choice(["accord", "note", "season", "gender", "time"])
        if variant == "accord":
            accord = rng.choice(indexes["accords"][:30])
            criteria = Criteria(accords=(accord,))
            question = SINGLE_FILTER_TEMPLATES[0].format(accord=accord)
        elif variant == "note":
            note = rng.choice(indexes["notes"][:40])
            criteria = Criteria(notes=(note,))
            question = SINGLE_FILTER_TEMPLATES[1].format(note=note)
        elif variant == "season":
            season = rng.choice(indexes["seasons"])
            criteria = Criteria(seasons=(season,))
            question = SINGLE_FILTER_TEMPLATES[2].format(season=season)
        elif variant == "gender":
            gender = rng.choice(indexes["genders"])
            criteria = Criteria(gender=gender)
            question = SINGLE_FILTER_TEMPLATES[3].format(gender=gender)
        else:
            time = rng.choice(indexes["times"])
            criteria = Criteria(time_profile=(time,))
            question = SINGLE_FILTER_TEMPLATES[4].format(time=time)

        sample = _sample_answer(perfumes, indexes, criteria, use_rag, rng)
        if sample:
            return question, sample[0], sample[1], {"criteria": criteria.__dict__}
    raise RuntimeError("Could not generate L2 single_filter sample")


def make_multi_filter(perfumes: list[Perfume], indexes: dict[str, Any], use_rag: bool, rng: random.Random):
    for _ in range(300):
        ref = rng.choice(perfumes)
        meta = ref["metadata"]
        gender = meta.get("gender")
        seasons = meta.get("best_seasons") or []
        times = meta.get("time_profile") or []
        accords = meta.get("accords_list") or []
        notes = meta.get("notes_list") or []
        if not gender or not seasons or not accords:
            continue

        variant = rng.choice(["gender_season_accord", "gender_season_note", "gender_time_accord", "season_accord_note"])
        if variant == "gender_season_accord":
            season = rng.choice(seasons)
            accord = rng.choice(accords[:5])
            criteria = Criteria(gender=gender, seasons=(season,), accords=(accord,))
            question = MULTI_FILTER_TEMPLATES[0].format(gender=gender, season=season, accord=accord)
        elif variant == "gender_season_note" and notes:
            season = rng.choice(seasons)
            note = rng.choice(notes[:8])
            criteria = Criteria(gender=gender, seasons=(season,), notes=(note,))
            question = MULTI_FILTER_TEMPLATES[1].format(gender=gender, season=season, note=note)
        elif variant == "gender_time_accord" and times:
            time = rng.choice(times)
            accord = rng.choice(accords[:5])
            criteria = Criteria(gender=gender, time_profile=(time,), accords=(accord,))
            question = MULTI_FILTER_TEMPLATES[2].format(gender=gender, time=time, accord=accord)
        elif notes:
            season = rng.choice(seasons)
            accord = rng.choice(accords[:5])
            note = rng.choice(notes[:8])
            criteria = Criteria(seasons=(season,), accords=(accord,), notes=(note,))
            question = MULTI_FILTER_TEMPLATES[3].format(season=season, accord=accord, note=note)
        else:
            continue

        sample = _sample_answer(perfumes, indexes, criteria, use_rag, rng)
        if sample:
            return question, sample[0], sample[1], {"criteria": criteria.__dict__}
    raise RuntimeError("Could not generate L2 multi_filter sample")


def make_negative_filter(perfumes: list[Perfume], indexes: dict[str, Any], use_rag: bool, rng: random.Random):
    for _ in range(300):
        ref = rng.choice(perfumes)
        meta = ref["metadata"]
        gender = meta.get("gender")
        seasons = meta.get("best_seasons") or []
        accords = meta.get("accords_list") or []
        notes = meta.get("notes_list") or []
        if not gender or not seasons or not accords:
            continue

        accord = rng.choice(accords[:5])
        season = rng.choice(seasons)
        available_negatives = [n for n in indexes["notes"][:50] if n not in meta["_notes_set"] and n not in meta["_accords_set"]]
        if not available_negatives:
            continue
        neg = rng.choice(available_negatives)

        if notes and rng.random() < 0.45:
            note = rng.choice(notes[:8])
            criteria = Criteria(accords=(accord,), notes=(note,), not_notes=(neg,))
            question = NEGATIVE_FILTER_TEMPLATES[1].format(accord=accord, note=note, neg=neg)
        elif rng.random() < 0.50:
            criteria = Criteria(gender=gender, seasons=(season,), accords=(accord,), not_notes=(neg,))
            question = NEGATIVE_FILTER_TEMPLATES[0].format(gender=gender, season=season, accord=accord, neg=neg)
        else:
            criteria = Criteria(seasons=(season,), accords=(accord,), not_notes=(neg,))
            question = NEGATIVE_FILTER_TEMPLATES[2].format(season=season, accord=accord, neg=neg)

        sample = _sample_answer(perfumes, indexes, criteria, use_rag, rng)
        if sample:
            return question, sample[0], sample[1], {"criteria": criteria.__dict__}
    raise RuntimeError("Could not generate L2 negative_filter sample")


def make_numeric_filter(perfumes: list[Perfume], indexes: dict[str, Any], use_rag: bool, rng: random.Random):
    for _ in range(300):
        variant = rng.choice(["rating", "votes", "rating_votes_accord", "longevity_gender", "sillage"])
        if variant == "rating":
            min_rating = round(rng.uniform(3.7, 4.4), 1)
            criteria = Criteria(min_rating=min_rating)
            question = NUMERIC_FILTER_TEMPLATES[0].format(min_rating=min_rating)
        elif variant == "votes":
            min_votes = rng.choice([50, 100, 500, 1000, 5000])
            criteria = Criteria(min_votes=min_votes)
            question = NUMERIC_FILTER_TEMPLATES[1].format(min_votes=min_votes)
        elif variant == "rating_votes_accord":
            accord = rng.choice(indexes["accords"][:25])
            min_rating = round(rng.uniform(3.7, 4.3), 1)
            min_votes = rng.choice([50, 100, 500])
            criteria = Criteria(accords=(accord,), min_rating=min_rating, min_votes=min_votes)
            question = NUMERIC_FILTER_TEMPLATES[2].format(accord=accord, min_rating=min_rating, min_votes=min_votes)
        elif variant == "longevity_gender":
            gender = rng.choice(["male", "female", "unisex"])
            min_longevity = round(rng.uniform(3.0, 4.0), 1)
            criteria = Criteria(gender=gender, min_longevity=min_longevity)
            question = NUMERIC_FILTER_TEMPLATES[3].format(gender=gender, min_longevity=min_longevity)
        else:
            min_sillage = round(rng.uniform(2.0, 3.2), 1)
            criteria = Criteria(min_sillage=min_sillage)
            question = NUMERIC_FILTER_TEMPLATES[4].format(min_sillage=min_sillage)

        sample = _sample_answer(perfumes, indexes, criteria, use_rag, rng)
        if sample:
            return question, sample[0], sample[1], {"criteria": criteria.__dict__}
    raise RuntimeError("Could not generate L2 numeric_filter sample")


def make_ranking(perfumes: list[Perfume], indexes: dict[str, Any], use_rag: bool, rng: random.Random):
    for _ in range(300):
        accord = rng.choice(indexes["accords"][:25])
        gender = rng.choice(indexes["genders"])
        season = rng.choice(indexes["seasons"])
        variants = [
            (
                "rating",
                Criteria(accords=(accord,), min_votes=10, sort_by="rating"),
                RANKING_TEMPLATES["rating"][0].format(accord=accord),
            ),
            (
                "rating",
                Criteria(gender=gender, min_votes=10, sort_by="rating"),
                RANKING_TEMPLATES["rating"][1].format(gender=gender),
            ),
            (
                "popularity",
                Criteria(accords=(accord,), sort_by="popularity"),
                RANKING_TEMPLATES["popularity"][0].format(accord=accord),
            ),
            (
                "popularity",
                Criteria(seasons=(season,), sort_by="popularity"),
                RANKING_TEMPLATES["popularity"][1].format(season=season),
            ),
            (
                "longevity",
                Criteria(accords=(accord,), sort_by="longevity"),
                RANKING_TEMPLATES["longevity"][0].format(accord=accord),
            ),
            (
                "longevity",
                Criteria(gender=gender, sort_by="longevity"),
                RANKING_TEMPLATES["longevity"][1].format(gender=gender),
            ),
            (
                "sillage",
                Criteria(accords=(accord,), sort_by="sillage"),
                RANKING_TEMPLATES["sillage"][0].format(accord=accord),
            ),
            (
                "sillage",
                Criteria(seasons=(season,), sort_by="sillage"),
                RANKING_TEMPLATES["sillage"][1].format(season=season),
            ),
        ]
        sort_by, criteria, question = rng.choice(variants)

        sample = _sample_answer(perfumes, indexes, criteria, use_rag, rng, min_results=3, answer_limit=5, keep_top_order=True)
        if sample:
            return question, sample[0], sample[1], {"criteria": criteria.__dict__, "sort_by": sort_by}
    raise RuntimeError("Could not generate L2 ranking sample")


def make_comparison(perfumes: list[Perfume], indexes: dict[str, Any], use_rag: bool, rng: random.Random):
    p1, p2 = rng.sample(perfumes, 2)
    comp_type = rng.choice(list(COMPARISON_TEMPLATES.keys()))
    question = COMPARISON_TEMPLATES[comp_type].format(
        name1=p1["name"],
        brand1=p1["brand"],
        name2=p2["name"],
        brand2=p2["brand"],
    )
    answer = format_comparison_answer(p1, p2, comp_type)
    context_perfumes = [p1, p2] if use_rag else []
    return question, answer, context_perfumes, {"comparison_type": comp_type, "source_ids": [p1.get("id"), p2.get("id")]}


def make_no_match(perfumes: list[Perfume], indexes: dict[str, Any], use_rag: bool, rng: random.Random):
    for _ in range(300):
        strategy = rng.choice(["contradictory_note", "impossible_rating", "overconstrained"])
        if strategy == "contradictory_note":
            note = rng.choice(indexes["notes"][:30])
            accord = rng.choice(indexes["accords"][:25])
            criteria = Criteria(accords=(accord,), notes=(note,), not_notes=(note,))
            question = f"Find perfumes with {accord} accords containing {note} but without {note}."
        elif strategy == "impossible_rating":
            accord = rng.choice(indexes["accords"][:25])
            criteria = Criteria(accords=(accord,), min_rating=5.1)
            question = f"Show {accord} perfumes rated above 5.0."
        else:
            note1 = rng.choice(indexes["notes"][:20])
            note2 = rng.choice(indexes["notes"][40:60] or indexes["notes"][:20])
            criteria = Criteria(notes=(note1, note2), min_rating=4.6, min_votes=10000)
            question = f"Find perfumes with {note1} and {note2} notes, rated above 4.6 with at least 10000 votes."

        if _find_l2_matches(perfumes, indexes, criteria, limit=1):
            continue
        context_perfumes = _build_no_match_context(perfumes, criteria, rng) if use_rag else []
        return question, NO_MATCH_TEXT, context_perfumes, {"criteria": criteria.__dict__, "strategy": strategy}
    raise RuntimeError("Could not generate L2 no_match sample")


def make_inclusive_gender_explicit(perfumes: list[Perfume], indexes: dict[str, Any], use_rag: bool, rng: random.Random):
    for _ in range(300):
        variant = rng.randrange(len(INCLUSIVE_GENDER_TEMPLATES))
        if variant == 0:
            accord = rng.choice(indexes["accords"][:25])
            criteria = Criteria(gender_any_of=("male", "unisex"), accords=(accord,))
            question = INCLUSIVE_GENDER_TEMPLATES[0].format(accord=accord)
        elif variant == 1:
            season = rng.choice(indexes["seasons"])
            criteria = Criteria(gender_any_of=("female", "unisex"), seasons=(season,))
            question = INCLUSIVE_GENDER_TEMPLATES[1].format(season=season)
        elif variant == 2:
            note = rng.choice(indexes["notes"][:40])
            criteria = Criteria(gender_any_of=("male", "unisex"), notes=(note,))
            question = INCLUSIVE_GENDER_TEMPLATES[2].format(note=note)
        else:
            accord = rng.choice(indexes["accords"][:25])
            criteria = Criteria(gender_any_of=("female", "unisex"), accords=(accord,))
            question = INCLUSIVE_GENDER_TEMPLATES[3].format(accord=accord)

        sample = _sample_answer(perfumes, indexes, criteria, use_rag, rng)
        if sample:
            return question, sample[0], sample[1], {"criteria": criteria.__dict__}
    raise RuntimeError("Could not generate L2 inclusive_gender_explicit sample")


def _sample_answer(
    perfumes: list[Perfume],
    indexes: dict[str, Any],
    criteria: Criteria,
    use_rag: bool,
    rng: random.Random,
    min_results: int = 1,
    answer_limit: int = 5,
    keep_top_order: bool = False,
) -> tuple[str, list[Perfume]] | None:
    matches = _find_l2_matches(perfumes, indexes, criteria, limit=30)
    if len(matches) < min_results:
        return None

    selected = _select_l2_answers(matches, criteria, answer_limit, min_results, rng, keep_top_order)

    if not use_rag:
        return format_perfume_list(selected), []

    context_perfumes = build_l2_context(selected, perfumes, criteria, rng)
    match_criteria = _normalized_criteria(criteria)
    final = [p for p in context_perfumes if _match_l2_perfume(p, match_criteria)]
    if criteria.sort_by:
        final.sort(key=lambda p: sort_value(p, criteria.sort_by), reverse=criteria.descending)
    else:
        final.sort(key=lambda p: p["metadata"].get("popularity", 0), reverse=True)
    final = final[:answer_limit]
    if len(final) < min_results:
        return None
    return format_perfume_list(final), context_perfumes


def _select_l2_answers(
    matches: list[Perfume],
    criteria: Criteria,
    answer_limit: int,
    min_results: int,
    rng: random.Random,
    keep_top_order: bool,
) -> list[Perfume]:
    size = max(min_results, min(answer_limit, len(matches)))
    if keep_top_order or criteria.sort_by:
        return matches[:size]

    pool = matches[: min(len(matches), max(answer_limit * 6, 12))]
    selected = rng.sample(pool, size) if len(pool) > size else list(pool)
    selected.sort(key=lambda p: p["metadata"].get("popularity", 0), reverse=True)
    return selected[:size]


def _find_l2_matches(
    perfumes: list[Perfume],
    indexes: dict[str, Any],
    criteria: Criteria,
    limit: int | None = None,
) -> list[Perfume]:
    criteria = _normalized_criteria(criteria)
    candidates = _candidate_pool(perfumes, indexes, criteria)
    key_fn = (
        (lambda p: sort_value(p, criteria.sort_by))
        if criteria.sort_by
        else (lambda p: float(p["metadata"].get("popularity", 0) or 0))
    )

    if limit is None:
        matches = [p for p in candidates if _match_l2_perfume(p, criteria)]
        matches.sort(key=key_fn, reverse=criteria.descending)
        return matches

    scored = (
        (key_fn(perfume), index, perfume)
        for index, perfume in enumerate(candidates)
        if _match_l2_perfume(perfume, criteria)
    )
    if criteria.descending:
        best = heapq.nlargest(limit, scored, key=lambda item: (item[0], -item[1]))
    else:
        best = heapq.nsmallest(limit, scored, key=lambda item: (item[0], item[1]))
    return [perfume for _score, _index, perfume in best]


def _candidate_pool(perfumes: list[Perfume], indexes: dict[str, Any], criteria: Criteria) -> list[Perfume]:
    pools: list[list[Perfume]] = []
    if criteria.gender:
        pools.append(indexes["_by_gender"].get(criteria.gender.lower(), []))
    if criteria.gender_any_of:
        pools.append(_merged_index_pool(indexes["_by_gender"], criteria.gender_any_of))
    if criteria.seasons:
        pools.append(_merged_index_pool(indexes["_by_season"], criteria.seasons))
    if criteria.time_profile:
        pools.append(_merged_index_pool(indexes["_by_time"], criteria.time_profile))
    for accord in criteria.accords:
        pools.append(indexes["_by_accord"].get(accord.lower(), []))
    for note in criteria.notes:
        pools.append(indexes["_by_note"].get(note.lower(), []))

    non_empty_pools = [pool for pool in pools if pool]
    if pools and not non_empty_pools:
        return []
    return min(non_empty_pools, key=len) if non_empty_pools else perfumes


def _merged_index_pool(index: dict[str, list[Perfume]], terms: tuple[str, ...]) -> list[Perfume]:
    merged: list[Perfume] = []
    seen: set[int] = set()
    for term in terms:
        for perfume in index.get(term.lower(), []):
            perfume_id = id(perfume)
            if perfume_id in seen:
                continue
            seen.add(perfume_id)
            merged.append(perfume)
    return merged


def _normalized_criteria(criteria: Criteria) -> Criteria:
    return Criteria(
        gender=criteria.gender.lower() if criteria.gender else None,
        gender_any_of=tuple(g.lower() for g in criteria.gender_any_of),
        seasons=tuple(s.lower() for s in criteria.seasons),
        time_profile=tuple(t.lower() for t in criteria.time_profile),
        accords=tuple(a.lower() for a in criteria.accords),
        notes=tuple(n.lower() for n in criteria.notes),
        not_accords=tuple(a.lower() for a in criteria.not_accords),
        not_notes=tuple(n.lower() for n in criteria.not_notes),
        min_rating=criteria.min_rating,
        min_votes=criteria.min_votes,
        min_longevity=criteria.min_longevity,
        min_sillage=criteria.min_sillage,
        min_value=criteria.min_value,
        sort_by=criteria.sort_by,
        descending=criteria.descending,
        tags=criteria.tags,
    )


def _match_l2_perfume(perfume: Perfume, criteria: Criteria) -> bool:
    meta = perfume["metadata"]
    gender = meta.get("_gender_low") or str(meta.get("gender") or "").lower()
    if criteria.gender and criteria.gender != gender:
        return False

    if criteria.gender_any_of and gender not in criteria.gender_any_of:
        return False

    if criteria.seasons and not any(season in meta["_seasons_set"] for season in criteria.seasons):
        return False

    if criteria.time_profile and not any(time in meta["_times_set"] for time in criteria.time_profile):
        return False

    if criteria.accords and not all(accord in meta["_accords_set"] for accord in criteria.accords):
        return False

    if criteria.notes and not all(note in meta["_notes_set"] for note in criteria.notes):
        return False

    if criteria.not_accords and _contains_forbidden_l2(meta, criteria.not_accords):
        return False

    if criteria.not_notes and _contains_forbidden_l2(meta, criteria.not_notes):
        return False

    if criteria.min_rating is not None:
        rating = meta.get("rating")
        votes = meta.get("popularity", 0)
        if rating is None or rating < criteria.min_rating or votes <= 0:
            return False

    if criteria.min_votes is not None and meta.get("popularity", 0) < criteria.min_votes:
        return False

    if criteria.min_longevity is not None and _positive_metric(perfume, "longevity") < criteria.min_longevity:
        return False

    if criteria.min_sillage is not None and _positive_metric(perfume, "sillage") < criteria.min_sillage:
        return False

    if criteria.min_value is not None and _positive_metric(perfume, "price_value") < criteria.min_value:
        return False

    return True


def _contains_forbidden_l2(meta: dict[str, Any], forbidden_items: tuple[str, ...]) -> bool:
    if not forbidden_items:
        return False

    notes = meta["_notes_set"]
    accords = meta["_accords_set"]
    combined = " ".join(notes) + " " + " ".join(accords)
    for item in forbidden_items:
        item_low = item.lower()
        if item_low in notes or item_low in accords:
            return True
        if re.search(rf"\b{re.escape(item_low)}\b", combined):
            return True
    return False


def build_l2_context(selected: list[Perfume], perfumes: list[Perfume], criteria: Criteria, rng: random.Random, context_size: int = 12) -> list[Perfume]:
    context = list(selected)
    seen = {id(p) for p in context}
    criteria = _normalized_criteria(criteria)

    hard_negatives = []
    sample_pool = rng.sample(perfumes, min(800, len(perfumes)))
    for perfume in sample_pool:
        if id(perfume) in seen:
            continue
        if not _match_l2_perfume(perfume, criteria):
            hard_negatives.append(perfume)
            seen.add(id(perfume))
        if len(hard_negatives) >= 5:
            break

    context.extend(hard_negatives)
    needed = context_size - len(context)
    if needed > 0:
        fillers = [p for p in rng.sample(perfumes, min(400, len(perfumes))) if id(p) not in seen]
        context.extend(fillers[:needed])

    rng.shuffle(context)
    return context[:context_size]


def _build_no_match_context(perfumes: list[Perfume], criteria: Criteria, rng: random.Random, context_size: int = 10) -> list[Perfume]:
    criteria = _normalized_criteria(criteria)
    candidates = [p for p in rng.sample(perfumes, min(1000, len(perfumes))) if not _match_l2_perfume(p, criteria)]
    return candidates[:context_size]


def format_perfume_list(perfumes: list[Perfume]) -> str:
    if not perfumes:
        return NO_MATCH_TEXT
    lines: list[str] = []
    for idx, perfume in enumerate(perfumes, 1):
        meta = perfume["metadata"]
        rating = fmt_rating(meta.get("rating"), meta.get("popularity", 0))
        seasons = fmt_list(title_items(meta.get("best_seasons") or []))
        times = fmt_list(title_items(meta.get("time_profile") or []))
        accords = fmt_list((meta.get("accords_list") or [])[:6])
        notes = fmt_list((meta.get("notes_list") or [])[:8])
        lines.extend(
            [
                f"{idx}. {perfume['name']} by {perfume['brand']}",
                f"- Gender: {meta.get('gender', 'N/A')}",
                f"- Rating: {rating}",
                f"- Seasons: {seasons}",
                f"- Time: {times}",
                f"- Accords: {accords}",
                f"- Notes: {notes}",
            ]
        )
    return "\n".join(lines)


def format_comparison_answer(p1: Perfume, p2: Perfume, comp_type: str) -> str:
    if comp_type == "rating":
        return _compare_numeric(p1, p2, "rating", _rating_value, lambda p: fmt_rating(p["metadata"].get("rating"), p["metadata"].get("popularity", 0)))
    if comp_type == "popularity":
        return _compare_numeric(p1, p2, "popularity", lambda p: p["metadata"].get("popularity", 0), lambda p: f"{p['metadata'].get('popularity', 0)} votes")
    if comp_type == "longevity":
        return _compare_numeric(p1, p2, "longevity", lambda p: _positive_metric(p, "longevity"), lambda p: fmt_val(p["metadata"].get("longevity"), "/5"))
    if comp_type == "sillage":
        return _compare_numeric(p1, p2, "sillage", lambda p: _positive_metric(p, "sillage"), lambda p: fmt_val(p["metadata"].get("sillage"), "/4"))
    if comp_type == "accords":
        return (
            f"{p1['name']} by {p1['brand']} accords: {fmt_list((p1['metadata'].get('accords_list') or [])[:8])}\n"
            f"{p2['name']} by {p2['brand']} accords: {fmt_list((p2['metadata'].get('accords_list') or [])[:8])}"
        )
    return (
        f"{p1['name']} by {p1['brand']} seasons: {fmt_list(title_items(p1['metadata'].get('best_seasons') or []))}\n"
        f"{p2['name']} by {p2['brand']} seasons: {fmt_list(title_items(p2['metadata'].get('best_seasons') or []))}"
    )


def _compare_numeric(p1: Perfume, p2: Perfume, label: str, value_fn, display_fn) -> str:
    v1 = value_fn(p1)
    v2 = value_fn(p2)
    n1 = f"{p1['name']} by {p1['brand']}"
    n2 = f"{p2['name']} by {p2['brand']}"
    if not v1 and not v2:
        return f"Neither {n1} nor {n2} has recorded {label} data."
    if not v1:
        return f"{n2} has higher recorded {label} with {display_fn(p2)}. {n1} has no recorded {label}."
    if not v2:
        return f"{n1} has higher recorded {label} with {display_fn(p1)}. {n2} has no recorded {label}."
    if v1 > v2:
        return f"{n1} has higher recorded {label} with {display_fn(p1)} compared to {n2} with {display_fn(p2)}."
    if v2 > v1:
        return f"{n2} has higher recorded {label} with {display_fn(p2)} compared to {n1} with {display_fn(p1)}."
    return f"Both {n1} and {n2} have the same recorded {label}: {display_fn(p1)}."


def _rating_value(perfume: Perfume) -> float:
    meta = perfume["metadata"]
    rating = meta.get("rating")
    return float(rating or 0.0) if meta.get("popularity", 0) > 0 else 0.0


def _positive_metric(perfume: Perfume, key: str) -> float:
    value = perfume["metadata"].get(key)
    return float(value or 0.0) if value and value > 0.0 else 0.0
