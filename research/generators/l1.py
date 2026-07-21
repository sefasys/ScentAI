from __future__ import annotations

import random
from typing import Any

from research.core.cards import (
    build_perfume_context,
    parse_card_accords,
    parse_card_metrics,
    parse_card_note_groups,
    parse_card_notes,
)
from research.core.config import DatasetConfig, L1_DEFAULT_RATIOS, counts_from_ratios
from research.core.data import Perfume
from research.core.formatting import fmt_list, fmt_rating, fmt_val, title_items
from research.core.messages import build_messages


INFO_QUESTIONS = [
    "Provide the database record for {name} by {brand}.",
    "Show the full specifications of {name} by {brand}.",
    "What are the recorded parameters for {name} by {brand}?",
    "Give me all stored data for {name} by {brand}.",
    "Retrieve the complete database entry for {name} by {brand}.",
    "What does the database say about {name} by {brand}?",
    "Display all fields for {name} by {brand}.",
]

NOTES_QUESTIONS = [
    "List all notes present in {name} by {brand}.",
    "What notes are recorded for {name} by {brand}?",
    "What notes are listed in {name} by {brand}?",
    "Show the note breakdown for {name} by {brand}.",
    "What are the top, middle, and base notes of {name} by {brand}?",
    "Which notes does {name} by {brand} contain according to the database?",
]

ACCORDS_QUESTIONS = [
    "What accords are present in {name} by {brand}?",
    "List the accord profile of {name} by {brand}.",
    "Show the accords for {name} by {brand}.",
    "What is the accord breakdown of {name} by {brand}?",
    "Which fragrance families does {name} by {brand} belong to?",
    "What are the dominant accords in {name} by {brand} according to the database?",
]

SEASONS_QUESTIONS = [
    "Which seasons are voted suitable for {name} by {brand}?",
    "What is the season profile of {name} by {brand}?",
    "What seasons are recommended for {name} by {brand} according to community votes?",
    "What seasons does the database list for {name} by {brand}?",
    "Is {name} by {brand} a summer, spring, autumn, or winter fragrance?",
    "What seasons of the year is {name} by {brand} recommended for?",
    "Show the seasonal suitability data for {name} by {brand}.",
]

TIME_QUESTIONS = [
    "Is {name} by {brand} a day or night fragrance?",
    "What time of day is {name} by {brand} best suited for?",
    "Show the day/night time profile for {name} by {brand}.",
    "Is {name} by {brand} recommended for daytime or evening wear?",
]

RATING_QUESTIONS = [
    "What is the average rating and vote count for {name} by {brand}?",
    "Show the community rating statistics for {name} by {brand}.",
    "How is {name} by {brand} rated by users?",
    "What rating does {name} by {brand} have in the database?",
    "What are the rating and performance scores for {name} by {brand}?",
    "Show the rating, longevity, sillage, and value scores for {name} by {brand}.",
]

COMPARISON_QUESTIONS = {
    "rated_higher": "Which is rated higher, {name1} by {brand1} or {name2} by {brand2}?",
    "lasts_longer": "Which has better longevity, {name1} by {brand1} or {name2} by {brand2}?",
    "more_sillage": "Which has stronger sillage, {name1} by {brand1} or {name2} by {brand2}?",
    "general": "Compare {name1} by {brand1} and {name2} by {brand2}.",
    "accords_diff": "How do the accords of {name1} by {brand1} and {name2} by {brand2} differ?",
    "seasons_diff": "Which seasons suit {name1} by {brand1} vs {name2} by {brand2}?",
}


def generate_l1_records(perfumes: list[Perfume], config: DatasetConfig) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(config.seed)
    counts = config.category_counts or counts_from_ratios(config.total, L1_DEFAULT_RATIOS)
    schedule = [category for category, count in counts.items() for _ in range(count)]
    rng.shuffle(schedule)

    records: list[dict[str, Any]] = []
    produced = {category: 0 for category in counts}

    for category in schedule:
        use_rag = rng.random() < config.rag_ratio
        record = _make_record(category, perfumes, use_rag, rng, config.include_debug_meta)
        records.append(record)
        produced[category] += 1

    rng.shuffle(records)
    return records, produced


def _make_record(
    category: str,
    perfumes: list[Perfume],
    use_rag: bool,
    rng: random.Random,
    include_debug_meta: bool,
) -> dict[str, Any]:
    p1 = rng.choice(perfumes)

    if category == "info":
        question, answer = generate_info_qa(p1, use_rag, rng)
        context_perfumes = [p1] if use_rag else []
    elif category == "notes":
        question, answer = generate_notes_qa(p1, use_rag, rng)
        context_perfumes = [p1] if use_rag else []
    elif category == "accords":
        question, answer = generate_accords_qa(p1, use_rag, rng)
        context_perfumes = [p1] if use_rag else []
    elif category == "seasons":
        question, answer = generate_seasons_qa(p1, rng)
        context_perfumes = [p1] if use_rag else []
    elif category == "rating":
        question, answer = generate_rating_qa(p1, use_rag, rng)
        context_perfumes = [p1] if use_rag else []
    elif category == "comparison":
        p2 = _pick_other_perfume(p1, perfumes, rng)
        question, answer = generate_comparison_qa(p1, p2, use_rag, rng)
        context_perfumes = [p1, p2] if use_rag else []
    else:
        raise ValueError(f"Unknown L1 category: {category}")

    context = build_perfume_context(context_perfumes)
    debug_meta = {
        "level": "L1",
        "category": category,
        "rag": bool(context),
        "source_ids": [p["id"] for p in context_perfumes if p.get("id")],
    }
    return build_messages(question, answer, context, debug_meta, include_debug_meta)


def generate_info_qa(p: Perfume, use_rag: bool, rng: random.Random) -> tuple[str, str]:
    name, brand, meta = p["name"], p["brand"], p["metadata"]
    card = p.get("card_text", "")
    question = rng.choice(INFO_QUESTIONS).format(name=name, brand=brand)

    if use_rag and card:
        accords = parse_card_accords(card)
        notes = parse_card_notes(card)
        longevity, sillage, value = parse_card_metrics(card)
    else:
        accords = meta.get("accords_list") or []
        notes = meta.get("notes_list") or []
        longevity = meta.get("longevity")
        sillage = meta.get("sillage")
        value = meta.get("price_value")

    lines = [
        f"Database Record - {name} by {brand}:",
        f"- Brand: {brand}",
        f"- Name: {name}",
        f"- Gender: {meta.get('gender', 'N/A')}",
        f"- Launch Year: {meta.get('year') or 'N/A'}",
        f"- Rating: {fmt_rating(meta.get('rating'), meta.get('popularity', 0))}",
        f"- Longevity: {fmt_val(longevity, '/5')}",
        f"- Sillage: {fmt_val(sillage, '/4')}",
        f"- Value: {fmt_val(value, '/5')}",
        f"- Best Seasons: {fmt_list(title_items(meta.get('best_seasons') or []))}",
        f"- Time Profile: {fmt_list(title_items(meta.get('time_profile') or []))}",
    ]
    if accords:
        lines.append(f"- Accords: {fmt_list(accords)}")
    if notes:
        lines.append(f"- Notes: {fmt_list(notes)}")
    if meta.get("perfumer"):
        lines.append(f"- Perfumer: {meta['perfumer']}")

    return question, "\n".join(lines)


def generate_notes_qa(p: Perfume, use_rag: bool, rng: random.Random) -> tuple[str, str]:
    name, brand, meta = p["name"], p["brand"], p["metadata"]
    card = p.get("card_text", "")
    question = rng.choice(NOTES_QUESTIONS).format(name=name, brand=brand)

    if use_rag and card:
        groups = parse_card_note_groups(card)
        if groups["top"] or groups["middle"] or groups["base"]:
            parts = []
            if groups["top"]:
                parts.append(f"Top: {', '.join(groups['top'])}")
            if groups["middle"]:
                parts.append(f"Middle: {', '.join(groups['middle'])}")
            if groups["base"]:
                parts.append(f"Base: {', '.join(groups['base'])}")
            return question, f"The notes recorded for {name} by {brand} are:\n" + "\n".join(parts)
        if groups["flat"]:
            return question, f"The recorded notes for {name} by {brand} are: {fmt_list(groups['flat'])}."

    notes = meta.get("notes_list") or []
    if notes:
        return question, f"The recorded notes for {name} by {brand} are: {fmt_list(notes)}."
    return question, f"There are no notes recorded in the database for {name} by {brand}."


def generate_accords_qa(p: Perfume, use_rag: bool, rng: random.Random) -> tuple[str, str]:
    name, brand, meta = p["name"], p["brand"], p["metadata"]
    card = p.get("card_text", "")
    question = rng.choice(ACCORDS_QUESTIONS).format(name=name, brand=brand)
    accords = parse_card_accords(card) if use_rag and card else (meta.get("accords_list") or [])

    if accords:
        return question, f"The accord profile for {name} by {brand} consists of: {fmt_list(accords)}."
    return question, f"There is no accord profile recorded in the database for {name} by {brand}."


def generate_seasons_qa(p: Perfume, rng: random.Random) -> tuple[str, str]:
    name, brand, meta = p["name"], p["brand"], p["metadata"]
    seasons = meta.get("best_seasons") or []
    times = meta.get("time_profile") or []

    if rng.random() < 0.70 or not times:
        question = rng.choice(SEASONS_QUESTIONS).format(name=name, brand=brand)
        if seasons:
            return question, f"The suitable seasons for {name} by {brand} are: {fmt_list(title_items(seasons))}."
        return question, f"There are no specific seasons recommended in the database for {name} by {brand}."

    question = rng.choice(TIME_QUESTIONS).format(name=name, brand=brand)
    if times:
        return question, f"The time profile for {name} by {brand} is: {fmt_list(title_items(times))}."
    return question, f"There is no time profile recorded in the database for {name} by {brand}."


def generate_rating_qa(p: Perfume, use_rag: bool, rng: random.Random) -> tuple[str, str]:
    name, brand, meta = p["name"], p["brand"], p["metadata"]
    card = p.get("card_text", "")
    question = rng.choice(RATING_QUESTIONS).format(name=name, brand=brand)
    rating = meta.get("rating")
    popularity = meta.get("popularity", 0)

    if use_rag and card:
        longevity, sillage, value = parse_card_metrics(card)
    else:
        longevity = meta.get("longevity")
        sillage = meta.get("sillage")
        value = meta.get("price_value")

    if rating is not None and rating > 0.0 and popularity > 0:
        answer = (
            f"{name} by {brand} has a community rating of {rating:.2f}/5 "
            f"based on {popularity} votes.\n"
            f"- Longevity: {fmt_val(longevity, '/5')}\n"
            f"- Sillage: {fmt_val(sillage, '/4')}\n"
            f"- Value: {fmt_val(value, '/5')}"
        )
        return question, answer

    return question, f"There are no rating statistics recorded for {name} by {brand}."


def generate_comparison_qa(p1: Perfume, p2: Perfume, use_rag: bool, rng: random.Random) -> tuple[str, str]:
    q_type = rng.choice(list(COMPARISON_QUESTIONS.keys()))
    question = COMPARISON_QUESTIONS[q_type].format(
        name1=p1["name"],
        brand1=p1["brand"],
        name2=p2["name"],
        brand2=p2["brand"],
    )

    metrics1 = _comparison_values(p1, use_rag)
    metrics2 = _comparison_values(p2, use_rag)

    if q_type == "rated_higher":
        answer = _compare_numeric(p1, p2, "rating", metrics1["rating_value"], metrics2["rating_value"], metrics1["rating"], metrics2["rating"])
    elif q_type == "lasts_longer":
        answer = _compare_numeric(p1, p2, "longevity", metrics1["longevity_value"], metrics2["longevity_value"], metrics1["longevity"], metrics2["longevity"])
    elif q_type == "more_sillage":
        answer = _compare_numeric(p1, p2, "sillage", metrics1["sillage_value"], metrics2["sillage_value"], metrics1["sillage"], metrics2["sillage"])
    elif q_type == "accords_diff":
        answer = (
            "The accords of the two perfumes differ as follows:\n"
            f"- {p1['name']} by {p1['brand']} accords: {fmt_list(metrics1['accords'])}\n"
            f"- {p2['name']} by {p2['brand']} accords: {fmt_list(metrics2['accords'])}"
        )
    elif q_type == "seasons_diff":
        answer = (
            "The suitable seasons comparison shows:\n"
            f"- {p1['name']} by {p1['brand']} is best worn during: {fmt_list(title_items(metrics1['seasons']))}\n"
            f"- {p2['name']} by {p2['brand']} is best worn during: {fmt_list(title_items(metrics2['seasons']))}"
        )
    else:
        answer = "Database Comparison:\n\n" + _perfume_summary(p1, metrics1) + "\n\n" + _perfume_summary(p2, metrics2)

    return question, answer


def _comparison_values(p: Perfume, use_rag: bool) -> dict[str, Any]:
    meta = p["metadata"]
    card = p.get("card_text", "")
    if use_rag and card:
        accords = parse_card_accords(card)
        longevity, sillage, _value = parse_card_metrics(card)
    else:
        accords = meta.get("accords_list") or []
        longevity = meta.get("longevity")
        sillage = meta.get("sillage")

    rating = meta.get("rating")
    popularity = meta.get("popularity", 0)
    return {
        "accords": accords,
        "seasons": meta.get("best_seasons") or [],
        "rating": fmt_rating(rating, popularity),
        "rating_value": rating if rating is not None and rating > 0.0 and popularity > 0 else None,
        "longevity": fmt_val(longevity, "/5"),
        "longevity_value": longevity if longevity is not None and longevity > 0.0 else None,
        "sillage": fmt_val(sillage, "/4"),
        "sillage_value": sillage if sillage is not None and sillage > 0.0 else None,
    }


def _compare_numeric(
    p1: Perfume,
    p2: Perfume,
    label: str,
    value1: float | None,
    value2: float | None,
    display1: str,
    display2: str,
) -> str:
    name1 = f"{p1['name']} by {p1['brand']}"
    name2 = f"{p2['name']} by {p2['brand']}"
    if value1 is None and value2 is None:
        return f"{label.capitalize()} information is not recorded for either {name1} or {name2}."
    if value1 is None:
        return f"{name2} has higher recorded {label} with {display2}. {name1} has no recorded {label}."
    if value2 is None:
        return f"{name1} has higher recorded {label} with {display1}. {name2} has no recorded {label}."
    if value1 > value2:
        return f"{name1} has higher recorded {label} with {display1} compared to {name2} with {display2}."
    if value2 > value1:
        return f"{name2} has higher recorded {label} with {display2} compared to {name1} with {display1}."
    return f"Both {name1} and {name2} have the same recorded {label}: {display1}."


def _perfume_summary(p: Perfume, values: dict[str, Any]) -> str:
    return (
        f"{p['name']} by {p['brand']}:\n"
        f"  Accords: {fmt_list(values['accords'][:4])}\n"
        f"  Rating: {values['rating']}\n"
        f"  Longevity: {values['longevity']} | Sillage: {values['sillage']}\n"
        f"  Best Seasons: {fmt_list(title_items(values['seasons']))}"
    )


def _pick_other_perfume(p1: Perfume, perfumes: list[Perfume], rng: random.Random) -> Perfume:
    p2 = rng.choice(perfumes)
    while p2.get("id") == p1.get("id"):
        p2 = rng.choice(perfumes)
    return p2
