from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from research.runtime.catalog import RuntimeCatalog
from research.runtime.query_analyzer import QueryAnalysis


UNSUPPORTED_PATTERNS = [
    (
        "live_price",
        re.compile(r"(?:[$€£₺]\s*\d|\b(?:price|cost|dollars?|euros?|lira|tl|under \d+|cheap|cheaper|affordable)\b|\b(?:kaç para|fiyatı ne|ucuz|daha ucuz)\b)", re.I),
        "I do not have a current price source, so I cannot safely answer price or budget questions.",
    ),
    (
        "availability",
        re.compile(r"\b(?:in stock|available now|availability|discontinued|where can i buy|still sold)\b|\b(?:stokta|satışta mı|satista mi|nereden alabilirim|üretiliyor mu)\b", re.I),
        "I do not have a live availability source, so I cannot verify stock or discontinued status.",
    ),
    (
        "medical_safety",
        re.compile(r"\b(?:asthma|allergy|allergic|pregnan|eczema|migraine|safe for skin)\b|\b(?:astım|astim|alerji|hamile|egzama|migren)\b", re.I),
        "I cannot guarantee medical or allergy safety from perfume database fields. Check the ingredient label and consult a qualified professional when needed.",
    ),
    (
        "compliments",
        re.compile(r"\b(?:most complimented|compliment getter|gets? compliments?)\b|\ben çok iltifat\b", re.I),
        "The database does not record compliment outcomes, so I cannot rank perfumes by compliments.",
    ),
    (
        "layering",
        re.compile(r"\b(?:layer|layering|mix together)\b|\b(?:katmanla|katmanlama|birlikte sık)\b", re.I),
        "The database does not contain verified layering guidance, so I cannot make a grounded layering claim.",
    ),
]


def unsupported_intent_answer(query: str) -> tuple[str, str] | None:
    for route, pattern, answer in UNSUPPORTED_PATTERNS:
        if pattern.search(query):
            return route, answer
    return None


def contradiction_answer(analysis: QueryAnalysis) -> str | None:
    if not analysis.contradictions:
        return None
    terms = ", ".join(analysis.contradictions)
    return (
        f"Your request both asks for and excludes the same trait(s): {terms}. "
        "Please tell me which side of that preference is more important."
    )


COLLECTION_PROFILES = (
    {
        "name": "fresh warm-weather daytime",
        "accords": ("fresh", "citrus", "aromatic", "aquatic"),
        "season": "summer",
        "time_profile": "day",
    },
    {
        "name": "warm cold-weather evening",
        "accords": ("amber", "warm spicy", "vanilla", "sweet"),
        "season": "winter",
        "time_profile": "night",
    },
    {
        "name": "elegant woody versatile",
        "accords": ("woody", "iris", "powdery", "musky"),
        "season": "autumn",
        "time_profile": "day",
    },
    {
        "name": "green aromatic spring daytime",
        "accords": ("green", "aromatic", "lavender", "fresh spicy"),
        "season": "spring",
        "time_profile": "day",
    },
)


def prepare_collection_gap(
    catalog: RuntimeCatalog,
    analysis: QueryAnalysis,
) -> tuple[QueryAnalysis | None, str]:
    """Select the least-covered broad wear profile from explicitly owned bottles."""
    if not analysis.owned_perfumes:
        return None, (
            "List the perfumes you own in a clear form, for example: "
            "'My collection: Aventus, Bleu de Chanel, Prada L'Homme. What is missing from my collection?'"
        )

    rows, missing = catalog.compare(list(analysis.owned_perfumes))
    if not rows:
        return None, "I could not resolve any perfume names from the collection list in the database."

    owned_accords = set()
    owned_seasons = set()
    owned_times = set()
    for row in rows:
        owned_accords.update(csv_terms(row.get("accords_csv")))
        owned_seasons.update(csv_terms(row.get("seasons_csv")))
        owned_times.update(csv_terms(row.get("time_profile_csv")))

    def coverage(profile: dict[str, Any]) -> float:
        accord_terms = set(profile["accords"])
        accord_coverage = len(accord_terms & owned_accords) / len(accord_terms)
        season_coverage = 1.0 if profile["season"] in owned_seasons else 0.0
        time_coverage = 1.0 if profile["time_profile"] in owned_times else 0.0
        return accord_coverage * 0.70 + season_coverage * 0.20 + time_coverage * 0.10

    target = min(COLLECTION_PROFILES, key=coverage)
    labels = tuple(f"{row['name']} by {row['brand']}" for row in rows)
    debug = {
        **analysis.debug,
        "collection_resolved": list(labels),
        "collection_unresolved": missing,
        "collection_gap_target": target["name"],
    }
    updated = replace(
        analysis,
        wanted_accords=tuple(sorted(set(analysis.wanted_accords) | set(target["accords"]))),
        season=analysis.season or str(target["season"]),
        time_profile=analysis.time_profile or str(target["time_profile"]),
        owned_perfumes=labels,
        debug=debug,
    )
    note = (
        f"Resolved owned collection: {', '.join(labels)}. "
        f"The least-covered broad wear profile is {target['name']}; recommend additions for that gap."
    )
    if missing:
        note += " Unresolved collection entries: " + ", ".join(missing) + "."
    return updated, note


def render_comparison(catalog: RuntimeCatalog, analysis: QueryAnalysis) -> tuple[str, list[dict[str, Any]]] | None:
    if len(analysis.comparison_perfumes) < 2:
        return None
    rows, missing = catalog.compare(list(analysis.comparison_perfumes))
    if missing:
        return f"I could not resolve these perfume names in the database: {', '.join(missing)}.", rows
    if len(rows) != 2:
        return "I need two distinct perfume names for a deterministic comparison.", rows

    first, second = rows
    lines = [
        "The generated comparison did not pass grounding, so here is a conservative card-based interpretation.",
        "",
    ]
    lines.extend(render_comparison_record(first, 1))
    lines.append("")
    lines.extend(render_comparison_record(second, 2))
    lines.append("")
    lines.append("How to choose:")
    lines.append(f"- Choose {first['name']} by {first['brand']} for {comparison_advantages(first, second)}.")
    lines.append(f"- Choose {second['name']} by {second['brand']} for {comparison_advantages(second, first)}.")
    return "\n".join(lines), rows


def render_comparison_record(row: dict[str, Any], index: int) -> list[str]:
    label = f"{row['name']} by {row['brand']}"
    accords = ordered_csv_terms(row.get("accords_csv"))
    notes = ordered_csv_terms(row.get("notes_csv"))
    character = natural_join(accords[:3]) if accords else "an unspecified recorded accord profile"
    note_support = f", supported by listed {natural_join(notes[:3])} notes" if notes else ""
    return [
        f"{index}. {label}",
        f"Character: Its card reads as a {character} profile{note_support}.",
        "Wear: " + comparison_wear_text(row),
        "Performance: " + comparison_performance_text(row),
    ]


def comparison_wear_text(row: dict[str, Any]) -> str:
    seasons = ordered_csv_terms(row.get("seasons_csv"))
    times = ordered_csv_terms(row.get("time_profile_csv"))
    if seasons and times:
        base = f"The recorded range covers {natural_join(seasons)} with {natural_join(times)} wear"
    elif seasons:
        base = f"The recorded range covers {natural_join(seasons)}"
    elif times:
        base = f"The card records {natural_join(times)} wear"
    else:
        return "The card does not record a season or time profile."
    if len(seasons) >= 3 and len(times) >= 2:
        return base + ", which makes it the more broadly positioned kind of option."
    if len(times) == 1:
        return base + ", pointing to a more focused role."
    return base + "."


def comparison_performance_text(row: dict[str, Any]) -> str:
    longevity = row.get("longevity")
    sillage = row.get("sillage")
    parts = []
    if longevity is not None:
        score = float(longevity)
        label = "strong" if score >= 3.75 else "solid" if score >= 3.10 else "moderate" if score >= 2.50 else "lighter"
        parts.append(f"{score:.2f}/5 longevity suggests {label} recorded staying power")
    if sillage is not None:
        score = float(sillage)
        label = "noticeable" if score >= 2.70 else "moderate" if score >= 2.10 else "reserved"
        parts.append(f"{score:.2f}/4 sillage suggests a {label} presence")
    if not parts:
        return "The card does not provide longevity or sillage scores."
    return natural_join(parts).capitalize() + "."


def comparison_advantages(row: dict[str, Any], other: dict[str, Any]) -> str:
    advantages = []
    for field, phrase in (
        ("longevity", "stronger recorded staying power"),
        ("sillage", "a more noticeable recorded presence"),
        ("value_score", "higher recorded value"),
    ):
        left = row.get(field)
        right = other.get(field)
        if left is not None and right is not None and float(left) > float(right):
            advantages.append(phrase)
    row_breadth = len(ordered_csv_terms(row.get("seasons_csv"))) + len(ordered_csv_terms(row.get("time_profile_csv")))
    other_breadth = len(ordered_csv_terms(other.get("seasons_csv"))) + len(ordered_csv_terms(other.get("time_profile_csv")))
    if row_breadth > other_breadth:
        advantages.append("a broader recorded wear range")
    if not advantages:
        accords = ordered_csv_terms(row.get("accords_csv"))
        advantages.append(f"its {natural_join(accords[:3])} direction" if accords else "its recorded profile")
    return natural_join(advantages[:3])


def compare_metric(first: dict[str, Any], second: dict[str, Any], field: str, label: str, suffix: str) -> str:
    left = first.get(field)
    right = second.get(field)
    first_label = f"{first['name']} by {first['brand']}"
    second_label = f"{second['name']} by {second['brand']}"
    if left is None or right is None:
        return f"{label.capitalize()}: insufficient recorded data for a direct comparison."
    if float(left) == float(right):
        return f"{label.capitalize()}: tied at {float(left):.2f}{suffix}."
    winner, loser = (first_label, second_label) if float(left) > float(right) else (second_label, first_label)
    high, low = (float(left), float(right)) if float(left) > float(right) else (float(right), float(left))
    return f"{winner} has higher recorded {label} ({high:.2f}{suffix} vs {low:.2f}{suffix} for {loser})."


def metric_text(value: object, suffix: str) -> str:
    return "not recorded" if value is None else f"{float(value):.2f}{suffix}"


def ordered_csv_terms(value: object) -> list[str]:
    if not value:
        return []
    return [part.strip().lower() for part in str(value).split(",") if part.strip()]


def natural_join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def csv_terms(value: object) -> set[str]:
    if not value:
        return set()
    return {part.strip().lower() for part in str(value).split(",") if part.strip()}
