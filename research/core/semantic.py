from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research.core.data import Perfume


@dataclass(frozen=True)
class SemanticProfile:
    label: str
    gender_any_of: tuple[str, ...] = ()
    prefer_gender: str | None = None
    hard_seasons: tuple[str, ...] = ()
    hard_times: tuple[str, ...] = ()
    wanted_accords: tuple[str, ...] = ()
    wanted_notes: tuple[str, ...] = ()
    soft_accords: tuple[str, ...] = ()
    soft_notes: tuple[str, ...] = ()
    avoid_accords: tuple[str, ...] = ()
    avoid_notes: tuple[str, ...] = ()
    soft_avoid_accords: tuple[str, ...] = ()
    soft_avoid_notes: tuple[str, ...] = ()
    reference_name: str | None = None
    reference_brand: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


VIBE_PROFILES: dict[str, dict[str, tuple[str, ...]]] = {
    "cozy": {
        "soft_accords": ("vanilla", "amber", "warm spicy", "sweet", "woody"),
        "soft_notes": ("vanilla", "tonka bean", "amber", "sandalwood"),
        "soft_avoid_accords": ("aquatic", "ozonic"),
    },
    "clean": {
        "soft_accords": ("fresh", "musky", "citrus", "soapy", "powdery"),
        "soft_notes": ("musk", "bergamot", "lavender", "iris"),
        "soft_avoid_accords": ("animalic", "smoky", "oud"),
    },
    "fresh": {
        "soft_accords": ("citrus", "fresh", "aromatic", "green", "aquatic"),
        "soft_notes": ("bergamot", "lemon", "mint", "grapefruit"),
    },
    "dark": {
        "soft_accords": ("woody", "smoky", "leather", "oud", "amber", "earthy"),
        "soft_notes": ("patchouli", "incense", "oud", "tobacco", "leather"),
    },
    "bright": {
        "soft_accords": ("citrus", "fresh", "fruity", "white floral"),
        "soft_notes": ("orange", "mandarin orange", "bergamot", "pear"),
    },
    "expensive": {
        "soft_accords": ("woody", "amber", "powdery", "iris", "musky", "leather"),
        "soft_notes": ("iris", "sandalwood", "amber", "musk", "cedar"),
    },
    "creamy": {
        "soft_accords": ("vanilla", "lactonic", "coconut", "sweet", "powdery"),
        "soft_notes": ("vanilla", "coconut", "tonka bean", "milk", "sandalwood"),
    },
    "elegant": {
        "soft_accords": ("floral", "powdery", "woody", "musky", "rose"),
        "soft_notes": ("rose", "iris", "jasmine", "musk", "sandalwood"),
    },
}


OCCASION_PROFILES: dict[str, dict[str, tuple[str, ...]]] = {
    "office": {
        "hard_times": ("day",),
        "soft_accords": ("fresh", "musky", "citrus", "woody", "powdery"),
        "soft_avoid_accords": ("animalic", "oud", "smoky"),
    },
    "date_night": {
        "hard_times": ("night",),
        "soft_accords": ("amber", "vanilla", "sweet", "warm spicy", "woody"),
        "soft_notes": ("vanilla", "tonka bean", "amber", "jasmine"),
    },
    "summer_day": {
        "hard_seasons": ("summer",),
        "hard_times": ("day",),
        "soft_accords": ("citrus", "fresh", "aquatic", "green", "aromatic"),
        "soft_avoid_accords": ("heavy", "smoky", "oud"),
    },
    "winter_evening": {
        "hard_seasons": ("winter",),
        "hard_times": ("night",),
        "soft_accords": ("amber", "vanilla", "warm spicy", "woody", "sweet"),
    },
    "wedding": {
        "soft_accords": ("floral", "white floral", "musky", "fresh", "powdery"),
        "soft_notes": ("jasmine", "rose", "musk", "orange blossom"),
    },
    "gym_after": {
        "hard_times": ("day",),
        "soft_accords": ("fresh", "citrus", "aquatic", "aromatic", "green"),
        "soft_avoid_accords": ("sweet", "oud", "smoky"),
    },
}


CONTRADICTION_PROFILES: dict[str, dict[str, tuple[str, ...]]] = {
    "sweet but not sugary": {
        "soft_accords": ("sweet", "vanilla", "amber", "woody"),
        "soft_avoid_accords": ("caramel", "candy", "syrupy"),
    },
    "fresh but warm": {
        "soft_accords": ("fresh spicy", "warm spicy", "citrus", "amber"),
    },
    "dark but clean": {
        "soft_accords": ("woody", "musky", "amber", "fresh", "powdery"),
        "soft_avoid_accords": ("animalic", "smoky"),
    },
    "woody but airy": {
        "soft_accords": ("woody", "fresh", "aromatic", "citrus"),
    },
    "soft leather": {
        "soft_accords": ("leather", "powdery", "musky", "floral"),
    },
}


def score_perfume(perfume: Perfume, profile: SemanticProfile) -> float:
    meta = perfume["metadata"]
    gender = meta.get("gender", "").lower()

    if profile.gender_any_of and gender not in {g.lower() for g in profile.gender_any_of}:
        return -1.0

    if profile.hard_seasons and not any(s.lower() in meta["_seasons_set"] for s in profile.hard_seasons):
        return -1.0

    if profile.hard_times and not any(t.lower() in meta["_times_set"] for t in profile.hard_times):
        return -1.0

    if _contains_any(meta, profile.avoid_accords) or _contains_any(meta, profile.avoid_notes):
        return -1.0

    score = 0.0
    score += _score_terms(meta["_accords_set"], profile.wanted_accords, 4.0)
    score += _score_terms(meta["_notes_set"], profile.wanted_notes, 4.0)
    score += _score_terms(meta["_accords_set"], profile.soft_accords, 2.0)
    score += _score_terms(meta["_notes_set"], profile.soft_notes, 1.5)

    score -= _score_terms(meta["_accords_set"], profile.soft_avoid_accords, 2.5)
    score -= _score_terms(meta["_notes_set"], profile.soft_avoid_notes, 2.0)

    if profile.prefer_gender and gender == profile.prefer_gender.lower():
        score += 0.75
    if profile.hard_seasons:
        score += 1.0
    if profile.hard_times:
        score += 0.75

    rating = meta.get("rating")
    votes = meta.get("popularity", 0) or 0
    if rating and rating > 0 and votes > 0:
        score += min((rating - 3.5) * 0.6, 0.8)
    score += min(votes / 50000, 0.8)

    return score


def find_semantic_matches(
    perfumes: list[Perfume],
    profile: SemanticProfile,
    limit: int = 10,
    min_score: float = 3.0,
) -> list[Perfume]:
    scored = []
    for perfume in perfumes:
        score = score_perfume(perfume, profile)
        if score >= min_score:
            scored.append((score, perfume))
    scored.sort(key=lambda item: (item[0], item[1]["metadata"].get("popularity", 0)), reverse=True)
    return [perfume for _score, perfume in scored[:limit]]


def profile_to_debug(profile: SemanticProfile) -> dict[str, Any]:
    data = profile.__dict__.copy()
    data["extra"] = dict(profile.extra)
    return data


def _score_terms(available: set[str], wanted: tuple[str, ...], weight: float) -> float:
    return sum(weight for term in wanted if term.lower() in available)


def _contains_any(meta: dict, terms: tuple[str, ...]) -> bool:
    available = meta["_accords_set"] | meta["_notes_set"]
    return any(term.lower() in available for term in terms)
