from __future__ import annotations

import heapq
import re
from dataclasses import dataclass, field
from typing import Literal

from research.core.data import Perfume


SortField = Literal["popularity", "rating", "longevity", "sillage", "value"]


@dataclass(frozen=True)
class Criteria:
    gender: str | None = None
    gender_any_of: tuple[str, ...] = ()
    seasons: tuple[str, ...] = ()
    time_profile: tuple[str, ...] = ()
    accords: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    not_accords: tuple[str, ...] = ()
    not_notes: tuple[str, ...] = ()
    min_rating: float | None = None
    min_votes: int | None = None
    min_longevity: float | None = None
    min_sillage: float | None = None
    min_value: float | None = None
    sort_by: SortField | None = None
    descending: bool = True
    tags: tuple[str, ...] = field(default_factory=tuple)


def match_perfume(perfume: Perfume, criteria: Criteria) -> bool:
    meta = perfume["metadata"]
    card = perfume.get("card_text", "").lower()

    gender = meta.get("gender", "").lower()
    if criteria.gender and criteria.gender.lower() != gender:
        return False

    if criteria.gender_any_of:
        allowed = {g.lower() for g in criteria.gender_any_of}
        if gender not in allowed:
            return False

    if criteria.seasons and not any(s.lower() in meta["_seasons_set"] for s in criteria.seasons):
        return False

    if criteria.time_profile and not any(t.lower() in meta["_times_set"] for t in criteria.time_profile):
        return False

    if criteria.accords and not all(a.lower() in meta["_accords_set"] for a in criteria.accords):
        return False

    if criteria.notes and not all(n.lower() in meta["_notes_set"] for n in criteria.notes):
        return False

    if _contains_forbidden(meta, card, criteria.not_accords):
        return False

    if _contains_forbidden(meta, card, criteria.not_notes):
        return False

    if criteria.min_rating is not None:
        rating = meta.get("rating")
        votes = meta.get("popularity", 0)
        if rating is None or rating < criteria.min_rating or votes <= 0:
            return False

    if criteria.min_votes is not None and meta.get("popularity", 0) < criteria.min_votes:
        return False

    if criteria.min_longevity is not None and _metric(meta, "longevity") < criteria.min_longevity:
        return False

    if criteria.min_sillage is not None and _metric(meta, "sillage") < criteria.min_sillage:
        return False

    if criteria.min_value is not None and _metric(meta, "price_value") < criteria.min_value:
        return False

    return True


def find_matches(perfumes: list[Perfume], criteria: Criteria, limit: int | None = None) -> list[Perfume]:
    key_fn = (
        (lambda p: sort_value(p, criteria.sort_by))
        if criteria.sort_by
        else (lambda p: float(p["metadata"].get("popularity", 0) or 0))
    )

    if limit is None:
        matches = [p for p in perfumes if match_perfume(p, criteria)]
        matches.sort(key=key_fn, reverse=criteria.descending)
        return matches

    scored = (
        (key_fn(perfume), index, perfume)
        for index, perfume in enumerate(perfumes)
        if match_perfume(perfume, criteria)
    )
    if criteria.descending:
        best = heapq.nlargest(limit, scored, key=lambda item: (item[0], -item[1]))
    else:
        best = heapq.nsmallest(limit, scored, key=lambda item: (item[0], item[1]))
    return [perfume for _score, _index, perfume in best]


def sort_value(perfume: Perfume, sort_by: SortField) -> float:
    meta = perfume["metadata"]
    if sort_by == "popularity":
        return float(meta.get("popularity", 0) or 0)
    if sort_by == "rating":
        rating = meta.get("rating")
        votes = meta.get("popularity", 0)
        return float(rating or 0.0) if votes > 0 else 0.0
    if sort_by == "longevity":
        return _metric(meta, "longevity")
    if sort_by == "sillage":
        return _metric(meta, "sillage")
    if sort_by == "value":
        return _metric(meta, "price_value")
    raise ValueError(f"Unsupported sort field: {sort_by}")


def violates_any_filter(perfume: Perfume, criteria: Criteria) -> bool:
    return not match_perfume(perfume, criteria)


def _contains_forbidden(meta: dict, card: str, forbidden_items: tuple[str, ...]) -> bool:
    if not forbidden_items:
        return False

    notes = meta["_notes_set"]
    accords = meta["_accords_set"]
    combined_meta = " ".join(notes) + " " + " ".join(accords)

    for item in forbidden_items:
        item_low = item.lower()
        if item_low in notes or item_low in accords:
            return True
        pattern = rf"\b{re.escape(item_low)}\b"
        if re.search(pattern, combined_meta) or re.search(pattern, card):
            return True
    return False


def _metric(meta: dict, key: str) -> float:
    value = meta.get(key)
    return float(value) if value is not None and value > 0.0 else 0.0
