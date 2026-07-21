from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research.core.data import Perfume


@dataclass(frozen=True)
class UserPreferenceProfile:
    liked_perfumes: tuple[Perfume, ...] = ()
    disliked_perfumes: tuple[Perfume, ...] = ()
    liked_notes: tuple[str, ...] = ()
    disliked_notes: tuple[str, ...] = ()
    liked_accords: tuple[str, ...] = ()
    disliked_accords: tuple[str, ...] = ()
    preferred_gender_any_of: tuple[str, ...] = ()
    preferred_seasons: tuple[str, ...] = ()
    preferred_times: tuple[str, ...] = ()
    previously_recommended: tuple[Perfume, ...] = ()
    confidence: str = "medium"
    notes: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.liked_perfumes,
                self.disliked_perfumes,
                self.liked_notes,
                self.disliked_notes,
                self.liked_accords,
                self.disliked_accords,
                self.preferred_gender_any_of,
                self.preferred_seasons,
                self.preferred_times,
                self.previously_recommended,
                self.notes,
            )
        )


def build_user_profile_context(profile: UserPreferenceProfile) -> str:
    lines = ["[USER PROFILE]"]
    if profile.is_empty:
        lines.append("No stable preferences recorded yet.")
    else:
        _add_perfume_line(lines, "Liked perfumes", profile.liked_perfumes)
        _add_perfume_line(lines, "Disliked perfumes", profile.disliked_perfumes)
        _add_terms_line(lines, "Liked notes", profile.liked_notes)
        _add_terms_line(lines, "Disliked notes", profile.disliked_notes)
        _add_terms_line(lines, "Liked accords", profile.liked_accords)
        _add_terms_line(lines, "Disliked accords", profile.disliked_accords)
        _add_terms_line(lines, "Preferred gender", profile.preferred_gender_any_of)
        _add_terms_line(lines, "Preferred seasons", profile.preferred_seasons)
        _add_terms_line(lines, "Preferred times", profile.preferred_times)
        _add_perfume_line(lines, "Previously recommended", profile.previously_recommended)
        lines.append(f"Profile confidence: {profile.confidence}")
        _add_terms_line(lines, "Profile notes", profile.notes)
    lines.append("[/USER PROFILE]")
    return "\n".join(lines) + "\n\n"


def profile_to_debug(profile: UserPreferenceProfile) -> dict[str, Any]:
    return {
        "liked_perfume_ids": [p.get("id") for p in profile.liked_perfumes],
        "disliked_perfume_ids": [p.get("id") for p in profile.disliked_perfumes],
        "liked_notes": list(profile.liked_notes),
        "disliked_notes": list(profile.disliked_notes),
        "liked_accords": list(profile.liked_accords),
        "disliked_accords": list(profile.disliked_accords),
        "preferred_gender_any_of": list(profile.preferred_gender_any_of),
        "preferred_seasons": list(profile.preferred_seasons),
        "preferred_times": list(profile.preferred_times),
        "previously_recommended_ids": [p.get("id") for p in profile.previously_recommended],
        "confidence": profile.confidence,
        "notes": list(profile.notes),
        "extra": dict(profile.extra),
        "empty": profile.is_empty,
    }


def perfume_label(perfume: Perfume) -> str:
    return f"{perfume['name']} by {perfume['brand']}"


def _add_perfume_line(lines: list[str], label: str, perfumes: tuple[Perfume, ...]) -> None:
    if perfumes:
        lines.append(f"{label}: {'; '.join(perfume_label(p) for p in perfumes)}")


def _add_terms_line(lines: list[str], label: str, terms: tuple[str, ...]) -> None:
    if terms:
        lines.append(f"{label}: {'; '.join(terms)}")
