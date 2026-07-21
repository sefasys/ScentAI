from __future__ import annotations


def fmt_list(items: list[str] | tuple[str, ...]) -> str:
    return ", ".join(items) if items else "none recorded"


def fmt_val(value: float | None, suffix: str = "") -> str:
    return f"{value:.2f}{suffix}" if value is not None and value > 0.0 else "N/A"


def fmt_rating(rating: float | None, popularity: int | None) -> str:
    votes = popularity or 0
    if rating is not None and rating > 0.0 and votes > 0:
        return f"{rating:.2f}/5 ({votes} votes)"
    return "N/A"


def title_items(items: list[str]) -> list[str]:
    return [item.capitalize() for item in items]
