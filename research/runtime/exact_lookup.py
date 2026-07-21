from __future__ import annotations

import re
from typing import Any

from research.runtime.grounding_checker import parse_context_cards


LOOKUP_PATTERNS = (
    re.compile(r"\b(database record|database says|recorded parameters|all fields)\b", re.I),
    re.compile(r"\b(display|show|list|give me|what are|what is)\b.*\b(fields|notes|accords|rating|longevity|sillage|value)\b", re.I),
    re.compile(r"\b(display|show|list|give me|what are|what is)\b.*\b(accord profile|seasonal suitability|suitability data|seasonal data|time profile)\b", re.I),
    re.compile(r"\b(exact|verbatim|full)\b.*\b(record|notes|accords|fields|database)\b", re.I),
)


def maybe_answer_exact_lookup(user_message: str) -> str | None:
    """Return a deterministic database answer when the user asks for exact fields."""
    if not is_exact_lookup_query(user_message):
        return None
    return render_database_lookup_answer(user_message)


def is_exact_lookup_query(user_message: str) -> bool:
    """Detect requests that should bypass the LLM and copy database fields exactly."""
    tail = user_message.split("[/PERFUMES]")[-1] if "[/PERFUMES]" in user_message else user_message
    return any(pattern.search(tail) for pattern in LOOKUP_PATTERNS)


def render_database_lookup_answer(user_or_context: str) -> str | None:
    context = extract_perfume_context(user_or_context)
    if not context:
        return None
    cards = parse_context_cards(context)
    if not cards:
        return None

    selected = select_requested_card(user_or_context, cards)
    if selected is None:
        if len(cards) != 1:
            return None
        selected = cards[0]
    return format_database_record(selected)


def extract_perfume_context(text: str) -> str:
    match = re.search(r"\[PERFUMES\].*?\[/PERFUMES\]", text, flags=re.S)
    return match.group(0) if match else ""


def select_requested_card(text: str, cards: list[dict[str, Any]]) -> dict[str, Any] | None:
    tail = text.split("[/PERFUMES]")[-1] if "[/PERFUMES]" in text else text
    tail_norm = normalize_name(tail)
    matches = [card for card in cards if normalize_name(card["name"]) in tail_norm]
    if len(matches) == 1:
        return matches[0]
    return cards[0] if len(cards) == 1 else None


def format_database_record(card: dict[str, Any]) -> str:
    display_name = card["name"]
    name, brand, launch_year = split_perfume_label(display_name)
    fields = card.get("fields", {})
    launch_year = launch_year or fields.get("launch_year")

    lines = [f"Database Record - {display_name}:"]
    if brand:
        lines.append(f"- Brand: {brand}")
    if name:
        lines.append(f"- Name: {name}")
    lines.append(f"- Gender: {card.get('gender', '')}")
    if launch_year:
        lines.append(f"- Launch Year: {launch_year}")

    append_scalar(lines, "Rating", fields.get("rating"))
    append_scalar(lines, "Longevity", fields.get("longevity"))
    append_scalar(lines, "Sillage", fields.get("sillage"))
    append_scalar(lines, "Value", fields.get("value"))
    append_scalar(lines, "Best Seasons", fields.get("best_seasons"))
    append_scalar(lines, "Time Profile", fields.get("time_profile"))

    if card.get("accords"):
        lines.append("- Accords: " + ", ".join(card["accords"]))
    if card.get("notes"):
        lines.append("- Notes: " + ", ".join(card["notes"]))
    append_scalar(lines, "Description", fields.get("description"))
    return "\n".join(lines)


def append_scalar(lines: list[str], label: str, value: object) -> None:
    if value is None or value == "":
        return
    lines.append(f"- {label}: {value}")


def split_perfume_label(label: str) -> tuple[str, str, str | None]:
    year_match = re.search(r"\((?P<year>(?:19|20)\d{2})\)\s*$", label)
    launch_year = year_match.group("year") if year_match else None
    clean_label = re.sub(r"\s*\((?:19|20)\d{2}\)\s*$", "", label).strip()
    if " by " not in clean_label:
        return clean_label, "", launch_year
    name, brand = clean_label.rsplit(" by ", 1)
    return name.strip(), brand.strip(), launch_year


def normalize_name(value: str) -> str:
    value = re.sub(r"\(\d{4}\)", " ", value.lower())
    value = re.sub(r"\b(?:19|20)\d{2}\b", " ", value)
    value = re.sub(r"[^\w\s']+", " ", value)
    return " ".join(value.split())
