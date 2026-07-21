"""Lightweight grounding checks for ScentAI generation smoke tests."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CARD_HEADER_RE = re.compile(
    r"^(?P<name>.+?)\s+(?:-|—)\s+(?P<gender>male|female|unisex)\s*$",
    re.I,
)
PERFUME_LINE_RE = re.compile(
    r"^\s*(?:\d+[\.)]|[-*])\s+(?P<name>[^:\n]+?\s+by\s+[^,:;—\n]+?)(?:\s*[,;—|]\s*.*)?\s*$",
    re.I,
)
BEST_PICK_RE = re.compile(
    r"^\s*Best pick:\s*(?P<name>[^:\n]+?\s+by\s+[^,:;—\n]+?)(?:\s*[,;—|]\s*.*)?\s*$",
    re.I,
)
ANSWER_FIELD_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?P<field>Brand|Name|Gender|Launch Year|Rating|Longevity|Sillage|Value|Best Seasons|Time Profile|Accords|Notes)\s*:\s*(?P<value>.+?)\s*$",
    re.I,
)
NEGATIVE_PATTERNS = [
    re.compile(
        r"\b(?:without|no|not|avoid|excluding|hate|dislike|not a fan of|don't like|do not like)\s+"
        r"(?:anything with|anything too|too much|a lot of|any)?\s*"
        r"([a-zA-ZğüşöçıİĞÜŞÖÇ\s-]{2,35})",
        re.I,
    ),
    re.compile(r"\b(?:less|not too|not so)\s+([a-zA-ZğüşöçıİĞÜŞÖÇ\s-]{2,35})", re.I),
]
KNOWN_NEGATIVE_TERMS = {
    "aldehydic",
    "amber",
    "animalic",
    "aquatic",
    "aromatic",
    "bergamot",
    "cardamom",
    "cedar",
    "cinnamon",
    "coconut",
    "citrus",
    "coffee",
    "fresh",
    "fresh spicy",
    "grapefruit",
    "green",
    "incense",
    "iris",
    "jasmine",
    "lavender",
    "leather",
    "lemon",
    "mandarin orange",
    "musky",
    "musk",
    "oakmoss",
    "orange",
    "oud",
    "patchouli",
    "pink pepper",
    "powdery",
    "rose",
    "sandalwood",
    "smoky",
    "suede",
    "sweet",
    "tobacco",
    "tonka bean",
    "vanilla",
    "vetiver",
    "warm spicy",
    "woody",
}
NOTE_ALIASES = {"vanille": "vanilla", "vanilya": "vanilla", "agarwood": "oud", "agarwood oud": "oud"}


def parse_context_cards(context: str) -> list[dict[str, Any]]:
    """Parse the compact [PERFUMES] context format used by runtime prompts."""
    body = context.replace("[PERFUMES]", "").replace("[/PERFUMES]", "").strip()
    if not body:
        return []

    cards: list[dict[str, Any]] = []
    for block in re.split(r"\n\s*\n", body):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        match = CARD_HEADER_RE.match(lines[0])
        if not match:
            continue

        card: dict[str, Any] = {
            "name": match.group("name").strip(),
            "gender": match.group("gender").lower(),
            "accords": [],
            "notes": [],
            "fields": {"gender": match.group("gender").lower()},
            "raw": block,
        }
        for line in lines[1:]:
            lower = line.lower()
            for key, value in _parse_context_field_segments(line):
                if key == "accords":
                    card["accords"] = _split_fact_list(value)
                    card["fields"]["accords"] = card["accords"]
                elif key in {"notes", "top_notes", "middle_notes", "base_notes"}:
                    card["notes"].extend(_split_fact_list(value.replace("|", ",")))
                    card["fields"]["notes"] = card["notes"]
                elif key == "time":
                    card["fields"]["time_profile"] = _normalize_scalar_fact(value)
                else:
                    card["fields"][key] = _normalize_scalar_fact(value)
            if lower.startswith(("top notes:", "middle notes:", "base notes:")):
                card["fields"]["notes"] = card["notes"]
        cards.append(card)
    return cards


def score_quick_inference_payload(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", [])
    case_reports = [score_case_result(item) for item in results]
    hard_failures = sum(1 for item in case_reports if not item["pass"])
    return {
        "case_count": len(case_reports),
        "pass_count": len(case_reports) - hard_failures,
        "hard_failure_count": hard_failures,
        "pass_rate": (len(case_reports) - hard_failures) / max(len(case_reports), 1),
        "cases": case_reports,
    }


def score_case_result(item: dict[str, Any]) -> dict[str, Any]:
    context = item.get("context", "")
    answer = item.get("answer", "")
    answer_lower = answer.lower()
    cards = parse_context_cards(context)
    context_names = [card["name"] for card in cards]

    mentioned_context_perfumes = [name for name in context_names if _contains_name(answer_lower, name)]
    mentioned_perfume_like_names = _extract_perfume_like_names(answer)
    unsupported_perfume_mentions = [
        name
        for name in mentioned_perfume_like_names
        if not any(_names_match(name, context_name) for context_name in context_names)
    ]

    forbidden_perfumes = item.get("forbidden_perfumes", [])
    forbidden_perfume_mentions = [
        name for name in forbidden_perfumes if _contains_name(answer_lower, name)
    ]

    excluded_accords = [term.lower() for term in item.get("excluded_accords", [])]
    excluded_terms = sorted(set(excluded_accords + item.get("excluded_terms", [])))
    context_filter_leaks = []
    for card in cards:
        card_terms = {term.lower() for term in card.get("accords", []) + card.get("notes", [])}
        leaks = sorted(set(excluded_terms) & card_terms)
        if leaks:
            context_filter_leaks.append({"name": card["name"], "excluded_accords": leaks})

    strict_filter_violations = []
    if excluded_terms:
        by_name = {card["name"]: card for card in cards}
        for name in mentioned_context_perfumes:
            card = by_name.get(name)
            if not card:
                continue
            card_terms = {term.lower() for term in card.get("accords", []) + card.get("notes", [])}
            leaks = sorted(set(excluded_terms) & card_terms)
            if leaks:
                strict_filter_violations.append({"name": name, "excluded_terms": leaks})

    unsupported_note_claims = _find_unsupported_note_claims(answer, cards)
    field_copy_violations = _find_field_copy_violations(answer, cards)

    warning_terms = _find_unsupported_fact_terms(answer, cards)
    warnings = []
    if not mentioned_context_perfumes:
        warnings.append("No known context perfume name was detected in the answer.")
    if warning_terms:
        warnings.append("Answer mentions accord/note-like terms not attached to any detected recommended card.")

    hard_fail_reasons = []
    if forbidden_perfume_mentions:
        hard_fail_reasons.append("Forbidden perfume appeared in the answer.")
    if unsupported_perfume_mentions:
        hard_fail_reasons.append("Answer mentioned a perfume that is not in the provided context.")
    if strict_filter_violations:
        hard_fail_reasons.append("Answer recommended a perfume containing an excluded term.")
    if unsupported_note_claims:
        hard_fail_reasons.append("Answer claimed notes not shown in the matching perfume card.")
    if field_copy_violations:
        hard_fail_reasons.append("Answer changed explicit database fields from the context card.")

    return {
        "name": item.get("name"),
        "pass": not hard_fail_reasons,
        "hard_fail_reasons": hard_fail_reasons,
        "warnings": warnings,
        "mentioned_context_perfumes": mentioned_context_perfumes,
        "unsupported_perfume_mentions": unsupported_perfume_mentions,
        "forbidden_perfume_mentions": forbidden_perfume_mentions,
        "excluded_terms": excluded_terms,
        "context_filter_leaks": context_filter_leaks,
        "strict_filter_violations": strict_filter_violations,
        "unsupported_note_claims": unsupported_note_claims,
        "field_copy_violations": field_copy_violations,
        "unsupported_fact_terms_warning": warning_terms,
    }


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _split_fact_list(value: str) -> list[str]:
    return [_normalize_list_fact(part) for part in _split_csv(value) if _normalize_list_fact(part)]


def _contains_name(answer_lower: str, name: str) -> bool:
    if name.lower() in answer_lower:
        return True
    normalized_answer = _normalize_name(answer_lower)
    normalized_name = _normalize_name(name)
    return bool(normalized_name and normalized_name in normalized_answer)


def _names_match(left: str, right: str) -> bool:
    return _normalize_name(left) == _normalize_name(right)


def _normalize_name(name: str) -> str:
    value = re.sub(r"\(\d{4}\)", " ", name.lower())
    value = re.sub(r"\b(?:19|20)\d{2}\b", " ", value)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", " ", value)).strip()


def _normalize_field_key(field: str) -> str:
    return field.strip().lower().replace(" ", "_")


def _normalize_scalar_fact(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip().lower())
    return value.strip(" .")


def _normalize_list_fact(value: str) -> str:
    value = re.sub(r"\([^)]*\)", " ", value.lower())
    value = re.sub(r"\s+", " ", value).strip(" .;-")
    return NOTE_ALIASES.get(value, value)


def _parse_context_field_segments(line: str) -> list[tuple[str, str]]:
    segments = []
    for part in line.split("|"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        normalized_key = _normalize_field_key(key)
        segments.append((normalized_key, value.strip()))
    return segments


def _extract_perfume_like_names(answer: str) -> list[str]:
    names = []
    for line in answer.splitlines():
        stripped = line.strip()
        match = PERFUME_LINE_RE.search(stripped) or BEST_PICK_RE.search(stripped)
        if not match:
            continue
        name = match.group("name").strip(" -*")
        if len(name.split()) >= 3:
            names.append(name)
    return names


def _extract_answer_fields(answer: str) -> dict[str, str]:
    fields = {}
    for raw_line in answer.splitlines():
        match = ANSWER_FIELD_RE.match(raw_line.strip())
        if not match:
            continue
        fields[_normalize_field_key(match.group("field"))] = match.group("value").strip()
    return fields


def _find_field_copy_violations(answer: str, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Catch database-record style answers that alter explicit card fields."""
    answer_fields = _extract_answer_fields(answer)
    if not answer_fields or len(cards) != 1:
        return []

    card = cards[0]
    card_fields = card.get("fields", {})
    violations = []

    for field in ("accords", "notes"):
        if field not in answer_fields:
            continue
        claimed = _split_fact_list(answer_fields[field])
        expected = [_normalize_list_fact(value) for value in card.get(field, [])]
        expected_set = set(expected)
        unsupported = [value for value in claimed if value not in expected_set]
        missing = [value for value in expected if value not in set(claimed)]
        if unsupported or missing:
            violations.append(
                {
                    "field": field,
                    "unsupported_values": unsupported,
                    "missing_values": missing,
                    "expected_values": expected,
                    "claimed_values": claimed,
                }
            )

    for field in ("gender", "rating", "longevity", "sillage", "value", "best_seasons", "time_profile"):
        if field not in answer_fields:
            continue
        expected = card_fields.get(field)
        claimed = _normalize_scalar_fact(answer_fields[field])
        if expected is None:
            violations.append({"field": field, "claimed_value": claimed, "reason": "field not present in context"})
            continue
        if field in {"best_seasons", "time_profile"}:
            expected_parts = set(_split_fact_list(str(expected)))
            claimed_parts = set(_split_fact_list(claimed))
            if expected_parts != claimed_parts:
                violations.append(
                    {
                        "field": field,
                        "expected_value": expected,
                        "claimed_value": claimed,
                    }
                )
        elif claimed != str(expected):
            violations.append(
                {
                    "field": field,
                    "expected_value": expected,
                    "claimed_value": claimed,
                }
            )

    return violations


def _find_unsupported_note_claims(answer: str, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find explicit 'notes like ...' claims that are not supported by the card."""
    findings = []
    current_card = None

    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        for card in cards:
            if _contains_name(lowered, card["name"]):
                current_card = card
                break
        if "notes like" not in lowered:
            continue
        if current_card is None:
            findings.append({"name": None, "claimed_notes": [], "reason": "notes-like claim has no matched perfume"})
            continue
        card = current_card
        supported_notes = {note.lower() for note in card.get("notes", [])}
        if not supported_notes:
            findings.append({
                "name": card["name"],
                "claimed_notes": _extract_notes_like_terms(line),
                "reason": "perfume card has no Notes line",
            })
            continue
        claimed = _extract_notes_like_terms(line)
        unsupported = [note for note in claimed if note.lower() not in supported_notes]
        if unsupported:
            findings.append({
                "name": card["name"],
                "claimed_notes": claimed,
                "unsupported_notes": unsupported,
                "reason": "claimed note not present in perfume card",
            })
    return findings


def _extract_notes_like_terms(line: str) -> list[str]:
    match = re.search(r"notes like\s+(.+?)(?:,?\s+and\s+|\s+and\s+)?(?:a\s+|an\s+)?(?:day|night|spring|summer|autumn|winter|wear|profile|suitability|rating|accord|accords|$)", line, re.I)
    if not match:
        return []
    text = match.group(1)
    text = re.split(r"\band\b|;", text)[0]
    return [part.strip(" .,-") for part in text.split(",") if part.strip(" .,-")]


def extract_excluded_terms_from_user(user_content: str) -> list[str]:
    terms = set()
    vocabulary = _negative_vocabulary_from_user(user_content)
    block_match = re.search(r"\[STRICT FILTERS\](.*?)\[/STRICT FILTERS\]", user_content, flags=re.S | re.I)
    if block_match:
        for line in block_match.group(1).splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if "excluded" not in key.lower():
                continue
            for part in value.split(","):
                term = _normalize_negative_term(part, vocabulary)
                if term:
                    terms.add(term)

    tail = user_content.split("[/PERFUMES]")[-1]
    for pattern in NEGATIVE_PATTERNS:
        for match in pattern.finditer(tail):
            phrase = _normalize_negative_phrase(match.group(1))
            found = False
            for term in sorted(vocabulary, key=len, reverse=True):
                if _contains_term(phrase, term):
                    terms.add(NOTE_ALIASES.get(term, term))
                    found = True
            # If the phrase is not an actual accord/note from this context, keep it out of
            # strict-filter scoring. Phrases like "not too overpowering" are style
            # preferences, not database terms we can deterministically validate.
    return sorted(terms)


def _negative_vocabulary_from_user(user_content: str) -> set[str]:
    vocabulary = {NOTE_ALIASES.get(term, term) for term in KNOWN_NEGATIVE_TERMS}
    match = re.search(r"\[PERFUMES\].*?\[/PERFUMES\]", user_content, flags=re.S)
    if not match:
        return vocabulary
    for card in parse_context_cards(match.group(0)):
        for term in card.get("accords", []) + card.get("notes", []):
            normalized = _normalize_list_fact(term)
            if normalized:
                vocabulary.add(normalized)
    return vocabulary


def _normalize_negative_phrase(value: str) -> str:
    value = re.split(r"\b(?:but|ama)\b", value, maxsplit=1, flags=re.I)[0]
    term = re.sub(r"[^a-zA-ZğüşöçıİĞÜŞÖÇ\s-]+", " ", value.lower())
    return re.sub(r"\s+", " ", term).strip(" -")


def _normalize_negative_term(value: str, vocabulary: set[str] | None = None) -> str:
    term = _normalize_negative_phrase(value)
    if not term or term in {"none", "n/a"}:
        return ""
    for known in sorted(vocabulary or KNOWN_NEGATIVE_TERMS, key=len, reverse=True):
        if _contains_term(term, known):
            return NOTE_ALIASES.get(known, known)
    return NOTE_ALIASES.get(term, term)


def _contains_term(text: str, term: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
    return bool(re.search(pattern, text))


def _find_unsupported_fact_terms(answer: str, cards: list[dict[str, Any]]) -> list[str]:
    """Conservative warning only; not a pass/fail metric."""
    known_terms = set()
    for card in cards:
        known_terms.update(term.lower() for term in card.get("accords", []))
        known_terms.update(term.lower() for term in card.get("notes", []))

    # Terms from our domain vocabulary that commonly get hallucinated.
    vocabulary = {
        "aldehydic",
        "amber",
        "animalic",
        "aquatic",
        "aromatic",
        "citrus",
        "clean",
        "earthy",
        "floral",
        "fresh",
        "fresh spicy",
        "green",
        "iris",
        "lavender",
        "leather",
        "musky",
        "neroli",
        "oud",
        "powdery",
        "rose",
        "smoky",
        "sweet",
        "tobacco",
        "vanilla",
        "warm spicy",
        "woody",
    }
    answer_lower = answer.lower()
    return sorted(term for term in vocabulary if term in answer_lower and term not in known_terms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score ScentAI quick inference grounding.")
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.json_path.read_text(encoding="utf-8"))
    report = score_quick_inference_payload(payload)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.write_report:
        args.write_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
