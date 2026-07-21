from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from research.runtime.grounding_checker import (
    extract_excluded_terms_from_user,
    parse_context_cards,
)


DEFAULT_INPUT = Path("train_set/finetune/gemma/full_validation.jsonl")
DEFAULT_OUTPUT = Path("train_set/eval/scentai_eval_v2.jsonl")

CATEGORY_TARGETS = {
    "strict_filter": 30,
    "database_lookup": 25,
    "reference_similarity": 25,
    "occasion_context": 25,
    "collection_advice": 20,
    "long_context": 20,
    "general_recommendation": 35,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed stratified ScentAI eval set.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cases", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    records = load_records(args.input)
    candidates = [make_case(item, args.input, idx) for idx, item in enumerate(records)]
    candidates = [case for case in candidates if case is not None]

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in candidates:
        by_category[case["category"]].append(case)

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    for category, target in CATEGORY_TARGETS.items():
        bucket = by_category.get(category, [])
        rng.shuffle(bucket)
        selected.extend(bucket[:target])

    if len(selected) < args.max_cases:
        used = {case["source_index"] for case in selected}
        leftovers = [case for case in candidates if case["source_index"] not in used]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: args.max_cases - len(selected)])

    selected = selected[: args.max_cases]
    for index, case in enumerate(selected, 1):
        case["id"] = f"scentai_eval_v2_{index:04d}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, selected)

    manifest = {
        "version": "scentai_eval_v2",
        "source": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "case_count": len(selected),
        "category_counts": category_counts(selected),
        "difficulty_counts": value_counts(selected, "difficulty"),
        "mode_counts": value_counts(selected, "mode"),
        "expected_forbidden_removed_count": sum(
            len(case["checks"].get("expected_forbidden_removed", [])) for case in selected
        ),
        "category_targets": CATEGORY_TARGETS,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def make_case(item: dict[str, Any], source: Path, source_index: int) -> dict[str, Any] | None:
    messages = item.get("messages", [])
    system = next((msg.get("content", "") for msg in messages if msg.get("role") == "system"), "")
    user = next((msg.get("content", "") for msg in messages if msg.get("role") == "user"), "")
    expected = next(
        (msg.get("content", "") for msg in messages if msg.get("role") in {"assistant", "model"}),
        "",
    )
    context = extract_context(user)
    if not user or not expected or not context:
        return None

    cards = parse_context_cards(context)
    if not cards:
        return None

    excluded_terms = extract_excluded_terms_from_user(user)
    raw_expected_perfumes = extract_context_perfumes_mentioned(expected, cards)
    category = classify_case(user, expected, cards, excluded_terms)
    forbidden_perfumes = forbidden_perfumes_for_terms(cards, excluded_terms)
    forbidden_normalized = {normalize_name(name) for name in forbidden_perfumes}
    expected_perfumes = [
        name for name in raw_expected_perfumes if normalize_name(name) not in forbidden_normalized
    ]
    expected_forbidden_removed = [
        name for name in raw_expected_perfumes if normalize_name(name) in forbidden_normalized
    ]
    mode = infer_mode(category)

    return {
        "id": "",
        "version": "scentai_eval_v2",
        "category": category,
        "mode": mode,
        "difficulty": infer_difficulty(cards, excluded_terms, expected, category),
        "tags": infer_tags(user, expected, cards, excluded_terms, category),
        "source_file": str(source),
        "source_index": source_index,
        "system": system,
        "user": user,
        "context": context,
        "expected": expected,
        "checks": {
            "context_only": True,
            "strict_filter": bool(excluded_terms),
            "field_copy": should_check_field_copy(expected, cards),
            "expected_perfumes": expected_perfumes,
            "raw_expected_perfumes": raw_expected_perfumes,
            "expected_forbidden_removed": expected_forbidden_removed,
            "forbidden_perfumes": forbidden_perfumes,
            "excluded_terms": excluded_terms,
            "context_perfume_count": len(cards),
            "minimum_context_mentions": 1,
        },
    }


def classify_case(
    user: str,
    expected: str,
    cards: list[dict[str, Any]],
    excluded_terms: list[str],
) -> str:
    tail = user.split("[/PERFUMES]")[-1].lower()
    expected_lower = expected.lower()
    if should_check_field_copy(expected, cards):
        return "database_lookup"
    if excluded_terms:
        return "strict_filter"
    if re.search(r"\b(like|similar|alternative|instead of|reminds me of|dupe|clone)\b", tail):
        return "reference_similarity"
    if re.search(r"\b(office|work|date|wedding|party|gym|school|summer|winter|spring|autumn|fall|night|day)\b", tail):
        return "occasion_context"
    if re.search(r"\b(collection|wardrobe|rotation|gap|own|already have|next addition)\b", tail + " " + expected_lower):
        return "collection_advice"
    if len(cards) >= 10:
        return "long_context"
    return "general_recommendation"


def should_check_field_copy(expected: str, cards: list[dict[str, Any]]) -> bool:
    if len(cards) != 1:
        return False
    lower = expected.lower()
    return "database record" in lower or bool(re.search(r"(?m)^-\s*(brand|name|gender|rating|accords|notes):", lower))


def infer_mode(category: str) -> str:
    if category == "database_lookup":
        return "database_lookup"
    if category == "collection_advice":
        return "collection_advice"
    return "recommendation"


def infer_difficulty(
    cards: list[dict[str, Any]],
    excluded_terms: list[str],
    expected: str,
    category: str,
) -> str:
    score = 0
    if len(cards) >= 10:
        score += 1
    if excluded_terms:
        score += 1
    if category in {"database_lookup", "reference_similarity", "collection_advice"}:
        score += 1
    if len(expected) > 900:
        score += 1
    if score >= 3:
        return "hard"
    if score == 2:
        return "medium"
    return "easy"


def infer_tags(
    user: str,
    expected: str,
    cards: list[dict[str, Any]],
    excluded_terms: list[str],
    category: str,
) -> list[str]:
    text = f"{user} {expected}".lower()
    tags = {category}
    if excluded_terms:
        tags.add("negative_constraints")
    if len(cards) == 1:
        tags.add("single_card")
    if len(cards) >= 10:
        tags.add("long_context")
    if "0.00/5" in user or "0 votes" in user:
        tags.add("zero_rating")
    if re.search(r"\b(office|work|date|wedding|party|gym|school)\b", text):
        tags.add("occasion")
    if re.search(r"\b(summer|winter|spring|autumn|fall)\b", text):
        tags.add("season")
    if re.search(r"\b(day|night)\b", text):
        tags.add("time_profile")
    if re.search(r"\b(like|similar|alternative|dupe|clone)\b", text):
        tags.add("reference")
    return sorted(tags)


def forbidden_perfumes_for_terms(cards: list[dict[str, Any]], excluded_terms: list[str]) -> list[str]:
    if not excluded_terms:
        return []
    excluded = {term.lower() for term in excluded_terms}
    forbidden = []
    for card in cards:
        card_terms = {term.lower() for term in card.get("accords", []) + card.get("notes", [])}
        if excluded & card_terms:
            forbidden.append(card["name"])
    return forbidden


def extract_context(user: str) -> str:
    match = re.search(r"\[PERFUMES\].*?\[/PERFUMES\]", user, flags=re.S)
    return match.group(0) if match else ""


def extract_context_perfumes_mentioned(expected: str, cards: list[dict[str, Any]]) -> list[str]:
    lower = expected.lower()
    return [card["name"] for card in cards if normalize_name(card["name"]) in normalize_name(lower)]


def normalize_name(value: str) -> str:
    value = re.sub(r"\(\d{4}\)", " ", value.lower())
    value = re.sub(r"\b(?:19|20)\d{2}\b", " ", value)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", " ", value)).strip()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    return dict(sorted(counts.items()))


def value_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    main()
