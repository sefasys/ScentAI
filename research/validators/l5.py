from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.core.config import DEFAULT_CLEAN_FILE


UNSUPPORTED_PATTERNS = [
    re.compile(r"\$\d+"),
    re.compile(r"\bprice\b", re.I),
    re.compile(r"\bbudget\b", re.I),
    re.compile(r"\bguaranteed compliments?\b", re.I),
    re.compile(r"\bcompliment beast\b", re.I),
]


def validate_l5_dataset(path: Path, clean_file: Path = DEFAULT_CLEAN_FILE) -> int:
    perfumes_by_id = _load_perfumes_by_id(clean_file)
    violations = {
        "json": 0,
        "format": 0,
        "profile_block": 0,
        "rag_subset": 0,
        "best_pick": 0,
        "unsupported_claim": 0,
        "disliked_perfume": 0,
        "previous_repeat": 0,
        "negative": 0,
        "empty_profile_misuse": 0,
        "conflict": 0,
    }
    total = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                violations["json"] += 1
                continue

            total += 1
            messages = row.get("messages") or []
            if len(messages) != 3:
                violations["format"] += 1
                continue

            user = messages[1].get("content", "")
            answer = messages[2].get("content", "")
            answer_items = _parse_answer_items(answer)
            best_pick = _parse_best_pick(answer)

            if "[USER PROFILE]" not in user or "[/USER PROFILE]" not in user:
                violations["profile_block"] += 1

            if not answer_items or "Why:" not in answer:
                violations["format"] += 1

            answer_item_set = {item.lower() for item in answer_items}
            if not best_pick or best_pick.lower() not in answer_item_set:
                violations["best_pick"] += 1

            if "[PERFUMES]" in user:
                context_items = _parse_context_items(user)
                if any(item.lower() not in context_items for item in answer_items):
                    violations["rag_subset"] += 1

            if any(pattern.search(answer) for pattern in UNSUPPORTED_PATTERNS):
                violations["unsupported_claim"] += 1

            if "_meta" in row:
                _validate_debug_meta(row["_meta"], answer, perfumes_by_id, violations)

    print("L5 validation report")
    print(f"Total records          : {total}")
    for key, value in violations.items():
        print(f"{key:22s}: {value}")
    return sum(violations.values())


def _load_perfumes_by_id(clean_file: Path) -> dict[Any, dict]:
    by_id = {}
    with clean_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            perfume = json.loads(line)
            by_id[perfume.get("id")] = perfume
    return by_id


def _parse_answer_items(answer: str) -> list[str]:
    items = []
    for line in answer.splitlines():
        match = re.match(r"^\d+\.\s+(.+?\s+by\s+.+)$", line.strip())
        if match:
            items.append(match.group(1).strip())
    return items


def _parse_best_pick(answer: str) -> str:
    for line in answer.splitlines():
        match = re.match(r"^Best pick:\s+(.+?\s+by\s+.+)$", line.strip())
        if match:
            return match.group(1).strip()
    return ""


def _parse_context_items(user: str) -> set[str]:
    block_match = re.search(r"\[PERFUMES\](.*?)\[/PERFUMES\]", user, re.DOTALL)
    if not block_match:
        return set()
    context_items = set()
    for match in re.finditer(
        r"^(.+?\s+by\s+.+?)(?:\s+\(\d{4}\))?\s+[—-]\s+(?:male|female|unisex)$",
        block_match.group(1),
        re.M,
    ):
        context_items.add(match.group(1).strip().lower())
    return context_items


def _validate_debug_meta(meta: dict, answer: str, perfumes_by_id: dict[Any, dict], violations: dict[str, int]) -> None:
    user_profile = meta.get("user_profile") or {}
    answer_ids = set(meta.get("answer_ids") or [])
    disliked_ids = set(user_profile.get("disliked_perfume_ids") or [])
    previous_ids = set(user_profile.get("previously_recommended_ids") or [])

    if disliked_ids & answer_ids:
        violations["disliked_perfume"] += 1

    if meta.get("category") == "avoid_previous_recommendations" and previous_ids & answer_ids:
        violations["previous_repeat"] += 1

    if user_profile.get("empty") and re.search(r"\byour (known|usual|stored) taste\b", answer, re.I):
        violations["empty_profile_misuse"] += 1

    if meta.get("conflict") and not re.search(r"\b(conflict|compromise|dislike|tradeoff|safer)\b", answer, re.I):
        violations["conflict"] += 1

    avoid_terms = {term.lower() for term in (user_profile.get("disliked_notes") or [])}
    avoid_terms |= {term.lower() for term in (user_profile.get("disliked_accords") or [])}
    if not avoid_terms or meta.get("profile_update") or meta.get("conflict"):
        return

    for perfume_id in answer_ids:
        perfume = perfumes_by_id.get(perfume_id)
        if not perfume:
            continue
        pmeta = perfume["metadata"]
        terms = {x.lower() for x in (pmeta.get("accords_list") or [])}
        terms |= {x.lower() for x in (pmeta.get("notes_list") or [])}
        if avoid_terms & terms:
            violations["negative"] += 1
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ScentAI L5 JSONL records.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--clean-file", type=Path, default=DEFAULT_CLEAN_FILE)
    args = parser.parse_args()
    raise SystemExit(validate_l5_dataset(args.path, args.clean_file))


if __name__ == "__main__":
    main()
