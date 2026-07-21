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

INTERNAL_LABEL_PATTERNS = [
    re.compile(r"\bwinter_evening\b"),
    re.compile(r"\bdate_night\b"),
    re.compile(r"\bgym_after\b"),
    re.compile(r"\bsummer_day\b"),
]

REPETITIVE_QUALITY_PATTERNS = [
    re.compile(r"which makes it a grounded fit for the request", re.I),
    re.compile(r"Based on your preferences, I would prioritize perfumes that fit the .* direction", re.I),
]


def validate_l4_dataset(path: Path, clean_file: Path = DEFAULT_CLEAN_FILE) -> int:
    perfumes_by_id = _load_perfumes_by_id(clean_file)
    violations = {
        "json": 0,
        "format": 0,
        "rag_subset": 0,
        "best_pick": 0,
        "unsupported_claim": 0,
        "internal_label": 0,
        "repetitive_phrase": 0,
        "negative": 0,
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

            if not answer_items or "Why:" not in answer:
                violations["format"] += 1

            if best_pick and best_pick.lower() not in {item.lower() for item in answer_items}:
                violations["best_pick"] += 1
            elif not best_pick:
                violations["best_pick"] += 1

            if "[PERFUMES]" in user:
                context_items = _parse_context_items(user)
                if any(item.lower() not in context_items for item in answer_items):
                    violations["rag_subset"] += 1

            if any(pattern.search(answer) for pattern in UNSUPPORTED_PATTERNS):
                violations["unsupported_claim"] += 1

            if any(pattern.search(answer) for pattern in INTERNAL_LABEL_PATTERNS):
                violations["internal_label"] += 1

            if any(pattern.search(answer) for pattern in REPETITIVE_QUALITY_PATTERNS):
                violations["repetitive_phrase"] += 1

            if "_meta" in row:
                _validate_debug_meta(row["_meta"], perfumes_by_id, violations)

    print("L4 validation report")
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


def _validate_debug_meta(meta: dict, perfumes_by_id: dict[Any, dict], violations: dict[str, int]) -> None:
    profile = meta.get("profile") or {}
    avoid_terms = {term.lower() for term in (profile.get("avoid_accords") or [])}
    avoid_terms |= {term.lower() for term in (profile.get("avoid_notes") or [])}

    best_pick_id = meta.get("best_pick_id")
    if best_pick_id not in set(meta.get("answer_ids") or []):
        violations["best_pick"] += 1

    if not avoid_terms:
        return

    for perfume_id in meta.get("answer_ids") or []:
        perfume = perfumes_by_id.get(perfume_id)
        if not perfume:
            continue
        pmeta = perfume["metadata"]
        terms = {x.lower() for x in (pmeta.get("accords_list") or [])}
        terms |= {x.lower() for x in (pmeta.get("notes_list") or [])}
        if any(term in terms for term in avoid_terms):
            violations["negative"] += 1
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ScentAI L4 JSONL records.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--clean-file", type=Path, default=DEFAULT_CLEAN_FILE)
    args = parser.parse_args()
    raise SystemExit(validate_l4_dataset(args.path, args.clean_file))


if __name__ == "__main__":
    main()
