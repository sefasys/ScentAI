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


def validate_l3_dataset(path: Path, clean_file: Path = DEFAULT_CLEAN_FILE) -> int:
    perfumes_by_id = _load_perfumes_by_id(clean_file)
    violations = {
        "json": 0,
        "list_format": 0,
        "rag_subset": 0,
        "gender": 0,
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
                violations["list_format"] += 1
                continue

            user = messages[1].get("content", "")
            answer = messages[2].get("content", "")
            answer_names = _parse_answer_names(answer)

            if not answer_names or len(answer_names) != len([line for line in answer.splitlines() if line.strip()]):
                violations["list_format"] += 1

            if "[PERFUMES]" in user:
                context_names = _parse_context_names(user)
                if any(name.lower() not in context_names for name in answer_names):
                    violations["rag_subset"] += 1

            if "_meta" in row:
                _validate_debug_meta(row["_meta"], perfumes_by_id, violations)

    print("L3 validation report")
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


def _parse_answer_names(answer: str) -> list[str]:
    names = []
    for line in answer.splitlines():
        match = re.match(r"^\d+\.\s+(.+?)\s+by\s+.+$", line.strip())
        if match:
            names.append(match.group(1).strip())
    return names


def _parse_context_names(user: str) -> set[str]:
    block_match = re.search(r"\[PERFUMES\](.*?)\[/PERFUMES\]", user, re.DOTALL)
    if not block_match:
        return set()
    context_names = set()
    for match in re.finditer(
        r"^(.+?)\s+by\s+.+?(?:\s+\(\d{4}\))?\s+[—-]\s+(?:male|female|unisex)$",
        block_match.group(1),
        re.M,
    ):
        context_names.add(match.group(1).strip().lower())
    return context_names


def _validate_debug_meta(meta: dict, perfumes_by_id: dict[Any, dict], violations: dict[str, int]) -> None:
    profile = meta.get("profile") or {}
    allowed_genders = set(profile.get("gender_any_of") or [])
    avoid_terms = {term.lower() for term in (profile.get("avoid_accords") or [])}
    avoid_terms |= {term.lower() for term in (profile.get("avoid_notes") or [])}

    for perfume_id in meta.get("answer_ids") or []:
        perfume = perfumes_by_id.get(perfume_id)
        if not perfume:
            continue
        pmeta = perfume["metadata"]
        if allowed_genders and pmeta.get("gender") not in allowed_genders:
            violations["gender"] += 1
            break
        if avoid_terms:
            terms = {x.lower() for x in (pmeta.get("accords_list") or [])}
            terms |= {x.lower() for x in (pmeta.get("notes_list") or [])}
            if any(term in terms for term in avoid_terms):
                violations["negative"] += 1
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ScentAI L3 JSONL records.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--clean-file", type=Path, default=DEFAULT_CLEAN_FILE)
    args = parser.parse_args()
    raise SystemExit(validate_l3_dataset(args.path, args.clean_file))


if __name__ == "__main__":
    main()
