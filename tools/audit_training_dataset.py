from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable


LEVEL_PATTERNS = {
    "L1": "train_set/L1/training_L1_4000.jsonl",
    "L2": "train_set/L2/l2_main_*.jsonl",
    "L3": "train_set/L3/l3_main_*_debug.jsonl",
    "L4": "train_set/L4/l4_main_*_debug.jsonl",
    "L5": "train_set/L5/l5_main_*_debug.jsonl",
}
EXCLUDED_SOURCE_NAMES = {"l3_main_014_debug.jsonl"}
EXPECTED_LEVEL_COUNTS = {"L1": 4_000, "L2": 6_000, "L3": 7_000, "L4": 10_000, "L5": 5_000}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def message_signature(record: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("Every public record must contain exactly three messages")
    signature = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str) or not content.strip():
            raise ValueError("Every message must contain a non-empty role and content")
        signature.append((role, content))
    return tuple(signature)


def analyze_public_exports(dataset_root: Path) -> dict[str, Any]:
    openai_path = dataset_root / "train_set/final/training_main_32000_openai.jsonl"
    gemma_path = dataset_root / "train_set/final/training_main_32000_gemini.jsonl"
    role_signatures: Counter[tuple[str, ...]] = Counter()
    user_prompts: Counter[str] = Counter()
    full_examples: Counter[tuple[str, str, str]] = Counter()
    total_chars = Counter()
    count = 0

    for openai_record, gemma_record in zip_longest(iter_jsonl(openai_path), iter_jsonl(gemma_path)):
        if openai_record is None or gemma_record is None:
            raise ValueError("The OpenAI and Gemma exports have different record counts")
        openai_messages = message_signature(openai_record)
        gemma_messages = message_signature(gemma_record)
        if tuple(content for _, content in openai_messages) != tuple(content for _, content in gemma_messages):
            raise ValueError(f"Export content mismatch at record {count + 1}")
        if tuple(role for role, _ in openai_messages) != ("system", "user", "assistant"):
            raise ValueError(f"Unexpected OpenAI role signature at record {count + 1}")
        if tuple(role for role, _ in gemma_messages) != ("system", "user", "model"):
            raise ValueError(f"Unexpected Gemma role signature at record {count + 1}")

        role_signatures[tuple(role for role, _ in openai_messages)] += 1
        system, user, answer = (content for _, content in openai_messages)
        user_prompts[user] += 1
        full_examples[(system, user, answer)] += 1
        total_chars["system"] += len(system)
        total_chars["user"] += len(user)
        total_chars["answer"] += len(answer)
        count += 1

    return {
        "record_count": count,
        "role_signatures": {"/".join(key): value for key, value in sorted(role_signatures.items())},
        "exports_content_equivalent": True,
        "duplicate_user_prompt_instances": sum(value - 1 for value in user_prompts.values() if value > 1),
        "duplicate_full_example_instances": sum(value - 1 for value in full_examples.values() if value > 1),
        "average_characters": {
            key: round(value / count, 2) for key, value in sorted(total_chars.items())
        },
        "files": {
            str(openai_path.relative_to(dataset_root)): {
                "bytes": openai_path.stat().st_size,
                "sha256": sha256_file(openai_path),
            },
            str(gemma_path.relative_to(dataset_root)): {
                "bytes": gemma_path.stat().st_size,
                "sha256": sha256_file(gemma_path),
            },
        },
    }


def analyze_generation_sources(dataset_root: Path) -> dict[str, Any]:
    level_counts: Counter[str] = Counter()
    category_counts: dict[str, Counter[str]] = {}
    provider_counts: Counter[str] = Counter()
    rag_counts: Counter[str] = Counter()
    duplicate_instances: dict[str, int] = {}
    duplicate_groups: dict[str, int] = {}
    source_files: list[str] = []

    for level, pattern in LEVEL_PATTERNS.items():
        paths = [
            path for path in sorted(dataset_root.glob(pattern))
            if path.name not in EXCLUDED_SOURCE_NAMES
        ]
        if not paths:
            raise FileNotFoundError(f"No source files matched {pattern}")
        category_counts[level] = Counter()
        level_examples: Counter[tuple[tuple[str, str], ...]] = Counter()
        for path in paths:
            source_files.append(str(path.relative_to(dataset_root)))
            for record in iter_jsonl(path):
                level_counts[level] += 1
                level_examples[message_signature(record)] += 1
                meta = record.get("_meta") or {}
                if category := meta.get("category"):
                    category_counts[level][str(category)] += 1
                if query_source := meta.get("query_source"):
                    provider_counts[str(query_source)] += 1
                if "rag" in meta:
                    rag_counts[f"{level}:{bool(meta['rag'])}"] += 1

        duplicate_instances[level] = sum(value - 1 for value in level_examples.values() if value > 1)
        duplicate_groups[level] = sum(value > 1 for value in level_examples.values())

    if dict(level_counts) != EXPECTED_LEVEL_COUNTS:
        raise ValueError(
            f"Unexpected level counts: expected {EXPECTED_LEVEL_COUNTS}, found {dict(level_counts)}"
        )

    return {
        "level_counts": dict(sorted(level_counts.items())),
        "category_counts": {
            level: dict(sorted(counts.items())) for level, counts in category_counts.items()
        },
        "query_source_counts_l3_l5": dict(sorted(provider_counts.items())),
        "rag_counts": dict(sorted(rag_counts.items())),
        "duplicate_full_example_instances_by_level": duplicate_instances,
        "duplicate_full_example_groups_by_level": duplicate_groups,
        "source_file_count": len(source_files),
        "excluded_sources": sorted(EXCLUDED_SOURCE_NAMES),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the final ScentAI 32K training corpus")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "ScentAI 32K Grounded Perfume Conversations",
        "license": "CC-BY-NC-SA-4.0",
        "source_dataset": {
            "title": "Fragrantica Perfumes: Ratings, Notes, Votes & More",
            "publisher": "Le Decanteur",
            "url": "https://www.kaggle.com/datasets/ledecanteur/fragrantica-perfumes",
            "version": 2,
            "license": "CC-BY-NC-SA-4.0",
            "catalog_records": 131_930,
        },
        "public_exports": analyze_public_exports(dataset_root),
        "generation_sources": analyze_generation_sources(dataset_root),
        "split": {
            "seed": 20260629,
            "train_records": 30_400,
            "validation_records": 1_600,
            "validation_ratio": 0.05,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "records": report["public_exports"]["record_count"],
        "level_counts": report["generation_sources"]["level_counts"],
        "providers": report["generation_sources"]["query_source_counts_l3_l5"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
