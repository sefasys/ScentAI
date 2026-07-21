from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.core.config import PROJECT_ROOT


DEFAULT_MANIFEST = PROJECT_ROOT / "train_set" / "final" / "MANIFEST.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "train_set" / "finetune" / "gemma"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified fine-tuning splits from ScentAI L1-L5 chunks.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--pilot-total", type=int, default=5000)
    parser.add_argument("--role-style", choices=["gemma", "openai"], default="gemma")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty: {args.output_dir}. Use --overwrite.")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records_by_level = load_records_by_level(manifest, role_style=args.role_style)
    rng = random.Random(args.seed)

    train_pools: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    validation_pools: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    split_counts: dict[str, dict[str, int]] = {}

    for level, records in sorted(records_by_level.items()):
        shuffled = list(records)
        rng.shuffle(shuffled)
        val_count = round(len(shuffled) * args.val_ratio)
        validation_pools[level] = shuffled[:val_count]
        train_pools[level] = shuffled[val_count:]
        split_counts[level] = {"train": len(shuffled) - val_count, "validation": val_count}

    full_train = [row for pool in train_pools.values() for _level, row in pool]
    full_val = [row for pool in validation_pools.values() for _level, row in pool]
    rng.shuffle(full_train)
    rng.shuffle(full_val)

    pilot_val_total = max(100, round(args.pilot_total * args.val_ratio))
    pilot_train_total = args.pilot_total - pilot_val_total
    pilot_train_tagged = sample_stratified(train_pools, pilot_train_total, rng)
    pilot_val_tagged = sample_stratified(validation_pools, pilot_val_total, rng)
    rng.shuffle(pilot_train_tagged)
    rng.shuffle(pilot_val_tagged)
    pilot_train = [row for _level, row in pilot_train_tagged]
    pilot_val = [row for _level, row in pilot_val_tagged]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "full_train": args.output_dir / "full_train.jsonl",
        "full_validation": args.output_dir / "full_validation.jsonl",
        "pilot_train": args.output_dir / f"pilot_train_{len(pilot_train)}.jsonl",
        "pilot_validation": args.output_dir / f"pilot_validation_{len(pilot_val)}.jsonl",
    }
    write_jsonl(outputs["full_train"], full_train)
    write_jsonl(outputs["full_validation"], full_val)
    write_jsonl(outputs["pilot_train"], pilot_train)
    write_jsonl(outputs["pilot_validation"], pilot_val)

    output_manifest = {
        "seed": args.seed,
        "role_style": args.role_style,
        "val_ratio": args.val_ratio,
        "pilot_total_requested": args.pilot_total,
        "source_manifest": str(args.manifest),
        "level_counts": {level: len(records) for level, records in sorted(records_by_level.items())},
        "split_counts": split_counts,
        "outputs": {name: {"path": str(path), "records": count_jsonl(path)} for name, path in outputs.items()},
        "pilot_counts": {
            "train": count_tagged_levels(pilot_train_tagged),
            "validation": count_tagged_levels(pilot_val_tagged),
        },
        "format": {
            "messages_only": True,
            "debug_meta_stripped": True,
            "roles": ["system", "user", "model"] if args.role_style == "gemma" else ["system", "user", "assistant"],
        },
    }
    manifest_path = args.output_dir / "SPLIT_MANIFEST.json"
    manifest_path.write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Fine-tune splits created")
    print(f"Output dir       : {args.output_dir}")
    print(f"Role style       : {args.role_style}")
    for name, path in outputs.items():
        print(f"{name:16s}: {count_jsonl(path)} records -> {path}")
    print(f"Manifest         : {manifest_path}")


def load_records_by_level(manifest: dict[str, Any], role_style: str) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    records_by_level: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for raw_path in manifest["source_counts"]:
        path = PROJECT_ROOT / raw_path
        level = path.parent.name
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                clean = {"messages": convert_roles(row["messages"], role_style)}
                validate_record(clean, path, line_no, role_style)
                records_by_level[level].append((level, clean))
    return dict(records_by_level)


def convert_roles(messages: list[dict[str, str]], role_style: str) -> list[dict[str, str]]:
    converted = []
    for message in messages:
        item = {"role": message["role"], "content": message["content"]}
        if role_style == "openai" and item["role"] == "model":
            item["role"] = "assistant"
        if role_style == "gemma" and item["role"] == "assistant":
            item["role"] = "model"
        converted.append(item)
    return converted


def validate_record(row: dict[str, Any], path: Path, line_no: int, role_style: str) -> None:
    expected = ["system", "user", "model"] if role_style == "gemma" else ["system", "user", "assistant"]
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"Invalid messages at {path}:{line_no}")
    roles = [message.get("role") for message in messages]
    if roles != expected:
        raise ValueError(f"Invalid roles at {path}:{line_no}: {roles}, expected {expected}")
    if any(not str(message.get("content") or "").strip() for message in messages):
        raise ValueError(f"Empty message content at {path}:{line_no}")


def sample_stratified(
    pools_by_level: dict[str, list[tuple[str, dict[str, Any]]]],
    total: int,
    rng: random.Random,
) -> list[tuple[str, dict[str, Any]]]:
    level_totals = {level: len(pool) for level, pool in pools_by_level.items()}
    quotas = allocate_counts(total, level_totals)
    sampled: list[tuple[str, dict[str, Any]]] = []
    for level, quota in quotas.items():
        pool = pools_by_level[level]
        if quota > len(pool):
            raise ValueError(f"Quota {quota} exceeds available {len(pool)} for {level}")
        sampled.extend(rng.sample(pool, quota))
    return sampled


def allocate_counts(total: int, level_totals: dict[str, int]) -> dict[str, int]:
    grand_total = sum(level_totals.values())
    raw = {level: total * count / grand_total for level, count in level_totals.items()}
    allocated = {level: int(value) for level, value in raw.items()}
    remainder = total - sum(allocated.values())
    order = sorted(raw, key=lambda level: raw[level] - allocated[level], reverse=True)
    for level in order[:remainder]:
        allocated[level] += 1
    return allocated


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def count_tagged_levels(records: list[tuple[str, dict[str, Any]]]) -> dict[str, int]:
    counts = Counter(level for level, _row in records)
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    main()
