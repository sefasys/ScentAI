from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.core.config import DatasetConfig, PROJECT_ROOT
from research.core.data import load_perfumes
from research.core.messages import write_jsonl
from research.generators.l2 import generate_l2_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ScentAI L2 dataset records.")
    parser.add_argument("--clean-file", type=Path, default=PROJECT_ROOT / "perfumes_clean.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "training_L2_v2.jsonl")
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--rag-ratio", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--category-counts",
        default="",
        help=(
            "Optional comma-separated counts, for example: "
            "single_filter=10,multi_filter=20,negative_filter=10"
        ),
    )
    parser.add_argument("--include-debug-meta", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DatasetConfig(
        clean_file=args.clean_file,
        output_file=args.output,
        total=args.total,
        rag_ratio=args.rag_ratio,
        seed=args.seed,
        include_debug_meta=args.include_debug_meta,
        category_counts=parse_category_counts(args.category_counts),
    )

    perfumes = load_perfumes(config.clean_file)
    records, counts = generate_l2_records(perfumes, config)
    write_jsonl(records, config.output_file)

    rag_count = sum(1 for r in records if "[PERFUMES]" in r["messages"][1]["content"])
    print("L2 dataset generation complete")
    print(f"Total records : {len(records)}")
    print(f"Output        : {config.output_file}")
    print(f"RAG ratio     : {rag_count}/{len(records)} ({rag_count / len(records) * 100:.1f}%)")
    print("Category breakdown:")
    for category, count in counts.items():
        print(f"  {category:28s}: {count}")


def parse_category_counts(raw: str) -> dict[str, int]:
    if not raw.strip():
        return {}

    counts: dict[str, int] = {}
    for item in raw.split(","):
        if "=" not in item:
            raise ValueError(f"Invalid category count item: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        try:
            count = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid count for category {key!r}: {value!r}") from exc
        if count < 0:
            raise ValueError(f"Category count cannot be negative: {key}={count}")
        counts[key] = count
    return counts


if __name__ == "__main__":
    main()
