from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.core.config import DatasetConfig, PROJECT_ROOT
from research.core.data import load_perfumes
from research.core.llm_query import QueryGenerator
from research.generators.l5 import (
    build_generation_indexes,
    build_l5_schedule,
    l5_category_counts,
    make_l5_record,
    record_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ScentAI L5 preference-aware dataset records.")
    parser.add_argument("--clean-file", type=Path, default=PROJECT_ROOT / "perfumes_clean.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "training_L5_v2.jsonl")
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--rag-ratio", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--category-counts",
        default="",
        help="Optional comma-separated counts, for example: empty_profile=10,profile_likes=20",
    )
    parser.add_argument("--include-debug-meta", action="store_true")
    parser.add_argument(
        "--query-provider",
        choices=["auto", "gemini", "compat", "pool", "fallback"],
        default="auto",
        help="Query generation provider. 'auto' prefers compat, Gemini, then fallback.",
    )
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--gemini-sleep", type=float, default=4.0)
    parser.add_argument("--compat-base-url", default="", help="OpenAI-compatible base URL, for example https://api.groq.com/openai/v1")
    parser.add_argument("--compat-model", default="llama-3.3-70b-versatile")
    parser.add_argument("--compat-api-key-env", default="OPENAI_COMPAT_API_KEY")
    parser.add_argument("--provider-pool", type=Path, default=None, help="JSON provider pool for --query-provider pool.")
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true", help="Append missing records if the output JSONL already exists.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output before generation.")
    parser.add_argument(
        "--fallback-policy",
        choices=["template", "fail"],
        default="template",
        help="Use template fallback, or fail if the selected LLM provider cannot produce a valid query.",
    )
    parser.add_argument("--no-llm", action="store_true", help="Deprecated alias for --query-provider fallback.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    category_counts = parse_category_counts(args.category_counts)
    total = sum(category_counts.values()) if category_counts else args.total
    config = DatasetConfig(
        clean_file=args.clean_file,
        output_file=args.output,
        total=total,
        rag_ratio=args.rag_ratio,
        seed=args.seed,
        include_debug_meta=args.include_debug_meta,
        category_counts=category_counts,
    )
    provider = "fallback" if args.no_llm else args.query_provider
    query_generator = QueryGenerator(
        provider=provider,
        gemini_model=args.gemini_model,
        compat_base_url=args.compat_base_url or None,
        compat_api_key=get_env_value(args.compat_api_key_env),
        compat_model=args.compat_model,
        provider_pool_path=args.provider_pool,
        sleep_seconds=args.gemini_sleep,
        use_api=not args.no_llm,
        max_retries=args.llm_retries,
        fallback_policy=args.fallback_policy,
    )
    print(f"Query provider : {query_generator.provider}")

    perfumes = load_perfumes(config.clean_file)
    total_records, counts, rag_count = generate_l5_streaming(
        perfumes,
        config,
        query_generator,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        overwrite=args.overwrite,
    )

    print("L5 dataset generation complete")
    print(f"Total records : {total_records}")
    print(f"Output        : {config.output_file}")
    print(f"RAG ratio     : {rag_count}/{total_records} ({rag_count / total_records * 100:.1f}%)")
    print("Category breakdown:")
    for category, count in counts.items():
        print(f"  {category:32s}: {count}")


def parse_category_counts(raw: str) -> dict[str, int]:
    if not raw.strip():
        return {}
    counts: dict[str, int] = {}
    for item in raw.split(","):
        if "=" not in item:
            raise ValueError(f"Invalid category count item: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        count = int(value.strip())
        if count < 0:
            raise ValueError(f"Category count cannot be negative: {key}={count}")
        counts[key] = count
    return counts


def get_env_value(name: str) -> str | None:
    return os.environ.get(name) if name else None


def generate_l5_streaming(
    perfumes,
    config: DatasetConfig,
    query_generator: QueryGenerator,
    checkpoint_every: int,
    resume: bool,
    overwrite: bool,
):
    if resume and overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    if checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be greater than 0")

    rng = random.Random(config.seed)
    counts = l5_category_counts(config)
    schedule = build_l5_schedule(counts, rng)
    indexes = build_generation_indexes(perfumes)

    output = config.output_file
    output.parent.mkdir(parents=True, exist_ok=True)

    existing = 0
    rag_count = 0
    if output.exists() and resume:
        existing, rag_count, truncate_offset = scan_jsonl_progress(output)
        if truncate_offset is not None:
            with output.open("rb+") as handle:
                handle.truncate(truncate_offset)
            print(f"Resume       : trimmed invalid tail at byte {truncate_offset}")

    if existing > len(schedule):
        raise ValueError(f"Output already has {existing} records, target schedule has {len(schedule)}")

    if overwrite and output.exists():
        output.unlink()
        existing = 0
    elif resume and existing:
        print(f"Resume       : found {existing}/{len(schedule)} existing records")

    produced = {category: 0 for category in counts}
    for category in schedule[:existing]:
        produced[category] += 1

    mode = "a" if existing else "w"
    with output.open(mode, encoding="utf-8") as handle:
        for idx, category in enumerate(schedule[existing:], existing + 1):
            record_rng = random.Random(record_seed(config.seed or 0, idx))
            record = make_l5_record(category, perfumes, indexes, config, record_rng, query_generator)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if "[PERFUMES]" in record["messages"][1]["content"]:
                rag_count += 1
            produced[category] += 1

            if idx % checkpoint_every == 0 or idx == len(schedule):
                handle.flush()
                os.fsync(handle.fileno())
                print(f"Checkpoint: {idx}/{len(schedule)} records written")

    return len(schedule), produced, rag_count


def scan_jsonl_progress(path: Path) -> tuple[int, int, int | None]:
    valid_count = 0
    rag_count = 0
    offset = 0
    last_good_offset = 0

    with path.open("rb") as handle:
        for raw_line in handle:
            next_offset = offset + len(raw_line)
            if not raw_line.strip():
                offset = next_offset
                last_good_offset = next_offset
                continue
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return valid_count, rag_count, last_good_offset

            valid_count += 1
            if "[PERFUMES]" in record["messages"][1]["content"]:
                rag_count += 1
            offset = next_offset
            last_good_offset = next_offset

    return valid_count, rag_count, None


if __name__ == "__main__":
    main()
