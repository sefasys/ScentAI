from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run checked L5 main dataset chunks.")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--seed-base", type=int, default=5500)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "train_set" / "L5")
    parser.add_argument("--provider-pool", type=Path, default=PROJECT_ROOT / "research" / "provider_pool.json")
    parser.add_argument("--api-key-file", type=Path, default=PROJECT_ROOT / "openrouter_api.txt")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--stall-after-complete-seconds", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start > args.end:
        raise ValueError("--start must be <= --end")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.log_file or args.output_dir / f"l5_auto_{args.start:03d}_{args.end:03d}.log"
    logger = Logger(log_file)

    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = read_api_key(args.api_key_file)

    logger.write(f"Started L5 auto run at {datetime.now().isoformat(timespec='seconds')}")
    logger.write(f"Chunks: {args.start:03d}-{args.end:03d}, chunk_size={args.chunk_size}")

    for chunk in range(args.start, args.end + 1):
        seed = args.seed_base + chunk
        output = args.output_dir / f"l5_main_{chunk:03d}_debug.jsonl"
        existing_lines = count_lines(output) if output.exists() else 0
        write_mode = "--resume" if 0 < existing_lines < args.chunk_size else "--overwrite"
        logger.write("")
        logger.write(f"=== L5 chunk {chunk:03d} seed={seed} output={output} ===")
        if write_mode == "--resume":
            logger.write(f"Resume partial chunk: found {existing_lines}/{args.chunk_size} existing lines.")

        run_command(
            [
                sys.executable,
                "research/generate_l5.py",
                "--total",
                str(args.chunk_size),
                "--seed",
                str(seed),
                "--query-provider",
                "pool",
                "--provider-pool",
                str(args.provider_pool),
                "--fallback-policy",
                "fail",
                "--checkpoint-every",
                "100",
                write_mode,
                "--include-debug-meta",
                "--gemini-sleep",
                str(args.sleep),
                "--output",
                str(output),
            ],
            env,
            logger,
            watched_output=output,
            expected_lines=args.chunk_size,
            poll_seconds=args.poll_seconds,
            stall_after_complete_seconds=args.stall_after_complete_seconds,
        )

        line_count = count_lines(output)
        logger.write(f"Line count: {line_count}")
        if line_count != args.chunk_size:
            raise RuntimeError(f"{output} has {line_count} lines, expected {args.chunk_size}")

        run_command([sys.executable, "research/validators/l5.py", str(output)], env, logger)
        run_command([sys.executable, "research/validators/query_quality.py", str(output)], env, logger)

        duplicate_queries = count_duplicate_queries(output)
        logger.write(f"Duplicate queries enforced: {duplicate_queries}")
        if duplicate_queries:
            raise RuntimeError(f"{output} has {duplicate_queries} duplicate queries")

        fingerprint_report = fingerprint_l5_chunks(args.output_dir)
        logger.write(f"Fingerprint total rows: {fingerprint_report['total_rows']}")
        for name, rows, fp in fingerprint_report["chunks"]:
            logger.write(f"  {rows:5d} {fp[:12]} {name}")
        if fingerprint_report["duplicate_groups"]:
            logger.write("Duplicate fingerprint groups:")
            for names in fingerprint_report["duplicate_groups"]:
                logger.write(f"  {names}")
            raise RuntimeError("Duplicate L5 chunk fingerprint detected")

        logger.write(f"Chunk {chunk:03d} passed all checks.")

    logger.write("")
    logger.write(f"Completed L5 auto run at {datetime.now().isoformat(timespec='seconds')}")


class Logger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def read_api_key(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"API key file not found: {path}")
    api_key = path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError(f"API key file is empty: {path}")
    return api_key


def run_command(
    command: list[str],
    env: dict[str, str],
    logger: Logger,
    watched_output: Path | None = None,
    expected_lines: int | None = None,
    poll_seconds: float = 60.0,
    stall_after_complete_seconds: float = 180.0,
) -> None:
    logger.write(f"$ {' '.join(command)}")
    if watched_output is None:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if process.stdout:
            for line in process.stdout.rstrip().splitlines():
                logger.write(line)
        if process.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")
        return

    completed_at: float | None = None
    terminated_after_complete = False
    with logger.path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            lines = count_lines(watched_output) if watched_output.exists() else 0
            if expected_lines is not None and lines >= expected_lines:
                if completed_at is None:
                    completed_at = time.monotonic()
                    logger.write(f"Observed {lines}/{expected_lines} lines; waiting for process shutdown.")
                elif time.monotonic() - completed_at >= stall_after_complete_seconds:
                    logger.write("Process did not exit after expected lines; sending SIGINT and continuing to validation.")
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        logger.write("Process ignored SIGINT; sending SIGKILL.")
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=30)
                    terminated_after_complete = True
                    break
            time.sleep(poll_seconds)

    if process.returncode not in (0, None) and not terminated_after_complete:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def normalized_query(row: dict) -> str:
    return re.sub(r"\s+", " ", row["messages"][1]["content"].strip().lower())


def count_duplicate_queries(path: Path) -> int:
    seen: set[str] = set()
    duplicates = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            query = normalized_query(json.loads(line))
            if query in seen:
                duplicates += 1
            seen.add(query)
    return duplicates


def fingerprint_payload(row: dict) -> tuple[object, ...]:
    meta = row.get("_meta") or {}
    answer = re.sub(r"\s+", " ", row["messages"][2]["content"].strip().lower())
    profile = meta.get("user_profile") or {}
    profile_key = (
        bool(profile.get("empty")),
        tuple(profile.get("liked_perfume_ids") or []),
        tuple(profile.get("disliked_perfume_ids") or []),
        tuple(profile.get("previously_recommended_ids") or []),
        tuple(profile.get("liked_notes") or []),
        tuple(profile.get("disliked_notes") or []),
        tuple(profile.get("liked_accords") or []),
        tuple(profile.get("disliked_accords") or []),
    )
    return (meta.get("category"), profile_key, tuple(meta.get("answer_ids") or []), answer)


def fingerprint_l5_chunks(output_dir: Path) -> dict[str, object]:
    groups: dict[tuple[int, str], list[str]] = {}
    chunks: list[tuple[str, int, str]] = []
    total_rows = 0

    for path in sorted(output_dir.glob("l5_main_*_debug.jsonl")):
        digest = hashlib.sha256()
        rows = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows += 1
                total_rows += 1
                digest.update(repr(fingerprint_payload(row)).encode("utf-8"))
        fp = digest.hexdigest()
        chunks.append((path.name, rows, fp))
        groups.setdefault((rows, fp), []).append(path.name)

    duplicate_groups = [names for names in groups.values() if len(names) > 1]
    return {"total_rows": total_rows, "chunks": chunks, "duplicate_groups": duplicate_groups}


if __name__ == "__main__":
    main()
