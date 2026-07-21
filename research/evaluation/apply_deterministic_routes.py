from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.runtime.exact_lookup import is_exact_lookup_query, render_database_lookup_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply deterministic runtime routes to eval outputs.")
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    args = parser.parse_args()

    cases = {case["id"]: case for case in read_jsonl(args.eval_set)}
    outputs = {row.get("id") or row.get("case_id"): row for row in read_jsonl(args.outputs)}
    routed_rows, route_counts = apply_routes(cases, outputs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, routed_rows)

    metadata_path = args.metadata or args.output.with_suffix(".metadata.json")
    metadata = {
        "eval_set": str(args.eval_set),
        "source_outputs": str(args.outputs),
        "routed_outputs": str(args.output),
        "case_count": len(routed_rows),
        "route_counts": route_counts,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def apply_routes(
    cases: dict[str, dict[str, Any]],
    outputs: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = []
    route_counts = {"llm": 0, "database_lookup": 0, "missing_output": 0}
    for case_id in sorted(cases):
        case = cases[case_id]
        row = dict(outputs.get(case_id, {"id": case_id, "answer": ""}))
        if should_route_database_lookup(case):
            answer = render_database_lookup_answer(case.get("user", "") or case.get("context", ""))
            if answer:
                row["answer"] = answer
                row["routed_by"] = "deterministic_database_lookup"
                route_counts["database_lookup"] += 1
            else:
                row["routed_by"] = "llm_database_lookup_fallback"
                route_counts["llm"] += 1
        elif case_id in outputs:
            row["routed_by"] = row.get("routed_by", "llm")
            route_counts["llm"] += 1
        else:
            row["routed_by"] = "missing_output"
            route_counts["missing_output"] += 1
        rows.append(row)
    return rows, route_counts


def should_route_database_lookup(case: dict[str, Any]) -> bool:
    if case.get("mode") == "database_lookup" or case.get("category") == "database_lookup":
        return True
    return is_exact_lookup_query(case.get("user", ""))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
