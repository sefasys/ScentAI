from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_REPORT = Path("train_set/eval/scentai_eval_v2_report.json")
DEFAULT_OUTPUT = Path("train_set/eval/scentai_eval_v2_human_review.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a compact human review pack from an eval report.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    selected = select_review_cases(report.get("cases", []), args.limit)
    args.output.write_text(render_review_pack(report, selected), encoding="utf-8")
    print(f"Selected {len(selected)} cases for review.")
    print("Saved:", args.output)


def select_review_cases(cases: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    def priority(case: dict[str, Any]) -> tuple[int, int, str]:
        reasons = set(case.get("hard_fail_reasons", []))
        grounding = case.get("grounding", {})
        if grounding.get("field_copy_violations"):
            return (0, len(reasons), case["id"])
        if grounding.get("strict_filter_violations") or grounding.get("forbidden_perfume_mentions"):
            return (1, len(reasons), case["id"])
        if grounding.get("unsupported_perfume_mentions") or grounding.get("unsupported_note_claims"):
            return (2, len(reasons), case["id"])
        if not case.get("pass"):
            return (3, len(reasons), case["id"])
        if case.get("acceptable_recommendation_hit") is False:
            return (4, 0, case["id"])
        return (9, 0, case["id"])

    interesting = [
        case
        for case in cases
        if not case.get("pass") or case.get("acceptable_recommendation_hit") is False
    ]
    return sorted(interesting, key=priority)[:limit]


def render_review_pack(report: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    lines = [
        "# ScentAI Human Review Pack",
        "",
        "## Summary",
        "",
    ]
    for key, value in report.get("summary", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Review Cases", ""])

    if not cases:
        lines.append("No cases selected for review.")
        return "\n".join(lines)

    for index, case in enumerate(cases, 1):
        grounding = case.get("grounding", {})
        lines.extend(
            [
                f"## {index}. {case['id']} - {case.get('category')} / {case.get('difficulty')}",
                "",
                f"- pass: `{case.get('pass')}`",
                f"- reasons: `{case.get('hard_fail_reasons', [])}`",
                f"- dimension_passes: `{case.get('dimension_passes', {})}`",
                f"- expected_perfumes: `{case.get('expected_perfumes', [])}`",
                "",
            ]
        )
        for key in (
            "unsupported_perfume_mentions",
            "forbidden_perfume_mentions",
            "strict_filter_violations",
            "field_copy_violations",
            "unsupported_note_claims",
        ):
            if grounding.get(key):
                lines.append(f"- {key}: `{grounding[key]}`")
        lines.extend(["", "### Model Answer", "", "```text", case.get("answer", "")[:2000], "```", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()

