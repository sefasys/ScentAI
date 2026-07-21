from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from research.runtime.grounding_checker import score_case_result


DEFAULT_EVAL_SET = Path("train_set/eval/scentai_eval_v2.jsonl")
DEFAULT_OUTPUTS = Path("train_set/eval/scentai_eval_v2_outputs.jsonl")
DEFAULT_REPORT = Path("train_set/eval/scentai_eval_v2_report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score ScentAI eval outputs.")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    cases = {case["id"]: case for case in read_jsonl(args.eval_set)}
    outputs = read_jsonl(args.outputs)
    report = score_outputs(cases, outputs)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = args.report.with_suffix(".md")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("Saved JSON report:", args.report)
    print("Saved Markdown report:", md_path)


def score_outputs(cases: dict[str, dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    case_reports = []
    missing = set(cases)

    for output in outputs:
        case_id = output.get("id") or output.get("case_id")
        if case_id not in cases:
            case_reports.append(
                {
                    "id": case_id,
                    "category": "unknown",
                    "pass": False,
                    "hard_fail_reasons": ["unknown_case_id"],
                    "answer": output.get("answer", ""),
                }
            )
            continue

        missing.discard(case_id)
        case = cases[case_id]
        answer = output.get("answer", "")
        checks = case.get("checks", {})
        minimum_mentions = int(checks.get("minimum_context_mentions", 1))
        grounding = score_case_result(
            {
                "name": case_id,
                "context": case.get("context", ""),
                "answer": answer,
                "excluded_terms": checks.get("excluded_terms", []),
                "forbidden_perfumes": checks.get("forbidden_perfumes", []),
            }
        )
        acceptable = acceptable_recommendation_hit(answer, checks.get("expected_perfumes", []))
        nonempty = bool(answer.strip())
        hard_reasons = list(grounding.get("hard_fail_reasons", []))
        if not nonempty:
            hard_reasons.append("empty_answer")
        if len(grounding.get("mentioned_context_perfumes", [])) < minimum_mentions:
            hard_reasons.append("minimum_context_mentions_not_met")

        report = {
            "id": case_id,
            "category": case.get("category"),
            "mode": case.get("mode"),
            "difficulty": case.get("difficulty"),
            "tags": case.get("tags", []),
            "pass": not hard_reasons,
            "hard_fail_reasons": hard_reasons,
            "answer": answer,
            "grounding": grounding,
            "dimension_passes": dimension_passes(grounding, acceptable, nonempty, minimum_mentions),
            "acceptable_recommendation_hit": acceptable,
            "expected_perfumes": checks.get("expected_perfumes", []),
            "strict_filter_case": bool(checks.get("strict_filter")),
            "field_copy_case": bool(checks.get("field_copy")),
            "forbidden_perfumes": checks.get("forbidden_perfumes", []),
        }
        case_reports.append(report)

    for case_id in sorted(missing):
        case = cases[case_id]
        case_reports.append(
            {
                "id": case_id,
                "category": case.get("category"),
                "mode": case.get("mode"),
                "difficulty": case.get("difficulty"),
                "tags": case.get("tags", []),
                "pass": False,
                "hard_fail_reasons": ["missing_output"],
                "answer": "",
                "grounding": {},
                "dimension_passes": {},
                "acceptable_recommendation_hit": False,
                "expected_perfumes": case.get("checks", {}).get("expected_perfumes", []),
                "strict_filter_case": bool(case.get("checks", {}).get("strict_filter")),
                "field_copy_case": bool(case.get("checks", {}).get("field_copy")),
                "forbidden_perfumes": case.get("checks", {}).get("forbidden_perfumes", []),
            }
        )

    return {
        "summary": summarize(case_reports),
        "by_category": summarize_by_category(case_reports),
        "by_mode": summarize_by_key(case_reports, "mode"),
        "by_difficulty": summarize_by_key(case_reports, "difficulty"),
        "by_tag": summarize_by_tag(case_reports),
        "gates": evaluate_gates(case_reports),
        "cases": case_reports,
    }


def acceptable_recommendation_hit(answer: str, expected_perfumes: list[str]) -> bool | None:
    if not expected_perfumes:
        return None
    lower = normalize(answer)
    return any(normalize(name) in lower for name in expected_perfumes)


def dimension_passes(
    grounding: dict[str, Any],
    acceptable_hit: bool | None,
    nonempty: bool,
    minimum_mentions: int,
) -> dict[str, bool | None]:
    mentioned_count = len(grounding.get("mentioned_context_perfumes", []))
    return {
        "nonempty": nonempty,
        "context_perfume_mentioned": mentioned_count >= minimum_mentions,
        "context_only": not grounding.get("unsupported_perfume_mentions"),
        "no_forbidden_perfume": not grounding.get("forbidden_perfume_mentions"),
        "strict_filter": not grounding.get("strict_filter_violations"),
        "unsupported_note_claims": not grounding.get("unsupported_note_claims"),
        "field_copy": not grounding.get("field_copy_violations"),
        "expected_overlap": acceptable_hit,
    }


def summarize(reports: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(reports)
    pass_count = sum(1 for report in reports if report["pass"])
    strict_cases = [report for report in reports if report.get("strict_filter_case")]
    field_cases = [report for report in reports if report.get("field_copy_case")]
    acceptable_cases = [report for report in reports if report.get("acceptable_recommendation_hit") is not None]
    reason_counts = Counter(reason for report in reports for reason in report.get("hard_fail_reasons", []))
    dimension_rates = summarize_dimensions(reports)

    return {
        "case_count": total,
        "pass_count": pass_count,
        "hard_failure_count": total - pass_count,
        "pass_rate": ratio(pass_count, total),
        "strict_filter_case_count": len(strict_cases),
        "strict_filter_pass_rate": ratio(
            sum(1 for report in strict_cases if not report["grounding"].get("strict_filter_violations")),
            len(strict_cases),
        ),
        "field_copy_case_count": len(field_cases),
        "field_copy_pass_rate": ratio(
            sum(1 for report in field_cases if not report["grounding"].get("field_copy_violations")),
            len(field_cases),
        ),
        "context_filter_leak_count": sum(
            1 for report in reports if report.get("grounding", {}).get("context_filter_leaks")
        ),
        "acceptable_case_count": len(acceptable_cases),
        "acceptable_hit_rate": ratio(
            sum(1 for report in acceptable_cases if report.get("acceptable_recommendation_hit")),
            len(acceptable_cases),
        ),
        "dimension_rates": dimension_rates,
        "failure_reasons": dict(reason_counts),
    }


def summarize_by_category(reports: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_by_key(reports, "category")


def summarize_by_key(reports: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        buckets[report.get(key) or "unknown"].append(report)
    return {category: summarize(items) for category, items in sorted(buckets.items())}


def summarize_by_tag(reports: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        for tag in report.get("tags", []) or ["untagged"]:
            buckets[tag].append(report)
    return {tag: summarize(items) for tag, items in sorted(buckets.items())}


def summarize_dimensions(reports: list[dict[str, Any]]) -> dict[str, float]:
    totals: Counter[str] = Counter()
    passes: Counter[str] = Counter()
    for report in reports:
        for name, value in report.get("dimension_passes", {}).items():
            if value is None:
                continue
            totals[name] += 1
            if value:
                passes[name] += 1
    return {name: ratio(passes[name], totals[name]) for name in sorted(totals)}


def evaluate_gates(reports: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize(reports)
    gates = {
        "overall_pass_rate": {"actual": summary["pass_rate"], "threshold": 0.90},
        "strict_filter_pass_rate": {"actual": summary["strict_filter_pass_rate"], "threshold": 0.95},
        "field_copy_pass_rate": {"actual": summary["field_copy_pass_rate"], "threshold": 0.90},
        "context_only_rate": {
            "actual": summary["dimension_rates"].get("context_only", 0.0),
            "threshold": 0.98,
        },
        "minimum_context_mention_rate": {
            "actual": summary["dimension_rates"].get("context_perfume_mentioned", 0.0),
            "threshold": 0.95,
        },
    }
    for gate in gates.values():
        gate["pass"] = gate["actual"] >= gate["threshold"]
    return {
        "pass": all(gate["pass"] for gate in gates.values()),
        "items": gates,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# ScentAI Eval Report", "", "## Summary", ""]
    for key, value in report["summary"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Gates", ""])
    lines.append(f"- overall: `{report['gates']['pass']}`")
    for key, gate in report["gates"]["items"].items():
        lines.append(f"- `{key}`: `{gate['actual']}` >= `{gate['threshold']}` -> `{gate['pass']}`")
    lines.extend(["", "## By Category", ""])
    for category, summary in report["by_category"].items():
        lines.append(f"### {category}")
        lines.append("")
        lines.append(f"- pass_rate: `{summary['pass_rate']}` ({summary['pass_count']}/{summary['case_count']})")
        if summary["failure_reasons"]:
            lines.append(f"- failure_reasons: `{summary['failure_reasons']}`")
        lines.append("")

    failures = [case for case in report["cases"] if not case["pass"]]
    lines.extend(["## Failures", ""])
    if not failures:
        lines.append("No hard failures.")
    for case in failures[:50]:
        lines.append(f"### {case['id']} ({case['category']})")
        lines.append("")
        lines.append(f"- reasons: `{case['hard_fail_reasons']}`")
        lines.append(f"- dimension_passes: `{case.get('dimension_passes', {})}`")
        grounding = case.get("grounding", {})
        for key in ("unsupported_perfume_mentions", "strict_filter_violations", "field_copy_violations", "unsupported_note_claims"):
            if grounding.get(key):
                lines.append(f"- {key}: `{grounding[key]}`")
        lines.append("")
        lines.append("```text")
        lines.append(case.get("answer", "")[:1200])
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def normalize(value: str) -> str:
    value = value.lower().replace("’", "'")
    value = re.sub(r"\(\d{4}\)", " ", value)
    value = re.sub(r"\b(?:19|20)\d{2}\b", " ", value)
    value = re.sub(r"[^\w\s']+", " ", value)
    return " ".join(value.split())


if __name__ == "__main__":
    main()
