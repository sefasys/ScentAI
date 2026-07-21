from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FINAL_EVAL_GATES = {
    "overall_pass_rate": (">=", 0.95),
    "language_pass_rate": (">=", 1.0),
    "requested_count_pass_rate": (">=", 1.0),
    "hard_filter_pass_rate": (">=", 1.0),
    "entity_resolution_pass_rate": (">=", 1.0),
    "unsupported_route_pass_rate": (">=", 1.0),
    "conversation_no_repeat_pass_rate": (">=", 1.0),
    "performance_calibration_pass_rate": (">=", 1.0),
    "first_attempt_rate": (">=", 0.90),
    "fallback_rate": ("<=", 0.05),
    "p50_latency_seconds": ("<=", 12.0),
    "p95_latency_seconds": ("<=", 20.0),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(row)
    return rows


def read_completed_outputs(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    if not path.exists():
        return {}, 0
    completed: dict[str, dict[str, Any]] = {}
    malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            case_id = str(row.get("id") or "")
            if case_id:
                completed[case_id] = row
    return completed, malformed


def write_completed_snapshot(
    path: Path,
    cases: list[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            row = completed.get(case["id"])
            if row:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def selected_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
    validation = result.get("validation") or {}
    labels = validation.get("numbered_recommendations") or validation.get("mentioned_candidates") or []
    selected = [candidate for candidate in result.get("candidates", []) if candidate.get("label") in labels]
    return selected


def selected_candidate_ids(result: dict[str, Any]) -> list[int]:
    return [int(candidate["perfume_id"]) for candidate in selected_candidates(result)]


def restore_session_state(session: Any, row: dict[str, Any]) -> None:
    result = row.get("result") or {}
    session.last_result = result
    session.recommendation_ids = [int(value) for value in row.get("session_recommendation_ids", [])]


def normalized_label_set(candidates: list[dict[str, Any]]) -> set[str]:
    return {str(candidate.get("label") or "") for candidate in candidates if candidate.get("label")}


def evaluation_trait_matches(expected: str, actual: str) -> bool:
    expected_norm = " ".join(str(expected or "").lower().replace("-", " ").split())
    actual_norm = " ".join(str(actual or "").lower().replace("-", " ").split())
    if expected_norm == actual_norm:
        return True
    if expected_norm == "spicy" and actual_norm in {"fresh spicy", "soft spicy", "warm spicy"}:
        return True
    families = (
        {"smoke", "smoky"},
        {"musk", "musky", "white musk"},
        {"leather", "leathery", "suede"},
        {"oud", "aoud", "agarwood"},
    )
    return any(expected_norm in family and actual_norm in family for family in families)


def plan_contains_traits(expected_terms: list[str], actual_terms: list[str]) -> bool:
    return all(
        any(evaluation_trait_matches(expected, actual) for actual in actual_terms)
        for expected in expected_terms
    )


def score_final_case(
    case: dict[str, Any],
    result: dict[str, Any],
    *,
    previous_selected_ids: set[int] | None = None,
) -> dict[str, Any]:
    expected = case["expected"]
    plan = result.get("plan") or {}
    validation = result.get("validation") or {}
    candidates = result.get("candidates") or []
    selected = selected_candidates(result)
    selected_ids = set(selected_candidate_ids(result))
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    checks["pipeline_validation"] = bool(validation.get("pass"))
    checks["response_language"] = (
        result.get("response_language") == expected["response_language"]
        and response_language_matches(result.get("answer", ""), expected["response_language"])
    )
    checks["intent"] = plan.get("intent") in expected.get("intent_in", [])
    checks["safe_route"] = result.get("route") not in {"retrieval_error", "comparison_unresolved", "no_safe_match"}

    if "route" in expected:
        checks["expected_route"] = result.get("route") == expected["route"]
    if "generation_attempts" in expected:
        checks["generation_attempts"] = int(result.get("generation_attempts") or 0) == expected["generation_attempts"]

    if "requested_count" in expected:
        count = int(expected["requested_count"])
        checks["requested_count"] = (
            int(plan.get("requested_count") or 0) == count
            and len(validation.get("numbered_recommendations") or []) == count
        )

    expected_labels = expected.get("resolved_labels") or []
    if expected_labels:
        actual_labels = [candidate.get("label") for candidate in candidates]
        checks["entity_resolution"] = len(actual_labels) == len(expected_labels) and set(actual_labels) == set(expected_labels)
        details["expected_labels"] = expected_labels
        details["actual_labels"] = actual_labels

    reference_label = expected.get("reference_label")
    if reference_label:
        reference = result.get("reference") or {}
        checks["reference_resolution"] = reference.get("label") == reference_label
        details["expected_reference"] = reference_label
        details["actual_reference"] = reference.get("label")

    required_terms = expected.get("required_terms") or []
    if required_terms:
        checks["required_terms_in_plan"] = plan_contains_traits(
            required_terms,
            plan.get("required_terms") or [],
        )
        checks["required_terms_in_results"] = bool(selected) and all(
            all(candidate_has_term(candidate, term) for term in required_terms)
            for candidate in selected
        )

    excluded_terms = expected.get("excluded_terms") or []
    actual_excluded_terms = plan.get("excluded_terms") or []
    filter_relevant_intents = {"recommendation", "similarity", "alternative"}
    if excluded_terms or (actual_excluded_terms and plan.get("intent") in filter_relevant_intents):
        checks["no_unexpected_excluded_terms"] = all(
            any(evaluation_trait_matches(expected_term, actual_term) for expected_term in excluded_terms)
            for actual_term in actual_excluded_terms
        )
        if not checks["no_unexpected_excluded_terms"]:
            details["unexpected_excluded_terms"] = [
                actual_term
                for actual_term in actual_excluded_terms
                if not any(
                    evaluation_trait_matches(expected_term, actual_term)
                    for expected_term in excluded_terms
                )
            ]
    if excluded_terms:
        checks["excluded_terms_in_plan"] = plan_contains_traits(
            excluded_terms,
            actual_excluded_terms,
        )
        checks["excluded_terms_absent"] = all(
            all(not candidate_has_term(candidate, term) for term in excluded_terms)
            for candidate in selected
        )

    requested_brand = expected.get("requested_brand")
    if requested_brand:
        checks["requested_brand"] = plan.get("requested_brand") == requested_brand
        checks["brand_results"] = bool(selected) and all(candidate.get("brand") == requested_brand for candidate in selected)

    for field in ("gender", "season", "time_profile", "discovery_mode"):
        if field in expected:
            checks[field] = plan.get(field) == expected[field]
    if "conversation_action" in expected:
        expected_action = expected["conversation_action"]
        actual_action = plan.get("conversation_action")
        follow_up_actions = {"more_options", "refine_previous"}
        checks["conversation_action"] = (
            actual_action == expected_action
            or {actual_action, expected_action}.issubset(follow_up_actions)
        )

    if expected.get("no_repeat_previous"):
        previous = previous_selected_ids or set()
        checks["conversation_no_repeat"] = bool(previous) and bool(selected_ids) and previous.isdisjoint(selected_ids)
        details["previous_selected_ids"] = sorted(previous)
        details["current_selected_ids"] = sorted(selected_ids)

    performance_violations = validation.get("performance_calibration_violations") or []
    checks["performance_calibration"] = not performance_violations
    details["performance_calibration_violations"] = performance_violations

    failures = [name for name, passed in checks.items() if not passed]
    return {
        "pass": not failures,
        "failures": failures,
        "checks": checks,
        "details": details,
        "selected_candidate_ids": sorted(selected_ids),
        "selected_candidate_labels": [candidate["label"] for candidate in selected],
    }


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 1.0


def metric_rate(rows: list[dict[str, Any]], check_names: set[str]) -> float:
    applicable = [row for row in rows if any(name in row["score"]["checks"] for name in check_names)]
    passed = [
        row for row in applicable
        if all(row["score"]["checks"].get(name, True) for name in check_names)
    ]
    return rate(len(passed), len(applicable))


def grouped_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key) or "unknown")].append(row)
    return {
        name: {
            "count": len(bucket),
            "pass_count": sum(row["score"]["pass"] for row in bucket),
            "pass_rate": rate(sum(row["score"]["pass"] for row in bucket), len(bucket)),
        }
        for name, bucket in sorted(buckets.items())
    }


def evaluate_gate(value: float, operator: str, threshold: float) -> bool:
    return value >= threshold if operator == ">=" else value <= threshold


def summarize_final_eval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["result"].get("timings", {}).get("total_seconds") or 0.0) for row in rows]
    total = len(rows)
    pass_count = sum(row["score"]["pass"] for row in rows)
    attempts = [int(row["result"].get("generation_attempts") or 0) for row in rows]
    generated_rows = [row for row in rows if int(row["result"].get("generation_attempts") or 0) > 0]
    fallback_count = sum(row["result"].get("route") == "validated_template_fallback" for row in rows)
    failure_counter = Counter(
        failure for row in rows for failure in row["score"].get("failures", [])
    )
    validator_reason_counter = Counter(
        reason
        for row in rows
        for reason in row["result"].get("validation", {}).get("reasons", [])
    )

    metrics = {
        "overall_pass_rate": rate(pass_count, total),
        "language_pass_rate": metric_rate(rows, {"response_language"}),
        "requested_count_pass_rate": metric_rate(rows, {"requested_count"}),
        "hard_filter_pass_rate": metric_rate(rows, {
            "required_terms_in_plan", "required_terms_in_results",
            "excluded_terms_in_plan", "excluded_terms_absent", "no_unexpected_excluded_terms",
        }),
        "hard_filter_plan_pass_rate": metric_rate(rows, {
            "required_terms_in_plan", "excluded_terms_in_plan", "no_unexpected_excluded_terms",
        }),
        "hard_filter_output_pass_rate": metric_rate(rows, {
            "required_terms_in_results", "excluded_terms_absent",
        }),
        "entity_resolution_pass_rate": metric_rate(rows, {"entity_resolution", "reference_resolution"}),
        "unsupported_route_pass_rate": metric_rate(rows, {"expected_route", "generation_attempts"}),
        "conversation_no_repeat_pass_rate": metric_rate(rows, {"conversation_no_repeat"}),
        "performance_calibration_pass_rate": metric_rate(rows, {"performance_calibration"}),
        "first_attempt_rate": rate(
            sum(int(row["result"].get("generation_attempts") or 0) == 1 for row in generated_rows),
            len(generated_rows),
        ),
        "fallback_rate": rate(fallback_count, total),
        "average_latency_seconds": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "p50_latency_seconds": round(percentile(latencies, 0.50), 4),
        "p95_latency_seconds": round(percentile(latencies, 0.95), 4),
        "max_latency_seconds": round(max(latencies), 4) if latencies else 0.0,
    }
    gates = {
        name: {
            "value": metrics[name],
            "operator": operator,
            "threshold": threshold,
            "pass": evaluate_gate(float(metrics[name]), operator, threshold),
        }
        for name, (operator, threshold) in FINAL_EVAL_GATES.items()
    }
    return {
        "version": "final_eval_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": total,
        "pass_count": pass_count,
        "failure_count": total - pass_count,
        "metrics": metrics,
        "quality_gates": gates,
        "all_quality_gates_passed": all(item["pass"] for item in gates.values()),
        "category_summary": grouped_summary(rows, "category"),
        "language_summary": grouped_summary(rows, "language"),
        "route_counts": dict(Counter(row["result"].get("route") for row in rows)),
        "generation_attempt_counts": dict(Counter(attempts)),
        "failure_counts": dict(failure_counter),
        "validator_reason_counts": dict(validator_reason_counter),
        "failed_case_ids": [row["id"] for row in rows if not row["score"]["pass"]],
    }


def select_human_review_rows(rows: list[dict[str, Any]], limit: int = 40) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not row["score"]["pass"]:
            selected.append(row)
            seen.add(row["id"])
            if len(selected) >= limit:
                return selected
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["id"] not in seen:
            buckets[row["category"]].append(row)
    categories = sorted(buckets)
    while len(selected) < limit and any(buckets.values()):
        for category in categories:
            if buckets[category] and len(selected) < limit:
                row = buckets[category].pop(0)
                selected.append(row)
                seen.add(row["id"])
    return selected


def write_human_review_csv(path: Path, rows: list[dict[str, Any]], limit: int = 40) -> None:
    selected = select_human_review_rows(rows, limit=limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "category", "language", "query", "answer", "candidate_labels",
        "route", "latency_seconds", "auto_pass", "auto_failures",
        "grounding_1_5", "technical_accuracy_1_5", "advisor_value_1_5",
        "naturalness_1_5", "review_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            result = row["result"]
            writer.writerow({
                "id": row["id"],
                "category": row["category"],
                "language": row["language"],
                "query": row["query"],
                "answer": result.get("answer", ""),
                "candidate_labels": " | ".join(candidate.get("label", "") for candidate in result.get("candidates", [])),
                "route": result.get("route"),
                "latency_seconds": result.get("timings", {}).get("total_seconds"),
                "auto_pass": row["score"]["pass"],
                "auto_failures": " | ".join(row["score"]["failures"]),
                "grounding_1_5": "",
                "technical_accuracy_1_5": "",
                "advisor_value_1_5": "",
                "naturalness_1_5": "",
                "review_notes": "",
            })


def run_final_evaluation(
    pipeline: Any,
    cases: list[dict[str, Any]],
    *,
    outputs_path: Path,
    summary_path: Path,
    human_review_path: Path,
    metadata_path: Path,
    resume: bool = True,
    limit: int | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_cases = cases[:limit] if limit is not None else list(cases)
    valid_ids = {case["id"] for case in active_cases}
    completed, malformed_lines = read_completed_outputs(outputs_path) if resume else ({}, 0)
    completed = {case_id: row for case_id, row in completed.items() if case_id in valid_ids}
    write_completed_snapshot(outputs_path, active_cases, completed)
    sessions: dict[str, Any] = {}
    session_previous_ids: dict[str, set[int]] = defaultdict(set)
    metadata = {
        "version": "final_eval_v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(active_cases),
        "resume": resume,
        "already_completed": len(completed),
        "malformed_output_lines_skipped": malformed_lines,
        "outputs_path": str(outputs_path),
        "runtime": run_metadata or {},
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    mode = "a" if completed else "w"
    with outputs_path.open(mode, encoding="utf-8") as handle:
        for index, case in enumerate(active_cases, 1):
            session_id = case.get("session_id")
            session = None
            if session_id:
                session = sessions.setdefault(session_id, ScentAISession(pipeline))
            existing = completed.get(case["id"])
            if existing:
                if session is not None:
                    restore_session_state(session, existing)
                    session_previous_ids[session_id].update(existing.get("score", {}).get("selected_candidate_ids", []))
                print(f"[{index}/{len(active_cases)}] resume {case['id']}")
                continue

            started = time.perf_counter()
            result = session.run(case["query"]) if session is not None else pipeline.run(case["query"])
            runner_seconds = round(time.perf_counter() - started, 4)
            previous_ids = set(session_previous_ids.get(session_id, set())) if session_id else set()
            score = score_final_case(case, result, previous_selected_ids=previous_ids)
            if session_id:
                session_previous_ids[session_id].update(score["selected_candidate_ids"])
            row = {
                "id": case["id"],
                "version": case["version"],
                "category": case["category"],
                "language": case["language"],
                "query": case["query"],
                "tags": case.get("tags", []),
                "session_id": session_id,
                "turn": case.get("turn"),
                "expected": case["expected"],
                "result": result,
                "score": score,
                "session_recommendation_ids": list(session.recommendation_ids) if session is not None else [],
                "runner_seconds": runner_seconds,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            completed[case["id"]] = row
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(
                f"[{index}/{len(active_cases)}] {case['id']} {case['category']} "
                f"pass={score['pass']} route={result.get('route')} "
                f"attempts={result.get('generation_attempts')} seconds={runner_seconds}"
            )
            if score["failures"]:
                print("  failures:", score["failures"])

    ordered_rows = [completed[case["id"]] for case in active_cases if case["id"] in completed]
    write_completed_snapshot(outputs_path, active_cases, completed)
    summary = summarize_final_eval(ordered_rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_human_review_csv(human_review_path, ordered_rows, limit=min(40, len(ordered_rows)))
    metadata.update({
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_count": len(ordered_rows),
        "summary_path": str(summary_path),
        "human_review_path": str(human_review_path),
    })
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
