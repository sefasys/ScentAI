from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SMOKE_CASE_IDS = (
    "fev1_rec_001",
    "fev1_rec_002",
    "fev1_profile_001",
    "fev1_cmp_001",
    "fev1_sim_001",
    "fev1_filter_001",
    "fev1_entity_004",
    "fev1_unsupported_001",
    "fev1_noisy_001",
    "fev1_conv_001",
    "fev1_conv_002",
    "fev1_filter_002",
)


def load_cases(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is None or limit >= len(cases):
        return cases
    if limit == len(SMOKE_CASE_IDS):
        by_id = {case["id"]: case for case in cases}
        selected = [by_id[case_id] for case_id in SMOKE_CASE_IDS if case_id in by_id]
        if len(selected) == limit:
            return selected
    return cases[:limit]


def _set_equal(actual: Any, expected: Any) -> bool:
    return {str(item) for item in actual or []} == {str(item) for item in expected or []}


def _contains_expected(actual: Any, expected: Any) -> bool:
    return {str(item) for item in expected or []}.issubset({str(item) for item in actual or []})


def score_case(
    case: dict[str, Any],
    result: dict[str, Any],
    previous_recommendation_ids: set[int] | None = None,
) -> dict[str, Any]:
    expected = case["expected"]
    plan = result.get("plan") or {}
    validation = result.get("validation") or {}
    failures: list[str] = []

    if not validation.get("pass"):
        failures.append("pipeline_validation_failed")
    if result.get("response_language") != expected.get("response_language"):
        failures.append("response_language_mismatch")
    if plan.get("intent") not in set(expected.get("intent_in") or []):
        failures.append("intent_mismatch")

    scalar_plan_fields = (
        "requested_count",
        "gender",
        "season",
        "time_profile",
        "discovery_mode",
        "requested_brand",
    )
    for field in scalar_plan_fields:
        if field in expected and plan.get(field) != expected[field]:
            failures.append(f"{field}_mismatch")

    if "conversation_action" in expected:
        expected_action = expected["conversation_action"]
        actual_action = plan.get("conversation_action")
        follow_up_actions = {"more_options", "refine_previous"}
        if not (
            actual_action == expected_action
            or {actual_action, expected_action}.issubset(follow_up_actions)
        ):
            failures.append("conversation_action_mismatch")

    # A phrase such as "woody fragrances" may be safely promoted from semantic
    # preference to a required catalog trait. Missing a frozen required term is
    # a regression; adding another evidenced required trait is not.
    if "required_terms" in expected and not _contains_expected(
        plan.get("required_terms"), expected["required_terms"]
    ):
        failures.append("required_terms_mismatch")
    if "excluded_terms" in expected and not _set_equal(plan.get("excluded_terms"), expected["excluded_terms"]):
        failures.append("excluded_terms_mismatch")

    candidate_labels = {str(item.get("label")) for item in result.get("candidates") or []}
    if "resolved_labels" in expected and not set(expected["resolved_labels"]).issubset(candidate_labels):
        failures.append("resolved_labels_mismatch")

    if "reference_label" in expected:
        reference_label = str((result.get("reference") or {}).get("label") or "")
        if reference_label != expected["reference_label"]:
            failures.append("reference_label_mismatch")

    if "route" in expected and result.get("route") != expected["route"]:
        failures.append("route_mismatch")
    if "generation_attempts" in expected and result.get("generation_attempts") != expected["generation_attempts"]:
        failures.append("generation_attempts_mismatch")

    mentioned_ids = {
        int(candidate["perfume_id"])
        for candidate in result.get("candidates") or []
        if candidate.get("label") in set(validation.get("mentioned_candidates") or [])
    }
    if expected.get("no_repeat_previous") and previous_recommendation_ids:
        if mentioned_ids & previous_recommendation_ids:
            failures.append("conversation_repeated_previous")

    return {
        "pass": not failures,
        "failures": failures,
        "mentioned_candidate_ids": sorted(mentioned_ids),
    }


def run_regression(
    runtime: Any,
    cases_path: Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    cases = load_cases(cases_path, limit)
    logical_sessions: dict[str, str] = {}
    prior_ids: dict[str, set[int]] = defaultdict(set)
    outputs = []
    started = time.perf_counter()

    for index, case in enumerate(cases, 1):
        logical_session = str(case.get("session_id") or "")
        actual_session = logical_sessions.get(logical_session) if logical_session else None
        case_started = time.perf_counter()
        actual_session, result = runtime.sessions.run(case["query"], actual_session)
        if logical_session:
            logical_sessions[logical_session] = actual_session
        score = score_case(case, result, prior_ids.get(logical_session))
        if logical_session:
            prior_ids[logical_session].update(score["mentioned_candidate_ids"])
        outputs.append(
            {
                "id": case["id"],
                "category": case["category"],
                "query": case["query"],
                "expected": case["expected"],
                "score": score,
                "runner_seconds": round(time.perf_counter() - case_started, 4),
                "result": result,
            }
        )
        print(
            json.dumps(
                {
                    "case": f"{index}/{len(cases)}",
                    "id": case["id"],
                    "pass": score["pass"],
                    "failures": score["failures"],
                }
            ),
            flush=True,
        )

    category_counts: dict[str, Counter] = defaultdict(Counter)
    for output in outputs:
        category_counts[output["category"]]["total"] += 1
        category_counts[output["category"]]["passed" if output["score"]["pass"] else "failed"] += 1
    passed = sum(output["score"]["pass"] for output in outputs)
    fallback_count = sum(
        str(output["result"].get("route") or "").startswith("validated_template_fallback")
        for output in outputs
    )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "stage6_modal_regression",
        "case_count": len(outputs),
        "pass_count": passed,
        "failure_count": len(outputs) - passed,
        "pass_rate": passed / len(outputs) if outputs else 0.0,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / len(outputs) if outputs else 0.0,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "categories": {name: dict(counts) for name, counts in sorted(category_counts.items())},
        "outputs": outputs,
    }


def rescore_report(report: dict[str, Any], cases_path: Path) -> dict[str, Any]:
    """Reapply corrected evaluation contracts without repeating model inference."""
    cases_by_id = {case["id"]: case for case in load_cases(cases_path)}
    prior_ids: dict[str, set[int]] = defaultdict(set)
    outputs = []
    for original in report.get("outputs") or []:
        case = cases_by_id.get(original.get("id"))
        if case is None:
            raise KeyError(f"Unknown evaluation case in report: {original.get('id')!r}")
        logical_session = str(case.get("session_id") or "")
        score = score_case(case, original.get("result") or {}, prior_ids.get(logical_session))
        if logical_session:
            prior_ids[logical_session].update(score["mentioned_candidate_ids"])
        outputs.append({**original, "expected": case["expected"], "score": score})

    category_counts: dict[str, Counter] = defaultdict(Counter)
    for output in outputs:
        category_counts[output["category"]]["total"] += 1
        category_counts[output["category"]]["passed" if output["score"]["pass"] else "failed"] += 1
    passed = sum(output["score"]["pass"] for output in outputs)
    fallback_count = sum(
        str(output["result"].get("route") or "").startswith("validated_template_fallback")
        for output in outputs
    )
    return {
        **report,
        "rescored_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(outputs),
        "pass_count": passed,
        "failure_count": len(outputs) - passed,
        "pass_rate": passed / len(outputs) if outputs else 0.0,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / len(outputs) if outputs else 0.0,
        "categories": {name: dict(counts) for name, counts in sorted(category_counts.items())},
        "outputs": outputs,
    }
