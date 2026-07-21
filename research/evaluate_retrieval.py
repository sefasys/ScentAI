from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.runtime.rag import DEFAULT_DB_DIR, DEFAULT_MODEL, PerfumeCandidate, ScentRetriever
from research.runtime.catalog import RuntimeCatalog
from research.runtime.query_analyzer import entity_matches


DEFAULT_CASES = Path(__file__).resolve().parent / "runtime" / "retrieval_eval_cases.json"
DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "scentai_catalog.sqlite3"


@dataclass
class CaseResult:
    case_id: str
    query: str
    passed: bool
    failures: list[str]
    hit_count: int
    violation_count: int
    avg_popularity: float
    top_labels: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ScentAI retrieval quality.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--fetch-k", type=int, default=120)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reranker", default="", help="Optional CrossEncoder reranker model, e.g. BAAI/bge-reranker-base.")
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--no-local-files-only", action="store_true")
    parser.add_argument("--show-passed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    catalog = RuntimeCatalog(args.catalog) if args.catalog.exists() else None
    retriever = ScentRetriever(
        db_dir=args.db_dir,
        model_name=args.model,
        device=args.device,
        local_files_only=not args.no_local_files_only,
        reranker_model_name=args.reranker or None,
        reranker_device=args.reranker_device,
        catalog=catalog,
    )
    print(f"Collection count : {retriever.collection.count()}")
    print(f"Cases            : {len(cases)}")
    print(f"Top K / Fetch K  : {args.top_k}/{args.fetch_k}")

    results: list[CaseResult] = []
    for case in cases:
        candidates, analysis = retriever.retrieve(case["query"], top_k=args.top_k, fetch_k=args.fetch_k)
        result = evaluate_case(case, candidates, analysis)
        results.append(result)
        if args.show_passed or not result.passed:
            print_case(result, candidates, case, analysis)

    passed = sum(1 for result in results if result.passed)
    total = len(results)
    avg_hits = sum(result.hit_count for result in results) / total if total else 0.0
    avg_violations = sum(result.violation_count for result in results) / total if total else 0.0
    avg_popularity = sum(result.avg_popularity for result in results) / total if total else 0.0

    print("\n" + "=" * 100)
    print("Retrieval evaluation summary")
    print(f"Passed          : {passed}/{total} ({passed / total * 100:.1f}%)")
    print(f"Avg hit count   : {avg_hits:.2f}/{args.top_k}")
    print(f"Avg violations  : {avg_violations:.2f}")
    print(f"Avg popularity  : {avg_popularity:.1f}")
    if passed != total:
        raise SystemExit(1)


def evaluate_case(case: dict[str, Any], candidates: list[PerfumeCandidate], analysis: Any) -> CaseResult:
    failures: list[str] = []
    violations = 0

    allowed_genders = {x.lower() for x in case.get("allowed_genders", [])}
    expected_season = case.get("expected_season")
    expected_time = case.get("expected_time")
    must_exclude = {x.lower() for x in case.get("must_exclude", [])}
    term_min_hits = {str(k).lower(): int(v) for k, v in (case.get("term_min_hits") or {}).items()}
    min_popularity = int(case.get("min_popularity") or 0)
    allow_low_popularity = bool(case.get("allow_low_popularity"))
    must_include_any = {x.lower() for x in case.get("must_include_any", [])}
    min_hits = int(case.get("min_hits_in_top_k") or 0)
    required_brand = str(case.get("required_brand") or "").strip()
    excluded_brand_terms = [str(item) for item in case.get("excluded_brand_terms", [])]
    expected_reference = str(case.get("expected_resolved_reference") or "").strip()
    exclude_exact_reference = bool(case.get("exclude_exact_reference"))

    hit_count = 0
    term_hits = {term: 0 for term in term_min_hits}
    popularity_total = 0
    for candidate in candidates:
        terms = candidate_terms(candidate)
        meta = candidate.metadata
        popularity = int(meta.get("popularity") or 0)
        popularity_total += popularity

        if must_include_any and terms & must_include_any:
            hit_count += 1
        for term in term_hits:
            if term_conflicts(term, terms):
                term_hits[term] += 1

        if allowed_genders and str(meta.get("gender") or "").lower() not in allowed_genders:
            violations += 1
        if expected_season and not bool(meta.get(f"season_{expected_season}")):
            violations += 1
        if expected_time and not bool(meta.get(f"time_{expected_time}")):
            violations += 1
        if must_exclude and any(term_conflicts(term, terms) for term in must_exclude):
            violations += 1
        if min_popularity and not allow_low_popularity and popularity < min_popularity:
            violations += 1
        if required_brand and not entity_matches(required_brand, candidate.brand):
            violations += 1
        if any(entity_matches(term, candidate.brand) for term in excluded_brand_terms):
            violations += 1

    if must_include_any and hit_count < min_hits:
        failures.append(f"hit_count {hit_count} < {min_hits}")
    for term, min_count in term_min_hits.items():
        if term_hits[term] < min_count:
            failures.append(f"term_hit {term} {term_hits[term]} < {min_count}")
    if violations:
        failures.append(f"violations {violations}")
    if len(candidates) < min(5, min_hits or 5):
        failures.append(f"too_few_candidates {len(candidates)}")
    if expected_reference and str(getattr(analysis, "resolved_reference", "") or "").lower() != expected_reference.lower():
        failures.append(
            f"resolved_reference {getattr(analysis, 'resolved_reference', None)!r} != {expected_reference!r}"
        )
    if exclude_exact_reference and expected_reference:
        leaked = [candidate.label for candidate in candidates if candidate.label.lower() == expected_reference.lower()]
        if leaked:
            failures.append(f"exact_reference_leaked {leaked}")

    return CaseResult(
        case_id=str(case["id"]),
        query=str(case["query"]),
        passed=not failures,
        failures=failures,
        hit_count=hit_count,
        violation_count=violations,
        avg_popularity=popularity_total / len(candidates) if candidates else 0.0,
        top_labels=[candidate.label for candidate in candidates[:5]],
    )


def print_case(result: CaseResult, candidates: list[PerfumeCandidate], case: dict[str, Any], analysis: Any) -> None:
    status = "PASS" if result.passed else "FAIL"
    print("\n" + "=" * 100)
    print(f"{status} {result.case_id}: {result.query}")
    print(f"Failures: {result.failures or '-'}")
    print(f"Analysis: {analysis}")
    print(f"Expected: {case}")
    for rank, candidate in enumerate(candidates, 1):
        meta = candidate.metadata
        print(
            f"{rank:02d}. {candidate.label} [{meta.get('gender')}] "
            f"score={candidate.final_score:.3f} dist={candidate.distance:.3f} "
            f"rating={float(meta.get('rating') or 0):.2f} votes={meta.get('popularity')}"
        )
        print(f"    seasons={meta.get('seasons_csv')} time={meta.get('time_profile_csv')}")
        print(f"    accords={meta.get('accords_csv')}")
        print(f"    notes={meta.get('notes_csv')}")


def candidate_terms(candidate: PerfumeCandidate) -> set[str]:
    meta = candidate.metadata
    terms = set(csv_terms(meta.get("accords_csv")))
    terms |= set(csv_terms(meta.get("notes_csv")))
    terms.add(str(meta.get("top_accord") or "").lower())
    terms.add(str(meta.get("top_note") or "").lower())
    return {term for term in terms if term}


def csv_terms(value: object) -> list[str]:
    if not value:
        return []
    return [part.strip().lower() for part in str(value).split(",") if part.strip()]


def term_conflicts(term: str, terms: set[str]) -> bool:
    return any(term == item or term in item or item in term for item in terms)


if __name__ == "__main__":
    main()
