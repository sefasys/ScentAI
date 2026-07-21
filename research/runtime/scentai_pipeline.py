from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from research.runtime.exact_lookup import is_exact_lookup_query, maybe_answer_exact_lookup, render_database_lookup_answer
from research.runtime.prompts import build_generation_prompt, build_strict_filter_block
from research.runtime.query_analyzer import QueryAnalysis
from research.runtime.rag import DEFAULT_DB_DIR, DEFAULT_MODEL, PerfumeCandidate, ScentRetriever, build_perfume_context


@dataclass
class PipelineResult:
    query: str
    route: str
    answer: str | None
    prompt: str | None
    analysis: dict[str, Any]
    candidates: list[dict[str, Any]]


class ScentAIPipeline:
    def __init__(
        self,
        *,
        db_dir: Path | str = DEFAULT_DB_DIR,
        embedding_model: str = DEFAULT_MODEL,
        device: str | None = None,
        local_files_only: bool = True,
        top_k: int = 8,
        fetch_k: int = 80,
    ) -> None:
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.retriever = ScentRetriever(
            db_dir=db_dir,
            model_name=embedding_model,
            device=device,
            local_files_only=local_files_only,
        )

    def run(
        self,
        query: str,
        *,
        profile_text: str = "No preferences recorded yet.",
        prompt_only: bool = False,
        template_answer: bool = True,
    ) -> PipelineResult:
        candidates, analysis = self.retriever.retrieve(query, top_k=self.top_k, fetch_k=self.fetch_k)
        prompt = build_generation_prompt(query, candidates, profile_text=profile_text, analysis=analysis)

        context = build_perfume_context(candidates)
        exact_answer = maybe_answer_exact_lookup(context + "\n\n" + query)
        if exact_answer is None and is_exact_lookup_query(query) and candidates:
            exact_answer = render_database_lookup_answer(build_perfume_context(candidates[:1]) + "\n\n" + query)
        if exact_answer:
            route = "deterministic_exact_lookup"
            answer = exact_answer
        elif prompt_only:
            route = "prompt_only"
            answer = None
        elif template_answer:
            route = "template_recommendation"
            answer = build_template_recommendation(query, candidates, analysis)
        else:
            route = "llm_required"
            answer = None

        return PipelineResult(
            query=query,
            route=route,
            answer=answer,
            prompt=prompt,
            analysis=analysis_to_dict(analysis),
            candidates=[candidate_to_dict(candidate) for candidate in candidates],
        )


def build_template_recommendation(query: str, candidates: list[PerfumeCandidate], analysis: QueryAnalysis) -> str:
    if not candidates:
        return "I could not find a safe grounded match in the database context."

    lines = []
    is_profile = analysis.model_intent == "perfume_profile"
    if is_profile:
        lines.append(f"Here is a grounded profile of {candidates[0].label}.")
    elif analysis.resolved_reference:
        lines.append(f"Here are the closest grounded matches to {analysis.resolved_reference}.")
    elif analysis.negative_accords or analysis.negative_notes:
        avoided = ", ".join([*analysis.negative_accords, *analysis.negative_notes])
        lines.append(f"I filtered out options that list: {avoided}.")
    else:
        lines.append("Here are the strongest grounded matches from the local retrieval set.")

    result_count = 1 if is_profile else (analysis.requested_count or 5)
    for index, candidate in enumerate(candidates[:result_count], 1):
        meta = candidate.metadata
        lines.append(f"{index}. {candidate.label}")
        lines.append(template_character_sentence(meta))
        evidence = evidence_terms(candidate, analysis)
        if evidence:
            lines.append(f"That directly supports the request through {evidence.removeprefix('listed ')}.")
        wear = template_wear_sentence(meta)
        if wear:
            lines.append(wear)
        performance = template_performance_sentence(meta)
        if performance:
            lines.append(performance)

    if not is_profile:
        lines.append(f"Best pick: {candidates[0].label}")
        lines.append("Its complete recorded profile is the strongest match among these retrieved options.")
    return "\n".join(lines)


def template_character_sentence(meta: dict[str, Any]) -> str:
    accords = terms(meta.get("accords_csv"))[:3]
    notes = terms(meta.get("notes_csv"))[:3]
    if accords:
        sentence = f"Its card suggests a {natural_join(accords)}-led character"
    else:
        sentence = "Its database card provides the closest grounded profile in this shortlist"
    if notes:
        sentence += f", with {natural_join(notes)} shaping the recorded note structure"
    return sentence + "."


def template_wear_sentence(meta: dict[str, Any]) -> str:
    seasons = terms(meta.get("seasons_csv"))
    times = terms(meta.get("time_profile_csv"))
    if not seasons and not times:
        return ""
    season_text = natural_join([season.title() for season in seasons])
    time_text = natural_join([time.lower() for time in times])
    if season_text and time_text:
        recorded = f"{season_text}, with {time_text} wear"
    elif season_text:
        recorded = season_text
    else:
        recorded = f"{time_text} wear"
    if len(seasons) >= 3 and len(times) >= 2:
        interpretation = "That broad recorded wear window makes it a versatile option."
    elif len(times) >= 2:
        interpretation = "That gives it flexibility across the recorded seasons."
    else:
        interpretation = "That points to a more focused role rather than an all-purpose choice."
    return f"The card places it in {recorded}. {interpretation}"


def template_performance_sentence(meta: dict[str, Any]) -> str:
    longevity = optional_float(meta.get("longevity"))
    sillage = optional_float(meta.get("sillage"))
    parts = []
    if longevity is not None:
        if longevity >= 3.75:
            label = "strong recorded staying power"
        elif longevity >= 3.10:
            label = "solid recorded staying power"
        elif longevity >= 2.50:
            label = "moderate recorded staying power"
        else:
            label = "lighter recorded staying power"
        parts.append(f"{longevity:.2f}/5 longevity suggests {label}")
    if sillage is not None:
        if sillage >= 2.70:
            label = "a noticeable presence"
        elif sillage >= 2.10:
            label = "a moderate presence"
        else:
            label = "a more reserved presence"
        parts.append(f"{sillage:.2f}/4 sillage suggests {label}")
    return (natural_join(parts).capitalize() + ".") if parts else ""


def evidence_terms(candidate: PerfumeCandidate, analysis: QueryAnalysis) -> str:
    meta = candidate.metadata
    accords = terms(meta.get("accords_csv"))
    notes = terms(meta.get("notes_csv"))
    hits = []
    for term in analysis.wanted_accords:
        if term in accords:
            hits.append(term)
    for term in analysis.wanted_notes:
        if term in notes:
            hits.append(term)
    if not hits:
        return ""
    return "listed " + ", ".join(unique(hits[:5]))


def compact_terms(value: object, *, limit: int) -> str:
    selected = list(terms(value))[:limit]
    return "listed " + ", ".join(selected) if selected else ""


def optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def natural_join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def terms(value: object) -> list[str]:
    if not value:
        return []
    return [part.strip().lower() for part in str(value).split(",") if part.strip()]


def unique(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def analysis_to_dict(analysis: QueryAnalysis) -> dict[str, Any]:
    data = asdict(analysis)
    data["wanted_accords"] = list(analysis.wanted_accords)
    data["wanted_notes"] = list(analysis.wanted_notes)
    data["negative_accords"] = list(analysis.negative_accords)
    data["negative_notes"] = list(analysis.negative_notes)
    data["reference_perfumes"] = list(analysis.reference_perfumes)
    data["resolved_references"] = list(analysis.resolved_references)
    data["comparison_perfumes"] = list(analysis.comparison_perfumes)
    data["target_perfumes"] = list(analysis.target_perfumes)
    data["owned_perfumes"] = list(analysis.owned_perfumes)
    return data


def candidate_to_dict(candidate: PerfumeCandidate) -> dict[str, Any]:
    meta = candidate.metadata
    return {
        "perfume_id": candidate.perfume_id,
        "label": candidate.label,
        "gender": meta.get("gender"),
        "rating": meta.get("rating"),
        "popularity": meta.get("popularity"),
        "accords": meta.get("accords_csv"),
        "notes": meta.get("notes_csv"),
        "seasons": meta.get("seasons_csv"),
        "time_profile": meta.get("time_profile_csv"),
        "year": meta.get("year"),
        "longevity": meta.get("longevity"),
        "sillage": meta.get("sillage"),
        "value_score": meta.get("value_score"),
        "community_similarity": meta.get("community_similarity"),
        "score": candidate.final_score,
        "distance": candidate.distance,
        "reasons": list(candidate.reasons),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local ScentAI retrieval/runtime pipeline.")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--fetch-k", type=int, default=80)
    parser.add_argument("--no-local-files-only", action="store_true")
    parser.add_argument("--prompt-only", action="store_true")
    parser.add_argument("--no-template-answer", action="store_true")
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = " ".join(args.query).strip()
    pipeline = ScentAIPipeline(
        db_dir=args.db_dir,
        embedding_model=args.embedding_model,
        device=args.device,
        local_files_only=not args.no_local_files_only,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
    )
    result = pipeline.run(
        query,
        prompt_only=args.prompt_only,
        template_answer=not args.no_template_answer,
    )

    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return

    print(f"Route: {result.route}")
    print("Analysis:", json.dumps(result.analysis, ensure_ascii=False, indent=2))
    print("\nCandidates:")
    for index, candidate in enumerate(result.candidates, 1):
        print(
            f"{index:02d}. {candidate['label']} [{candidate['gender']}] "
            f"score={candidate['score']:.3f} rating={candidate['rating']} votes={candidate['popularity']}"
        )
        print(f"    accords={candidate['accords']}")
        print(f"    reasons={', '.join(candidate['reasons'])}")
    if result.answer:
        print("\nAnswer:")
        print(result.answer)
    if args.show_prompt and result.prompt:
        print("\nPrompt:")
        print(result.prompt)


if __name__ == "__main__":
    main()
