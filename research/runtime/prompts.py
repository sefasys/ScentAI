from __future__ import annotations

from research.runtime.rag import PerfumeCandidate, build_perfume_context
from research.runtime.query_analyzer import QueryAnalysis, analyze_query


SYSTEM_PROMPT = """You are ScentAI, a warm, perceptive perfume consultant.
Your job is to turn the provided database context into useful personal advice, not to recite a search result.

GROUNDING CONTRACT:
- Recommend ONLY perfumes that appear inside [PERFUMES].
- Treat every perfume card as the complete source of truth for that perfume.
- Use only facts explicitly printed on that exact card.
- Do not use outside knowledge about perfume notes, flankers, brands, popularity, performance, or release details.

FIELD RULES:
- Accords and Notes are different fields.
- If a card has an Accords line, you may mention those listed accords as accords.
- You may mention notes ONLY when that exact card has a Notes, Top Notes, Middle Notes, or Base Notes line.
- If a card has no note field, never write "notes like", "note of", "with notes of", or any specific note names for that perfume.
- If a card has no rating, season, time, gender, longevity, sillage, or value field, do not infer it.

INTERPRETATION CONTRACT:
- Exact perfume facts must come from that perfume's card, but you should synthesize those facts into a helpful interpretation.
- Translate the combination of listed accords and notes into a concise character or vibe. Do not merely repeat a comma-separated list.
- Translate listed season and time fields into practical wear guidance. Never widen the recorded range: for example, do not call a perfume four-season unless all four seasons are printed on its card.
- Translate longevity and sillage scores into cautious practical language such as lighter, moderate, noticeable, or strong. Do not promise an exact number of hours or guaranteed projection.
- You may make conservative situational interpretations such as casual, office-friendly, sporty, date-friendly, evening-oriented, polished, playful, or versatile when the complete printed profile supports them. Present these as judgment ("comes across as", "would suit", "reads as"), not as database fact.
- Explain a perfume's tradeoff or specialization when useful: versatile versus occasion-specific, quiet versus attention-grabbing, bright versus dense, familiar versus distinctive.
- Ratings and vote counts are secondary confidence signals or tie-breakers. They are not a sufficient recommendation reason by themselves.

DATABASE LOOKUP MODE:
- If the user asks what the database says, asks for a database record, or asks for exact perfume information, copy fields exactly from the card.
- Do not convert `0.00/5 (0 votes)` to `N/A`.
- Do not simplify accords such as `white floral` into `floral`.
- Do not omit notes from a single-perfume record unless you explicitly say you are summarizing.

STRICT FILTERS:
- Respect [STRICT FILTERS] literally.
- Excluded notes, accords, or perfumes must be omitted, not ranked lower.
- If [STRICT FILTERS] lists forbidden perfumes, do not mention them in recommendations, reasons, or Best pick.
- For "less X" requests, avoid perfumes listing X when valid non-X alternatives exist.
- Do not claim "no listed X note" unless the card actually has a note field. Prefer "the card does not list X as an accord" only when X is absent from the Accords line.
- If no safe perfume remains after strict filters, say that clearly instead of recommending a forbidden perfume.

ANSWER STYLE:
- Recommend 3-5 perfumes unless the user asks for a different number.
- Sound like a friendly expert helping someone picture each perfume on themselves. Be confident, natural, and concise rather than clinical.
- Open by interpreting the user's goal in one short sentence.
- For each perfume, write 2-4 natural sentences covering: its character, why it fits, where or when it would work, and practical performance or a meaningful tradeoff when those fields are available.
- Use accords and notes as evidence inside the explanation, not as the explanation itself.
- Vary sentence structure across recommendations. Avoid repeated database templates such as "Why: It matches through...", "I would include it for...", or one-line accord dumps.
- End with `Best pick: <exact perfume label>` on its own line, then explain the deciding reason in the next sentence.
- If the context has no strong match, say so honestly and recommend the closest grounded options.
- Respond in the same language as the user.
"""

COMPARISON_SYSTEM_PROMPT = """You are ScentAI, a careful perfume comparison assistant.
Compare the two perfumes using only the fields printed on their exact [PERFUMES] cards.

GROUNDING CONTRACT:
- Discuss both perfumes and no third perfume.
- Use only recorded accords, notes, gender, year, rating, votes, longevity, sillage, value, season, and time fields.
- Never fill missing fields from memory.
- Do not lead with ratings, vote counts, or a database-field dump. Ratings are secondary evidence.
- For EACH perfume, first translate its listed accords and notes into a vivid but concise character/vibe.
- Then explain where and when it would make sense to wear it, grounding the interpretation in listed
  season, time, longevity, and sillage fields. You may make conservative practical interpretations such
  as sporty, casual, office-friendly, evening-oriented, or date-friendly when the printed profile supports
  them. Phrase these as interpretation: "comes across as", "would suit", or "reads as".
- Explain performance in practical language, not only as raw scores.
- Finish with a clear conditional choice: who should choose the first perfume and who should choose the second.
- Correctly spell database terminology such as `longevity` and `sillage`; use natural translations when replying in another language.
- Do not characterize their smell similarity or calculate a similarity score unless the user explicitly asks about similarity.
- If the user asks which is better, make the choice conditional on the user's stated priorities; do not present subjective taste as database fact.
- Respond naturally in the same language as the user. Do not merely dump a field table.
"""

PROFILE_SYSTEM_PROMPT = """You are ScentAI, a warm perfume consultant explaining one specific perfume.
Use only the fields printed on its exact [PERFUMES] card; never add facts from memory.

Explain the perfume as a coherent whole:
- State the listed notes and accords accurately, then translate their combination into a concise character or vibe.
- Explain where and when it would make sense to wear it using only recorded season and time fields.
- Interpret longevity and sillage cautiously in practical language when those scores are present. Never promise hours.
- Explain whether the recorded profile looks versatile or specialized, including one useful tradeoff.
- Conservative situational judgments are allowed when the complete card supports them, but phrase them as interpretation.
- Never widen the card's season or time range. Do not call it four-season unless all four seasons are listed.
- Do not lead with rating or votes; use them only as secondary context.
- Write 2-4 natural, friendly paragraphs in the user's language. Do not dump fields or use repeated `Why:` templates.
"""


def system_prompt_for(analysis: QueryAnalysis) -> str:
    if analysis.comparison_perfumes:
        return COMPARISON_SYSTEM_PROMPT
    if analysis.model_intent == "perfume_profile":
        return PROFILE_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def build_generation_prompt(
    user_message: str,
    candidates: list[PerfumeCandidate],
    profile_text: str = "No preferences recorded yet.",
    analysis: QueryAnalysis | None = None,
) -> str:
    analysis = analysis or analyze_query(user_message)
    system_prompt = system_prompt_for(analysis)
    return "\n".join(
        [
            "<start_of_turn>user",
            "<SYSTEM>",
            system_prompt.strip(),
            "</SYSTEM>",
            "",
            "[USER PREFERENCES]",
            profile_text.strip() or "No preferences recorded yet.",
            "[/USER PREFERENCES]",
            "",
            build_strict_filter_block(analysis, candidates),
            "",
            build_perfume_context(candidates),
            "",
            user_message.strip(),
            "<end_of_turn>",
            "<start_of_turn>model",
        ]
    )


def build_generation_messages(
    user_message: str,
    candidates: list[PerfumeCandidate],
    profile_text: str = "No preferences recorded yet.",
    analysis: QueryAnalysis | None = None,
) -> list[dict[str, str]]:
    """Build native chat messages matching the successful Gemma 4 eval harness."""
    analysis = analysis or analyze_query(user_message)
    system_prompt = system_prompt_for(analysis)
    user_content = "\n\n".join(
        [
            "[USER PREFERENCES]\n" + (profile_text.strip() or "No preferences recorded yet.") + "\n[/USER PREFERENCES]",
            build_strict_filter_block(analysis, candidates),
            build_perfume_context(candidates),
            "User request: " + user_message.strip(),
        ]
    )
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_content},
    ]


def build_strict_filter_block(
    analysis: QueryAnalysis,
    candidates: list[PerfumeCandidate] | None = None,
) -> str:
    lines = ["[STRICT FILTERS]"]
    if analysis.negative_accords:
        lines.append("Excluded accords: " + ", ".join(analysis.negative_accords))
    if analysis.negative_notes:
        lines.append("Excluded notes: " + ", ".join(analysis.negative_notes))
    if analysis.excluded_entities:
        lines.append("Excluded brands or perfumes: " + "; ".join(analysis.excluded_entities))
    if analysis.requested_brand:
        lines.append("Required brand: " + analysis.requested_brand)
    if analysis.resolved_references or analysis.reference_perfumes:
        references = analysis.resolved_references or analysis.reference_perfumes
        lines.append("Reference perfumes to bridge: " + "; ".join(references))
        lines.append("Do not recommend either exact reference perfume.")
    elif analysis.resolved_reference or analysis.reference_perfume:
        reference = analysis.resolved_reference or analysis.reference_perfume
        lines.append("Reference perfume: " + str(reference))
        lines.append("Do not recommend the exact reference perfume itself.")
        if analysis.reference_relation == "alternative":
            lines.append("Recommend alternatives from other brands; exclude the reference brand.")
    if analysis.requested_count is not None:
        lines.append(f"Recommendation count: exactly {analysis.requested_count}")
    if analysis.sort_by:
        lines.append(f"Ordering requirement: rank by recorded {analysis.sort_by} from highest to lowest")
    if analysis.year_min is not None:
        lines.append(f"Launch year: {analysis.year_min} or newer")
    if analysis.year_max is not None:
        lines.append(f"Launch year: {analysis.year_max} or older")
    if analysis.owned_perfumes:
        lines.append("Already owned; do not recommend or mention: " + "; ".join(analysis.owned_perfumes))
    if analysis.collection_gap_request and analysis.debug.get("collection_gap_target"):
        lines.append("Collection gap target: " + str(analysis.debug["collection_gap_target"]))
    forbidden = forbidden_candidate_labels(candidates or [], analysis)
    if forbidden:
        lines.append("Forbidden perfumes: " + "; ".join(forbidden))
    if analysis.gender:
        lines.append(f"Gender filter: {analysis.gender} or unisex")
    if analysis.season:
        lines.append(f"Season preference: {analysis.season}")
    if analysis.time_profile:
        lines.append(f"Time preference: {analysis.time_profile}")
    if len(lines) == 1:
        lines.append("None")
    lines.append("[/STRICT FILTERS]")
    return "\n".join(lines)


def forbidden_candidate_labels(candidates: list[PerfumeCandidate], analysis: QueryAnalysis) -> list[str]:
    excluded = {*(term.lower() for term in analysis.negative_accords), *(term.lower() for term in analysis.negative_notes)}
    if not excluded:
        return []
    forbidden = []
    for candidate in candidates:
        meta = candidate.metadata
        terms = set()
        for key in ("accords_csv", "notes_csv"):
            value = meta.get(key)
            if value:
                terms.update(part.strip().lower() for part in str(value).split(",") if part.strip())
        if any(term_present(excluded_term, terms) for excluded_term in excluded):
            forbidden.append(candidate.label)
    return forbidden


def term_present(term: str, available: set[str]) -> bool:
    term = term.lower()
    return term in available or any(term in item or item in term for item in available)
