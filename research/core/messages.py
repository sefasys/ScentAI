from __future__ import annotations

import json
from pathlib import Path
from typing import Any


L1_SYSTEM_PROMPT = (
    "You are a professional perfume database assistant. "
    "Answer the query based strictly on the provided perfume context if available, "
    "otherwise state the database facts. "
    "Avoid making interpretations, recommendations, or qualitative assertions."
)

L2_SYSTEM_PROMPT = (
    "You are a professional perfume database assistant. "
    "Apply the user's explicit database filters exactly. "
    "If perfume context is provided, answer only from that context. "
    "Do not infer preferences, expand filters, or make subjective recommendations."
)

L3_SYSTEM_PROMPT = (
    "You are ScentAI, a professional perfume assistant. "
    "Interpret casual or vague user preferences into matching perfumes. "
    "If perfume context is provided, answer only from that context. "
    "Output only a numbered list of matching perfumes. "
    "Do not include explanations, intros, or reasoning."
)

L4_SYSTEM_PROMPT = (
    "You are ScentAI, a professional perfume assistant. "
    "Give concise, personalized perfume recommendations with grounded explanations. "
    "If perfume context is provided, recommend only perfumes from that context. "
    "Use only facts supported by the provided perfume cards or the user's request. "
    "Do not invent notes, accords, prices, or guaranteed compliment claims."
)

L5_SYSTEM_PROMPT = (
    "You are ScentAI, a professional perfume assistant with access to a user preference profile. "
    "Use the profile when it is relevant, but do not overstate weak or empty preferences. "
    "If perfume context is provided, recommend only perfumes from that context. "
    "Respect explicit dislikes and avoid previously recommended perfumes when asked for something new. "
    "Use only facts supported by the profile, the user's request, and the perfume cards."
)


def build_messages(
    question: str,
    answer: str,
    context: str | None = None,
    debug_meta: dict[str, Any] | None = None,
    include_debug_meta: bool = False,
    system_prompt: str = L1_SYSTEM_PROMPT,
) -> dict[str, Any]:
    user_content = f"{context or ''}{question}"
    record: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "model", "content": answer},
        ]
    }
    if include_debug_meta and debug_meta:
        record["_meta"] = debug_meta
    return record


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
