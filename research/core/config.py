from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEAN_FILE = PROJECT_ROOT / "perfumes_clean.jsonl"


@dataclass(frozen=True)
class DatasetConfig:
    clean_file: Path = DEFAULT_CLEAN_FILE
    output_file: Path = PROJECT_ROOT / "training_L1_v2.jsonl"
    total: int = 100
    rag_ratio: float = 0.60
    seed: int | None = 42
    include_debug_meta: bool = False
    category_counts: dict[str, int] = field(default_factory=dict)


L1_DEFAULT_RATIOS = {
    "info": 0.20,
    "notes": 0.16,
    "accords": 0.16,
    "seasons": 0.16,
    "rating": 0.16,
    "comparison": 0.16,
}


L2_DEFAULT_RATIOS = {
    "single_filter": 0.12,
    "multi_filter": 0.25,
    "negative_filter": 0.20,
    "numeric_filter": 0.13,
    "ranking": 0.12,
    "comparison": 0.10,
    "no_match": 0.05,
    "inclusive_gender_explicit": 0.03,
}


L3_DEFAULT_RATIOS = {
    "casual_vibe": 0.22,
    "occasion": 0.14,
    "likes_dislikes": 0.16,
    "negative_preference": 0.12,
    "reference_similarity": 0.14,
    "messy_query": 0.14,
    "conceptual_contradiction": 0.08,
}


L4_DEFAULT_RATIOS = {
    "persona_upgrade": 0.18,
    "occasion_with_tradeoffs": 0.18,
    "compare_and_decide": 0.14,
    "avoidance_reasoning": 0.14,
    "style_translation": 0.16,
    "collection_gap": 0.15,
    "no_strong_match": 0.05,
}


L5_DEFAULT_RATIOS = {
    "empty_profile": 0.10,
    "profile_likes": 0.18,
    "profile_dislikes": 0.16,
    "profile_likes_and_dislikes": 0.20,
    "avoid_previous_recommendations": 0.12,
    "profile_query_conflict": 0.12,
    "profile_update_request": 0.06,
    "low_confidence_profile": 0.06,
}


def counts_from_ratios(total: int, ratios: dict[str, float]) -> dict[str, int]:
    counts = {name: round(total * ratio) for name, ratio in ratios.items()}
    diff = total - sum(counts.values())
    if diff:
        first_key = next(iter(counts))
        counts[first_key] += diff
    return counts
