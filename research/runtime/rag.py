from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from research.runtime.catalog import RuntimeCatalog
from research.runtime.query_analyzer import (
    AMBIGUOUS_BRAND_TERMS,
    QueryAnalysis,
    analyze_query,
    entity_matches,
    normalize_match_text,
)


DEFAULT_DB_DIR = Path(__file__).resolve().parents[2] / "chroma_db_bge_m3"
DEFAULT_COLLECTION = "scentai_perfumes"
DEFAULT_MODEL = "BAAI/bge-m3"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class PerfumeCandidate:
    perfume_id: int
    name: str
    brand: str
    document: str
    metadata: dict[str, Any]
    distance: float
    semantic_score: float
    final_score: float
    reasons: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{self.name} by {self.brand}"


class ScentRetriever:
    def __init__(
        self,
        db_dir: Path | str = DEFAULT_DB_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        local_files_only: bool = True,
        use_query_prefix: bool = True,
        reranker_model_name: str | None = None,
        reranker_device: str | None = None,
        catalog: RuntimeCatalog | None = None,
    ) -> None:
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.db_dir = Path(db_dir)
        self.collection_name = collection_name
        self.model_name = model_name
        self.use_query_prefix = use_query_prefix
        self.catalog = catalog
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        self.collection = self.client.get_collection(collection_name)
        self.model = SentenceTransformer(model_name, device=device, local_files_only=local_files_only)
        self.reranker = None
        if reranker_model_name:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(reranker_model_name, device=reranker_device or device)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        fetch_k: int = 80,
        analysis: QueryAnalysis | None = None,
        min_popularity: int | None = None,
        min_results_before_fallback: int = 20,
    ) -> tuple[list[PerfumeCandidate], QueryAnalysis]:
        analysis = analysis or analyze_query(query)
        where = build_metadata_filter(analysis, min_popularity=min_popularity)

        if analysis.reference_perfumes:
            multi_similarity_result = self._retrieve_multi_reference_similarity(
                query,
                analysis,
                where=where,
                top_k=top_k,
                fetch_k=fetch_k,
            )
            if multi_similarity_result is not None:
                return multi_similarity_result

        if analysis.reference_perfume:
            similarity_result = self._retrieve_reference_similarity(
                query,
                analysis,
                where=where,
                top_k=top_k,
                fetch_k=fetch_k,
            )
            if similarity_result is not None:
                return similarity_result

        raw = self._query_chroma(query, fetch_k, where)
        requested_brand = resolve_requested_brand(analysis.requested_brand, raw)
        brand_filter = {"brand": requested_brand} if requested_brand else None
        if requested_brand:
            raw = self._query_chroma(query, fetch_k, combine_filters(where, brand_filter))
        elif analysis.requested_brand:
            raw = [candidate for candidate in raw if entity_matches(analysis.requested_brand, candidate.brand)]

        if len(raw) < min_results_before_fallback and where:
            relaxed = build_metadata_filter(analysis, min_popularity=0, include_soft_filters=False)
            raw = self._query_chroma(query, fetch_k, combine_filters(relaxed, brand_filter))
            if analysis.requested_brand and not requested_brand:
                raw = [candidate for candidate in raw if entity_matches(analysis.requested_brand, candidate.brand)]

        filtered = [candidate for candidate in raw if not violates_negative(candidate.metadata, analysis)]
        if len(filtered) < top_k and where:
            relaxed = build_metadata_filter(analysis, min_popularity=0, include_soft_filters=False)
            relaxed_raw = self._query_chroma(query, fetch_k, combine_filters(relaxed, brand_filter))
            if analysis.requested_brand and not requested_brand:
                relaxed_raw = [candidate for candidate in relaxed_raw if entity_matches(analysis.requested_brand, candidate.brand)]
            merged = merge_candidates(raw, relaxed_raw)
            filtered = [candidate for candidate in merged if not violates_negative(candidate.metadata, analysis)]

        scored = [score_candidate(candidate, analysis) for candidate in filtered]
        scored.sort(key=lambda item: item.final_score, reverse=True)
        if analysis.sort_by:
            scored = explicit_metric_sort(scored, analysis.sort_by)
        elif self.reranker and scored:
            scored = self._rerank(query, scored[: min(len(scored), 40)])
        collision_brands = {
            normalize_match_text(candidate.brand)
            for candidate in scored
            if has_accidental_brand_collision(candidate, analysis)
        }
        deduped = brand_dedup(
            scored,
            max_per_brand=top_k if analysis.requested_brand else 2,
            per_brand_limits={brand: 1 for brand in collision_brands},
        )
        return deduped[:top_k], analysis

    def _retrieve_multi_reference_similarity(
        self,
        query: str,
        analysis: QueryAnalysis,
        *,
        where: dict | None,
        top_k: int,
        fetch_k: int,
    ) -> tuple[list[PerfumeCandidate], QueryAnalysis] | None:
        references = [self._resolve_reference(hint) for hint in analysis.reference_perfumes]
        if any(reference is None for reference in references):
            return None
        resolved = [reference for reference in references if reference is not None]
        if len(resolved) < 2:
            return None

        query_raw = self._query_chroma(query, fetch_k, where)
        reference_raw = []
        for reference in resolved:
            reference_raw.extend(
                self._query_chroma(reference.document, max(fetch_k * 2, 180), where, use_query_prefix=False)
            )

        direct_maps: list[dict[int, dict[str, Any]]] = []
        direct_ids: set[int] = set()
        if self.catalog:
            for reference in resolved:
                rows = self.catalog.direct_similarity(reference.perfume_id, limit=160)
                row_map = {int(row["perfume_id"]): row for row in rows}
                direct_maps.append(row_map)
                direct_ids.update(row_map)
        direct_raw = self._candidates_by_ids(list(direct_ids))
        if direct_maps:
            direct_raw = [attach_multi_community_similarity(candidate, direct_maps) for candidate in direct_raw]

        reference_ids = {reference.perfume_id for reference in resolved}
        merged = merge_candidates(direct_raw, reference_raw, query_raw)
        filtered = [
            candidate
            for candidate in merged
            if candidate.perfume_id not in reference_ids and not violates_negative(candidate.metadata, analysis)
        ]
        scored = [score_multi_similarity_candidate(candidate, resolved, analysis) for candidate in filtered]
        scored.sort(key=lambda item: item.final_score, reverse=True)
        deduped = brand_dedup(scored, max_per_brand=1)
        labels = tuple(reference.label for reference in resolved)
        updated_analysis = replace(
            analysis,
            resolved_reference=" + ".join(labels),
            resolved_references=labels,
            debug={
                **analysis.debug,
                "retrieval_route": "multi_reference_similarity",
                "resolved_references": list(labels),
                "reference_perfume_ids": [reference.perfume_id for reference in resolved],
            },
        )
        return deduped[:top_k], updated_analysis

    def _retrieve_reference_similarity(
        self,
        query: str,
        analysis: QueryAnalysis,
        *,
        where: dict | None,
        top_k: int,
        fetch_k: int,
    ) -> tuple[list[PerfumeCandidate], QueryAnalysis] | None:
        reference = self._resolve_reference(analysis.reference_perfume or "")
        if reference is None:
            return None

        query_raw = self._query_chroma(query, fetch_k, where)
        requested_brand = resolve_requested_brand(analysis.requested_brand, query_raw)
        brand_filter = {"brand": requested_brand} if requested_brand else None
        candidate_where = combine_filters(where, brand_filter)
        similarity_fetch_k = max(fetch_k * 3, 240)
        reference_raw = self._query_chroma(
            reference.document,
            similarity_fetch_k,
            candidate_where,
            use_query_prefix=False,
        )
        direct_raw: list[PerfumeCandidate] = []
        if self.catalog:
            direct_rows = self.catalog.direct_similarity(reference.perfume_id, limit=160)
            direct_scores = {int(row["perfume_id"]): row for row in direct_rows}
            direct_raw = self._candidates_by_ids(list(direct_scores))
            direct_raw = [attach_community_similarity(candidate, direct_scores[candidate.perfume_id]) for candidate in direct_raw]
        merged = merge_candidates(direct_raw, reference_raw, query_raw)
        if analysis.requested_brand and not requested_brand:
            merged = [candidate for candidate in merged if entity_matches(analysis.requested_brand, candidate.brand)]
        filtered = [
            candidate
            for candidate in merged
            if candidate.perfume_id != reference.perfume_id
            and not violates_negative(candidate.metadata, analysis)
            and not (
                analysis.reference_relation == "alternative"
                and normalize_match_text(candidate.brand) == normalize_match_text(reference.brand)
            )
        ]
        scored = [score_similarity_candidate(candidate, reference, analysis) for candidate in filtered]
        scored.sort(key=lambda item: item.final_score, reverse=True)
        if analysis.requested_brand:
            max_per_brand = top_k
        else:
            max_per_brand = 1
        deduped = brand_dedup(scored, max_per_brand=max_per_brand)
        updated_analysis = replace(
            analysis,
            resolved_reference=reference.label,
            debug={
                **analysis.debug,
                "retrieval_route": "reference_similarity",
                "resolved_reference": reference.label,
                "reference_perfume_id": reference.perfume_id,
            },
        )
        return deduped[:top_k], updated_analysis

    def _resolve_reference(self, hint: str) -> PerfumeCandidate | None:
        if self.catalog:
            row = self.catalog.resolve(hint)
            if row:
                candidates = self._candidates_by_ids([int(row["perfume_id"])])
                if candidates:
                    return candidates[0]
        candidates = self._query_chroma(f"Name: {hint}", 60, None)
        if not candidates:
            return None
        ranked = sorted(
            candidates,
            key=lambda candidate: reference_resolution_score(hint, candidate),
            reverse=True,
        )
        best = ranked[0]
        return best if reference_resolution_score(hint, best) >= 0.62 else None

    def _candidates_by_ids(self, perfume_ids: list[int]) -> list[PerfumeCandidate]:
        if not perfume_ids:
            return []
        results = self.collection.get(
            ids=[str(perfume_id) for perfume_id in perfume_ids],
            include=["documents", "metadatas"],
        )
        by_id = {}
        for item_id, document, metadata in zip(results["ids"], results["documents"], results["metadatas"]):
            perfume_id = int(item_id)
            meta = dict(metadata)
            if self.catalog:
                meta = self.catalog.enrich_metadata(perfume_id, meta)
            by_id[perfume_id] = PerfumeCandidate(
                perfume_id=perfume_id,
                name=str(meta.get("name") or ""),
                brand=str(meta.get("brand") or ""),
                document=document,
                metadata=meta,
                distance=1.0,
                semantic_score=0.0,
                final_score=0.0,
                reasons=("catalog",),
            )
        return [by_id[perfume_id] for perfume_id in perfume_ids if perfume_id in by_id]

    def candidates_by_ids(self, perfume_ids: list[int]) -> list[PerfumeCandidate]:
        return self._candidates_by_ids(perfume_ids)

    def _query_chroma(
        self,
        query: str,
        fetch_k: int,
        where: dict | None,
        *,
        use_query_prefix: bool | None = None,
    ) -> list[PerfumeCandidate]:
        should_prefix = self.use_query_prefix if use_query_prefix is None else use_query_prefix
        encoded_query = BGE_QUERY_PREFIX + query if should_prefix else query
        embedding = [list(self._encode_query(encoded_query))]
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=fetch_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        candidates: list[PerfumeCandidate] = []
        for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            perfume_id = int(meta.get("perfume_id") or 0)
            enriched_meta = dict(meta)
            if self.catalog:
                enriched_meta = self.catalog.enrich_metadata(perfume_id, enriched_meta)
            candidates.append(
                PerfumeCandidate(
                    perfume_id=perfume_id,
                    name=str(enriched_meta.get("name") or ""),
                    brand=str(enriched_meta.get("brand") or ""),
                    document=doc,
                    metadata=enriched_meta,
                    distance=float(distance),
                    semantic_score=max(0.0, 1.0 - float(distance)),
                    final_score=max(0.0, 1.0 - float(distance)),
                    reasons=("semantic",),
                )
            )
        return candidates

    @lru_cache(maxsize=512)
    def _encode_query(self, encoded_query: str) -> tuple[float, ...]:
        vector = self.model.encode(encoded_query, normalize_embeddings=True)
        return tuple(float(value) for value in vector)

    def _rerank(self, query: str, candidates: list[PerfumeCandidate]) -> list[PerfumeCandidate]:
        assert self.reranker is not None
        pairs = [(query, candidate.document) for candidate in candidates]
        raw_scores = [float(score) for score in self.reranker.predict(pairs)]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        span = max(max_score - min_score, 1e-9)
        reranked: list[PerfumeCandidate] = []
        for candidate, raw_score in zip(candidates, raw_scores):
            normalized = (raw_score - min_score) / span
            final_score = candidate.final_score * 0.55 + normalized * 0.45
            reranked.append(replace_candidate_score(candidate, final_score, f"reranker={normalized:.3f}"))
        reranked.sort(key=lambda item: item.final_score, reverse=True)
        return reranked


def build_metadata_filter(
    analysis: QueryAnalysis,
    *,
    min_popularity: int | None = None,
    include_soft_filters: bool = True,
) -> dict | None:
    filters: list[dict[str, object]] = []
    if analysis.gender:
        allowed = [analysis.gender] if analysis.gender == "unisex" else [analysis.gender, "unisex"]
        filters.append({"gender": {"$in": allowed}})
    if analysis.season:
        filters.append({f"season_{analysis.season}": True})
    if analysis.time_profile:
        filters.append({f"time_{analysis.time_profile}": True})
    if analysis.min_rating is not None:
        filters.append({"rating": {"$gte": analysis.min_rating}})
    if analysis.year_min is not None:
        filters.append({"year": {"$gte": analysis.year_min}})
    if analysis.year_max is not None:
        filters.append({"year": {"$lte": analysis.year_max}})

    popularity = analysis.min_popularity if min_popularity is None else min_popularity
    if include_soft_filters and popularity:
        filters.append({"popularity": {"$gte": popularity}})

    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def combine_filters(base_filter: dict | None, extra_filter: dict | None) -> dict | None:
    filters = [item for item in (base_filter, extra_filter) if item]
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def resolve_requested_brand(hint: str | None, candidates: list[PerfumeCandidate]) -> str | None:
    if not hint:
        return None
    brands = sorted({candidate.brand for candidate in candidates if candidate.brand})
    exact_matches = [brand for brand in brands if entity_matches(hint, brand)]
    if not exact_matches:
        return None
    exact_matches.sort(key=lambda brand: (len(brand), brand.lower()))
    return exact_matches[0]


def infer_requested_brand(query: str, candidates: list[PerfumeCandidate]) -> str | None:
    if not re.search(r"\b(perfumes?|fragrances?|scents?|parfüm|parfum|kokular?)\b", query, re.I):
        return None
    matches = []
    for brand in {candidate.brand for candidate in candidates if candidate.brand}:
        brand_norm = normalize_match_text(brand)
        if brand_norm and brand_norm in normalize_match_text(query):
            matches.append(brand)
    if not matches:
        return None
    matches.sort(key=lambda value: (-len(normalize_match_text(value)), value.lower()))
    return matches[0]


def explicit_metric_sort(candidates: list[PerfumeCandidate], field: str) -> list[PerfumeCandidate]:
    metadata_field = "popularity" if field == "popularity" else field
    available = [candidate for candidate in candidates if candidate.metadata.get(metadata_field) is not None]
    missing = [candidate for candidate in candidates if candidate.metadata.get(metadata_field) is None]

    def sort_value(candidate: PerfumeCandidate) -> float:
        if field == "rating":
            return confidence_adjusted_rating(
                float(candidate.metadata.get("rating") or 0.0),
                int(candidate.metadata.get("popularity") or 0),
            )
        return float(candidate.metadata.get(metadata_field) or 0.0)

    available.sort(
        key=lambda candidate: (
            sort_value(candidate),
            candidate.final_score,
        ),
        reverse=True,
    )
    return available + missing


def violates_negative(candidate_meta: dict[str, Any], analysis: QueryAnalysis) -> bool:
    accords = _csv_terms(candidate_meta.get("accords_csv"))
    notes = _csv_terms(candidate_meta.get("notes_csv"))
    if any(_term_present(term, accords) for term in analysis.negative_accords):
        return True
    if any(_term_present(term, notes) for term in analysis.negative_notes):
        return True
    candidate_identity = " ".join(
        str(candidate_meta.get(key) or "")
        for key in ("name", "brand")
    )
    candidate_traits = " ".join([*accords, *notes])
    if any(
        entity_matches(entity, candidate_identity) or entity_matches(entity, candidate_traits)
        for entity in analysis.excluded_entities
    ):
        return True
    if any(entity_matches(owned, candidate_identity) for owned in analysis.owned_perfumes):
        return True
    return False


def reference_resolution_score(hint: str, candidate: PerfumeCandidate) -> float:
    hint_norm = normalize_match_text(hint)
    name_norm = normalize_match_text(candidate.name)
    label_norm = normalize_match_text(candidate.label).replace(" by ", " ")
    if not hint_norm or not name_norm:
        return 0.0

    hint_tokens = set(hint_norm.split())
    name_tokens = set(name_norm.split())
    label_tokens = set(label_norm.split()) - {"by"}
    if hint_norm == name_norm:
        lexical = 1.0
    elif hint_tokens == label_tokens or hint_tokens == name_tokens:
        lexical = 0.98
    else:
        token_overlap = len(hint_tokens & label_tokens) / max(len(hint_tokens | label_tokens), 1)
        lexical = max(
            difflib.SequenceMatcher(None, hint_norm, name_norm).ratio(),
            difflib.SequenceMatcher(None, hint_norm, label_norm).ratio(),
            token_overlap,
        )
    popularity = int(candidate.metadata.get("popularity") or 0)
    popularity_tiebreaker = min(math.log10(popularity + 1) / 100.0, 0.05)
    return lexical + popularity_tiebreaker


def score_similarity_candidate(
    candidate: PerfumeCandidate,
    reference: PerfumeCandidate,
    analysis: QueryAnalysis,
) -> PerfumeCandidate:
    candidate_meta = candidate.metadata
    reference_meta = reference.metadata

    accord_similarity = set_similarity(
        _csv_terms(candidate_meta.get("accords_csv")),
        _csv_terms(reference_meta.get("accords_csv")),
    )
    note_similarity = set_similarity(
        _csv_terms(candidate_meta.get("notes_csv")),
        _csv_terms(reference_meta.get("notes_csv")),
    )
    season_similarity = set_similarity(
        _csv_terms(candidate_meta.get("seasons_csv")),
        _csv_terms(reference_meta.get("seasons_csv")),
    )
    time_similarity = set_similarity(
        _csv_terms(candidate_meta.get("time_profile_csv")),
        _csv_terms(reference_meta.get("time_profile_csv")),
    )
    gender_similarity = gender_compatibility(
        str(candidate_meta.get("gender") or ""),
        str(reference_meta.get("gender") or ""),
    )

    components = [(accord_similarity, 0.50)]
    if _csv_terms(candidate_meta.get("notes_csv")) and _csv_terms(reference_meta.get("notes_csv")):
        components.append((note_similarity, 0.27))
    if _csv_terms(candidate_meta.get("seasons_csv")) and _csv_terms(reference_meta.get("seasons_csv")):
        components.append((season_similarity, 0.10))
    if _csv_terms(candidate_meta.get("time_profile_csv")) and _csv_terms(reference_meta.get("time_profile_csv")):
        components.append((time_similarity, 0.08))
    components.append((gender_similarity, 0.05))
    structural_similarity = weighted_average(components)

    candidate_terms = _csv_terms(candidate_meta.get("accords_csv")) | _csv_terms(candidate_meta.get("notes_csv"))
    wanted_terms = {*analysis.wanted_accords, *analysis.wanted_notes}
    preference_score = (
        sum(1 for term in wanted_terms if _term_present(term, candidate_terms)) / len(wanted_terms)
        if wanted_terms
        else 0.0
    )
    popularity = int(candidate_meta.get("popularity") or 0)
    rating = confidence_adjusted_rating(
        float(candidate_meta.get("rating") or 0.0),
        popularity,
    )
    popularity_score = min(math.log10(popularity + 1) / math.log10(50000), 1.0) if popularity else 0.0
    rating_score = min(max((rating - 3.2) / 1.3, 0.0), 1.0) if rating else 0.0
    quality_score = popularity_score * 0.55 + rating_score * 0.45
    community_similarity = float(candidate_meta.get("community_similarity") or 0.0)
    if community_similarity > 0:
        final_score = (
            community_similarity * 0.42
            + structural_similarity * 0.30
            + candidate.semantic_score * 0.16
            + quality_score * 0.08
            + preference_score * 0.04
        )
    else:
        final_score = (
            candidate.semantic_score * 0.34
            + structural_similarity * 0.50
            + quality_score * 0.08
            + preference_score * 0.08
        )
    reasons = (
        f"reference={reference.label}",
        f"semantic={candidate.semantic_score:.3f}",
        f"accord_similarity={accord_similarity:.3f}",
        f"note_similarity={note_similarity:.3f}",
        f"season_similarity={season_similarity:.3f}",
        f"time_similarity={time_similarity:.3f}",
        f"structural_similarity={structural_similarity:.3f}",
        f"preference={preference_score:.3f}",
        f"community_similarity={community_similarity:.3f}",
    )
    return PerfumeCandidate(
        perfume_id=candidate.perfume_id,
        name=candidate.name,
        brand=candidate.brand,
        document=candidate.document,
        metadata=candidate.metadata,
        distance=candidate.distance,
        semantic_score=candidate.semantic_score,
        final_score=final_score,
        reasons=reasons,
    )


def score_multi_similarity_candidate(
    candidate: PerfumeCandidate,
    references: list[PerfumeCandidate],
    analysis: QueryAnalysis,
) -> PerfumeCandidate:
    per_reference = [score_similarity_candidate(candidate, reference, analysis) for reference in references]
    scores = [item.final_score for item in per_reference]
    bridge_score = (sum(scores) / len(scores)) * 0.65 + min(scores) * 0.35
    reasons = ["multi_reference_bridge"]
    reasons.extend(
        f"similarity_to={reference.label}:{score:.3f}"
        for reference, score in zip(references, scores)
    )
    return PerfumeCandidate(
        perfume_id=candidate.perfume_id,
        name=candidate.name,
        brand=candidate.brand,
        document=candidate.document,
        metadata=candidate.metadata,
        distance=candidate.distance,
        semantic_score=candidate.semantic_score,
        final_score=bridge_score,
        reasons=tuple(reasons),
    )


def set_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    jaccard = intersection / len(left | right)
    overlap = intersection / min(len(left), len(right))
    return jaccard * 0.65 + overlap * 0.35


def weighted_average(components: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in components)
    if not total_weight:
        return 0.0
    return sum(value * weight for value, weight in components) / total_weight


def gender_compatibility(left: str, right: str) -> float:
    left = left.lower()
    right = right.lower()
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if "unisex" in {left, right}:
        return 0.75
    return 0.0


def attach_community_similarity(candidate: PerfumeCandidate, row: dict[str, Any]) -> PerfumeCandidate:
    metadata = dict(candidate.metadata)
    metadata["community_similarity"] = float(row.get("community_similarity") or 0.0)
    metadata["similarity_up_votes"] = int(row.get("up_votes") or 0)
    metadata["similarity_down_votes"] = int(row.get("down_votes") or 0)
    return PerfumeCandidate(
        perfume_id=candidate.perfume_id,
        name=candidate.name,
        brand=candidate.brand,
        document=candidate.document,
        metadata=metadata,
        distance=candidate.distance,
        semantic_score=candidate.semantic_score,
        final_score=candidate.final_score,
        reasons=candidate.reasons,
    )


def attach_multi_community_similarity(
    candidate: PerfumeCandidate,
    direct_maps: list[dict[int, dict[str, Any]]],
) -> PerfumeCandidate:
    rows = [row_map.get(candidate.perfume_id) for row_map in direct_maps]
    scores = [float(row.get("community_similarity") or 0.0) if row else 0.0 for row in rows]
    metadata = dict(candidate.metadata)
    metadata["community_similarity"] = sum(scores) / len(scores)
    metadata["similarity_up_votes"] = sum(int(row.get("up_votes") or 0) for row in rows if row)
    metadata["similarity_down_votes"] = sum(int(row.get("down_votes") or 0) for row in rows if row)
    return PerfumeCandidate(
        perfume_id=candidate.perfume_id,
        name=candidate.name,
        brand=candidate.brand,
        document=candidate.document,
        metadata=metadata,
        distance=candidate.distance,
        semantic_score=candidate.semantic_score,
        final_score=candidate.final_score,
        reasons=candidate.reasons,
    )


def score_candidate(candidate: PerfumeCandidate, analysis: QueryAnalysis) -> PerfumeCandidate:
    meta = candidate.metadata
    accords = _csv_terms(meta.get("accords_csv"))
    notes = _csv_terms(meta.get("notes_csv"))

    popularity = int(meta.get("popularity") or 0)
    rating = confidence_adjusted_rating(
        float(meta.get("rating") or 0.0),
        popularity,
    )
    popularity_score = min(math.log10(popularity + 1) / math.log10(50000), 1.0) if popularity > 0 else 0.0
    rating_score = min(max((rating - 3.2) / 1.3, 0.0), 1.0) if rating > 0 else 0.0

    wanted_accord_hits = sum(1 for term in analysis.wanted_accords if _term_present(term, accords))
    wanted_note_hits = sum(1 for term in analysis.wanted_notes if _term_present(term, notes))
    explicit_accord_hits = sum(
        1 for term in analysis.wanted_accords if _explicit_term(term, analysis.raw_query) and _term_present(term, accords)
    )
    explicit_note_hits = sum(
        1 for term in analysis.wanted_notes if _explicit_term(term, analysis.raw_query) and _term_present(term, notes)
    )
    intent_score = min(
        (wanted_accord_hits * 0.07)
        + (wanted_note_hits * 0.08)
        + (explicit_accord_hits * 0.12)
        + (explicit_note_hits * 0.20),
        0.48,
    )

    season_bonus = 0.04 if analysis.season and meta.get(f"season_{analysis.season}") else 0.0
    gender_bonus = 0.04 if analysis.gender and meta.get("gender") in {analysis.gender, "unisex"} else 0.0
    brand_collision_penalty = 0.14 if has_accidental_brand_collision(candidate, analysis) else 0.0

    final_score = (
        candidate.semantic_score * 0.66
        + popularity_score * 0.16
        + rating_score * 0.10
        + intent_score
        + season_bonus
        + gender_bonus
        - brand_collision_penalty
    )
    reasons = [
        f"semantic={candidate.semantic_score:.3f}",
        f"pop={popularity_score:.3f}",
        f"rating={rating_score:.3f}",
    ]
    if wanted_accord_hits:
        reasons.append(f"accord_hits={wanted_accord_hits}")
    if wanted_note_hits:
        reasons.append(f"note_hits={wanted_note_hits}")
    if explicit_accord_hits:
        reasons.append(f"explicit_accord_hits={explicit_accord_hits}")
    if explicit_note_hits:
        reasons.append(f"explicit_note_hits={explicit_note_hits}")
    if season_bonus:
        reasons.append("season")
    if gender_bonus:
        reasons.append("gender")
    if brand_collision_penalty:
        reasons.append(f"ambiguous_brand_collision=-{brand_collision_penalty:.2f}")

    return PerfumeCandidate(
        perfume_id=candidate.perfume_id,
        name=candidate.name,
        brand=candidate.brand,
        document=candidate.document,
        metadata=candidate.metadata,
        distance=candidate.distance,
        semantic_score=candidate.semantic_score,
        final_score=final_score,
        reasons=tuple(reasons),
    )


def brand_dedup(
    candidates: list[PerfumeCandidate],
    max_per_brand: int = 2,
    per_brand_limits: dict[str, int] | None = None,
) -> list[PerfumeCandidate]:
    counts: dict[str, int] = {}
    normalized_limits = {
        normalize_match_text(brand): limit
        for brand, limit in (per_brand_limits or {}).items()
    }
    seen_labels: set[str] = set()
    output: list[PerfumeCandidate] = []
    for candidate in candidates:
        label = normalize_match_text(candidate.label)
        if label in seen_labels:
            continue
        brand = normalize_match_text(candidate.brand)
        brand_limit = normalized_limits.get(brand, max_per_brand)
        if counts.get(brand, 0) >= brand_limit:
            continue
        output.append(candidate)
        seen_labels.add(label)
        counts[brand] = counts.get(brand, 0) + 1
    return output


def has_accidental_brand_collision(candidate: PerfumeCandidate, analysis: QueryAnalysis) -> bool:
    """Detect when a semantic query word happens to equal a catalog brand name."""
    if analysis.requested_brand:
        return False
    brand = normalize_match_text(candidate.brand)
    if not brand or brand not in {normalize_match_text(term) for term in AMBIGUOUS_BRAND_TERMS}:
        return False
    query = normalize_match_text(analysis.raw_query)
    return bool(re.search(rf"\b{re.escape(brand)}\b", query, re.I))


def confidence_adjusted_rating(
    rating: float,
    votes: int,
    *,
    prior_mean: float = 4.01,
    prior_weight: int = 50,
) -> float:
    """Shrink low-vote ratings toward the catalog's vote-weighted global mean."""
    if rating <= 0 or votes <= 0:
        return 0.0
    return (votes * rating + prior_weight * prior_mean) / (votes + prior_weight)


def merge_candidates(*candidate_lists: list[PerfumeCandidate]) -> list[PerfumeCandidate]:
    merged: list[PerfumeCandidate] = []
    seen: set[int] = set()
    for candidates in candidate_lists:
        for candidate in candidates:
            if candidate.perfume_id in seen:
                continue
            seen.add(candidate.perfume_id)
            merged.append(candidate)
    return merged


def replace_candidate_score(candidate: PerfumeCandidate, final_score: float, reason: str) -> PerfumeCandidate:
    return PerfumeCandidate(
        perfume_id=candidate.perfume_id,
        name=candidate.name,
        brand=candidate.brand,
        document=candidate.document,
        metadata=candidate.metadata,
        distance=candidate.distance,
        semantic_score=candidate.semantic_score,
        final_score=final_score,
        reasons=(*candidate.reasons, reason),
    )


def build_perfume_context(candidates: list[PerfumeCandidate]) -> str:
    cards = [build_grounded_card(candidate) for candidate in candidates]
    return "[PERFUMES]\n" + "\n\n".join(card for card in cards if card.strip()) + "\n[/PERFUMES]"


def build_grounded_card(candidate: PerfumeCandidate) -> str:
    """Build a compact card from structured metadata only.

    The generation model is prone to filling in familiar perfume facts from memory.
    Keeping the context field-based makes it easier to enforce "only from card" rules.
    """
    meta = candidate.metadata
    lines = [
        f"{candidate.label} - {meta.get('gender') or 'unknown'}",
        f"Accords: {_clean_csv(meta.get('accords_csv')) or 'not provided'}",
    ]
    if meta.get("year") is not None:
        lines.append(f"Launch Year: {int(meta['year'])}")
    notes = _clean_csv(meta.get("notes_csv"))
    if notes:
        lines.append(f"Notes: {notes}")
    rating = float(meta.get("rating") or 0.0)
    popularity = int(meta.get("popularity") or 0)
    if rating > 0 or popularity > 0:
        lines.append(f"Rating: {rating:.2f}/5 ({popularity} votes)")
    performance = []
    if meta.get("longevity") is not None:
        performance.append(f"Longevity: {float(meta['longevity']):.2f}/5")
    if meta.get("sillage") is not None:
        performance.append(f"Sillage: {float(meta['sillage']):.2f}/4")
    if meta.get("value_score") is not None:
        performance.append(f"Value: {float(meta['value_score']):.2f}/5")
    if performance:
        lines.append(" | ".join(performance))
    seasons = _title_csv(meta.get("seasons_csv"))
    times = _title_csv(meta.get("time_profile_csv"))
    if seasons or times:
        parts = []
        if seasons:
            parts.append(f"Best Seasons: {seasons}")
        if times:
            parts.append(f"Time: {times}")
        lines.append(" | ".join(parts))
    if meta.get("similarity_up_votes") is not None:
        lines.append(
            "Community similarity: "
            f"{int(meta.get('similarity_up_votes') or 0)} up / {int(meta.get('similarity_down_votes') or 0)} down votes"
        )
    lines.append(f"Retrieval evidence: {', '.join(candidate.reasons)}")
    return "\n".join(lines)


def _clean_csv(value: object) -> str:
    if not value:
        return ""
    return ", ".join(part.strip() for part in str(value).split(",") if part.strip())


def _title_csv(value: object) -> str:
    if not value:
        return ""
    return ", ".join(part.strip().title() for part in str(value).split(",") if part.strip())


def _csv_terms(value: object) -> set[str]:
    if not value:
        return set()
    return {part.strip().lower() for part in str(value).split(",") if part.strip()}


def _term_present(term: str, available: set[str]) -> bool:
    term_low = term.lower()
    if term_low in available:
        return True
    return any(term_low in item or item in term_low for item in available)


def _explicit_term(term: str, query: str) -> bool:
    return term.lower() in query.lower()
