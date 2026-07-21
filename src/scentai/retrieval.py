from __future__ import annotations

import difflib
import json
import math
import os
import re
import sqlite3
import threading
import time
import unicodedata
from collections import defaultdict
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_COLLECTION = "scentai_perfumes"
DEFAULT_MODEL = "BAAI/bge-m3"
VALID_GENDERS = {"male", "female", "unisex"}
VALID_SEASONS = {"spring", "summer", "autumn", "winter"}
VALID_TIMES = {"day", "night"}
VALID_DISCOVERY_MODES = {"balanced", "mainstream", "niche"}

CONCENTRATION_ALIAS_PATTERNS = (
    (re.compile(r"\be\s*d\s*p\b|\bepd\b"), "eau de parfum"),
    (re.compile(r"\be\s*d\s*t\b|\betd\b"), "eau de toilette"),
    (re.compile(r"\be\s*d\s*c\b"), "eau de cologne"),
    (re.compile(r"\bparfume\b"), "parfum"),
)
CONCENTRATION_SIGNATURES = (
    ("extrait de parfum", "extrait"),
    ("eau de parfum", "edp"),
    ("eau de toilette", "edt"),
    ("eau de cologne", "edc"),
)
CONCENTRATION_TOKENS = {"edp", "edt", "edc", "extrait", "parfum", "cologne"}
EDITION_TOKENS = {
    *CONCENTRATION_TOKENS,
    "absolu", "absolute", "elixir", "extreme", "fraiche", "intense",
    "limited", "edition", "oil", "sport", "spray",
}
IDENTITY_CONNECTORS = {"a", "by", "de", "des", "du", "for", "le", "la", "of", "pour", "the"}

# These are spelling/market-name exceptions that cannot be inferred from initials
# alone. Initialisms such as YSL, JPG, LV, MFK, PDM, and CDG are generated from
# the catalog dynamically below.
EXPLICIT_BRAND_ALIASES = {
    "armani": "giorgio armani",
    "cdg": "comme des garcons",
    "ch": "carolina herrera",
    "d g": "dolce gabbana",
    "dg": "dolce gabbana",
    "jpg": "jean paul gaultier",
    "lv": "louis vuitton",
    "mfk": "maison francis kurkdjian",
    "pdm": "parfums de marly",
    "tf": "tom ford",
    "ysl": "yves saint laurent",
}


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def expand_concentration_aliases(value: Any) -> str:
    """Expand common concentration abbreviations before identity matching."""
    normalized = normalize_text(value)
    for pattern, replacement in CONCENTRATION_ALIAS_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    return " ".join(normalized.split())


def identity_signature(value: Any) -> tuple[list[str], set[str]]:
    """Return identity-bearing name tokens and explicit edition qualifiers."""
    normalized = expand_concentration_aliases(value)
    for phrase, marker in CONCENTRATION_SIGNATURES:
        normalized = re.sub(rf"\b{re.escape(phrase)}\b", marker, normalized)
    tokens = normalized.split()
    qualifiers = {token for token in tokens if token in EDITION_TOKENS}
    core = [
        token for token in tokens
        if token not in EDITION_TOKENS and token not in IDENTITY_CONNECTORS
    ]
    return core, qualifiers


def perfume_label(name: Any, brand: Any) -> str:
    name_text = str(name or "").strip()
    brand_text = str(brand or "").strip()
    if not brand_text:
        return name_text
    normalized_name = normalize_text(name_text)
    normalized_brand = normalize_text(brand_text)
    if normalized_name == normalized_brand or normalized_name.endswith(f" by {normalized_brand}"):
        return name_text
    return f"{name_text} by {brand_text}"


def csv_terms(value: Any) -> set[str]:
    return {normalize_text(part) for part in str(value or "").split(",") if normalize_text(part)}


def trait_variants(value: Any) -> set[str]:
    """Return directional taxonomy variants for user-facing trait constraints."""
    normalized = normalize_text(value)
    return {
        "musk": {"musk", "musky", "white musk"},
        "musky": {"musk", "musky", "white musk"},
        "leather": {"leather", "leathery", "suede"},
        "leathery": {"leather", "leathery", "suede"},
        "oud": {"oud", "aoud", "agarwood"},
        "aoud": {"oud", "aoud", "agarwood"},
        "agarwood": {"oud", "aoud", "agarwood"},
        "smoke": {"smoke", "smoky"},
        "smoky": {"smoke", "smoky"},
    }.get(normalized, {normalized} if normalized else set())


def split_name_brand_hint(hint: str) -> tuple[str, str | None]:
    normalized = normalize_text(hint)
    if " by " not in normalized:
        return normalized, None
    name, brand = normalized.rsplit(" by ", 1)
    return (name.strip(), brand.strip()) if name.strip() and brand.strip() else (normalized, None)


def confidence_adjusted_rating(rating: float, popularity: int, prior: float = 3.8) -> float:
    if rating <= 0:
        return 0.0
    weight = popularity / (popularity + 250) if popularity > 0 else 0.0
    return rating * weight + prior * (1.0 - weight)


def community_similarity_score(up_votes: int, down_votes: int) -> float:
    total = up_votes + down_votes
    if total <= 0:
        return 0.0
    approval = (up_votes + 1) / (total + 2)
    confidence = min(math.log1p(total) / math.log(501), 1.0)
    return approval * 0.72 + confidence * 0.28


def set_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    jaccard = intersection / len(left | right)
    overlap = intersection / min(len(left), len(right))
    return jaccard * 0.65 + overlap * 0.35


def trait_match_strength(wanted: str, trait: str) -> float:
    """Score canonical trait matches without rewarding accidental substrings."""
    wanted_norm = normalize_text(wanted)
    trait_norm = normalize_text(trait)
    if not wanted_norm or not trait_norm:
        return 0.0
    if wanted_norm == trait_norm:
        return 1.0
    if trait_norm in trait_variants(wanted_norm):
        return 1.0

    if wanted_norm in {"spicy", "floral"} and wanted_norm in trait_norm.split():
        return 0.75

    # A multi-word planner term can be a useful refinement of a stored trait,
    # but a single token such as "fresh" must not match "fresh spicy".
    if len(wanted_norm.split()) > 1:
        padded_wanted = f" {wanted_norm} "
        padded_trait = f" {trait_norm} "
        if padded_wanted in padded_trait or padded_trait in padded_wanted:
            return 0.65
    return 0.0


def metadata_has_term(metadata: dict[str, Any], term: str) -> bool:
    term_norm = normalize_text(term)
    if not term_norm:
        return False
    searchable = normalize_text(" ".join(
        str(metadata.get(key) or "")
        for key in ("name", "brand", "accords_csv", "notes_csv")
    ))
    if f" {term_norm} " in f" {searchable} ":
        return True
    traits = csv_terms(metadata.get("accords_csv")) | csv_terms(metadata.get("notes_csv"))
    return any(trait_match_strength(term_norm, trait) > 0.0 for trait in traits)


class CatalogResolver:
    def __init__(self, catalog_path: Path | str) -> None:
        self.path = Path(catalog_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Catalog does not exist: {self.path}")
        self.connection = sqlite3.connect(
            f"file:{self.path}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self._brand_norm_to_name, self._brand_alias_to_norm = self._build_brand_alias_index()

    def _build_brand_alias_index(self) -> tuple[dict[str, str], dict[str, str]]:
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT brand, SUM(popularity) AS total_popularity
                FROM perfumes
                WHERE brand != ''
                GROUP BY brand
                ORDER BY total_popularity DESC
                """
            ).fetchall()

        norm_to_name: dict[str, str] = {}
        alias_targets: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            brand = str(row["brand"] or "").strip()
            brand_norm = normalize_text(brand)
            if not brand_norm:
                continue
            norm_to_name.setdefault(brand_norm, brand)
            alias_targets[brand_norm].add(brand_norm)
            words = brand_norm.split()
            if 2 <= len(words) <= 5:
                initials = "".join(word[0] for word in words)
                spaced_initials = " ".join(word[0] for word in words)
                if len(initials) >= 2:
                    alias_targets[initials].add(brand_norm)
                    alias_targets[spaced_initials].add(brand_norm)

        # Ambiguous initials are intentionally discarded. Guessing a brand is
        # worse than falling through to the lexical resolver.
        aliases = {
            alias: next(iter(targets))
            for alias, targets in alias_targets.items()
            if len(targets) == 1
        }
        # A compact curated list overrides ambiguous generated initials only
        # where the fragrance community has a stable, well-known shorthand.
        for alias, target in EXPLICIT_BRAND_ALIASES.items():
            if target in norm_to_name:
                aliases[normalize_text(alias)] = target
        return norm_to_name, aliases

    def _canonical_brand_norm(self, hint: Any) -> str | None:
        normalized = normalize_text(hint)
        return self._brand_alias_to_norm.get(normalized)

    def _extract_edge_brand(self, normalized_hint: str) -> tuple[str, str | None]:
        tokens = normalized_hint.split()
        if not tokens:
            return normalized_hint, None
        max_width = min(5, len(tokens))
        matches: list[tuple[int, str, str]] = []
        for width in range(1, max_width + 1):
            prefix = " ".join(tokens[:width])
            suffix = " ".join(tokens[-width:])
            if prefix in self._brand_alias_to_norm:
                remainder = tokens[width:]
                if remainder[:1] == ["s"]:
                    remainder = remainder[1:]
                matches.append((width, " ".join(remainder), self._brand_alias_to_norm[prefix]))
            if suffix in self._brand_alias_to_norm:
                matches.append((width, " ".join(tokens[:-width]), self._brand_alias_to_norm[suffix]))
        if not matches:
            return normalized_hint, None
        _, name_hint, brand_norm = max(matches, key=lambda item: (item[0], len(item[1])))
        return name_hint.strip(), brand_norm

    def _prepare_hint(self, hint: str) -> tuple[str, str | None, str]:
        normalized = expand_concentration_aliases(hint)
        name_hint, raw_brand_hint = split_name_brand_hint(normalized)
        if raw_brand_hint:
            brand_hint = self._canonical_brand_norm(raw_brand_hint)
            if brand_hint:
                return name_hint, brand_hint, f"{name_hint} {brand_hint}".strip()
        edge_name, edge_brand = self._extract_edge_brand(normalized)
        if edge_brand:
            return edge_name, edge_brand, f"{edge_name} {edge_brand}".strip()
        return normalized, None, normalized

    def counts(self) -> dict[str, int]:
        with self.lock:
            perfume_count = int(self.connection.execute("SELECT COUNT(*) FROM perfumes").fetchone()[0])
            edge_count = int(self.connection.execute("SELECT COUNT(*) FROM similarity_edges").fetchone()[0])
        return {"perfumes": perfume_count, "similarity_edges": edge_count}

    @lru_cache(maxsize=4096)
    def canonical_brand(self, hint: str) -> str | None:
        normalized = self._canonical_brand_norm(hint) or normalize_text(hint)
        if not normalized:
            return None
        with self.lock:
            row = self.connection.execute(
                """
                SELECT brand, SUM(popularity) AS total_popularity
                FROM perfumes
                WHERE brand_norm = ?
                GROUP BY brand
                ORDER BY total_popularity DESC
                LIMIT 1
                """,
                (normalized,),
            ).fetchone()
        return str(row["brand"]) if row else None

    @lru_cache(maxsize=4096)
    def resolve(self, hint: str) -> dict[str, Any] | None:
        normalized = expand_concentration_aliases(hint)
        if not normalized:
            return None

        # A product name may legitimately start or end with its own house name
        # (for example "Burberry Men" or "Versace Man"). Resolve an exact
        # catalog identity before edge-brand extraction can strip that token.
        with self.lock:
            literal_exact = self.connection.execute(
                """
                SELECT * FROM perfumes
                WHERE name_norm = ? OR label_norm = ?
                ORDER BY popularity DESC
                LIMIT 20
                """,
                (normalized, normalized),
            ).fetchall()
        if literal_exact:
            return dict(literal_exact[0])

        name_hint, brand_hint, label_hint = self._prepare_hint(hint)

        with self.lock:
            if brand_hint:
                exact = self.connection.execute(
                    """
                    SELECT * FROM perfumes
                    WHERE (name_norm = ? AND brand_norm = ?) OR label_norm = ?
                    ORDER BY popularity DESC LIMIT 20
                    """,
                    (name_hint, brand_hint, label_hint),
                ).fetchall()
            else:
                exact = self.connection.execute(
                    """
                    SELECT * FROM perfumes
                    WHERE name_norm = ? OR label_norm = ?
                    ORDER BY popularity DESC LIMIT 20
                    """,
                    (name_hint, label_hint),
                ).fetchall()
        if exact:
            return dict(exact[0])

        family_found, family = self._dominant_family_member(name_hint, brand_hint)
        if family_found:
            return family

        tokens = [token for token in name_hint.split() if len(token) >= 2]
        if not tokens and brand_hint and name_hint:
            tokens = [name_hint]
        if not tokens:
            return None
        clauses: list[str] = []
        values: list[str] = []
        for token in tokens[:5]:
            clauses.append("(name_norm LIKE ? OR brand_norm LIKE ?)")
            values.extend([f"%{token}%", f"%{token}%"])
        brand_clause = " AND brand_norm = ?" if brand_hint else ""
        if brand_hint:
            values.append(brand_hint)
        with self.lock:
            rows = self.connection.execute(
                "SELECT * FROM perfumes WHERE "
                + "(" + " OR ".join(clauses) + ")"
                + brand_clause
                + " ORDER BY popularity DESC LIMIT 500",
                values,
            ).fetchall()
            if not rows and brand_hint:
                # When the brand is certain, a misspelled short product name
                # should still be compared against that house's catalog.
                rows = self.connection.execute(
                    """
                    SELECT * FROM perfumes
                    WHERE brand_norm = ?
                    ORDER BY popularity DESC, rating DESC
                    LIMIT 500
                    """,
                    (brand_hint,),
                ).fetchall()
        ranked = sorted(
            (dict(row) for row in rows),
            key=lambda row: self._resolution_score(name_hint, brand_hint, row),
            reverse=True,
        )
        if not ranked or self._resolution_score(name_hint, brand_hint, ranked[0]) < 0.60:
            return None
        return ranked[0]

    def _dominant_family_member(
        self,
        name_hint: str,
        brand_hint: str | None,
    ) -> tuple[bool, dict[str, Any] | None]:
        if len(name_hint.split()) < 2:
            return False, None
        values: list[Any] = [f"{name_hint} %"]
        brand_clause = ""
        if brand_hint:
            brand_clause = " AND brand_norm = ?"
            values.append(brand_hint)
        with self.lock:
            rows = self.connection.execute(
                "SELECT * FROM perfumes WHERE name_norm LIKE ?"
                + brand_clause
                + " ORDER BY popularity DESC, rating DESC LIMIT 25",
                values,
            ).fetchall()
        if not rows:
            return False, None
        ranked = [dict(row) for row in rows]
        leader = ranked[0]
        leader_popularity = int(leader.get("popularity") or 0)
        runner_up = int(ranked[1].get("popularity") or 0) if len(ranked) > 1 else 0
        clearly_dominant = (
            leader_popularity >= 500
            and (runner_up == 0 or leader_popularity >= runner_up * 2)
        )
        return True, leader if clearly_dominant else None

    @staticmethod
    def _resolution_score(
        name_hint: str,
        brand_hint: str | None,
        row: dict[str, Any],
    ) -> float:
        name = str(row.get("name_norm") or "")
        row_brand = str(row.get("brand_norm") or "")
        hint_core, hint_qualifiers = identity_signature(name_hint)
        name_core, name_qualifiers = identity_signature(name)
        hint_core_text = " ".join(hint_core)
        name_core_text = " ".join(name_core)
        hint_tokens = set(hint_core)
        name_tokens = set(name_core)
        overlap = len(hint_tokens & name_tokens) / max(len(hint_tokens | name_tokens), 1)
        coverage = len(hint_tokens & name_tokens) / max(len(hint_tokens), 1)
        core_sequence = difflib.SequenceMatcher(None, hint_core_text, name_core_text).ratio()
        full_sequence = difflib.SequenceMatcher(
            None,
            expand_concentration_aliases(name_hint),
            name,
        ).ratio()
        core_lexical = max(core_sequence, overlap, coverage)
        # Edition words such as "eau de toilette" are common across thousands
        # of unrelated perfumes. The product-name core must dominate them.
        lexical = core_lexical * 0.90 + full_sequence * 0.10
        core_mismatch_penalty = (
            0.38
            if hint_tokens and name_tokens and not (hint_tokens & name_tokens) and core_sequence < 0.72
            else 0.0
        )

        qualifier_score = 0.5
        qualifier_penalty = 0.0
        if hint_qualifiers:
            qualifier_score = len(hint_qualifiers & name_qualifiers) / len(hint_qualifiers)
            # Some catalog base releases omit EDT/EDC from their display name.
            # Missing edition text is a moderate penalty; an explicitly
            # conflicting concentration remains a hard penalty below.
            qualifier_penalty += 0.06 * (
                len(hint_qualifiers - name_qualifiers) / len(hint_qualifiers)
            )
            requested_concentration = hint_qualifiers & CONCENTRATION_TOKENS
            candidate_concentration = name_qualifiers & CONCENTRATION_TOKENS
            if requested_concentration and candidate_concentration and not (
                requested_concentration & candidate_concentration
            ):
                qualifier_penalty += 0.32

        brand_score = 1.0 if brand_hint and row_brand == brand_hint else (0.5 if not brand_hint else 0.0)
        popularity = int(row.get("popularity") or 0)
        popularity_score = min(math.log10(popularity + 1) / math.log10(50000), 1.0) if popularity else 0.0
        return (
            lexical * 0.72
            + qualifier_score * core_lexical * 0.18
            + brand_score * 0.07
            + popularity_score * 0.03
            - qualifier_penalty
            - core_mismatch_penalty
        )

    def direct_similarity(self, source_id: int, limit: int = 200) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT p.*, e.up_votes, e.down_votes
                FROM similarity_edges e
                JOIN perfumes p ON p.perfume_id = e.target_id
                WHERE e.source_id = ?
                ORDER BY (e.up_votes - e.down_votes) DESC, e.up_votes DESC
                LIMIT ?
                """,
                (source_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def popular_candidates(
        self,
        filters: dict[str, Any],
        required_terms: list[str],
        exclude_terms: list[str],
        *,
        canonical_brand: str | None = None,
        limit: int = 160,
    ) -> list[dict[str, Any]]:
        """Return popular eligible rows independently of the ANN neighborhood."""
        clauses: list[str] = []
        values: list[Any] = []

        gender = normalize_text(filters.get("gender"))
        if gender:
            allowed = [gender] if gender == "unisex" else [gender, "unisex"]
            clauses.append("gender IN (" + ", ".join("?" for _ in allowed) + ")")
            values.extend(allowed)
        if canonical_brand:
            clauses.append("brand = ?")
            values.append(canonical_brand)
        if filters.get("min_rating") is not None:
            clauses.append("rating >= ?")
            values.append(float(filters["min_rating"]))
        if filters.get("min_popularity") is not None:
            clauses.append("popularity >= ?")
            values.append(int(filters["min_popularity"]))

        sql = "SELECT * FROM perfumes"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        # Scan a generous popularity-first window, then apply taxonomy-aware
        # trait and season/time checks in Python.
        sql += " ORDER BY popularity DESC, rating DESC LIMIT 4000"
        with self.lock:
            rows = [dict(row) for row in self.connection.execute(sql, values).fetchall()]

        season = normalize_text(filters.get("season"))
        time_profile = normalize_text(filters.get("time"))
        output: list[dict[str, Any]] = []
        for row in rows:
            if season and season not in csv_terms(row.get("seasons_csv")):
                continue
            if time_profile and time_profile not in csv_terms(row.get("time_profile_csv")):
                continue
            traits = csv_terms(row.get("accords_csv")) | csv_terms(row.get("notes_csv"))
            if any(
                not any(trait_match_strength(term, trait) > 0.0 for trait in traits)
                for term in required_terms
            ):
                continue
            if any(metadata_has_term(row, term) for term in exclude_terms):
                continue
            output.append(row)
            if len(output) >= limit:
                break
        return output


class RetrievalEngine:
    def __init__(
        self,
        db_dir: Path | str,
        catalog_path: Path | str,
        *,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.started_at = time.time()
        self.db_dir = Path(db_dir)
        self.collection_name = collection_name
        self.model_name = model_name
        self.catalog = CatalogResolver(catalog_path)
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        self.collection = self.client.get_collection(collection_name)
        self.model = SentenceTransformer(model_name, device="cpu")
        self.encode_lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.model_name,
            "device": "cpu",
            "collection": self.collection_name,
            "collection_count": self.collection.count(),
            "catalog": self.catalog.counts(),
            "embedding_cache": self._encode.cache_info()._asdict(),
            "uptime_seconds": round(time.time() - self.started_at, 2),
        }

    @lru_cache(maxsize=512)
    def _encode(self, text: str) -> tuple[float, ...]:
        with self.encode_lock:
            vector = self.model.encode(text, normalize_embeddings=True)
        return tuple(float(value) for value in vector)

    def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        query = str(payload.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        top_k = min(max(int(payload.get("top_k") or 10), 1), 30)
        fetch_k = min(max(int(payload.get("fetch_k") or 300), top_k), 300)
        filters = dict(payload.get("filters") or {})
        exclude_terms = self._clean_terms(payload.get("exclude_terms") or [])
        wanted_terms = self._clean_terms(payload.get("wanted_terms") or [])
        required_terms = self._clean_terms(payload.get("required_terms") or [])
        exclude_ids = self._clean_ids(payload.get("exclude_ids") or [])
        discovery_mode = normalize_text(payload.get("discovery_mode") or "balanced")
        if discovery_mode not in VALID_DISCOVERY_MODES:
            raise ValueError(f"Unsupported discovery mode: {discovery_mode}")
        where, canonical_brand = self._build_where(filters)
        normalized_query = normalize_text(query)
        padded_query = f" {normalized_query} "

        encoded_query = BGE_QUERY_PREFIX + query
        before = self._encode.cache_info().hits
        embedding = [list(self._encode(encoded_query))]
        cache_hit = self._encode.cache_info().hits > before
        raw = self.collection.query(
            query_embeddings=embedding,
            n_results=fetch_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        raw_rows = [(*row, {"semantic_ann"}) for row in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        )]

        # ANN relevance and catalog popularity are complementary candidate
        # generators. This branch prevents famous, eligible perfumes from
        # disappearing merely because their names/cards are not close enough
        # to a short lifestyle query in the first ANN neighborhood.
        if discovery_mode != "niche":
            popular_rows = self.catalog.popular_candidates(
                filters,
                required_terms,
                exclude_terms,
                canonical_brand=canonical_brand,
                limit=200 if discovery_mode == "mainstream" else 140,
            )
            popular_ids = [str(row["perfume_id"]) for row in popular_rows]
            if popular_ids:
                popular = self.collection.get(
                    ids=popular_ids,
                    include=["documents", "metadatas", "embeddings"],
                )
                existing = {int(metadata.get("perfume_id") or 0): index for index, (_, metadata, _, _) in enumerate(raw_rows)}
                for document, metadata, vector in zip(
                    popular.get("documents") or [],
                    popular.get("metadatas") or [],
                    popular.get("embeddings") if popular.get("embeddings") is not None else [],
                ):
                    perfume_id = int(metadata.get("perfume_id") or 0)
                    if perfume_id in existing:
                        raw_rows[existing[perfume_id]][3].add("catalog_popular")
                        continue
                    semantic = sum(float(left) * float(right) for left, right in zip(embedding[0], vector))
                    raw_rows.append((document, metadata, 1.0 - semantic, {"catalog_popular"}))
        fetched_trait_sets = [
            csv_terms(metadata.get("accords_csv")) | csv_terms(metadata.get("notes_csv"))
            for _, metadata, _, _ in raw_rows
        ]
        supported_wanted_terms = [
            term
            for term in wanted_terms
            if any(
                any(trait_match_strength(term, trait) > 0.0 for trait in traits)
                for traits in fetched_trait_sets
            )
        ]
        supported_required_terms = [
            term
            for term in required_terms
            if any(
                any(trait_match_strength(term, trait) > 0.0 for trait in traits)
                for traits in fetched_trait_sets
            )
        ]

        candidates: list[dict[str, Any]] = []
        collision_brands: set[str] = set()
        score_weights = {
            # Balanced keeps semantic fit as the largest signal. Popularity
            # earns a candidate entry into the race, not an automatic win.
            "balanced": (0.62, 0.16, 0.08, 0.14),
            "mainstream": (0.48, 0.29, 0.09, 0.14),
            "niche": (0.70, 0.05, 0.11, 0.14),
        }[discovery_mode]
        semantic_weight, popularity_weight, rating_weight, wanted_weight = score_weights
        for document, metadata, distance, source_pools in raw_rows:
            meta = dict(metadata)
            perfume_id = int(meta.get("perfume_id") or 0)
            if perfume_id in exclude_ids:
                continue
            if self._has_excluded_term(meta, exclude_terms):
                continue
            semantic = max(0.0, 1.0 - float(distance))
            popularity = int(meta.get("popularity") or 0)
            rating = confidence_adjusted_rating(float(meta.get("rating") or 0.0), popularity)
            popularity_score = min(math.log10(popularity + 1) / math.log10(50000), 1.0) if popularity else 0.0
            rating_score = min(max((rating - 3.2) / 1.3, 0.0), 1.0) if rating else 0.0
            traits = csv_terms(meta.get("accords_csv")) | csv_terms(meta.get("notes_csv"))
            if required_terms and (
                len(supported_required_terms) != len(required_terms)
                or any(
                    not any(trait_match_strength(term, trait) > 0.0 for trait in traits)
                    for term in supported_required_terms
                )
            ):
                continue
            wanted_matches = {
                term: max((trait_match_strength(term, trait) for trait in traits), default=0.0)
                for term in supported_wanted_terms
            }
            wanted_matches = {
                term: strength for term, strength in wanted_matches.items() if strength > 0.0
            }
            wanted_hits = len(wanted_matches)
            wanted_score = (
                sum(wanted_matches.values()) / len(supported_wanted_terms)
                if supported_wanted_terms
                else 0.0
            )
            brand_norm = normalize_text(meta.get("brand"))
            brand_collision = bool(
                not canonical_brand
                and len(brand_norm) >= 4
                and f" {brand_norm} " in padded_query
            )
            if brand_collision:
                collision_brands.add(brand_norm)
            collision_penalty = 0.14 if brand_collision else 0.0
            final_score = (
                semantic * semantic_weight
                + popularity_score * popularity_weight
                + rating_score * rating_weight
                + wanted_score * wanted_weight
                - collision_penalty
            )
            candidates.append(
                {
                    "perfume_id": perfume_id,
                    "name": str(meta.get("name") or ""),
                    "brand": str(meta.get("brand") or ""),
                    "label": perfume_label(meta.get("name"), meta.get("brand")),
                    "document": document,
                    "metadata": meta,
                    "distance": round(float(distance), 6),
                    "semantic_score": round(semantic, 6),
                    "score": round(final_score, 6),
                    "reasons": {
                        "semantic": round(semantic, 4),
                        "popularity": round(popularity_score, 4),
                        "rating": round(rating_score, 4),
                        "wanted_term_hits": wanted_hits,
                        "wanted_term_score": round(wanted_score, 4),
                        "wanted_term_matches": {
                            term: round(strength, 2)
                            for term, strength in wanted_matches.items()
                        },
                        "accidental_brand_collision_penalty": collision_penalty,
                        "candidate_sources": sorted(source_pools),
                    },
                }
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        selected = self._diversify(
            candidates,
            top_k,
            brand_limit=top_k if canonical_brand else 2,
            per_brand_limits={brand: 1 for brand in collision_brands},
        )
        return {
            "route": "semantic",
            "query": query,
            "filters": {**filters, "canonical_brand": canonical_brand},
            "exclude_terms": exclude_terms,
            "wanted_terms": wanted_terms,
            "supported_wanted_terms": supported_wanted_terms,
            "ignored_wanted_terms": [term for term in wanted_terms if term not in supported_wanted_terms],
            "required_terms": required_terms,
            "supported_required_terms": supported_required_terms,
            "exclude_ids": sorted(exclude_ids),
            "discovery_mode": discovery_mode,
            "accidental_brand_collisions": sorted(collision_brands),
            "embedding_cache_hit": cache_hit,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "result_count": len(selected),
            "results": selected,
        }

    def resolve(self, payload: dict[str, Any]) -> dict[str, Any]:
        hint = str(payload.get("hint") or "").strip()
        if not hint:
            raise ValueError("hint is required")
        row = self.catalog.resolve(hint)
        return {"hint": hint, "resolved": self._public_catalog_row(row) if row else None}

    def similar(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        hint = str(payload.get("hint") or "").strip()
        if not hint:
            raise ValueError("hint is required")
        source = self.catalog.resolve(hint)
        if not source:
            raise LookupError(f"Could not resolve perfume: {hint}")
        top_k = min(max(int(payload.get("top_k") or 10), 1), 30)
        exclude_terms = self._clean_terms(payload.get("exclude_terms") or [])
        required_terms = self._clean_terms(payload.get("required_terms") or [])
        exclude_ids = self._clean_ids(payload.get("exclude_ids") or [])
        exclude_source_brand = bool(payload.get("exclude_source_brand", False))
        source_accords = csv_terms(source.get("accords_csv"))
        source_notes = csv_terms(source.get("notes_csv"))

        scored: list[dict[str, Any]] = []
        for row in self.catalog.direct_similarity(int(source["perfume_id"]), limit=max(200, top_k * 15)):
            if int(row.get("perfume_id") or 0) in exclude_ids:
                continue
            if exclude_source_brand and normalize_text(row.get("brand")) == normalize_text(source.get("brand")):
                continue
            if self._has_excluded_term(row, exclude_terms):
                continue
            row_traits = csv_terms(row.get("accords_csv")) | csv_terms(row.get("notes_csv"))
            if any(
                not any(trait_match_strength(term, trait) > 0.0 for trait in row_traits)
                for term in required_terms
            ):
                continue
            graph = community_similarity_score(int(row.get("up_votes") or 0), int(row.get("down_votes") or 0))
            accord_similarity = set_similarity(source_accords, csv_terms(row.get("accords_csv")))
            note_similarity = set_similarity(source_notes, csv_terms(row.get("notes_csv")))
            structure = accord_similarity * 0.68 + note_similarity * 0.32
            popularity = int(row.get("popularity") or 0)
            quality = min(math.log10(popularity + 1) / math.log10(50000), 1.0) if popularity else 0.0
            final_score = graph * 0.68 + structure * 0.24 + quality * 0.08
            public = self._public_catalog_row(row)
            assert public is not None
            public.update(
                {
                    "score": round(final_score, 6),
                    "community_similarity": round(graph, 6),
                    "accord_similarity": round(accord_similarity, 6),
                    "note_similarity": round(note_similarity, 6),
                    "up_votes": int(row.get("up_votes") or 0),
                    "down_votes": int(row.get("down_votes") or 0),
                }
            )
            scored.append(public)
        scored.sort(key=lambda item: item["score"], reverse=True)
        selected = self._diversify(scored, top_k, brand_limit=2)
        return {
            "route": "community_similarity",
            "hint": hint,
            "source": self._public_catalog_row(source),
            "exclude_terms": exclude_terms,
            "required_terms": required_terms,
            "exclude_ids": sorted(exclude_ids),
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "result_count": len(selected),
            "results": selected,
        }

    def _build_where(self, filters: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        clauses: list[dict[str, Any]] = []
        gender = normalize_text(filters.get("gender"))
        if gender:
            if gender not in VALID_GENDERS:
                raise ValueError(f"Unsupported gender filter: {gender}")
            allowed = [gender] if gender == "unisex" else [gender, "unisex"]
            clauses.append({"gender": {"$in": allowed}})
        season = normalize_text(filters.get("season"))
        if season:
            if season not in VALID_SEASONS:
                raise ValueError(f"Unsupported season filter: {season}")
            clauses.append({f"season_{season}": True})
        time_profile = normalize_text(filters.get("time"))
        if time_profile:
            if time_profile not in VALID_TIMES:
                raise ValueError(f"Unsupported time filter: {time_profile}")
            clauses.append({f"time_{time_profile}": True})
        if filters.get("min_rating") is not None:
            clauses.append({"rating": {"$gte": float(filters["min_rating"])}})
        if filters.get("min_popularity") is not None:
            clauses.append({"popularity": {"$gte": int(filters["min_popularity"])}})

        canonical_brand = None
        brand_hint = str(filters.get("brand") or "").strip()
        if brand_hint:
            canonical_brand = self.catalog.canonical_brand(brand_hint)
            if not canonical_brand:
                raise LookupError(f"Unknown brand: {brand_hint}")
            clauses.append({"brand": canonical_brand})

        if not clauses:
            return None, canonical_brand
        if len(clauses) == 1:
            return clauses[0], canonical_brand
        return {"$and": clauses}, canonical_brand

    @staticmethod
    def _clean_terms(values: Any) -> list[str]:
        if not isinstance(values, list):
            raise ValueError("wanted_terms, required_terms, and exclude_terms must be JSON arrays")
        return list(dict.fromkeys(term for value in values if (term := normalize_text(value))))[:30]

    @staticmethod
    def _clean_ids(values: Any) -> set[int]:
        if not isinstance(values, list):
            raise ValueError("exclude_ids must be a JSON array")
        output: set[int] = set()
        for value in values[:200]:
            try:
                perfume_id = int(value)
            except (TypeError, ValueError):
                continue
            if perfume_id > 0:
                output.add(perfume_id)
        return output

    @staticmethod
    def _has_excluded_term(metadata: dict[str, Any], excluded: list[str]) -> bool:
        return any(metadata_has_term(metadata, term) for term in excluded)

    @staticmethod
    def _diversify(
        items: list[dict[str, Any]],
        top_k: int,
        *,
        brand_limit: int,
        per_brand_limits: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        seen_labels: set[str] = set()
        brand_counts: dict[str, int] = {}
        normalized_limits = {
            normalize_text(brand): int(limit)
            for brand, limit in (per_brand_limits or {}).items()
        }
        for item in items:
            perfume_id = int(item.get("perfume_id") or 0)
            label = normalize_text(perfume_label(item.get("name"), item.get("brand")))
            brand = normalize_text(item.get("brand"))
            current_limit = normalized_limits.get(brand, brand_limit)
            if (
                perfume_id in seen_ids
                or label in seen_labels
                or brand_counts.get(brand, 0) >= current_limit
            ):
                continue
            output.append(item)
            seen_ids.add(perfume_id)
            seen_labels.add(label)
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
            if len(output) >= top_k:
                break
        return output

    @staticmethod
    def _public_catalog_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if not row:
            return None
        return {
            "perfume_id": int(row["perfume_id"]),
            "name": str(row.get("name") or ""),
            "brand": str(row.get("brand") or ""),
            "label": perfume_label(row.get("name"), row.get("brand")),
            "gender": str(row.get("gender") or ""),
            "year": row.get("year"),
            "rating": row.get("rating"),
            "popularity": int(row.get("popularity") or 0),
            "longevity": row.get("longevity"),
            "sillage": row.get("sillage"),
            "value_score": row.get("value_score"),
            "accords_csv": str(row.get("accords_csv") or ""),
            "notes_csv": str(row.get("notes_csv") or ""),
            "seasons_csv": str(row.get("seasons_csv") or ""),
            "time_profile_csv": str(row.get("time_profile_csv") or ""),
        }


class RetrievalRequestHandler(BaseHTTPRequestHandler):
    engine: RetrievalEngine
    server_version = "ScentAIRetrieval/1.0"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, self.engine.health())
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
            if self.path == "/search":
                output = self.engine.search(payload)
            elif self.path == "/resolve":
                output = self.engine.resolve(payload)
            elif self.path == "/similar":
                output = self.engine.similar(payload)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._send_json(HTTPStatus.OK, output)
        except LookupError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except (TypeError, ValueError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": repr(exc)})

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 1_000_000:
            raise ValueError("Request body must contain at most 1 MB of JSON")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON request must be an object")
        return payload

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[retrieval-http] {self.address_string()} {format % args}", flush=True)


def main() -> None:
    db_dir = Path(os.environ["SCENTAI_CHROMA_DIR"])
    catalog_path = Path(os.environ["SCENTAI_CATALOG_PATH"])
    host = os.environ.get("SCENTAI_RETRIEVAL_HOST", "127.0.0.1")
    port = int(os.environ.get("SCENTAI_RETRIEVAL_PORT", "8020"))
    engine = RetrievalEngine(db_dir, catalog_path)
    RetrievalRequestHandler.engine = engine
    server = ThreadingHTTPServer((host, port), RetrievalRequestHandler)
    print(json.dumps({"event": "ready", "host": host, "port": port, **engine.health()}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
