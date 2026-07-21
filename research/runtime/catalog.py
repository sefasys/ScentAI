from __future__ import annotations

import difflib
import math
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from research.runtime.query_analyzer import entity_matches, normalize_match_text


class RuntimeCatalog:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Runtime catalog not found: {self.path}")
        self.connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row

    def get(self, perfume_id: int) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM perfumes WHERE perfume_id = ?", (perfume_id,)).fetchone()
        return dict(row) if row else None

    @lru_cache(maxsize=4096)
    def resolve(self, hint: str) -> dict[str, Any] | None:
        normalized = normalize_match_text(hint)
        if not normalized:
            return None
        name_hint, brand_hint = split_name_brand_hint(normalized)
        label_hint = f"{name_hint} {brand_hint}".strip() if brand_hint else normalized

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

        family_found, family = self._resolve_dominant_family_member(name_hint, brand_hint)
        if family_found:
            return family

        tokens = [token for token in label_hint.split() if len(token) >= 3]
        if not tokens:
            return None
        clauses = []
        values = []
        for token in tokens[:5]:
            clauses.append("(name_norm LIKE ? OR brand_norm LIKE ?)")
            values.extend([f"%{token}%", f"%{token}%"])
        rows = self.connection.execute(
            "SELECT * FROM perfumes WHERE " + " OR ".join(clauses) + " ORDER BY popularity DESC LIMIT 500",
            values,
        ).fetchall()
        ranked = sorted((dict(row) for row in rows), key=lambda row: resolution_score(label_hint, row), reverse=True)
        if not ranked or resolution_score(label_hint, ranked[0]) < 0.62:
            return None
        return ranked[0]

    def _resolve_dominant_family_member(
        self,
        name_hint: str,
        brand_hint: str | None,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Resolve a shortened product-family name only when usage clearly identifies a canonical member."""
        if len(name_hint.split()) < 2:
            return False, None
        values: list[Any] = [f"{name_hint} %"]
        brand_clause = ""
        if brand_hint:
            brand_clause = " AND brand_norm = ?"
            values.append(brand_hint)
        rows = self.connection.execute(
            """
            SELECT * FROM perfumes
            WHERE name_norm LIKE ?
            """ + brand_clause + " ORDER BY popularity DESC, rating DESC LIMIT 25",
            values,
        ).fetchall()
        if not rows:
            return False, None

        ranked = [dict(row) for row in rows]
        leader = ranked[0]
        leader_popularity = int(leader.get("popularity") or 0)
        runner_up_popularity = int(ranked[1].get("popularity") or 0) if len(ranked) > 1 else 0
        clearly_dominant = (
            leader_popularity >= 500
            and (runner_up_popularity == 0 or leader_popularity >= runner_up_popularity * 2)
        )
        return True, leader if clearly_dominant else None

    def enrich_metadata(self, perfume_id: int, metadata: dict[str, Any]) -> dict[str, Any]:
        row = self.get(perfume_id)
        if not row:
            return metadata
        enriched = dict(metadata)
        for key in ("year", "longevity", "sillage", "value_score"):
            if row.get(key) is not None:
                enriched[key] = row[key]
        return enriched

    @lru_cache(maxsize=4096)
    def direct_similarity(self, source_id: int, limit: int = 120) -> list[dict[str, Any]]:
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
        output = []
        for row in rows:
            item = dict(row)
            up = int(item["up_votes"])
            down = int(item["down_votes"])
            item["community_similarity"] = community_similarity_score(up, down)
            output.append(item)
        return output

    def compare(self, hints: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
        resolved = []
        missing = []
        seen = set()
        for hint in hints:
            row = self.resolve(hint)
            if not row:
                missing.append(hint)
                continue
            if row["perfume_id"] in seen:
                continue
            seen.add(row["perfume_id"])
            resolved.append(row)
        return resolved, missing

    @lru_cache(maxsize=2048)
    def extract_mentions(self, text: str, limit: int = 8) -> tuple[str, ...]:
        """Find exact catalog perfume names in free-form text without requiring intent keywords."""
        tokens = normalize_match_text(text).split()
        if not tokens:
            return ()
        ngrams = []
        max_width = min(8, len(tokens))
        for width in range(max_width, 0, -1):
            for start in range(len(tokens) - width + 1):
                value = " ".join(tokens[start:start + width])
                if width == 1 and len(value) < 5:
                    continue
                ngrams.append(value)
        ngrams = list(dict.fromkeys(ngrams))[:450]
        if not ngrams:
            return ()
        placeholders = ", ".join("?" for _ in ngrams)
        rows = self.connection.execute(
            f"""
            SELECT * FROM perfumes
            WHERE name_norm IN ({placeholders}) OR label_norm IN ({placeholders})
            ORDER BY LENGTH(name_norm) DESC, popularity DESC
            LIMIT 100
            """,
            [*ngrams, *ngrams],
        ).fetchall()
        output = []
        seen_names = set()
        selected_norms: list[str] = []
        query_norm = " ".join(tokens)
        for raw_row in rows:
            row = dict(raw_row)
            name_norm = str(row.get("name_norm") or "")
            if not name_norm or name_norm in seen_names:
                continue
            seen_names.add(name_norm)
            if any(name_norm in selected for selected in selected_norms) and query_norm.count(name_norm) == 1:
                continue
            output.append(str(row["name"]))
            selected_norms.append(name_norm)
            if len(output) >= limit:
                break
        return tuple(output)


def resolution_score(hint_norm: str, row: dict[str, Any]) -> float:
    name = str(row.get("name_norm") or "")
    label = str(row.get("label_norm") or "")
    hint_tokens = set(hint_norm.split())
    label_tokens = set(label.split())
    if hint_norm == name or hint_norm == label:
        lexical = 1.0
    else:
        overlap = len(hint_tokens & label_tokens) / max(len(hint_tokens | label_tokens), 1)
        lexical = max(
            difflib.SequenceMatcher(None, hint_norm, name).ratio(),
            difflib.SequenceMatcher(None, hint_norm, label).ratio(),
            overlap,
        )
    popularity = int(row.get("popularity") or 0)
    return lexical + min(math.log10(popularity + 1) / 100.0, 0.05)


def split_name_brand_hint(normalized_hint: str) -> tuple[str, str | None]:
    if " by " not in normalized_hint:
        return normalized_hint, None
    name, brand = normalized_hint.rsplit(" by ", 1)
    name = name.strip()
    brand = brand.strip()
    return (name, brand) if name and brand else (normalized_hint, None)


def community_similarity_score(up_votes: int, down_votes: int) -> float:
    total = up_votes + down_votes
    if total <= 0:
        return 0.0
    approval = (up_votes + 1) / (total + 2)
    confidence = min(math.log1p(total) / math.log(501), 1.0)
    return approval * 0.72 + confidence * 0.28


def row_matches_entity(row: dict[str, Any], entity: str) -> bool:
    return entity_matches(entity, f"{row.get('name', '')} {row.get('brand', '')}")
