from __future__ import annotations

import argparse
import json
import re
import sqlite3
import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "perfumes_clean.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "scentai_catalog.sqlite3"


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.lower())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def metric(card: str, label: str, denominator: int) -> float | None:
    match = re.search(rf"{re.escape(label)}:\s*([\d.]+)/{denominator}", card, re.I)
    return float(match.group(1)) if match else None


def build_catalog(source: Path, output: Path) -> None:
    if output.exists():
        output.unlink()
    connection = sqlite3.connect(output)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE perfumes (
            perfume_id INTEGER PRIMARY KEY,
            slug TEXT NOT NULL,
            name TEXT NOT NULL,
            brand TEXT NOT NULL,
            name_norm TEXT NOT NULL,
            brand_norm TEXT NOT NULL,
            label_norm TEXT NOT NULL,
            gender TEXT,
            year INTEGER,
            rating REAL,
            popularity INTEGER,
            longevity REAL,
            sillage REAL,
            value_score REAL,
            accords_csv TEXT,
            notes_csv TEXT,
            seasons_csv TEXT,
            time_profile_csv TEXT
        );
        CREATE TABLE similarity_edges (
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            up_votes INTEGER NOT NULL,
            down_votes INTEGER NOT NULL,
            PRIMARY KEY (source_id, target_id)
        );
        """
    )

    perfume_batch = []
    edge_batch = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            card = row.get("card_text") or ""
            name = str(row.get("name") or "")
            brand = str(row.get("brand") or "")
            perfume_id = int(row["id"])
            perfume_batch.append(
                (
                    perfume_id,
                    str(row.get("slug") or ""),
                    name,
                    brand,
                    normalize(name),
                    normalize(brand),
                    normalize(f"{name} {brand}"),
                    str(metadata.get("gender") or ""),
                    metadata.get("year"),
                    metadata.get("rating"),
                    int(metadata.get("popularity") or 0),
                    metric(card, "Longevity", 5),
                    metric(card, "Sillage", 4),
                    metric(card, "Value", 5),
                    ", ".join(metadata.get("accords_list") or []),
                    ", ".join(metadata.get("notes_list") or []),
                    ", ".join(metadata.get("best_seasons") or []),
                    ", ".join(metadata.get("time_profile") or []),
                )
            )
            for edge in (row.get("similar") or {}).get("reminds_me_of", []):
                target_id = edge.get("id")
                if target_id is None:
                    continue
                edge_batch.append(
                    (perfume_id, int(target_id), int(edge.get("up_votes") or 0), int(edge.get("down_votes") or 0))
                )

            if len(perfume_batch) >= 2000:
                insert_batches(connection, perfume_batch, edge_batch)
                perfume_batch.clear()
                edge_batch.clear()
                if line_number % 20000 == 0:
                    print(f"Indexed {line_number:,} perfumes")

    insert_batches(connection, perfume_batch, edge_batch)
    connection.executescript(
        """
        CREATE INDEX idx_perfumes_name_norm ON perfumes(name_norm);
        CREATE INDEX idx_perfumes_brand_norm ON perfumes(brand_norm);
        CREATE INDEX idx_perfumes_label_norm ON perfumes(label_norm);
        CREATE INDEX idx_perfumes_year ON perfumes(year);
        CREATE INDEX idx_perfumes_rating ON perfumes(rating DESC);
        CREATE INDEX idx_perfumes_popularity ON perfumes(popularity DESC);
        CREATE INDEX idx_perfumes_longevity ON perfumes(longevity DESC);
        CREATE INDEX idx_perfumes_sillage ON perfumes(sillage DESC);
        CREATE INDEX idx_perfumes_value ON perfumes(value_score DESC);
        CREATE INDEX idx_similarity_source ON similarity_edges(source_id);
        PRAGMA optimize;
        """
    )
    perfume_count = connection.execute("SELECT COUNT(*) FROM perfumes").fetchone()[0]
    edge_count = connection.execute("SELECT COUNT(*) FROM similarity_edges").fetchone()[0]
    connection.commit()
    connection.close()
    print(f"Catalog complete: {perfume_count:,} perfumes, {edge_count:,} similarity edges -> {output}")


def insert_batches(connection: sqlite3.Connection, perfumes: list[tuple], edges: list[tuple]) -> None:
    connection.executemany(
        "INSERT INTO perfumes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        perfumes,
    )
    connection.executemany(
        "INSERT OR REPLACE INTO similarity_edges VALUES (?, ?, ?, ?)",
        edges,
    )
    connection.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the compact deterministic ScentAI runtime catalog.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_catalog(args.source, args.output)


if __name__ == "__main__":
    main()
