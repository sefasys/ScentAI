from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.core.config import DEFAULT_CLEAN_FILE, PROJECT_ROOT


DEFAULT_DB_DIR = PROJECT_ROOT / "chroma_db_bge_m3"
DEFAULT_COLLECTION = "scentai_perfumes"
DEFAULT_MODEL = "BAAI/bge-m3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a persistent ChromaDB perfume vector database.")
    parser.add_argument("--clean-file", type=Path, default=DEFAULT_CLEAN_FILE)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--add-batch-size", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=0, help="Build only the first N records for smoke testing.")
    parser.add_argument("--rebuild", action="store_true", help="Delete the existing ChromaDB directory before building.")
    parser.add_argument("--resume", action="store_true", help="Resume a previous sequential build from the current collection count.")
    parser.add_argument("--device", default=None, help="Optional sentence-transformers device, for example cuda or cpu.")
    parser.add_argument("--local-files-only", action="store_true", help="Load the embedding model from the local HF cache only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rebuild and args.db_dir.exists():
        shutil.rmtree(args.db_dir)

    perfumes = load_perfumes(args.clean_file, args.limit)
    if not perfumes:
        raise SystemExit(f"No perfumes loaded from {args.clean_file}")

    import chromadb
    from sentence_transformers import SentenceTransformer

    print(f"Loaded perfumes : {len(perfumes)}")
    print(f"Embedding model : {args.model}")
    print(f"ChromaDB dir    : {args.db_dir}")
    print(f"Collection      : {args.collection}")

    if args.rebuild and args.resume:
        raise SystemExit("--rebuild and --resume cannot be used together.")

    model = SentenceTransformer(args.model, device=args.device, local_files_only=args.local_files_only)
    client = chromadb.PersistentClient(path=str(args.db_dir))

    start_index = 0
    if args.rebuild:
        collection = client.create_collection(name=args.collection, metadata={"hnsw:space": "cosine"})
    else:
        collection = client.get_or_create_collection(name=args.collection, metadata={"hnsw:space": "cosine"})
        existing_count = collection.count()
        if args.resume:
            start_index = existing_count
            print(f"Resuming from existing collection count: {existing_count}")
        elif existing_count:
            raise SystemExit(
                f"Collection {args.collection!r} already contains {existing_count} records. "
                "Use --rebuild to recreate it or --resume to continue a sequential build."
            )

    ids = [str(p["id"]) for p in perfumes]
    documents = [build_embedding_document(p) for p in perfumes]
    metadatas = [build_chroma_metadata(p) for p in perfumes]

    inserted = 0
    if start_index > len(perfumes):
        raise SystemExit(f"Existing collection count {start_index} is larger than source size {len(perfumes)}.")

    for start in range(start_index, len(perfumes), args.add_batch_size):
        end = min(start + args.add_batch_size, len(perfumes))
        batch_docs = documents[start:end]
        embeddings = model.encode(
            batch_docs,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).tolist()
        collection.add(
            ids=ids[start:end],
            documents=batch_docs,
            metadatas=metadatas[start:end],
            embeddings=embeddings,
        )
        inserted = end
        print(f"Inserted {inserted}/{len(perfumes)}")

    print("ChromaDB build complete")
    print(f"Collection count: {collection.count()}")


def load_perfumes(path: Path, limit: int) -> list[dict[str, Any]]:
    perfumes: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            perfumes.append(json.loads(line))
            if limit and len(perfumes) >= limit:
                break
    return perfumes


def build_embedding_document(perfume: dict[str, Any]) -> str:
    meta = perfume.get("metadata") or {}
    parts = [
        perfume.get("card_text") or "",
        f"Name: {perfume.get('name', '')}",
        f"Brand: {perfume.get('brand', '')}",
        f"Gender: {meta.get('gender', '')}",
        f"Accords: {', '.join(meta.get('accords_list') or [])}",
        f"Notes: {', '.join(meta.get('notes_list') or [])}",
        f"Best seasons: {', '.join(meta.get('best_seasons') or [])}",
        f"Time profile: {', '.join(meta.get('time_profile') or [])}",
    ]
    return "\n".join(part for part in parts if part.strip())


def build_chroma_metadata(perfume: dict[str, Any]) -> dict[str, str | int | float | bool]:
    meta = perfume.get("metadata") or {}
    seasons = {str(item).lower() for item in (meta.get("best_seasons") or [])}
    times = {str(item).lower() for item in (meta.get("time_profile") or [])}
    accords = [str(item).lower() for item in (meta.get("accords_list") or [])]
    notes = [str(item).lower() for item in (meta.get("notes_list") or [])]

    chroma_meta: dict[str, str | int | float | bool] = {
        "perfume_id": int(perfume["id"]),
        "slug": str(perfume.get("slug") or ""),
        "name": str(perfume.get("name") or ""),
        "brand": str(perfume.get("brand") or ""),
        "gender": str(meta.get("gender") or ""),
        "rating": float(meta.get("rating") or 0.0),
        "popularity": int(meta.get("popularity") or 0),
        "top_accord": accords[0] if accords else "",
        "top_note": notes[0] if notes else "",
        "accords_csv": ", ".join(accords),
        "notes_csv": ", ".join(notes[:40]),
        "seasons_csv": ", ".join(sorted(seasons)),
        "time_profile_csv": ", ".join(sorted(times)),
        "season_spring": "spring" in seasons,
        "season_summer": "summer" in seasons,
        "season_autumn": "autumn" in seasons,
        "season_winter": "winter" in seasons,
        "time_day": "day" in times,
        "time_night": "night" in times,
    }

    year = meta.get("year")
    if year is not None:
        chroma_meta["year"] = int(year)

    return chroma_meta


if __name__ == "__main__":
    main()
