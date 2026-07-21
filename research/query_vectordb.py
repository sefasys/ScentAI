from __future__ import annotations

import argparse
from pathlib import Path


if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.build_vectordb import DEFAULT_COLLECTION, DEFAULT_DB_DIR, DEFAULT_MODEL


BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the ScentAI ChromaDB perfume collection.")
    parser.add_argument("query", nargs="*", help="Query text. If omitted, runs built-in sanity queries.")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gender", choices=["male", "female", "unisex"])
    parser.add_argument("--season", choices=["spring", "summer", "autumn", "winter"])
    parser.add_argument("--min-rating", type=float)
    parser.add_argument("--min-popularity", type=int)
    parser.add_argument("--no-query-prefix", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", help="Load the embedding model from the local HF cache only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=str(args.db_dir))
    collection = client.get_collection(args.collection)
    model = SentenceTransformer(args.model, local_files_only=args.local_files_only)

    queries = [" ".join(args.query).strip()] if args.query else sanity_queries()
    where = build_where(args)

    print(f"Collection count: {collection.count()}")
    if where:
        print(f"Filter          : {where}")

    for query in queries:
        encoded_query = query if args.no_query_prefix else BGE_QUERY_PREFIX + query
        embedding = model.encode([encoded_query], normalize_embeddings=True).tolist()
        results = collection.query(
            query_embeddings=embedding,
            n_results=args.top_k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        print("\n" + "=" * 80)
        print(f"Query: {query}")
        for rank, (metadata, distance) in enumerate(zip(results["metadatas"][0], results["distances"][0]), 1):
            name = metadata.get("name", "")
            brand = metadata.get("brand", "")
            gender = metadata.get("gender", "")
            rating = metadata.get("rating", 0)
            popularity = metadata.get("popularity", 0)
            accords = metadata.get("accords_csv", "")
            seasons = metadata.get("seasons_csv", "")
            print(f"{rank}. {name} by {brand} [{gender}] distance={distance:.4f}")
            print(f"   rating={rating:.2f} votes={popularity} seasons={seasons}")
            print(f"   accords={accords}")


def sanity_queries() -> list[str]:
    return [
        "fresh citrus summer cologne",
        "warm spicy winter perfume",
        "pineapple woody fragrance",
        "sweet vanilla perfume for date night",
        "old school masculine fougere",
    ]


def build_where(args: argparse.Namespace) -> dict:
    filters = []
    if args.gender:
        filters.append({"gender": {"$in": [args.gender, "unisex"] if args.gender != "unisex" else ["unisex"]}})
    if args.season:
        filters.append({f"season_{args.season}": True})
    if args.min_rating is not None:
        filters.append({"rating": {"$gte": args.min_rating}})
    if args.min_popularity is not None:
        filters.append({"popularity": {"$gte": args.min_popularity}})

    if not filters:
        return {}
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


if __name__ == "__main__":
    main()
