from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.runtime.rag import DEFAULT_DB_DIR, DEFAULT_MODEL, ScentRetriever


DEFAULT_QUERIES = [
    "fresh citrus summer cologne",
    "warm spicy winter perfume",
    "pineapple woody fragrance",
    "sweet vanilla perfume for date night",
    "old school masculine fougere",
    "yaz için ferah temiz turunçgil erkek parfümü",
    "ofis için temiz sabunsu rahatsız etmeyen koku",
    "date night için tatlı vanilyalı ama ağır olmayan parfüm",
    "I want something like Aventus but less smoky",
    "Recommend a clean office scent without vanilla",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ScentAI retrieval sanity checks.")
    parser.add_argument("queries", nargs="*")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--fetch-k", type=int, default=80)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reranker", default="", help="Optional CrossEncoder reranker model, e.g. BAAI/bge-reranker-base.")
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--no-local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = args.queries or DEFAULT_QUERIES
    retriever = ScentRetriever(
        db_dir=args.db_dir,
        model_name=args.model,
        device=args.device,
        local_files_only=not args.no_local_files_only,
        reranker_model_name=args.reranker or None,
        reranker_device=args.reranker_device,
    )
    print(f"Collection count: {retriever.collection.count()}")
    print(f"DB dir          : {args.db_dir}")
    print(f"Model           : {args.model}")

    for query in queries:
        candidates, analysis = retriever.retrieve(query, top_k=args.top_k, fetch_k=args.fetch_k)
        print("\n" + "=" * 100)
        print(f"Query: {query}")
        print(
            "Analysis:",
            {
                "gender": analysis.gender,
                "season": analysis.season,
                "time": analysis.time_profile,
                "wanted_accords": analysis.wanted_accords,
                "wanted_notes": analysis.wanted_notes,
                "negative_accords": analysis.negative_accords,
                "negative_notes": analysis.negative_notes,
                "min_popularity": analysis.min_popularity,
            },
        )
        for rank, candidate in enumerate(candidates, 1):
            meta = candidate.metadata
            print(
                f"{rank:02d}. {candidate.label} [{meta.get('gender')}] "
                f"score={candidate.final_score:.3f} dist={candidate.distance:.3f} "
                f"rating={float(meta.get('rating') or 0):.2f} votes={meta.get('popularity')}"
            )
            print(f"    seasons={meta.get('seasons_csv')} accords={meta.get('accords_csv')}")
            print(f"    reasons={', '.join(candidate.reasons)}")


if __name__ == "__main__":
    main()
