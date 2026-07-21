from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scentai.retrieval import RetrievalEngine, metadata_has_term


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the frozen BGE-M3 retrieval snapshot")
    parser.add_argument("--chroma", type=Path, default=ROOT / "chroma_db_bge_m3")
    parser.add_argument("--catalog", type=Path, default=ROOT / "scentai_catalog.sqlite3")
    args = parser.parse_args()

    started = time.perf_counter()
    engine = RetrievalEngine(args.chroma, args.catalog)
    health = engine.health()
    resolved = engine.resolve({"hint": "YSL Y EDP"})["resolved"]
    search = engine.search({
        "query": "clean professional office fragrance",
        "top_k": 5,
        "filters": {},
        "wanted_terms": [],
        "required_terms": [],
        "exclude_terms": ["vanilla"],
        "exclude_ids": [],
        "discovery_mode": "balanced",
    })
    assert health["status"] == "ok"
    assert health["collection_count"] == 131_930
    assert health["catalog"] == {"perfumes": 131_930, "similarity_edges": 692_729}
    assert resolved and resolved["name"] == "Y Eau de Parfum"
    assert len(search["results"]) == 5
    assert all(
        not metadata_has_term(item.get("metadata") or item, "vanilla")
        for item in search["results"]
    )
    print(json.dumps({
        "status": "ok",
        "health": health,
        "resolved": resolved["label"],
        "results": [item["label"] for item in search["results"]],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

