from __future__ import annotations

import sqlite3
import tempfile
import threading
import unittest
import json
import io
from pathlib import Path

from scentai.retrieval import (
    CatalogResolver,
    RetrievalRequestHandler,
    RetrievalEngine,
    community_similarity_score,
    normalize_text,
    perfume_label,
    trait_match_strength,
)


class FakeEmbeddingModel:
    def encode(self, text, *, normalize_embeddings):
        assert normalize_embeddings is True
        return [0.1, 0.2, 0.3]


class FakeCollection:
    def count(self):
        return 3

    def query(self, **kwargs):
        self.last_query = kwargs
        return {
            "documents": [["clean card", "vanilla card", "fresh card"]],
            "metadatas": [[
                {"perfume_id": 1, "name": "Office One", "brand": "Clean", "accords_csv": "clean, soapy", "notes_csv": "iris", "rating": 4.2, "popularity": 1000},
                {"perfume_id": 2, "name": "Sweet One", "brand": "B", "accords_csv": "vanilla, sweet", "notes_csv": "vanilla", "rating": 4.4, "popularity": 5000},
                {"perfume_id": 3, "name": "Office Two", "brand": "C", "accords_csv": "fresh, aromatic", "notes_csv": "musk", "rating": 4.0, "popularity": 800},
            ]],
            "distances": [[0.1, 0.05, 0.2]],
        }


class FakeCatalog:
    def canonical_brand(self, hint):
        return "A" if normalize_text(hint) == "a" else None

    def counts(self):
        return {"perfumes": 3, "similarity_edges": 1}

    def popular_candidates(self, *args, **kwargs):
        return []


class CleanRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.temp_dir.name) / "catalog.sqlite3"
        connection = sqlite3.connect(self.catalog_path)
        connection.executescript(
            """
            CREATE TABLE perfumes (
                perfume_id INTEGER PRIMARY KEY,
                slug TEXT,
                name TEXT,
                brand TEXT,
                name_norm TEXT,
                brand_norm TEXT,
                label_norm TEXT,
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
            CREATE INDEX idx_perfumes_name_norm ON perfumes(name_norm);
            CREATE INDEX idx_perfumes_brand_norm ON perfumes(brand_norm);
            CREATE INDEX idx_perfumes_label_norm ON perfumes(label_norm);
            CREATE TABLE similarity_edges (
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                up_votes INTEGER NOT NULL,
                down_votes INTEGER NOT NULL,
                PRIMARY KEY (source_id, target_id)
            );
            CREATE INDEX idx_similarity_source ON similarity_edges(source_id);
            """
        )
        rows = [
            (1, "aventus", "Aventus", "Creed", "aventus", "creed", "aventus by creed", "male", 2010, 4.4, 26000, 4.0, 3.0, 3.0, "fruity, smoky", "pineapple", "spring, summer", "day, night"),
            (2, "cdnim", "Club de Nuit Intense Man", "Armaf", "club de nuit intense man", "armaf", "club de nuit intense man by armaf", "male", 2015, 4.3, 27000, 4.0, 3.0, 4.0, "fruity, smoky", "lemon", "spring, summer", "day, night"),
            (3, "cdn-bling", "Club de Nuit Bling", "Armaf", "club de nuit bling", "armaf", "club de nuit bling by armaf", "unisex", 2025, 3.7, 300, 3.0, 2.0, 3.0, "vanilla, citrus", "vanilla", "spring", "day"),
        ]
        connection.executemany(
            "INSERT INTO perfumes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        connection.execute("INSERT INTO similarity_edges VALUES (1, 2, 5900, 886)")
        connection.commit()
        connection.close()
        self.catalog = CatalogResolver(self.catalog_path)

    def tearDown(self):
        self.catalog.connection.close()
        self.temp_dir.cleanup()

    def test_normalization_matches_catalog_contract(self):
        self.assertEqual(normalize_text("Estée Lauder / L'Homme"), "estee lauder l homme")

    def test_label_does_not_repeat_embedded_brand(self):
        self.assertEqual(
            perfume_label("Laundry by Shelli Segal", "Laundry by Shelli Segal"),
            "Laundry by Shelli Segal",
        )

    def test_short_family_name_prefers_dominant_member(self):
        resolved = self.catalog.resolve("Club de Nuit by Armaf")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["name"], "Club de Nuit Intense Man")

    def test_exact_name_and_brand_resolution(self):
        resolved = self.catalog.resolve("Aventus by Creed")
        self.assertEqual(resolved["perfume_id"], 1)
        self.assertEqual(self.catalog.canonical_brand("ARMAF"), "Armaf")

    def test_community_edges_are_available(self):
        rows = self.catalog.direct_similarity(1)
        self.assertEqual(rows[0]["perfume_id"], 2)
        self.assertGreater(community_similarity_score(5900, 886), 0.7)

    def test_negative_term_matching_uses_word_boundaries(self):
        metadata = {
            "name": "Cloud",
            "brand": "Example",
            "accords_csv": "sweet, airy",
            "notes_csv": "vanilla",
        }
        self.assertFalse(RetrievalEngine._has_excluded_term(metadata, ["oud"]))
        self.assertTrue(RetrievalEngine._has_excluded_term(metadata, ["vanilla"]))

    def test_wanted_trait_matching_does_not_reward_substrings(self):
        self.assertEqual(trait_match_strength("fresh", "fresh"), 1.0)
        self.assertEqual(trait_match_strength("fresh", "fresh spicy"), 0.0)
        self.assertEqual(trait_match_strength("warm spicy", "spicy"), 0.65)
        self.assertEqual(trait_match_strength("spicy", "warm spicy"), 0.75)

    def test_diversification_caps_repeated_brands(self):
        items = [
            {"perfume_id": 1, "brand": "A", "score": 1.0},
            {"perfume_id": 2, "brand": "A", "score": 0.9},
            {"perfume_id": 3, "brand": "B", "score": 0.8},
        ]
        selected = RetrievalEngine._diversify(items, 3, brand_limit=1)
        self.assertEqual([item["perfume_id"] for item in selected], [1, 3])

    def test_semantic_search_applies_negative_filter_after_ann(self):
        engine = RetrievalEngine.__new__(RetrievalEngine)
        engine.model = FakeEmbeddingModel()
        engine.collection = FakeCollection()
        engine.catalog = FakeCatalog()
        engine.encode_lock = threading.Lock()
        result = engine.search(
            {
                "query": "clean office without vanilla",
                "top_k": 3,
                "wanted_terms": ["clean", "fresh"],
                "exclude_terms": ["vanilla"],
            }
        )
        self.assertEqual({item["perfume_id"] for item in result["results"]}, {1, 3})
        self.assertFalse(any("vanilla" in item["document"] for item in result["results"]))
        self.assertEqual(result["accidental_brand_collisions"], ["clean"])
        clean_item = next(item for item in result["results"] if item["brand"] == "Clean")
        self.assertEqual(clean_item["reasons"]["accidental_brand_collision_penalty"], 0.14)

    def test_unknown_wanted_terms_are_ignored_using_fetched_taxonomy(self):
        engine = RetrievalEngine.__new__(RetrievalEngine)
        engine.model = FakeEmbeddingModel()
        engine.collection = FakeCollection()
        engine.catalog = FakeCatalog()
        engine.encode_lock = threading.Lock()
        result = engine.search({
            "query": "clean office scent",
            "top_k": 3,
            "wanted_terms": ["clean", "office"],
        })
        self.assertEqual(result["supported_wanted_terms"], ["clean"])
        self.assertEqual(result["ignored_wanted_terms"], ["office"])

    def test_required_terms_are_hard_filtered(self):
        engine = RetrievalEngine.__new__(RetrievalEngine)
        engine.model = FakeEmbeddingModel()
        engine.collection = FakeCollection()
        engine.catalog = FakeCatalog()
        engine.encode_lock = threading.Lock()
        result = engine.search({
            "query": "clean office scent",
            "top_k": 3,
            "required_terms": ["clean"],
        })
        self.assertEqual([item["perfume_id"] for item in result["results"]], [1])

    def test_balanced_search_merges_catalog_popularity_pool(self):
        class PopularCatalog(FakeCatalog):
            def popular_candidates(self, *args, **kwargs):
                return [{"perfume_id": 4}]

        class PopularCollection(FakeCollection):
            def get(self, **kwargs):
                self.last_get = kwargs
                return {
                    "ids": ["4"],
                    "documents": ["famous vanilla card"],
                    "metadatas": [{
                        "perfume_id": 4,
                        "name": "Famous Vanilla",
                        "brand": "House D",
                        "accords_csv": "vanilla, amber",
                        "notes_csv": "vanilla",
                        "rating": 4.5,
                        "popularity": 20000,
                    }],
                    "embeddings": [[1.0, 1.0, 1.0]],
                }

        engine = RetrievalEngine.__new__(RetrievalEngine)
        engine.model = FakeEmbeddingModel()
        engine.collection = PopularCollection()
        engine.catalog = PopularCatalog()
        engine.encode_lock = threading.Lock()
        result = engine.search({
            "query": "romantic evening fragrance",
            "top_k": 4,
            "required_terms": ["vanilla"],
        })
        famous = next(item for item in result["results"] if item["perfume_id"] == 4)
        self.assertEqual(famous["reasons"]["candidate_sources"], ["catalog_popular"])
        self.assertEqual(result["discovery_mode"], "balanced")
        # A substantially stronger semantic match must still beat the more
        # popular candidate in balanced mode.
        self.assertEqual(result["results"][0]["perfume_id"], 2)

    def test_niche_mode_does_not_inject_popularity_pool(self):
        class GuardCatalog(FakeCatalog):
            def popular_candidates(self, *args, **kwargs):
                raise AssertionError("niche mode must not request the popularity pool")

        engine = RetrievalEngine.__new__(RetrievalEngine)
        engine.model = FakeEmbeddingModel()
        engine.collection = FakeCollection()
        engine.catalog = GuardCatalog()
        engine.encode_lock = threading.Lock()
        result = engine.search({"query": "obscure vanilla", "top_k": 3, "discovery_mode": "niche"})
        self.assertEqual(result["discovery_mode"], "niche")

    def test_http_handler_exposes_json_contract(self):
        class FakeEngine:
            def health(self):
                return {"status": "ok"}

            def search(self, payload):
                return {"query": payload["query"], "results": []}

            def resolve(self, payload):
                return {"hint": payload["hint"], "resolved": None}

            def similar(self, payload):
                return {"hint": payload["hint"], "results": []}

        body = json.dumps({"query": "test"}).encode()
        handler = object.__new__(RetrievalRequestHandler)
        handler.engine = FakeEngine()
        handler.path = "/search"
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        handler.wfile = io.BytesIO()
        status_codes = []
        handler.send_response = status_codes.append
        handler.send_header = lambda *_: None
        handler.end_headers = lambda: None

        handler.do_POST()

        self.assertEqual(status_codes, [200])
        self.assertEqual(json.loads(handler.wfile.getvalue())["query"], "test")


if __name__ == "__main__":
    unittest.main()
