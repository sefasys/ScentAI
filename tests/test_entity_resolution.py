from __future__ import annotations

import unittest
from pathlib import Path

from scentai.retrieval import (
    CatalogResolver,
    RetrievalEngine,
    metadata_has_term,
    trait_match_strength,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "scentai_catalog.sqlite3"


@unittest.skipUnless(CATALOG.exists(), "local runtime catalog is required")
class CatalogResolverRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resolver = CatalogResolver(CATALOG)

    def assert_resolves(self, hint: str, expected_label: str) -> None:
        row = self.resolver.resolve(hint)
        self.assertIsNotNone(row, hint)
        actual = f"{row['name']} by {row['brand']}"
        self.assertEqual(actual, expected_label, hint)

    def test_brand_and_concentration_aliases(self) -> None:
        cases = {
            "YSL Y EDP": "Y Eau de Parfum by Yves Saint Laurent",
            "Y EDP by YSL": "Y Eau de Parfum by Yves Saint Laurent",
            "YSL's Y EDP": "Y Eau de Parfum by Yves Saint Laurent",
            "Y EPD YSL": "Y Eau de Parfum by Yves Saint Laurent",
            "YSL Y EDT": "Y Eau de Toilette by Yves Saint Laurent",
            "Bleu de Chanel EDP": "Bleu de Chanel Eau de Parfum by Chanel",
            "Bleu de Chanel EDT": "Bleu de Chanel by Chanel",
            "Bleu EDP Chanel": "Bleu de Chanel Eau de Parfum by Chanel",
            "Blue de Chanel EDP": "Bleu de Chanel Eau de Parfum by Chanel",
        }
        for hint, expected in cases.items():
            with self.subTest(hint=hint):
                self.assert_resolves(hint, expected)

    def test_catalog_initialisms_and_curated_community_aliases(self) -> None:
        cases = {
            "JPG Le Male Le Parfum": "Le Male Le Parfum by Jean Paul Gaultier",
            "LV Imagination": "Imagination by Louis Vuitton",
            "MFK Grand Soir": "Grand Soir by Maison Francis Kurkdjian",
            "PDM Layton": "Layton by Parfums de Marly",
            "CDG Kyoto": "Comme des Garcons Series 3 Incense: Kyoto by Comme des Garcons",
            "TF Tobacco Vanille": "Tobacco Vanille by Tom Ford",
            "CH Good Girl": "Good Girl by Carolina Herrera",
            "D&G The One for Men EDP": "The One for Men Eau de Parfum by Dolce&Gabbana",
        }
        for hint, expected in cases.items():
            with self.subTest(hint=hint):
                self.assert_resolves(hint, expected)

    def test_short_family_name_still_uses_dominant_member(self) -> None:
        self.assert_resolves(
            "Club de Nuit by Armaf",
            "Club de Nuit Intense Man by Armaf",
        )
        self.assert_resolves("Santal 33 by Le Labo", "Santal 33 by Le Labo")

    def test_literal_product_name_wins_before_edge_brand_extraction(self) -> None:
        cases = {
            "Burberry Men": "Burberry Men by Burberry",
            "Versace Man": "Versace Man by Versace",
        }
        for hint, expected in cases.items():
            with self.subTest(hint=hint):
                self.assert_resolves(hint, expected)

    def test_known_brand_scopes_typo_recovery(self) -> None:
        cases = {
            "MFK Grnad Soir": "Grand Soir by Maison Francis Kurkdjian",
            "JPG Le Male Le Parfume": "Le Male Le Parfum by Jean Paul Gaultier",
            "LV Imaginaton": "Imagination by Louis Vuitton",
            "PDM Laytn": "Layton by Parfums de Marly",
            "Tom Ford Tobaco Vanille": "Tobacco Vanille by Tom Ford",
            "Adidas Team 5": "Team Five by Adidas",
        }
        for hint, expected in cases.items():
            with self.subTest(hint=hint):
                self.assert_resolves(hint, expected)

    def test_diversification_deduplicates_normalized_labels(self) -> None:
        items = [
            {"perfume_id": 1, "name": "Example", "brand": "House", "score": 1.0},
            {"perfume_id": 2, "name": "Example", "brand": "House", "score": 0.9},
            {"perfume_id": 3, "name": "Another", "brand": "Other", "score": 0.8},
        ]
        selected = RetrievalEngine._diversify(items, 3, brand_limit=3)
        self.assertEqual([item["perfume_id"] for item in selected], [1, 3])

    def test_taxonomy_families_cover_strict_filter_variants(self) -> None:
        self.assertEqual(trait_match_strength("musk", "musky"), 1.0)
        self.assertEqual(trait_match_strength("leather", "suede"), 1.0)
        self.assertEqual(trait_match_strength("oud", "agarwood"), 1.0)
        self.assertTrue(metadata_has_term({"accords_csv": "musky, citrus"}, "musk"))


if __name__ == "__main__":
    unittest.main()
