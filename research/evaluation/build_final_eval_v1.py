from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CATALOG = Path("scentai_catalog.sqlite3")
DEFAULT_OUTPUT = Path("research/evaluation/final_eval_v1.jsonl")

CATEGORY_TARGETS = {
    "recommendation": 20,
    "perfume_profile": 15,
    "comparison": 15,
    "similarity_alternative": 15,
    "hard_filters": 15,
    "entity_resolution": 10,
    "conversation": 10,
    "unsupported": 10,
    "multilingual_noisy": 10,
}


def make_case(
    case_id: str,
    category: str,
    query: str,
    language: str,
    *,
    expected: dict[str, Any],
    tags: list[str] | None = None,
    session_id: str | None = None,
    turn: int | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": case_id,
        "version": "final_eval_v1",
        "category": category,
        "language": language,
        "query": query,
        "tags": tags or [],
        "expected": {"response_language": language, **expected},
    }
    if session_id:
        row["session_id"] = session_id
        row["turn"] = turn
    return row


def recommendation_cases() -> list[dict[str, Any]]:
    specs = [
        ("I need exactly 3 clean professional office fragrances without vanilla.", "en", ["recommendation"], 3, [], ["vanilla"], {}),
        ("Date akşamı için vanilya ve baharat mutlaka bulunan tam 3 parfüm öner.", "tr", ["recommendation"], 3, ["vanilla", "warm spicy"], [], {"time_profile": "night"}),
        ("Recommend exactly 3 fresh citrus summer fragrances for men.", "en", ["recommendation"], 3, [], [], {"gender": "male", "season": "summer"}),
        ("Kadınlar için kış gecesine uygun, vanilyalı tam 3 parfüm öner.", "tr", ["recommendation"], 3, ["vanilla"], [], {"gender": "female", "season": "winter", "time_profile": "night"}),
        ("Recommend exactly 3 popular and recognizable woody fragrances for versatile wear.", "en", ["recommendation"], 3, [], [], {"discovery_mode": "mainstream"}),
        ("İlkbahar için az bilinen, niş ve yeşil karakterli tam 3 parfüm öner.", "tr", ["recommendation"], 3, [], [], {"discovery_mode": "niche", "season": "spring"}),
        ("Recommend exactly 3 unisex smoky fragrances for autumn nights.", "en", ["recommendation"], 3, [], [], {"gender": "unisex", "season": "autumn", "time_profile": "night"}),
        ("Hindistan cevizi mutlaka bulunan tropikal yazlık tam 3 parfüm öner.", "tr", ["recommendation"], 3, ["coconut"], [], {"season": "summer"}),
        ("Recommend exactly 3 spring fragrances for women that must feature rose.", "en", ["recommendation"], 3, ["rose"], [], {"gender": "female", "season": "spring"}),
        ("Erkekler için oud ve deri mutlaka bulunan, akşama uygun tam 3 parfüm öner.", "tr", ["recommendation"], 3, ["oud", "leather"], [], {"gender": "male", "time_profile": "night"}),
        ("Recommend exactly 3 cozy winter fragrances that must have coffee.", "en", ["recommendation"], 3, ["coffee"], [], {"season": "winter"}),
        ("Misk mutlaka bulunan, tene yakın ve temiz hissettiren tam 3 parfüm öner.", "tr", ["recommendation"], 3, ["musk"], [], {}),
        ("Recommend exactly 3 aromatic men's fragrances for daytime gym wear.", "en", ["recommendation"], 3, [], [], {"gender": "male", "time_profile": "day"}),
        ("Ofis için pudralı ve iris karakterli tam 3 parfüm öner.", "tr", ["recommendation"], 3, ["iris"], [], {}),
        ("Recommend exactly 3 unisex evening fragrances that must have amber.", "en", ["recommendation"], 3, ["amber"], [], {"gender": "unisex", "time_profile": "night"}),
        ("Günlük kullanım için ferah baharatlı tam 3 erkek parfümü öner.", "tr", ["recommendation"], 3, [], [], {"gender": "male"}),
        ("Recommend exactly 3 popular winter-night fragrances that must feature tobacco.", "en", ["recommendation"], 3, ["tobacco"], [], {"discovery_mode": "mainstream", "season": "winter", "time_profile": "night"}),
        ("Tütsü mutlaka bulunan, sıra dışı ve niş tam 3 parfüm öner.", "tr", ["recommendation"], 3, ["incense"], [], {"discovery_mode": "niche"}),
        ("Recommend exactly 3 fruity sweet date fragrances with no vanilla.", "en", ["recommendation"], 3, [], ["vanilla"], {"time_profile": "night"}),
        ("Yaz için narenciye mutlaka bulunan ama misk içermeyen tam 3 parfüm öner.", "tr", ["recommendation"], 3, ["citrus"], ["musk"], {"season": "summer"}),
    ]
    rows = []
    for index, (query, language, intents, count, required, excluded, extra) in enumerate(specs, 1):
        expected = {
            "intent_in": intents,
            "requested_count": count,
            "required_terms": required,
            "excluded_terms": excluded,
            **extra,
        }
        rows.append(make_case(f"fev1_rec_{index:03d}", "recommendation", query, language, expected=expected))
    return rows


PROFILE_SPECS = [
    ("Turkish Leather by Pryn Parfum", "Turkish Leather by Pryn Parfum nasıl bir parfüm? Karakterini ve performansını anlat.", "tr"),
    ("Tobacco Vanille by Tom Ford", "What is Tobacco Vanille by Tom Ford like? Explain its character and performance.", "en"),
    ("Team Five by Adidas", "Team Five by Adidas nasıl bir parfüm? En uygun kullanım alanını anlat.", "tr"),
    ("Versace Pour Homme by Versace", "Tell me what Versace Pour Homme by Versace is like and where it works best.", "en"),
    ("Imagination by Louis Vuitton", "Imagination by Louis Vuitton hakkında danışman gibi bilgi ver.", "tr"),
    ("Santal 33 by Le Labo", "Describe the vibe, performance, and best use cases of Santal 33 by Le Labo.", "en"),
    ("Coco Mademoiselle by Chanel", "Coco Mademoiselle by Chanel nasıl bir karaktere sahip?", "tr"),
    ("Terre d'Hermès by Hermès", "What does Terre d'Hermès by Hermès feel like, and when should it be worn?", "en"),
    ("Spicebomb Extreme by Viktor&Rolf", "Spicebomb Extreme by Viktor&Rolf'un karakterini ve kullanımını anlat.", "tr"),
    ("Grand Soir by Maison Francis Kurkdjian", "Give me a grounded profile of Grand Soir by Maison Francis Kurkdjian.", "en"),
    ("By the Fireplace by Maison Martin Margiela", "By the Fireplace by Maison Martin Margiela nasıl bir atmosfer yaratıyor?", "tr"),
    ("Prada L'Homme by Prada", "Explain the character and practical wear of Prada L'Homme by Prada.", "en"),
    ("Y Eau de Parfum by Yves Saint Laurent", "Y Eau de Parfum by Yves Saint Laurent hakkında detaylı ama kısa bilgi ver.", "tr"),
    ("Reflection Man by Amouage", "What is Reflection Man by Amouage like as a personal style choice?", "en"),
    ("Club de Nuit Intense Man by Armaf", "Club de Nuit Intense Man by Armaf nasıl bir parfüm ve performansı nasıl?", "tr"),
]


def profile_cases() -> list[dict[str, Any]]:
    return [
        make_case(
            f"fev1_profile_{index:03d}",
            "perfume_profile",
            query,
            language,
            expected={"intent_in": ["perfume_profile"], "resolved_labels": [label]},
        )
        for index, (label, query, language) in enumerate(PROFILE_SPECS, 1)
    ]


COMPARISON_SPECS = [
    ("Club de Nuit Intense Man by Armaf", "Team Five by Adidas", "Compare Club de Nuit by Armaf with Team Five by Adidas for vibe, performance, and use.", "en"),
    ("Hero by Burberry", "Tobacco Vanille by Tom Ford", "Burberry Hero ile Tobacco Vanille by Tom Ford'u karakter ve kullanım açısından karşılaştır.", "tr"),
    ("Imagination by Louis Vuitton", "Versace Pour Homme by Versace", "Compare Imagination by Louis Vuitton and Versace Pour Homme by Versace for office wear.", "en"),
    ("Santal 33 by Le Labo", "Prada L'Homme by Prada", "Santal 33 by Le Labo ile Prada L'Homme by Prada'yı tarz ve kullanım açısından karşılaştır.", "tr"),
    ("Spicebomb Extreme by Viktor&Rolf", "Le Male Le Parfum by Jean Paul Gaultier", "Compare Spicebomb Extreme by Viktor&Rolf with Le Male Le Parfum by Jean Paul Gaultier for a date night.", "en"),
    ("Grand Soir by Maison Francis Kurkdjian", "By the Fireplace by Maison Martin Margiela", "Grand Soir ile By the Fireplace'ı sıcaklık, karakter ve ortam açısından karşılaştır.", "tr"),
    ("Aventus by Creed", "Explorer by Montblanc", "Compare Aventus by Creed and Explorer by Montblanc without declaring a winner.", "en"),
    ("Coco Mademoiselle by Chanel", "Hypnotic Poison by Dior", "Coco Mademoiselle ile Hypnotic Poison'ı verdikleri izlenim açısından karşılaştır.", "tr"),
    ("Terre d'Hermès by Hermès", "Bleu de Chanel Eau de Parfum by Chanel", "Compare Terre d'Hermès by Hermès with Bleu de Chanel Eau de Parfum by Chanel for professional wear.", "en"),
    ("Eros by Versace", "Eros Flame by Versace", "Versace Eros ile Eros Flame'i karakter ve mevsim açısından karşılaştır.", "tr"),
    ("Reflection Man by Amouage", "Dior Homme Intense 2011 by Dior", "Compare Reflection Man by Amouage and Dior Homme Intense 2011 by Dior for formal occasions.", "en"),
    ("Light Blue pour Homme by Dolce&Gabbana", "Acqua di Giò Profondo by Giorgio Armani", "Light Blue pour Homme ile Acqua di Giò Profondo'yu yaz kullanımı için karşılaştır.", "tr"),
    ("Angels' Share by By Kilian", "Khamrah Qahwa by Lattafa Perfumes", "Compare Angels' Share by By Kilian and Khamrah Qahwa by Lattafa Perfumes as gourmand evening scents.", "en"),
    ("Black Orchid by Tom Ford", "Libre Intense by Yves Saint Laurent", "Black Orchid ile Libre Intense'i gece kullanımı ve karakter açısından karşılaştır.", "tr"),
    ("Oud for Greatness by Initio Parfums Prives", "Ombre Nomade by Louis Vuitton", "Compare Oud for Greatness and Ombre Nomade for oud character, performance, and occasion.", "en"),
]


def comparison_cases() -> list[dict[str, Any]]:
    return [
        make_case(
            f"fev1_cmp_{index:03d}",
            "comparison",
            query,
            language,
            expected={"intent_in": ["comparison"], "resolved_labels": [left, right]},
        )
        for index, (left, right, query, language) in enumerate(COMPARISON_SPECS, 1)
    ]


SIMILARITY_SPECS = [
    ("Aventus by Creed", "I want exactly 3 fragrances similar to Aventus by Creed but less smoky.", "en", ["smoky"]),
    ("Tobacco Vanille by Tom Ford", "Tobacco Vanille'e benzeyen ama tütünsüz tam 3 seçenek öner.", "tr", ["tobacco"]),
    ("Santal 33 by Le Labo", "Give me exactly 3 fragrances similar to Santal 33 by Le Labo but without leather.", "en", ["leather"]),
    ("Bleu de Chanel Eau de Parfum by Chanel", "Bleu de Chanel EDP'ye alternatif, daha narenciyeli tam 3 parfüm öner.", "tr", []),
    ("By the Fireplace by Maison Martin Margiela", "Find exactly 3 scents like By the Fireplace by Maison Martin Margiela but less smoky.", "en", ["smoky"]),
    ("Explorer by Montblanc", "Montblanc Explorer'a benzeyen ama daha meyvemsi tam 3 parfüm öner.", "tr", []),
    ("Grand Soir by Maison Francis Kurkdjian", "Give me exactly 3 alternatives to Grand Soir by Maison Francis Kurkdjian with a softer amber profile.", "en", []),
    ("Prada L'Homme by Prada", "Prada L'Homme'a benzeyen ama pudralı olmayan tam 3 parfüm öner.", "tr", ["powdery"]),
    ("Imagination by Louis Vuitton", "Recommend exactly 3 fragrances similar to Imagination by Louis Vuitton but woodier.", "en", []),
    ("Light Blue pour Homme by Dolce&Gabbana", "Light Blue pour Homme benzeri ama daha karakterli tam 3 seçenek ver.", "tr", []),
    ("Eros by Versace", "Find exactly 3 alternatives to Eros by Versace without vanilla.", "en", ["vanilla"]),
    ("Coco Mademoiselle by Chanel", "Coco Mademoiselle'e benzeyen ama gül içermeyen tam 3 parfüm öner.", "tr", ["rose"]),
    ("Spicebomb Extreme by Viktor&Rolf", "Recommend exactly 3 scents like Spicebomb Extreme by Viktor&Rolf without tobacco.", "en", ["tobacco"]),
    ("Reflection Man by Amouage", "Reflection Man'e benzeyen ama daha narenciyeli tam 3 seçenek öner.", "tr", []),
    ("Angels' Share by By Kilian", "Give me exactly 3 alternatives to Angels' Share by By Kilian without cinnamon.", "en", ["cinnamon"]),
]


def similarity_cases() -> list[dict[str, Any]]:
    return [
        make_case(
            f"fev1_sim_{index:03d}",
            "similarity_alternative",
            query,
            language,
            expected={
                "intent_in": ["similarity", "alternative"],
                "requested_count": 3,
                "reference_label": reference,
                "excluded_terms": excluded,
            },
        )
        for index, (reference, query, language, excluded) in enumerate(SIMILARITY_SPECS, 1)
    ]


HARD_FILTER_SPECS = [
    ("Recommend exactly 3 office fragrances with no vanilla, oud, or smoky accords.", "en", [], ["vanilla", "oud", "smoky"], {}),
    ("Date akşamı için vanilya mutlaka bulunan ama tütün içermeyen tam 3 parfüm öner.", "tr", ["vanilla"], ["tobacco"], {"time_profile": "night"}),
    ("Recommend exactly 3 rose fragrances that must not contain musk.", "en", ["rose"], ["musk"], {}),
    ("Narenciye mutlaka bulunan ama çiçeksi olmayan tam 3 parfüm öner.", "tr", ["citrus"], ["floral"], {}),
    ("Recommend exactly 3 coconut fragrances without vanilla.", "en", ["coconut"], ["vanilla"], {}),
    ("Kahve mutlaka bulunan ama karamel içermeyen tam 3 parfüm öner.", "tr", ["coffee"], ["caramel"], {}),
    ("Recommend exactly 4 men's fragrances from Versace without vanilla.", "en", [], ["vanilla"], {"requested_brand": "Versace", "gender": "male"}),
    ("Kadınlar için ilkbahara uygun ama paçuli içermeyen tam 3 parfüm öner.", "tr", [], ["patchouli"], {"gender": "female", "season": "spring"}),
    ("Recommend exactly 3 unisex winter fragrances that must have oud but no rose.", "en", ["oud"], ["rose"], {"gender": "unisex", "season": "winter"}),
    ("Erkekler için aromatik akor mutlaka bulunan ama deri içermeyen tam 3 parfüm öner.", "tr", ["aromatic"], ["leather"], {"gender": "male"}),
    ("Recommend exactly 3 iris fragrances without vanilla.", "en", ["iris"], ["vanilla"], {}),
    ("Ananas mutlaka bulunan ama dumanlı olmayan tam 3 parfüm öner.", "tr", ["pineapple"], ["smoky"], {}),
    ("Recommend exactly 3 incense fragrances without oud.", "en", ["incense"], ["oud"], {}),
    ("Yeşil karakterli ama tatlı olmayan tam 3 parfüm öner.", "tr", ["green"], ["sweet"], {}),
    ("Recommend exactly 3 leather fragrances without animalic accords.", "en", ["leather"], ["animalic"], {}),
]


def hard_filter_cases() -> list[dict[str, Any]]:
    rows = []
    for index, (query, language, required, excluded, extra) in enumerate(HARD_FILTER_SPECS, 1):
        count = 4 if "exactly 4" in query else 3
        rows.append(make_case(
            f"fev1_filter_{index:03d}",
            "hard_filters",
            query,
            language,
            expected={
                "intent_in": ["recommendation"],
                "requested_count": count,
                "required_terms": required,
                "excluded_terms": excluded,
                **extra,
            },
        ))
    return rows


ENTITY_SPECS = [
    ("Club de Nuit Intense Man by Armaf", "Club de Nuit by Armaf nasıl bir parfüm?", "tr"),
    ("Aventus by Creed", "Tell me about Aventus by Creed.", "en"),
    ("Hero by Burberry", "Burberry Hero hakkında bilgi ver.", "tr"),
    ("Y Eau de Parfum by Yves Saint Laurent", "What is YSL Y EDP like?", "en"),
    ("Prada L'Homme by Prada", "Prada L'Homme nasıl bir parfüm?", "tr"),
    ("Imagination by Louis Vuitton", "Tell me about LV Imagination.", "en"),
    ("Grand Soir by Maison Francis Kurkdjian", "MFK Grand Soir hakkında bilgi ver.", "tr"),
    ("Team Five by Adidas", "What is Adidas Team Five like?", "en"),
    ("Tobacco Vanille by Tom Ford", "Tom Ford Tobacco Vanille nasıl bir parfüm?", "tr"),
    ("Bleu de Chanel Eau de Parfum by Chanel", "Tell me about Bleu de Chanel EDP.", "en"),
]


def entity_cases() -> list[dict[str, Any]]:
    return [
        make_case(
            f"fev1_entity_{index:03d}",
            "entity_resolution",
            query,
            language,
            expected={"intent_in": ["perfume_profile"], "resolved_labels": [label]},
        )
        for index, (label, query, language) in enumerate(ENTITY_SPECS, 1)
    ]


def conversation_cases() -> list[dict[str, Any]]:
    specs = [
        ("conv_vanilla", 1, "Recommend exactly 3 popular fragrances that must have vanilla.", "en", "new_request", ["vanilla"], [], "mainstream", False),
        ("conv_vanilla", 2, "Give me three different options under the same requirements.", "en", "more_options", ["vanilla"], [], "mainstream", True),
        ("conv_office", 1, "Ofis için vanilyasız tam 3 temiz parfüm öner.", "tr", "new_request", [], ["vanilla"], None, False),
        ("conv_office", 2, "Aynı koşullarda daha sportif üç farklı seçenek ver.", "tr", "refine_previous", [], ["vanilla"], None, True),
        ("conv_summer", 1, "Recommend exactly 3 citrus summer fragrances for men.", "en", "new_request", [], [], None, False),
        ("conv_summer", 2, "Make them unisex and give me three new options.", "en", "refine_previous", [], [], None, True),
        ("conv_discovery", 1, "Bana popüler, vanilyalı tam 3 parfüm öner.", "tr", "new_request", ["vanilla"], [], "mainstream", False),
        ("conv_discovery", 2, "Şimdi aynı isteğe uygun ama daha niş üç farklı seçenek ver.", "tr", "refine_previous", ["vanilla"], [], "niche", True),
        ("conv_date", 1, "Recommend exactly 3 warm spicy fragrances for a date night.", "en", "new_request", [], [], None, False),
        ("conv_date", 2, "Keep the same request, but exclude tobacco and show three different choices.", "en", "refine_previous", [], ["tobacco"], None, True),
    ]
    rows = []
    for index, (session_id, turn, query, language, action, required, excluded, discovery, no_repeat) in enumerate(specs, 1):
        expected: dict[str, Any] = {
            "intent_in": ["recommendation"],
            "requested_count": 3,
            "conversation_action": action,
            "required_terms": required,
            "excluded_terms": excluded,
            "no_repeat_previous": no_repeat,
        }
        if discovery:
            expected["discovery_mode"] = discovery
        rows.append(make_case(
            f"fev1_conv_{index:03d}",
            "conversation",
            query,
            language,
            expected=expected,
            session_id=session_id,
            turn=turn,
        ))
    return rows


UNSUPPORTED_SPECS = [
    ("What is the current price of Aventus by Creed?", "en", "unsupported_price"),
    ("Tobacco Vanille şu anda Türkiye'de stokta mı?", "tr", "unsupported_availability"),
    ("Is Santal 33 safe for someone with a severe fragrance allergy?", "en", "unsupported_medical"),
    ("En çok iltifat getirecek parfüm hangisi?", "tr", "unsupported_social_claim"),
    ("How should I layer Aventus with Tobacco Vanille?", "en", "unsupported_layering"),
    ("Grand Soir'ın güncel fiyatı ne kadar?", "tr", "unsupported_price"),
    ("Can you confirm whether Prada L'Homme is discontinued and in stock?", "en", "unsupported_availability"),
    ("Bu parfüm astımımı tetikler mi, güvenli mi?", "tr", "unsupported_medical"),
    ("Which perfume is guaranteed to attract compliments on a date?", "en", "unsupported_social_claim"),
    ("Santal 33 ile Another 13'ü nasıl katmanlamalıyım?", "tr", "unsupported_layering"),
]


def unsupported_cases() -> list[dict[str, Any]]:
    return [
        make_case(
            f"fev1_unsupported_{index:03d}",
            "unsupported",
            query,
            language,
            expected={"intent_in": [route], "route": route, "generation_attempts": 0},
        )
        for index, (query, language, route) in enumerate(UNSUPPORTED_SPECS, 1)
    ]


NOISY_SPECS = [
    ("bna yaz icin ferah tam 3 erkek parfumu oner", "tr", {"intent_in": ["recommendation"], "requested_count": 3, "gender": "male"}),
    ("need exactly 3 ofice scents no vanila pls", "en", {"intent_in": ["recommendation"], "requested_count": 3, "excluded_terms": ["vanilla"]}),
    ("aventus nasil bi parfum", "tr", {"intent_in": ["perfume_profile"], "resolved_labels": ["Aventus by Creed"]}),
    ("cmpare eros by versace n eros flame by versace", "en", {"intent_in": ["comparison"], "resolved_labels": ["Eros by Versace", "Eros Flame by Versace"]}),
    ("bana date night icin baharatli tam 3 parfum", "tr", {"intent_in": ["recommendation"], "requested_count": 3}),
    ("wht is turkish leather by pryn parfum like", "en", {"intent_in": ["perfume_profile"], "resolved_labels": ["Turkish Leather by Pryn Parfum"]}),
    ("Versace'den tam 4 erkek parfumu oner", "tr", {"intent_in": ["recommendation"], "requested_count": 4, "requested_brand": "Versace", "gender": "male"}),
    ("clean ama sweet olmayan tam 3 bisey oner", "tr", {"intent_in": ["recommendation"], "requested_count": 3, "excluded_terms": ["sweet"]}),
    ("smth like santal 33 by le labo but less leathery exactly 3", "en", {"intent_in": ["similarity", "alternative"], "requested_count": 3, "reference_label": "Santal 33 by Le Labo", "excluded_terms": ["leather"]}),
    ("LV imagination hakkinda ne dusunuyosun", "tr", {"intent_in": ["perfume_profile"], "resolved_labels": ["Imagination by Louis Vuitton"]}),
]


def noisy_cases() -> list[dict[str, Any]]:
    return [
        make_case(
            f"fev1_noisy_{index:03d}",
            "multilingual_noisy",
            query,
            language,
            expected=expected,
            tags=["noisy_or_colloquial"],
        )
        for index, (query, language, expected) in enumerate(NOISY_SPECS, 1)
    ]


def all_cases() -> list[dict[str, Any]]:
    return [
        *recommendation_cases(),
        *profile_cases(),
        *comparison_cases(),
        *similarity_cases(),
        *hard_filter_cases(),
        *entity_cases(),
        *conversation_cases(),
        *unsupported_cases(),
        *noisy_cases(),
    ]


def expected_labels(cases: list[dict[str, Any]]) -> set[str]:
    labels: set[str] = set()
    for case in cases:
        expected = case["expected"]
        labels.update(expected.get("resolved_labels", []))
        reference = expected.get("reference_label")
        if reference:
            labels.add(reference)
    return labels


def catalog_labels(connection: sqlite3.Connection) -> set[str]:
    return {
        f"{name} by {brand}"
        for name, brand in connection.execute("SELECT name, brand FROM perfumes")
    }


def validate_cases(cases: list[dict[str, Any]], catalog: Path) -> dict[str, Any]:
    if len(cases) != 120:
        raise ValueError(f"Expected 120 cases, found {len(cases)}")
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate final-eval IDs")
    queries = [(case.get("session_id"), case["query"].casefold()) for case in cases]
    if len(queries) != len(set(queries)):
        raise ValueError("Duplicate final-eval query in the same session scope")
    counts = Counter(case["category"] for case in cases)
    if dict(counts) != CATEGORY_TARGETS:
        raise ValueError(f"Unexpected category counts: {dict(counts)}")
    if not catalog.exists():
        raise FileNotFoundError(catalog)
    with sqlite3.connect(catalog) as connection:
        available = catalog_labels(connection)
    missing = sorted(expected_labels(cases) - available)
    if missing:
        raise ValueError(f"Expected catalog labels are missing: {missing}")
    session_turns: dict[str, list[int]] = {}
    for case in cases:
        if case.get("session_id"):
            session_turns.setdefault(case["session_id"], []).append(int(case["turn"]))
    invalid_sessions = {
        session: turns for session, turns in session_turns.items() if sorted(turns) != list(range(1, len(turns) + 1))
    }
    if invalid_sessions:
        raise ValueError(f"Invalid conversation turn sequence: {invalid_sessions}")
    return {
        "version": "final_eval_v1",
        "case_count": len(cases),
        "category_counts": dict(counts),
        "language_counts": dict(Counter(case["language"] for case in cases)),
        "session_count": len(session_turns),
        "catalog": str(catalog),
        "expected_catalog_label_count": len(expected_labels(cases)),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fixed ScentAI final evaluation v1 set.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cases = all_cases()
    manifest = validate_cases(cases, args.catalog)
    write_jsonl(args.output, cases)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
