"""
generate_l2.py — ScentAI L2 Training Data Generator

Felsefe: L2 = Bilgiyi filtreleme.
Model asla yorum yapmaz, çıkarım yapmaz, subjektif ifade kullanmaz.
Sadece: Filtrele, Sırala, Karşılaştır.

Tip A — Multi Filter (%35)
Tip B — Negative Constraints (%25)
Tip C — Ranking (%15)
Tip D — Numeric Filters (%10)
Tip E — Comparison (%10)
Tip F — No Match (%5)

RAG: %80 context, %20 genel sorgu
"""

import json
import random
import os
import re
from collections import Counter

# ── CONFIG ──────────────────────────────────────────────────────────────────
IS_BASELINE = False
CLEAN_FILE = "/home/sefasys/Desktop/Perfume-Dataset/perfumes_clean.jsonl"
OUTPUT_FILE = (
    "/home/sefasys/Desktop/Perfume-Dataset/baseline_L2.jsonl"
    if IS_BASELINE
    else "/home/sefasys/Desktop/Perfume-Dataset/training_L2.jsonl"
)

SYSTEM_PROMPT = (
    "You are a professional perfume database assistant. "
    "Answer the query based strictly on the provided perfume context if available, "
    "otherwise state the database facts. "
    "Do not make subjective interpretations or qualitative assertions."
)

CATEGORY_RATIOS = {
    "multi_filter": 0.35,
    "negative_constraints": 0.25,
    "ranking": 0.15,
    "numeric_filters": 0.10,
    "comparison": 0.10,
    "no_match": 0.05,
}

# ── YAYIN ACCORD / NOTE LİSTELERİ ──────────────────────────────────────────
COMMON_ACCORDS = []
COMMON_NOTES = []

# ── VERİ YÜKLEME ───────────────────────────────────────────────────────────

def load_perfumes():
    perfumes = []
    with open(CLEAN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            meta = p["metadata"]
            # Ön-hesaplanmış küçük harfli kümeler (O(1) lookup)
            meta["_seasons_set"] = {s.lower() for s in (meta.get("best_seasons") or [])}
            meta["_times_set"] = {t.lower() for t in (meta.get("time_profile") or [])}
            meta["_accords_set"] = {a.lower() for a in (meta.get("accords_list") or [])}
            meta["_notes_set"] = {n.lower() for n in (meta.get("notes_list") or [])}
            perfumes.append(p)
    perfumes.sort(key=lambda x: x["metadata"].get("popularity", 0), reverse=True)
    return perfumes


def build_frequency_lists(perfumes):
    """En yaygın accord ve notaları topla — sorguları gerçekçi kılmak için."""
    acc_counter = Counter()
    note_counter = Counter()
    for p in perfumes:
        meta = p["metadata"]
        for a in (meta.get("accords_list") or []):
            acc_counter[a.lower()] += 1
        for n in (meta.get("notes_list") or []):
            note_counter[n.lower()] += 1
    return (
        [a for a, _ in acc_counter.most_common(30)],
        [n for n, _ in note_counter.most_common(40)],
    )


# ── FİLTRELEME MOTORU ──────────────────────────────────────────────────────

def match_perfume(p, criteria):
    """Tek bir parfümün verilen kriterlere uyup uymadığını kontrol eder."""
    meta = p["metadata"]
    card = p.get("card_text", "").lower()

    # Cinsiyet
    if criteria.get("gender"):
        q_g = criteria["gender"].lower()
        p_g = meta.get("gender", "").lower()
        if q_g != "any" and q_g != p_g:
            return False

    # Sezon (en az biri eşleşmeli)
    if criteria.get("seasons"):
        if not any(s.lower() in meta["_seasons_set"] for s in criteria["seasons"]):
            return False

    # Gün/gece
    if criteria.get("time_profile"):
        if not any(t.lower() in meta["_times_set"] for t in criteria["time_profile"]):
            return False

    # Pozitif akorlar (tümü olmalı)
    if criteria.get("accords"):
        if not all(a.lower() in meta["_accords_set"] for a in criteria["accords"]):
            return False

    # Pozitif notalar (tümü olmalı)
    if criteria.get("notes"):
        if not all(n.lower() in meta["_notes_set"] for n in criteria["notes"]):
            return False

    # Negatif akorlar — word boundary ile kontrol
    for forbidden in (criteria.get("not_accords") or []):
        f_low = forbidden.lower()
        if f_low in meta["_accords_set"]:
            return False
        if f_low in meta["_notes_set"]:
            return False
        if re.search(rf"\b{re.escape(f_low)}\b", card):
            return False
        # Metadata'daki diğer elemanlarda da regex ile ara (örn "white floral" içindeki "floral" için)
        combined_meta = " ".join(meta["_accords_set"]) + " " + " ".join(meta["_notes_set"])
        if re.search(rf"\b{re.escape(f_low)}\b", combined_meta):
            return False

    # Negatif notalar — word boundary ile kontrol
    for forbidden in (criteria.get("not_notes") or []):
        f_low = forbidden.lower()
        if f_low in meta["_notes_set"]:
            return False
        if f_low in meta["_accords_set"]:
            return False
        if re.search(rf"\b{re.escape(f_low)}\b", card):
            return False
        combined_meta = " ".join(meta["_accords_set"]) + " " + " ".join(meta["_notes_set"])
        if re.search(rf"\b{re.escape(f_low)}\b", combined_meta):
            return False

    # Minimum rating
    if criteria.get("min_rating"):
        r = meta.get("rating")
        pop = meta.get("popularity", 0)
        if r is None or r < criteria["min_rating"] or pop == 0:
            return False

    # Minimum oy
    if criteria.get("min_votes"):
        if meta.get("popularity", 0) < criteria["min_votes"]:
            return False

    return True


def find_matches(perfumes, criteria, limit=10):
    """Kriterlere uyan parfümleri döndürür (popularity sıralı)."""
    results = []
    for p in perfumes:
        if match_perfume(p, criteria):
            results.append(p)
            if len(results) >= limit:
                break
    return results


# ── FORMAT FONKSİYONLARI ───────────────────────────────────────────────────

def format_perfume_list(perfumes):
    """Parfüm listesini sade, faktüel formatta döndürür."""
    if not perfumes:
        return "No matching perfumes found."
    lines = []
    for idx, p in enumerate(perfumes, 1):
        meta = p["metadata"]
        rating = meta.get("rating")
        pop = meta.get("popularity", 0)
        seasons = meta.get("best_seasons") or []
        times = meta.get("time_profile") or []
        accords = meta.get("accords_list") or []

        rating_str = (
            f"{rating:.2f}/5 ({pop} votes)"
            if (rating is not None and rating > 0.0 and pop > 0)
            else "N/A"
        )
        seasons_str = ", ".join(s.capitalize() for s in seasons) if seasons else "N/A"
        times_str = ", ".join(t.capitalize() for t in times) if times else "N/A"
        accords_str = ", ".join(accords[:6]) if accords else "N/A"

        lines.append(f"{idx}. {p['name']} by {p['brand']}")
        lines.append(f"- Rating: {rating_str}")
        lines.append(f"- Seasons: {seasons_str}")
        lines.append(f"- Time: {times_str}")
        lines.append(f"- Accords: {accords_str}")

        # Nota hiyerarşisini koru (Top | Middle | Base)
        notes_line = None
        for line in p["card_text"].split("\n"):
            ls = line.strip()
            if ls.startswith("Top Notes:") or ls.startswith("Notes:"):
                notes_line = ls
                break
        if notes_line:
            if "|" in notes_line:
                for part in notes_line.split("|"):
                    lines.append(f"- {part.strip()}")
            else:
                lines.append(f"- {notes_line}")
        else:
            notes = meta.get("notes_list") or []
            lines.append(f"- Notes: {', '.join(notes[:8]) if notes else 'N/A'}")

    return "\n".join(lines)


def build_rag_context(selected, all_perfumes, context_size=12):
    """RAG context bloğu oluşturur: seçili parfümler + rastgele dolgu."""
    pool = set(id(p) for p in selected)
    fillers = [p for p in random.sample(all_perfumes[:5000], min(200, len(all_perfumes))) if id(p) not in pool]
    needed = context_size - len(selected)
    context_list = list(selected) + random.sample(fillers, min(needed, len(fillers)))
    random.shuffle(context_list)
    cards = [p["card_text"] for p in context_list]
    context_text = f"[PERFUMES]\n" + "\n\n".join(cards) + "\n[/PERFUMES]\n\n"
    return context_list, context_text


def build_messages(user_content, answer):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "model", "content": answer},
        ]
    }


# ── DOĞRULAYICILAR (VALIDATORS) ────────────────────────────────────────────

def validate_negative_constraints(matched, not_notes, not_accords):
    """Bulunan parfümlerde negatif kısıtların ihlal edilmediğini doğrular."""
    for p in matched:
        meta = p["metadata"]
        p_notes = {n.lower() for n in (meta.get("notes_list") or [])}
        p_acc = {a.lower() for a in (meta.get("accords_list") or [])}
        card = p.get("card_text", "").lower()
        combined_meta = " ".join(p_notes) + " " + " ".join(p_acc)
        all_forbidden = [(f, "note") for f in not_notes] + [(f, "accord") for f in not_accords]
        for forbidden, ftype in all_forbidden:
            f_low = forbidden.lower()
            if f_low in p_notes or f_low in p_acc or re.search(rf"\b{re.escape(f_low)}\b", card) or re.search(rf"\b{re.escape(f_low)}\b", combined_meta):
                raise AssertionError(
                    f"Excluded {ftype} '{forbidden}' found in '{p['name']} by {p['brand']}'. "
                    f"Notes: {p_notes}, Accords: {p_acc}"
                )


def validate_no_match(results):
    if len(results) > 0:
        raise AssertionError(f"Expected no matches, found {len(results)}")


# ── SORU TEMPLATE'LERİ ─────────────────────────────────────────────────────

MULTI_FILTER_TEMPLATES = [
    "List {gender} perfumes for {season} with {accord} accords.",
    "Show {gender} fragrances suitable for {season} featuring {accord}.",
    "Find {gender} {season} perfumes with {accord} accord.",
    "What {gender} perfumes are good for {season} and have {accord} accords?",
    "Recommend {gender} {accord} fragrances for {season}.",
    "Which {gender} perfumes match {season} and {accord}?",
    "Search for {gender} {season} {accord} perfumes.",
]

MULTI_FILTER_WITH_TIME_TEMPLATES = [
    "List {gender} perfumes for {season} {time} with {accord} accords.",
    "Find {gender} {time} fragrances for {season} featuring {accord}.",
    "Show {gender} {accord} perfumes suitable for {season} {time} use.",
]

MULTI_FILTER_WITH_NOTE_TEMPLATES = [
    "Find {gender} perfumes for {season} with {accord} accords containing {note}.",
    "List {gender} {season} fragrances featuring {accord} and {note} note.",
    "Show {gender} perfumes with {accord} and {note} for {season}.",
]

NEG_TEMPLATES = [
    "Recommend {gender} perfumes for {season} featuring {accord} but without {neg}.",
    "Find {gender} {season} {accord} fragrances excluding {neg}.",
    "List {gender} perfumes for {season} with {accord} accords, but {neg}.",
    "Show {gender} {accord} fragrances for {season} that do not contain {neg}.",
    "Which {gender} {season} perfumes have {accord} but {neg}?",
    "Search for {gender} {accord} perfumes suitable for {season}, {neg}.",
]


# ── TIP A: MULTI FILTER ────────────────────────────────────────────────────

def make_type_a(all_perfumes, use_rag):
    """Birden fazla filtreyi aynı anda uygulayan sorgular."""
    for _ in range(200):
        ref = random.choice(all_perfumes[:8000])
        meta = ref["metadata"]
        gender = meta.get("gender", "unisex")
        seasons = meta.get("best_seasons") or []
        times = meta.get("time_profile") or []
        accords = meta.get("accords_list") or []
        notes = meta.get("notes_list") or []

        if not seasons or not accords:
            continue

        season = random.choice(seasons)
        accord = random.choice(accords[:5])

        # %30 zaman filtresi ekle, %20 nota filtresi ekle
        variant = random.random()

        if variant < 0.30 and times:
            time = random.choice(times)
            criteria = {"gender": gender, "seasons": [season], "time_profile": [time], "accords": [accord]}
            question = random.choice(MULTI_FILTER_WITH_TIME_TEMPLATES).format(
                gender=gender, season=season, time=time, accord=accord
            )
        elif variant < 0.50 and notes:
            note = random.choice(notes[:8])
            criteria = {"gender": gender, "seasons": [season], "accords": [accord], "notes": [note]}
            question = random.choice(MULTI_FILTER_WITH_NOTE_TEMPLATES).format(
                gender=gender, season=season, accord=accord, note=note
            )
        else:
            criteria = {"gender": gender, "seasons": [season], "accords": [accord]}
            question = random.choice(MULTI_FILTER_TEMPLATES).format(
                gender=gender, season=season, accord=accord
            )

        matches = find_matches(all_perfumes, criteria, limit=15)
        if not matches:
            continue

        if use_rag:
            k = min(len(matches), random.randint(1, 5))
            subset = random.sample(matches, k)
            context_list, context_text = build_rag_context(subset, all_perfumes, context_size=12)
            final = [p for p in context_list if match_perfume(p, criteria)]
            if not final:
                continue
            return context_text + question, format_perfume_list(final)
        else:
            final = matches[:random.randint(1, 5)]
            return question, format_perfume_list(final)

    raise RuntimeError("Type A: could not generate sample after 200 attempts")


# ── TIP B: NEGATIVE CONSTRAINTS ────────────────────────────────────────────

def make_type_b(all_perfumes, use_rag):
    """Negatif filtreler — L2'nin kalbi."""
    for _ in range(300):
        ref = random.choice(all_perfumes[:8000])
        meta = ref["metadata"]
        gender = meta.get("gender", "unisex")
        seasons = meta.get("best_seasons") or []
        accords = meta.get("accords_list") or []

        if not seasons or not accords:
            continue

        season = random.choice(seasons)
        accord = random.choice(accords[:5])

        ref_notes_set = meta["_notes_set"]
        ref_accords_set = meta["_accords_set"]

        # Negatif adayları: referans parfümde OLMAYAN yaygın nota/akorlar
        cand_notes = [n for n in COMMON_NOTES if n not in ref_notes_set]
        cand_accords = [a for a in COMMON_ACCORDS if a not in ref_accords_set]

        if not cand_notes:
            continue

        not_notes = []
        not_accords = []

        # İleri seviye negatifler (%30 çoklu negatif)
        if random.random() < 0.30:
            not_notes = random.sample(cand_notes, min(random.randint(2, 3), len(cand_notes)))
            if cand_accords and random.random() < 0.5:
                not_accords = [random.choice(cand_accords)]
        else:
            not_notes = [random.choice(cand_notes)]
            if cand_accords and random.random() < 0.6:
                not_accords = [random.choice(cand_accords)]

        criteria = {
            "gender": gender,
            "seasons": [season],
            "accords": [accord],
            "not_notes": not_notes,
            "not_accords": not_accords,
        }

        matches = find_matches(all_perfumes, criteria, limit=15)
        if not matches:
            continue

        # Negatif string oluştur
        neg_parts = [f"without {n}" for n in not_notes] + [f"not {a}" for a in not_accords]
        neg_str = " and ".join(neg_parts)
        question = random.choice(NEG_TEMPLATES).format(
            gender=gender, season=season, accord=accord, neg=neg_str
        )

        if use_rag:
            k = min(len(matches), random.randint(1, 4))
            subset = random.sample(matches, k)
            context_list, context_text = build_rag_context(subset, all_perfumes, context_size=12)
            final = [p for p in context_list if match_perfume(p, criteria)]
            if not final:
                continue
            validate_negative_constraints(final, not_notes, not_accords)
            return context_text + question, format_perfume_list(final)
        else:
            final = matches[:random.randint(1, 4)]
            validate_negative_constraints(final, not_notes, not_accords)
            return question, format_perfume_list(final)

    raise RuntimeError("Type B: could not generate sample after 300 attempts")


# ── TIP C: RANKING ──────────────────────────────────────────────────────────

RANKING_TEMPLATES = [
    "Show the highest rated {accord} fragrances.",
    "List the top rated {accord} perfumes.",
    "What are the best rated {accord} fragrances in the database?",
    "Show the most popular {accord} perfumes by vote count.",
    "Which {accord} perfumes have the most votes?",
    "List the most popular {gender} {accord} fragrances.",
    "Show top rated {gender} perfumes for {season} with {accord} accords.",
    "What are the highest rated {season} {accord} fragrances?",
]


def make_type_c(all_perfumes, use_rag):
    """Sıralama sorguları: filter + sort."""
    for _ in range(200):
        accord = random.choice(COMMON_ACCORDS[:20])
        sort_by = random.choice(["rating", "popularity"])

        # %50 ek filtre ekle (gender + season)
        add_filters = random.random() < 0.50
        if add_filters:
            ref = random.choice(all_perfumes[:5000])
            gender = ref["metadata"].get("gender", "unisex")
            seasons = ref["metadata"].get("best_seasons") or []
            season = random.choice(seasons) if seasons else None
            criteria = {"accords": [accord], "gender": gender, "min_rating": 3.0, "min_votes": 10}
            if season:
                criteria["seasons"] = [season]
            
            if sort_by == "rating":
                template = random.choice([
                    "Show top rated {gender} perfumes for {season} with {accord} accords.",
                    "What are the highest rated {season} {accord} fragrances?",
                ])
            else:
                template = random.choice([
                    "List the most popular {gender} {season} {accord} fragrances.",
                    "Which {season} {accord} perfumes have the most votes?",
                ])
            question = template.format(gender=gender, season=season or "all-season", accord=accord)
        else:
            gender = random.choice(["male", "female", "unisex"])
            criteria = {"accords": [accord], "gender": gender, "min_rating": 3.0, "min_votes": 10}
            if sort_by == "rating":
                template = random.choice([
                    "Show the highest rated {accord} fragrances.",
                    "List the top rated {accord} perfumes.",
                    "What are the best rated {accord} fragrances in the database?",
                ])
            else:
                template = random.choice([
                    "Show the most popular {accord} perfumes by vote count.",
                    "Which {accord} perfumes have the most votes?",
                    "List the most popular {gender} {accord} fragrances.",
                ])
            question = template.format(gender=gender, accord=accord)

        matches = find_matches(all_perfumes, criteria, limit=50)
        if len(matches) < 3:
            continue

        # Sıralama
        if sort_by == "rating":
            matches.sort(key=lambda x: (x["metadata"].get("rating") or 0), reverse=True)
        else:
            matches.sort(key=lambda x: x["metadata"].get("popularity", 0), reverse=True)

        top_n = matches[:random.randint(3, 5)]

        if use_rag:
            context_list, context_text = build_rag_context(top_n, all_perfumes, context_size=12)
            # Sadece önceden sıralanmış top_n'i kullan (dolgu parfümlerini dahil etme)
            top_n_ids = set(id(p) for p in top_n)
            final = [p for p in top_n if id(p) in top_n_ids]
            if not final:
                continue
            return context_text + question, format_perfume_list(final)
        else:
            return question, format_perfume_list(top_n)

    raise RuntimeError("Type C: could not generate sample after 200 attempts")


# ── TIP D: NUMERIC FILTERS ─────────────────────────────────────────────────

NUMERIC_TEMPLATES = [
    "Show perfumes with a rating above {min_rating}.",
    "List fragrances rated higher than {min_rating} with at least {min_votes} votes.",
    "Find {accord} perfumes with a rating above {min_rating}.",
    "Show {gender} perfumes rated above {min_rating} for {season}.",
    "List perfumes with more than {min_votes} votes.",
    "Find {accord} fragrances with more than {min_votes} votes and rating above {min_rating}.",
    "Which {season} perfumes have a rating above {min_rating}?",
]


def make_type_d(all_perfumes, use_rag):
    """Sayısal filtre sorguları: rating > X, votes > Y."""
    for _ in range(200):
        min_rating = round(random.uniform(3.8, 4.5), 1)
        min_votes = random.choice([50, 100, 500, 1000, 5000])

        # %60 ek filtre ekle
        add_filters = random.random() < 0.60
        if add_filters:
            accord = random.choice(COMMON_ACCORDS[:15])
            gender = random.choice(["male", "female"])
            ref = random.choice(all_perfumes[:3000])
            seasons = ref["metadata"].get("best_seasons") or []
            season = random.choice(seasons) if seasons else None

            criteria = {"accords": [accord], "gender": gender, "min_rating": min_rating, "min_votes": min_votes}
            if season:
                criteria["seasons"] = [season]

            template = random.choice(NUMERIC_TEMPLATES[2:])
            question = template.format(
                accord=accord, gender=gender, season=season or "any",
                min_rating=min_rating, min_votes=min_votes
            )
        else:
            criteria = {"min_rating": min_rating, "min_votes": min_votes}
            template = random.choice(NUMERIC_TEMPLATES[:3])
            question = template.format(min_rating=min_rating, min_votes=min_votes, accord=random.choice(COMMON_ACCORDS[:10]))

        matches = find_matches(all_perfumes, criteria, limit=20)
        if not matches:
            continue

        selected = matches[:random.randint(2, 5)]

        if use_rag:
            context_list, context_text = build_rag_context(selected, all_perfumes, context_size=12)
            final = [p for p in context_list if match_perfume(p, criteria)]
            if not final:
                continue
            return context_text + question, format_perfume_list(final)
        else:
            return question, format_perfume_list(selected)

    raise RuntimeError("Type D: could not generate sample after 200 attempts")


# ── TIP E: COMPARISON ──────────────────────────────────────────────────────

COMPARISON_TEMPLATES = {
    "rating": [
        "Which perfume has a higher rating, {name1} by {brand1} or {name2} by {brand2}?",
        "Compare the ratings of {name1} by {brand1} and {name2} by {brand2}.",
    ],
    "popularity": [
        "Which is more popular, {name1} by {brand1} or {name2} by {brand2}?",
        "Compare the vote counts of {name1} by {brand1} and {name2} by {brand2}.",
    ],
    "accords": [
        "How do the accords of {name1} by {brand1} and {name2} by {brand2} differ?",
        "Compare the accord profiles of {name1} by {brand1} versus {name2} by {brand2}.",
    ],
    "seasons": [
        "Which seasons suit {name1} by {brand1} vs {name2} by {brand2}?",
        "Compare the seasonal suitability of {name1} by {brand1} and {name2} by {brand2}.",
    ],
}


def make_type_e(all_perfumes, use_rag):
    """Objektif karşılaştırma — sadece sayısal ve faktüel."""
    for _ in range(200):
        # Popüler parfümlerden iki farklısı
        pool = all_perfumes[:3000]
        p1, p2 = random.sample(pool, 2)
        m1, m2 = p1["metadata"], p2["metadata"]

        comp_type = random.choice(list(COMPARISON_TEMPLATES.keys()))
        template = random.choice(COMPARISON_TEMPLATES[comp_type])
        question = template.format(
            name1=p1["name"], brand1=p1["brand"],
            name2=p2["name"], brand2=p2["brand"]
        )

        # Cevap oluştur — tamamen objektif
        if comp_type == "rating":
            r1 = m1.get("rating")
            r2 = m2.get("rating")
            pop1, pop2 = m1.get("popularity", 0), m2.get("popularity", 0)
            r1_str = f"{r1:.2f}/5 ({pop1} votes)" if (r1 and r1 > 0 and pop1 > 0) else "N/A"
            r2_str = f"{r2:.2f}/5 ({pop2} votes)" if (r2 and r2 > 0 and pop2 > 0) else "N/A"
            r1_val = r1 if (r1 and r1 > 0 and pop1 > 0) else None
            r2_val = r2 if (r2 and r2 > 0 and pop2 > 0) else None

            if r1_val is None and r2_val is None:
                answer = f"Neither {p1['name']} nor {p2['name']} has a recorded rating."
            elif r1_val is None:
                answer = f"{p2['name']} by {p2['brand']} is rated {r2_str}. {p1['name']} by {p1['brand']} has no rating recorded."
            elif r2_val is None:
                answer = f"{p1['name']} by {p1['brand']} is rated {r1_str}. {p2['name']} by {p2['brand']} has no rating recorded."
            elif r1_val > r2_val:
                answer = f"{p1['name']} by {p1['brand']} has a higher rating of {r1_str} compared to {p2['name']} by {p2['brand']} with {r2_str}."
            elif r2_val > r1_val:
                answer = f"{p2['name']} by {p2['brand']} has a higher rating of {r2_str} compared to {p1['name']} by {p1['brand']} with {r1_str}."
            else:
                answer = f"Both have the same rating of {r1_str}."

        elif comp_type == "popularity":
            pop1, pop2 = m1.get("popularity", 0), m2.get("popularity", 0)
            if pop1 > pop2:
                answer = f"{p1['name']} by {p1['brand']} is more popular with {pop1} votes compared to {p2['name']} by {p2['brand']} with {pop2} votes."
            elif pop2 > pop1:
                answer = f"{p2['name']} by {p2['brand']} is more popular with {pop2} votes compared to {p1['name']} by {p1['brand']} with {pop1} votes."
            else:
                answer = f"Both have equal popularity with {pop1} votes each."

        elif comp_type == "accords":
            acc1 = m1.get("accords_list") or []
            acc2 = m2.get("accords_list") or []
            answer = (
                f"{p1['name']} by {p1['brand']} accords: {', '.join(acc1[:6]) if acc1 else 'N/A'}\n"
                f"{p2['name']} by {p2['brand']} accords: {', '.join(acc2[:6]) if acc2 else 'N/A'}"
            )

        else:  # seasons
            s1 = [s.capitalize() for s in (m1.get("best_seasons") or [])]
            s2 = [s.capitalize() for s in (m2.get("best_seasons") or [])]
            answer = (
                f"{p1['name']} by {p1['brand']} seasons: {', '.join(s1) if s1 else 'N/A'}\n"
                f"{p2['name']} by {p2['brand']} seasons: {', '.join(s2) if s2 else 'N/A'}"
            )

        if use_rag:
            cards = []
            if p1.get("card_text"):
                cards.append(p1["card_text"])
            if p2.get("card_text"):
                cards.append(p2["card_text"])
            context_text = f"[PERFUMES]\n" + "\n\n".join(cards) + "\n[/PERFUMES]\n\n"
            return context_text + question, answer
        else:
            return question, answer

    raise RuntimeError("Type E: could not generate sample after 200 attempts")


# ── TIP F: NO MATCH ────────────────────────────────────────────────────────

NO_MATCH_TEMPLATES = [
    "Find perfumes with {accord} accords but without {neg}.",
    "List {gender} {season} perfumes with {note} notes, excluding {neg}.",
    "Show perfumes rated above {min_rating} with {neg}.",
    "Find {accord} perfumes rated above {impossible_rating}.",
]


def make_type_f(all_perfumes, use_rag):
    """İmkansız sorgular — hallucination önleme."""
    for _ in range(200):
        strategy = random.choice(["contradictory", "impossible_rating", "impossible_combo"])

        if strategy == "contradictory":
            # Bir nota/akor iste VE aynı anda reddet
            target = random.choice(COMMON_NOTES[:20])
            accord = random.choice(COMMON_ACCORDS[:15])
            criteria = {"accords": [accord], "notes": [target], "not_notes": [target]}
            question = f"Find perfumes with {accord} accords containing {target} but without {target}."

        elif strategy == "impossible_rating":
            # Rating > 5.0 (imkansız)
            accord = random.choice(COMMON_ACCORDS[:15])
            criteria = {"accords": [accord], "min_rating": 5.1}
            question = f"Show {accord} perfumes rated above 5.0."

        else:  # impossible_combo
            # Çelişkili filtre kombinasyonu
            note1 = random.choice(COMMON_NOTES[:10])
            note2 = random.choice(COMMON_NOTES[25:40])
            neg_notes = random.sample(COMMON_NOTES[:15], min(5, len(COMMON_NOTES)))
            criteria = {"notes": [note1, note2], "not_notes": neg_notes, "min_rating": 4.5, "min_votes": 5000}
            neg_str = ", ".join(neg_notes)
            question = f"Find perfumes with {note1} and {note2} notes, excluding {neg_str}, rated above 4.5 with 5000+ votes."

        matches = find_matches(all_perfumes, criteria, limit=5)
        if len(matches) > 0:
            continue  # Sonuç varsa bu strateji işe yaramadı, tekrar dene

        validate_no_match(matches)

        if use_rag:
            # RAG context ver ama sonuç yok
            fillers = random.sample(all_perfumes[:5000], min(10, len(all_perfumes)))
            cards = [p["card_text"] for p in fillers]
            context_text = f"[PERFUMES]\n" + "\n\n".join(cards) + "\n[/PERFUMES]\n\n"
            return context_text + question, "No matching perfumes found."
        else:
            return question, "No matching perfumes found."

    raise RuntimeError("Type F: could not generate sample after 200 attempts")


# ── ANA FONKSİYON ──────────────────────────────────────────────────────────

GENERATORS = {
    "multi_filter": make_type_a,
    "negative_constraints": make_type_b,
    "ranking": make_type_c,
    "numeric_filters": make_type_d,
    "comparison": make_type_e,
    "no_match": make_type_f,
}


def generate_dataset():
    global COMMON_ACCORDS, COMMON_NOTES

    if not os.path.exists(CLEAN_FILE):
        print(f"Error: {CLEAN_FILE} not found")
        return

    print("Loading dataset...")
    all_perfumes = load_perfumes()
    print(f"Loaded {len(all_perfumes)} perfumes.")

    COMMON_ACCORDS, COMMON_NOTES = build_frequency_lists(all_perfumes)

    # Hedef sayıları hesapla
    if IS_BASELINE:
        total_target = 2250
    else:
        total_target = 4500

    print(f"Target count: {total_target} (IS_BASELINE = {IS_BASELINE})")

    category_counts = {}
    for cat, ratio in CATEGORY_RATIOS.items():
        category_counts[cat] = round(total_target * ratio)

    # Yuvarlama farkını düzelt
    diff = total_target - sum(category_counts.values())
    if diff != 0:
        category_counts["multi_filter"] += diff

    records = []

    for cat, target in category_counts.items():
        gen_func = GENERATORS[cat]
        print(f"Generating category '{cat}' ({target} records)...")
        for i in range(target):
            if (i + 1) % 500 == 0:
                print(f"  Progress: {i + 1}/{target}")

            use_rag = random.random() < 0.80
            user_content, answer = gen_func(all_perfumes, use_rag)
            records.append(build_messages(user_content, answer))

        print(f"  Progress: {target}/{target}")

    random.shuffle(records)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    rag_count = sum(1 for r in records if "[PERFUMES]" in r["messages"][1]["content"])

    print(f"\n✅ L2 Dataset Generation Complete!")
    print(f"   Total records : {len(records)}")
    print(f"   Output        : {OUTPUT_FILE}")
    print(f"   RAG ratio     : {rag_count}/{len(records)} ({rag_count/len(records)*100:.1f}%)")
    print(f"   Category breakdown:")
    for cat, cnt in category_counts.items():
        print(f"     {cat:25s}: {cnt}")


if __name__ == "__main__":
    generate_dataset()
