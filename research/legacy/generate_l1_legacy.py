"""
generate_l1_v2.py — ScentAI L1 Training Data Generator (v2)

Mevcut versiyona göre düzeltmeler:
1. Info cevabı artık longevity/sillage/value/accords/notes de içeriyor
2. Accord sırası context varsa strength'e göre, yoksa metadata'dan alınıyor
3. Similar (benzer) ve Comparison (karşılaştırma) kategorileri eklendi
4. RAG durumunda cevap context'ten türetiliyor (metadata'dan değil)
5. Soru template'leri genişletildi (her kategori için 5-7 varyasyon)
6. Hedef: 15K kayıt (tasarım dökümanıyla uyumlu)
"""

import json
import random
import os
import re

IS_BASELINE = True
CLEAN_FILE = "/home/sefasys/Desktop/Perfume-Dataset/perfumes_clean.jsonl"
OUTPUT_FILE = "/home/sefasys/Desktop/Perfume-Dataset/baseline_L1.jsonl" if IS_BASELINE else "/home/sefasys/Desktop/Perfume-Dataset/training_L1.jsonl"

SYSTEM_PROMPT = (
    "You are a professional perfume database assistant. "
    "Answer the query based strictly on the provided perfume context if available, "
    "otherwise state the database facts. "
    "Avoid making interpretations, recommendations, or qualitative assertions."
)


# ────────────────────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ────────────────────────────────────────────────────────────────────────────

def load_perfumes():
    perfumes_list = []
    with open(CLEAN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            perfumes_list.append(json.loads(line))
    return perfumes_list


def parse_card_accords(card_text):
    """card_text'ten strength sırasıyla accord listesi çıkar."""
    for line in card_text.split("\n"):
        if line.startswith("Accords:"):
            accords = []
            for part in line.replace("Accords:", "").split(","):
                part = part.strip()
                # "citrus (100%)" → "citrus"
                name = re.split(r"\s*\(", part)[0].strip()
                if name:
                    accords.append(name)
            return accords
    return []


def parse_card_notes(card_text):
    """card_text'ten top/mid/base notaları çıkar, düz liste döner."""
    notes = []
    for line in card_text.split("\n"):
        matched = False
        for prefix in ("Top Notes:", "Middle Notes:", "Base Notes:"):
            if prefix in line:
                segment = line.split(prefix, 1)[1].strip()
                # "| Middle Notes:" gibi devam varsa kes
                segment = segment.split("|")[0].strip()
                notes.extend([n.strip() for n in segment.split(",") if n.strip()])
                matched = True
        if not matched and "Notes:" in line:
            segment = line.split("Notes:", 1)[1].strip()
            segment = segment.split("|")[0].strip()
            notes.extend([n.strip() for n in segment.split(",") if n.strip()])
    return notes



def parse_card_metrics(card_text):
    """Rating satırından longevity/sillage/value değerlerini çıkar."""
    for line in card_text.split("\n"):
        if line.startswith("Rating:"):
            lon = sil = val = None
            m = re.search(r"Longevity:\s*([\d.]+)/5", line)
            if m:
                lon = float(m.group(1))
            m = re.search(r"Sillage:\s*([\d.]+)/4", line)
            if m:
                sil = float(m.group(1))
            m = re.search(r"Value:\s*([\d.]+)/5", line)
            if m:
                val = float(m.group(1))
            return lon, sil, val
    return None, None, None


def fmt_list(items):
    return ", ".join(items) if items else "none recorded"


def fmt_val(v, suffix=""):
    return f"{v:.2f}{suffix}" if (v is not None and v > 0.0) else "N/A"


# ────────────────────────────────────────────────────────────────────────────
# KATEGORİ 1: INFO — tam veritabanı kaydı
# ────────────────────────────────────────────────────────────────────────────

INFO_QUESTIONS = [
    "Provide the database record for {name} by {brand}.",
    "Show the full specifications of {name} by {brand}.",
    "What are the recorded parameters for {name} by {brand}?",
    "Give me all stored data for {name} by {brand}.",
    "Retrieve the complete database entry for {name} by {brand}.",
    "What does the database say about {name} by {brand}?",
    "Display all fields for {name} by {brand}.",
]


def generate_info_qa(p, use_rag):
    name, brand = p["name"], p["brand"]
    meta = p["metadata"]
    card = p.get("card_text", "")

    question = random.choice(INFO_QUESTIONS).format(name=name, brand=brand)

    year_val = str(meta.get("year")) if meta.get("year") else "N/A"
    rating = meta.get("rating")
    popularity = meta.get("popularity", 0)
    rating_str = f"{rating:.2f}/5 ({popularity} votes)" if (rating is not None and rating > 0.0 and popularity > 0) else "N/A"
    gender = meta.get("gender", "N/A")

    seasons = meta.get("best_seasons") or []
    seasons_str = fmt_list([s.capitalize() for s in seasons])

    times = meta.get("time_profile") or []
    times_str = fmt_list([t.capitalize() for t in times])

    # Accord sırası: context varsa strength'e göre, yoksa metadata'dan
    if use_rag and card:
        accords = parse_card_accords(card)
        notes = parse_card_notes(card)
        lon, sil, val = parse_card_metrics(card)
    else:
        accords = meta.get("accords_list") or []
        notes = meta.get("notes_list") or []
        lon = meta.get("longevity")
        sil = meta.get("sillage")
        val = meta.get("price_value")

    answer_lines = [
        f"Database Record — {name} by {brand}:",
        f"- Brand: {brand}",
        f"- Name: {name}",
        f"- Gender: {gender}",
        f"- Launch Year: {year_val}",
        f"- Rating: {rating_str}",
        f"- Longevity: {fmt_val(lon, '/5')}",
        f"- Sillage: {fmt_val(sil, '/4')}",
        f"- Value: {fmt_val(val, '/5')}",
        f"- Best Seasons: {seasons_str}",
        f"- Time Profile: {times_str}",
    ]
    if accords:
        answer_lines.append(f"- Accords: {fmt_list(accords)}")
    if notes:
        answer_lines.append(f"- Notes: {fmt_list(notes)}")

    perfumer = meta.get("perfumer")
    if perfumer:
        answer_lines.append(f"- Perfumer: {perfumer}")

    return question, "\n".join(answer_lines)


# ────────────────────────────────────────────────────────────────────────────
# KATEGORİ 2: NOTES
# ────────────────────────────────────────────────────────────────────────────

NOTES_QUESTIONS = [
    "List all notes present in {name} by {brand}.",
    "What notes are recorded for {name} by {brand}?",
    "What notes are listed in {name} by {brand}?",
    "Show the note breakdown for {name} by {brand}.",
    "What are the top, middle, and base notes of {name} by {brand}?",
    "Which notes does {name} by {brand} contain according to the database?",
]


def generate_notes_qa(p, use_rag):
    name, brand = p["name"], p["brand"]
    card = p.get("card_text", "")
    meta = p["metadata"]

    question = random.choice(NOTES_QUESTIONS).format(name=name, brand=brand)

    if use_rag and card:
        # Tiered format varsa ayrı ayrı listele
        top_notes = []
        mid_notes = []
        base_notes = []
        flat_notes = []
        for line in card.split("\n"):
            if "Top Notes:" in line:
                seg = line.split("Top Notes:")[1].split("|")[0].strip()
                top_notes = [n.strip() for n in seg.split(",") if n.strip()]
            if "Middle Notes:" in line:
                seg = line.split("Middle Notes:")[1].split("|")[0].strip()
                mid_notes = [n.strip() for n in seg.split(",") if n.strip()]
            if "Base Notes:" in line:
                seg = line.split("Base Notes:")[1].split("|")[0].strip()
                base_notes = [n.strip() for n in seg.split(",") if n.strip()]
            if line.startswith("Notes:"):
                seg = line.replace("Notes:", "").strip()
                flat_notes = [n.strip() for n in seg.split(",") if n.strip()]

        if top_notes or mid_notes or base_notes:
            parts = []
            if top_notes:
                parts.append(f"Top: {', '.join(top_notes)}")
            if mid_notes:
                parts.append(f"Middle: {', '.join(mid_notes)}")
            if base_notes:
                parts.append(f"Base: {', '.join(base_notes)}")
            answer = f"The notes recorded for {name} by {brand} are:\n" + "\n".join(parts)
        elif flat_notes:
            answer = f"The recorded notes for {name} by {brand} are: {fmt_list(flat_notes)}."
        else:
            answer = f"There are no notes recorded in the database for {name} by {brand}."
    else:
        notes = meta.get("notes_list") or []
        if notes:
            answer = f"The recorded notes for {name} by {brand} are: {fmt_list(notes)}."
        else:
            answer = f"There are no notes recorded in the database for {name} by {brand}."

    return question, answer


# ────────────────────────────────────────────────────────────────────────────
# KATEGORİ 3: ACCORDS
# ────────────────────────────────────────────────────────────────────────────

ACCORDS_QUESTIONS = [
    "What accords are present in {name} by {brand}?",
    "List the accord profile of {name} by {brand}.",
    "Show the accords for {name} by {brand}.",
    "What is the accord breakdown of {name} by {brand}?",
    "Which fragrance families does {name} by {brand} belong to?",
    "What are the dominant accords in {name} by {brand} according to the database?",
]


def generate_accords_qa(p, use_rag):
    name, brand = p["name"], p["brand"]
    card = p.get("card_text", "")
    meta = p["metadata"]

    question = random.choice(ACCORDS_QUESTIONS).format(name=name, brand=brand)

    # Accord sırası: context varsa strength'e göre (card'daki sıra), yoksa metadata
    if use_rag and card:
        accords = parse_card_accords(card)
    else:
        accords = meta.get("accords_list") or []

    if accords:
        answer = f"The accord profile for {name} by {brand} consists of: {fmt_list(accords)}."
    else:
        answer = f"There is no accord profile recorded in the database for {name} by {brand}."

    return question, answer


# ────────────────────────────────────────────────────────────────────────────
# KATEGORİ 4: SEASONS
# ────────────────────────────────────────────────────────────────────────────

SEASONS_QUESTIONS = [
    "Which seasons are voted suitable for {name} by {brand}?",
    "What is the season profile of {name} by {brand}?",
    "What seasons are recommended for {name} by {brand} according to community votes?",
    "What seasons does the database list for {name} by {brand}?",
    "Is {name} by {brand} a summer, spring, autumn, or winter fragrance?",
    "What seasons of the year is {name} by {brand} recommended for?",
    "Show the seasonal suitability data for {name} by {brand}.",
]

TIME_QUESTIONS = [
    "Is {name} by {brand} a day or night fragrance?",
    "What time of day is {name} by {brand} best suited for?",
    "Show the day/night time profile for {name} by {brand}.",
    "Is {name} by {brand} recommended for daytime or evening wear?",
]


def generate_seasons_qa(p, use_rag):
    name, brand = p["name"], p["brand"]
    meta = p["metadata"]
    seasons = meta.get("best_seasons") or []
    times = meta.get("time_profile") or []

    # Rastgele season veya time sorusu
    if random.random() < 0.7 or not times:
        question = random.choice(SEASONS_QUESTIONS).format(name=name, brand=brand)
        if seasons:
            answer = f"The suitable seasons for {name} by {brand} are: {fmt_list([s.capitalize() for s in seasons])}."
        else:
            answer = f"There are no specific seasons recommended in the database for {name} by {brand}."
    else:
        question = random.choice(TIME_QUESTIONS).format(name=name, brand=brand)
        if times:
            answer = f"The time profile for {name} by {brand} is: {fmt_list([t.capitalize() for t in times])}."
        else:
            answer = f"There is no time profile recorded in the database for {name} by {brand}."

    return question, answer


# ────────────────────────────────────────────────────────────────────────────
# KATEGORİ 5: RATING
# ────────────────────────────────────────────────────────────────────────────

RATING_QUESTIONS = [
    "What is the average rating and vote count for {name} by {brand}?",
    "Show the community rating statistics for {name} by {brand}.",
    "How is {name} by {brand} rated by users?",
    "What rating does {name} by {brand} have in the database?",
    "What are the rating and performance scores for {name} by {brand}?",
    "Show the rating, longevity, sillage, and value scores for {name} by {brand}.",
]


def generate_rating_qa(p, use_rag):
    name, brand = p["name"], p["brand"]
    meta = p["metadata"]
    card = p.get("card_text", "")

    question = random.choice(RATING_QUESTIONS).format(name=name, brand=brand)

    rating = meta.get("rating")
    popularity = meta.get("popularity", 0)

    if use_rag and card:
        lon, sil, val = parse_card_metrics(card)
    else:
        lon = meta.get("longevity")
        sil = meta.get("sillage")
        val = meta.get("price_value")

    if rating is not None and rating > 0.0 and popularity > 0:
        answer = (
            f"{name} by {brand} has a community rating of {rating:.2f}/5 "
            f"based on {popularity} votes.\n"
            f"- Longevity: {fmt_val(lon, '/5')}\n"
            f"- Sillage: {fmt_val(sil, '/4')}\n"
            f"- Value: {fmt_val(val, '/5')}"
        )
    else:
        answer = f"There are no rating statistics recorded for {name} by {brand}."

    return question, answer


# ────────────────────────────────────────────────────────────────────────────
# KATEGORİ 6: COMPARISON — iki parfümü karşılaştır
# ────────────────────────────────────────────────────────────────────────────

COMPARISON_QUESTIONS = {
    "rated_higher":  "which is rated higher, {name1} by {brand1} or {name2} by {brand2}?",
    "lasts_longer":  "which has better longevity, {name1} by {brand1} or {name2} by {brand2}?",
    "more_sillage":  "which has stronger sillage, {name1} by {brand1} or {name2} by {brand2}?",
    "general":       "compare {name1} by {brand1} and {name2} by {brand2}.",
    "accords_diff":  "how do the accords of {name1} by {brand1} and {name2} by {brand2} differ?",
    "seasons_diff":  "which seasons suit {name1} by {brand1} vs {name2} by {brand2}?",
}


def _perfume_summary(p, card=None):
    """Tek parfüm için karşılaştırma satırları üret."""
    name, brand = p["name"], p["brand"]
    meta = p["metadata"]

    if card:
        accords = parse_card_accords(card)
        lon, sil, val = parse_card_metrics(card)
    else:
        accords = meta.get("accords_list") or []
        lon = meta.get("longevity")
        sil = meta.get("sillage")
        val = meta.get("price_value")

    rating = meta.get("rating")
    pop = meta.get("popularity", 0)
    seasons = meta.get("best_seasons") or []

    rating_str = f"{rating:.2f}/5 ({pop} votes)" if (rating is not None and rating > 0.0 and pop > 0) else "N/A"

    return (
        f"{name} by {brand}:\n"
        f"  Accords: {fmt_list(accords[:4])}\n"
        f"  Rating: {rating_str}\n"
        f"  Longevity: {fmt_val(lon, '/5')} | Sillage: {fmt_val(sil, '/4')} | Value: {fmt_val(val, '/5')}\n"
        f"  Best Seasons: {fmt_list([s.capitalize() for s in seasons])}"
    )


def generate_comparison_qa(p1, p2, use_rag):
    name1, brand1 = p1["name"], p1["brand"]
    name2, brand2 = p2["name"], p2["brand"]

    q_type = random.choice(list(COMPARISON_QUESTIONS.keys()))
    question = COMPARISON_QUESTIONS[q_type].format(
        name1=name1, brand1=brand1, name2=name2, brand2=brand2
    )

    card1 = p1.get("card_text", "") if use_rag else None
    card2 = p2.get("card_text", "") if use_rag else None

    # Değerleri karşılaştırmak için çekelim
    r1 = p1["metadata"].get("rating")
    r2 = p2["metadata"].get("rating")
    pop1 = p1["metadata"].get("popularity", 0)
    pop2 = p2["metadata"].get("popularity", 0)

    if use_rag and card1:
        accords1 = parse_card_accords(card1)
        lon1, sil1, _ = parse_card_metrics(card1)
    else:
        accords1 = p1["metadata"].get("accords_list") or []
        lon1 = p1["metadata"].get("longevity")
        sil1 = p1["metadata"].get("sillage")

    if use_rag and card2:
        accords2 = parse_card_accords(card2)
        lon2, sil2, _ = parse_card_metrics(card2)
    else:
        accords2 = p2["metadata"].get("accords_list") or []
        lon2 = p2["metadata"].get("longevity")
        sil2 = p2["metadata"].get("sillage")

    seasons1 = p1["metadata"].get("best_seasons") or []
    seasons2 = p2["metadata"].get("best_seasons") or []

    # Valid values check (values <= 0.0 or 0 popularity are treated as None/N/A)
    r1_val_compare = r1 if (r1 is not None and r1 > 0.0 and pop1 > 0) else None
    r2_val_compare = r2 if (r2 is not None and r2 > 0.0 and pop2 > 0) else None

    lon1_val_compare = lon1 if (lon1 is not None and lon1 > 0.0) else None
    lon2_val_compare = lon2 if (lon2 is not None and lon2 > 0.0) else None

    sil1_val_compare = sil1 if (sil1 is not None and sil1 > 0.0) else None
    sil2_val_compare = sil2 if (sil2 is not None and sil2 > 0.0) else None

    # Formatlama
    r1_val = f"{r1:.2f}/5 ({pop1} votes)" if r1_val_compare is not None else "N/A"
    r2_val = f"{r2:.2f}/5 ({pop2} votes)" if r2_val_compare is not None else "N/A"
    lon1_val = fmt_val(lon1_val_compare, "/5")
    sil1_val = fmt_val(sil1_val_compare, "/4")
    lon2_val = fmt_val(lon2_val_compare, "/5")
    sil2_val = fmt_val(sil2_val_compare, "/4")

    if q_type == "rated_higher":
        if r1_val_compare is None and r2_val_compare is None:
            answer = f"Both {name1} by {brand1} and {name2} by {brand2} do not have community rating statistics recorded."
        elif r1_val_compare is None:
            answer = f"{name2} by {brand2} is rated higher with {r2_val} (since {name1} by {brand1} has no rating recorded)."
        elif r2_val_compare is None:
            answer = f"{name1} by {brand1} is rated higher with {r1_val} (since {name2} by {brand2} has no rating recorded)."
        elif r1_val_compare > r2_val_compare:
            answer = f"{name1} by {brand1} is rated higher with {r1_val} compared to {name2} by {brand2} with {r2_val}."
        elif r2_val_compare > r1_val_compare:
            answer = f"{name2} by {brand2} is rated higher with {r2_val} compared to {name1} by {brand1} with {r1_val}."
        else:
            answer = f"Both {name1} by {brand1} and {name2} by {brand2} have the same rating of {r1_val}."

    elif q_type == "lasts_longer":
        if lon1_val_compare is None and lon2_val_compare is None:
            answer = f"Longevity information is not recorded for either {name1} by {brand1} or {name2} by {brand2}."
        elif lon1_val_compare is None:
            answer = f"{name2} by {brand2} has better longevity with a score of {lon2_val} (since {name1} by {brand1} is not recorded)."
        elif lon2_val_compare is None:
            answer = f"{name1} by {brand1} has better longevity with a score of {lon1_val} (since {name2} by {brand2} is not recorded)."
        elif lon1_val_compare > lon2_val_compare:
            answer = f"{name1} by {brand1} has better longevity with a score of {lon1_val} compared to {name2} by {brand2} with {lon2_val}."
        elif lon2_val_compare > lon1_val_compare:
            answer = f"{name2} by {brand2} has better longevity with a score of {lon2_val} compared to {name1} by {brand1} with {lon1_val}."
        else:
            answer = f"Both {name1} by {brand1} and {name2} by {brand2} have the same longevity score of {lon1_val}."

    elif q_type == "more_sillage":
        if sil1_val_compare is None and sil2_val_compare is None:
            answer = f"Sillage information is not recorded for either {name1} by {brand1} or {name2} by {brand2}."
        elif sil1_val_compare is None:
            answer = f"{name2} by {brand2} has stronger sillage with a score of {sil2_val} (since {name1} by {brand1} is not recorded)."
        elif sil2_val_compare is None:
            answer = f"{name1} by {brand1} has stronger sillage with a score of {sil1_val} (since {name2} by {brand2} is not recorded)."
        elif sil1_val_compare > sil2_val_compare:
            answer = f"{name1} by {brand1} has stronger sillage with a score of {sil1_val} compared to {name2} by {brand2} with {sil2_val}."
        elif sil2_val_compare > sil1_val_compare:
            answer = f"{name2} by {brand2} has stronger sillage with a score of {sil2_val} compared to {name1} by {brand1} with {sil1_val}."
        else:
            answer = f"Both {name1} by {brand1} and {name2} by {brand2} have the same sillage score of {sil1_val}."

    elif q_type == "accords_diff":
        a1_str = fmt_list(accords1) if accords1 else "none recorded"
        a2_str = fmt_list(accords2) if accords2 else "none recorded"
        answer = (
            f"The accords of the two perfumes differ as follows:\n"
            f"- {name1} by {brand1} accords: {a1_str}\n"
            f"- {name2} by {brand2} accords: {a2_str}"
        )

    elif q_type == "seasons_diff":
        s1_str = fmt_list([s.capitalize() for s in seasons1]) if seasons1 else "none recorded"
        s2_str = fmt_list([s.capitalize() for s in seasons2]) if seasons2 else "none recorded"
        answer = (
            f"The suitable seasons comparison shows:\n"
            f"- {name1} by {brand1} is best worn during: {s1_str}\n"
            f"- {name2} by {brand2} is best worn during: {s2_str}"
        )

    else: # general
        summary1 = _perfume_summary(p1, card1)
        summary2 = _perfume_summary(p2, card2)
        answer = f"Database Comparison:\n\n{summary1}\n\n{summary2}"

    context_card = None
    if use_rag:
        card_text1 = p1.get("card_text", "")
        card_text2 = p2.get("card_text", "")
        if card_text1 or card_text2:
            context_card = f"{card_text1}\n\n{card_text2}".strip()

    return question, answer, context_card


# ────────────────────────────────────────────────────────────────────────────
# MESAJ OLUŞTURUCU
# ────────────────────────────────────────────────────────────────────────────

def build_messages(question, answer, context_card=None):
    user_content = ""
    if context_card:
        user_content += f"[PERFUMES]\n{context_card}\n[/PERFUMES]\n\n"
    user_content += question

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "model", "content": answer},
        ]
    }


# ────────────────────────────────────────────────────────────────────────────
# ANA FONKSİYON
# ────────────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(CLEAN_FILE):
        print(f"Error: Clean file not found at {CLEAN_FILE}")
        return

    print("Loading cleaned dataset...")
    perfumes_list = load_perfumes()
    print(f"Loaded {len(perfumes_list)} perfumes.")

    # Tasarım dökümanına göre hedef: 4K kayıt (veya baseline için 2K)
    if IS_BASELINE:
        category_targets = {
            "info": 400,
            "notes": 320,
            "accords": 320,
            "seasons": 320,
            "rating": 320,
            "comparison": 320,
        }
    else:
        category_targets = {
            "info": 800,
            "notes": 640,
            "accords": 640,
            "seasons": 640,
            "rating": 640,
            "comparison": 640,
        }
    total_needed = sum(category_targets.values())

    generated_records = []
    counts = {k: 0 for k in category_targets}

    # Kategori listesi, hedeflere göre genişlet
    category_schedule = []
    for cat, count in category_targets.items():
        category_schedule.extend([cat] * count)
    random.shuffle(category_schedule)

    print(f"Generating {total_needed} L1 samples...")

    for idx, category in enumerate(category_schedule):
        if idx % 1000 == 0:
            print(f"  Progress: {idx}/{total_needed}")

        use_rag = random.random() < 0.60
        p = random.choice(perfumes_list)

        if category == "info":
            q, a = generate_info_qa(p, use_rag)
            context = p.get("card_text") if use_rag else None
            record = build_messages(q, a, context)

        elif category == "notes":
            q, a = generate_notes_qa(p, use_rag)
            context = p.get("card_text") if use_rag else None
            record = build_messages(q, a, context)

        elif category == "accords":
            q, a = generate_accords_qa(p, use_rag)
            context = p.get("card_text") if use_rag else None
            record = build_messages(q, a, context)

        elif category == "seasons":
            q, a = generate_seasons_qa(p, use_rag)
            context = p.get("card_text") if use_rag else None
            record = build_messages(q, a, context)

        elif category == "rating":
            q, a = generate_rating_qa(p, use_rag)
            context = p.get("card_text") if use_rag else None
            record = build_messages(q, a, context)

        elif category == "comparison":
            p2 = random.choice(perfumes_list)
            while p2["name"] == p["name"]:
                p2 = random.choice(perfumes_list)
            q, a, context = generate_comparison_qa(p, p2, use_rag)
            record = build_messages(q, a, context)

        generated_records.append(record)
        counts[category] += 1

    # Shuffle
    random.shuffle(generated_records)

    # Yazma
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for rec in generated_records:
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ L1 Dataset Generation Complete!")
    print(f"   Total records : {len(generated_records)}")
    print(f"   Output        : {OUTPUT_FILE}")
    print(f"   Category breakdown:")
    for cat, cnt in counts.items():
        print(f"     {cat:12s}: {cnt}")

    # Hızlı kalite kontrolü
    rag_count = sum(1 for r in generated_records if "[PERFUMES]" in r["messages"][1]["content"])
    print(f"\n   RAG ratio     : {rag_count}/{total_needed} ({rag_count/total_needed*100:.0f}%)")


if __name__ == "__main__":
    main()
