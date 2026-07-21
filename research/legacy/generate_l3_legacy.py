"""
generate_l3.py — ScentAI L3 Training Data Generator

L3 is not a reasoning dataset.
L3 is a semantic translation dataset.

Felsefe: İnsan dili -> parfüm özellikleri -> öneri eşleşmesini öğretmek.
Gemini sadece soruyu (user query) üretmek için kullanılır. Cevap her zaman programatik ve güvenilirdir.
Cevap hiçbir açıklama, yorum veya giriş cümlesi (intro) içermez. Doğrudan listeye geçer.

Kategoriler:
Tip A (Casual): %25
Tip B (Reference): %20
Tip C (Likes/Dislikes): %20
Tip D (Messy): %20
Tip E (Contradiction): %15

Hedef: 2500 Kayıt (%70 RAG, %30 Non-RAG).
"""

import json
import random
import os
import re
import time
from google import genai
from google.genai import types

# ── CONFIG ──────────────────────────────────────────────────────────────────
IS_BASELINE = True
CLEAN_FILE = "/home/sefasys/Desktop/Perfume-Dataset/perfumes_clean.jsonl"
OUTPUT_FILE = (
    "/home/sefasys/Desktop/Perfume-Dataset/baseline_L3.jsonl"
    if IS_BASELINE
    else "/home/sefasys/Desktop/Perfume-Dataset/training_L3.jsonl"
)

TOTAL_TARGET = 50

SYSTEM_PROMPT = (
    "You are ScentAI, a professional perfume assistant. "
    "Interpret the user's casual or vague language into fragrance characteristics, "
    "and provide recommendations strictly from the provided database context (if any) or database facts. "
    "Do not provide reasoning chains, intros, or explain your interpretation. Simply output the matching perfumes directly."
)

API_KEY = os.environ.get("GEMINI_API_KEY")
if API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    client = None

ID_TO_PERFUME = {}

def load_perfumes():
    perfumes = []
    with open(CLEAN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            meta = p["metadata"]
            meta["_seasons_set"] = {s.lower() for s in (meta.get("best_seasons") or [])}
            meta["_times_set"] = {t.lower() for t in (meta.get("time_profile") or [])}
            meta["_accords_set"] = {a.lower() for a in (meta.get("accords_list") or [])}
            meta["_notes_set"] = {n.lower() for n in (meta.get("notes_list") or [])}
            perfumes.append(p)
            ID_TO_PERFUME[p["id"]] = p
            
    # Remove strict sorting by popularity so random.choice gets true diversity
    return perfumes

# ── GEMINI HELPER ──────────────────────────────────────────────────────────

def call_gemini(prompt):
    if not client:
        return "I am looking for something based on: " + prompt.split("User profile: ")[-1]
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.8,
                    max_output_tokens=100,
                )
            )
            time.sleep(4)  # Ensure we stay under 15 RPM for Free Tier
            return response.text.strip().replace('"', '')
        except Exception as e:
            if "503" in str(e):
                print(f"Gemini 503 High Demand... Retrying in {5 * (attempt + 1)} seconds.")
                time.sleep(5 * (attempt + 1))
            else:
                print(f"Gemini API Error: {e}")
                time.sleep(4)
                return "Can you recommend a perfume based on these preferences?"
                
    return "Can you recommend a perfume based on these preferences?"

# ── SCORING LOGIC ──────────────────────────────────────────────────────────

def score_perfume(p, criteria):
    """
    Returns a score. -1 means hard constraint violated.
    > 0 means matched soft constraints.
    """
    meta = p["metadata"]
    score = 0
    
    if criteria.get("gender") and criteria["gender"] != "any":
        q_g = criteria["gender"].lower()
        p_g = meta.get("gender", "").lower()
        if q_g == "male" and p_g not in ("male", "unisex"):
            return -1
        if q_g == "female" and p_g not in ("female", "unisex"):
            return -1
        if q_g == "unisex" and p_g != "unisex":
            return -1

    for forbidden in (criteria.get("not_accords") or []):
        f_low = forbidden.lower()
        if f_low in meta["_accords_set"] or f_low in meta["_notes_set"]:
            return -1
            
    for forbidden in (criteria.get("not_notes") or []):
        f_low = forbidden.lower()
        if f_low in meta["_notes_set"] or f_low in meta["_accords_set"]:
            return -1

    has_soft = False
    matched_soft = False

    if criteria.get("seasons"):
        has_soft = True
        for s in criteria["seasons"]:
            if s.lower() in meta["_seasons_set"]:
                score += 2
                matched_soft = True

    if criteria.get("accords"):
        has_soft = True
        for a in criteria["accords"]:
            if a.lower() in meta["_accords_set"]:
                score += 3
                matched_soft = True

    if criteria.get("notes"):
        has_soft = True
        for n in criteria["notes"]:
            if n.lower() in meta["_notes_set"]:
                score += 3
                matched_soft = True
                
    if has_soft and not matched_soft:
        return -1

    return score

def find_matches(all_perfumes, criteria, limit=10):
    scored = []
    for p in all_perfumes:
        s = score_perfume(p, criteria)
        if s >= 0:
            scored.append((s, p))
            
    # Sort by score descending, then by popularity fallback
    scored.sort(key=lambda x: (x[0], x[1]["metadata"].get("popularity", 0)), reverse=True)
    return [p for s, p in scored[:limit]]

def get_best_from_context(ctx_list, criteria, limit=3, min_score=5):
    scored = []
    for p in ctx_list:
        s = score_perfume(p, criteria)
        if s >= min_score: 
            scored.append((s, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for s, p in scored[:limit]]

def format_l3_answer(perfumes):
    if not perfumes:
        return "I wasn't able to find a strong match from the current selection."
    
    lines = []
    for idx, p in enumerate(perfumes, 1):
        lines.append(f"{idx}. {p['name']} by {p['brand']}")
    return "\n".join(lines)

def build_rag_context(selected, all_perfumes, criteria=None, context_size=8):
    pool = set(id(p) for p in selected)
    fillers = []
    
    # 1. Inject Hard Negatives to prevent data leakage (Needle already inserted problem)
    if criteria:
        sample_pool = random.sample(all_perfumes, min(500, len(all_perfumes)))
        for p in sample_pool:
            if id(p) in pool: continue
            
            # A hard negative shares an accord but violates a hard rule or fails the min_score
            meta = p["metadata"]
            shared_accord = False
            if criteria.get("accords"):
                shared_accord = any(a.lower() in meta["_accords_set"] for a in criteria["accords"])
            
            s = score_perfume(p, criteria)
            if shared_accord and (s == -1 or s < 5):
                fillers.append(p)
                pool.add(id(p))
            
            if len(fillers) >= 4: # Put up to 4 hard negatives
                break
                
    # 2. Fill the rest with random perfumes
    needed = context_size - len(selected) - len(fillers)
    if needed > 0:
        random_pool = [p for p in random.sample(all_perfumes, min(200, len(all_perfumes))) if id(p) not in pool]
        fillers.extend(random_pool[:needed])
        
    context_list = list(selected) + fillers
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

# ── TYPE GENERATORS ────────────────────────────────────────────────────────

def make_type_a(all_perfumes, use_rag):
    """Casual Language (25%)"""
    for _ in range(50):
        ref = random.choice(all_perfumes)
        meta = ref["metadata"]
        gender = meta.get("gender", "unisex")
        seasons = meta.get("best_seasons") or []
        accords = meta.get("accords_list") or []
        
        if not seasons or not accords:
            continue
            
        season = random.choice(seasons)
        accord = random.choice(accords[:3])
        
        criteria = {"gender": gender, "seasons": [season], "accords": [accord]}
        matches = find_matches(all_perfumes, criteria, limit=10)
        if not matches:
            continue
            
        profile_json = {"gender": gender, "season": season, "wanted_accord": accord}
        gemini_prompt = f"A user is looking for a perfume based on these hidden parameters: {json.dumps(profile_json)}. Write a 1-2 sentence casual, natural user query asking for this without using exact database jargon. Output only the query."
        user_query = call_gemini(gemini_prompt)
        
        if use_rag:
            subset = random.sample(matches, min(len(matches), 3))
            ctx_list, ctx_text = build_rag_context(subset, all_perfumes, criteria)
            final = get_best_from_context(ctx_list, criteria)
            if not final: final = subset
            return ctx_text + user_query, format_l3_answer(final)
        else:
            final = matches[:3]
            return user_query, format_l3_answer(final)
            
    raise Exception("Failed to generate Type A")

def make_type_b(all_perfumes, use_rag):
    """Reference (25%)"""
    for _ in range(50):
        ref = random.choice(all_perfumes)
        similar_list = ref.get("similar", {}).get("reminds_me_of", [])
        if not similar_list:
            continue
            
        sim_ids = [s["id"] for s in similar_list[:5]]
        matches = [ID_TO_PERFUME[sid] for sid in sim_ids if sid in ID_TO_PERFUME]
        
        if not matches:
            continue
            
        ref_accords = ref["metadata"].get("accords_list", [])[:3]
        criteria = {"accords": ref_accords}
            
        gemini_prompt = f"A user currently wears {ref['name']} by {ref['brand']}. They want something similar but maybe a little different. Write a short, casual query. Example: 'I love X but want to try something new that gives the same vibe.' Output only the query."
        user_query = call_gemini(gemini_prompt)
        
        if use_rag:
            subset = random.sample(matches, min(len(matches), 3))
            ctx_list, ctx_text = build_rag_context(subset, all_perfumes, criteria)
            final = get_best_from_context(ctx_list, criteria)
            if not final: final = subset
            return ctx_text + user_query, format_l3_answer(final)
        else:
            final = matches[:3]
            return user_query, format_l3_answer(final)
            
    raise Exception("Failed to generate Type B")

def make_type_c(all_perfumes, use_rag):
    """Likes / Dislikes (25%)"""
    for _ in range(50):
        ref = random.choice(all_perfumes)
        meta = ref["metadata"]
        gender = meta.get("gender", "unisex")
        accords = meta.get("accords_list") or []
        notes = meta.get("notes_list") or []
        
        if not accords or not notes:
            continue
            
        liked_accord = random.choice(accords[:3])
        liked_note = random.choice(notes[:5])
        
        all_accords = {"sweet", "vanilla", "citrus", "woody", "leather", "floral"}
        disliked = list(all_accords - set(accords) - {liked_accord})
        if not disliked:
            continue
        dislike = random.choice(disliked)
        
        criteria = {
            "gender": gender,
            "accords": [liked_accord],
            "notes": [liked_note],
            "not_accords": [dislike]
        }
        
        matches = find_matches(all_perfumes, criteria, limit=10)
        if not matches:
            continue
            
        profile_json = {"gender": gender, "liked_accord": liked_accord, "liked_note": liked_note, "hated_accord": dislike}
        gemini_prompt = f"A user is looking for a perfume based on these hidden parameters: {json.dumps(profile_json)}. Write a 1-2 sentence casual user query expressing what they like and explicitly stating what they hate/want to avoid. Output only the query."
        user_query = call_gemini(gemini_prompt)
        
        if use_rag:
            subset = random.sample(matches, min(len(matches), 3))
            ctx_list, ctx_text = build_rag_context(subset, all_perfumes, criteria)
            final = get_best_from_context(ctx_list, criteria)
            if not final: final = subset
            return ctx_text + user_query, format_l3_answer(final)
        else:
            final = matches[:3]
            return user_query, format_l3_answer(final)
            
    raise Exception("Failed to generate Type C")

def make_type_d(all_perfumes, use_rag):
    """Messy Queries (25%) - Reverse Engineered"""
    for _ in range(50):
        ref = random.choice(all_perfumes)
        meta = ref["metadata"]
        gender = meta.get("gender", "unisex")
        accords = meta.get("accords_list") or []
        
        if len(accords) < 2:
            continue
            
        top_accords = accords[:3]
        criteria = {"gender": gender, "accords": top_accords}
        matches = find_matches(all_perfumes, criteria, limit=10)
        
        if not matches:
            continue
            
        profile_json = {"gender": gender, "main_accords": top_accords}
        
        gemini_prompt = f"A user wants a perfume with these exact characteristics: {json.dumps(profile_json)}. Write a highly messy, casual user query in 1-2 sentences in English describing this vibe. Do not use exact words like 'accords' or 'gender'. Use slang like 'bro', 'man', 'stuff', or describe scenarios (e.g. 'doesn't give me a headache'). Include typos, vague wording, no capitalization, and incomplete thoughts. Output only the English query."
        user_query = call_gemini(gemini_prompt)
        
        if use_rag:
            subset = random.sample(matches, min(len(matches), 3))
            ctx_list, ctx_text = build_rag_context(subset, all_perfumes, criteria)
            final = get_best_from_context(ctx_list, criteria)
            if not final: final = subset
            return ctx_text + user_query, format_l3_answer(final)
        else:
            final = matches[:3]
            return user_query, format_l3_answer(final)
            
    raise Exception("Failed to generate Type D")

def make_type_e(all_perfumes, use_rag):
    """Contradiction Queries (15%)"""
    conflict_pairs = [
        (["fresh spicy", "warm spicy"], "fresh but warm"),
        (["sweet", "fresh spicy"], "sweet but fresh and not cloying"),
        (["woody", "aquatic"], "deep woody but fresh/aquatic"),
        (["leather", "powdery"], "tough leather but soft/powdery"),
        (["citrus", "woody"], "bright citrus but grounded and woody"),
    ]
    for _ in range(50):
        pair_accords, concept = random.choice(conflict_pairs)
        
        matches = []
        for p in all_perfumes:
            meta = p["metadata"]
            acc_lower = {a.lower() for a in meta.get("accords_list", [])}
            if all(a.lower() in acc_lower for a in pair_accords):
                matches.append(p)
                
        if not matches:
            continue
            
        # Top accords to score for retrieval
        criteria = {"accords": pair_accords}
        
        gemini_prompt = f"A user wants a perfume that balances two opposing qualities conceptually: '{concept}'. Write a natural, slightly contradictory 1-2 sentence user query expressing this desire. Include typos, vague wording, no capitalization, and incomplete thoughts. Output only the query."
        user_query = call_gemini(gemini_prompt)
        
        if use_rag:
            subset = random.sample(matches, min(len(matches), 3))
            ctx_list, ctx_text = build_rag_context(subset, all_perfumes, criteria)
            final = get_best_from_context(ctx_list, criteria)
            if not final: final = subset
            return ctx_text + user_query, format_l3_answer(final)
        else:
            final = matches[:3]
            return user_query, format_l3_answer(final)
            
    raise Exception("Failed to generate Type E")

# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(CLEAN_FILE):
        print(f"Error: {CLEAN_FILE} not found")
        return
        
    print("Loading dataset...")
    all_perfumes = load_perfumes()
    print(f"Loaded {len(all_perfumes)} perfumes.")
    
    if not API_KEY:
        print("WARNING: GEMINI_API_KEY NOT SET. THE SCRIPT WILL RUN WITH DUMMY TEXT.")
        print("Export your API key: export GEMINI_API_KEY='your_key'")
        return

    # Total 2500
    cat_counts = {
        "casual": int(TOTAL_TARGET * 0.25),    # 625
        "reference": int(TOTAL_TARGET * 0.20), # 500
        "likes": int(TOTAL_TARGET * 0.20),     # 500
        "messy": int(TOTAL_TARGET * 0.20),     # 500
        "contradiction": int(TOTAL_TARGET * 0.15) # 375
    }
    
    generators = {
        "casual": make_type_a,
        "reference": make_type_b,
        "likes": make_type_c,
        "messy": make_type_d,
        "contradiction": make_type_e
    }
    
    skipped = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for cat, count in cat_counts.items():
            print(f"Generating {cat} ({count} records)...")
            func = generators[cat]
            for i in range(count):
                use_rag = random.random() < 0.70 # 70% RAG, 30% Non-RAG
                
                try:
                    user_content, answer = func(all_perfumes, use_rag)
                    record = build_messages(user_content, answer)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                except Exception as e:
                    skipped += 1
                    print(f"Error in {cat} [{skipped} skipped]: {e}")
                    
                if (i+1) % 10 == 0:
                    print(f"  Progress: {i+1}/{count}")
                    
    print(f"Finished generating {TOTAL_TARGET} records at {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
