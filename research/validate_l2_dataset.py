"""
validate_l2_dataset.py — ScentAI L2 Dataset Validator

5 ayrı doğrulama katmanı:
1. RAG Validator      — cevaptaki parfümler ⊆ context parfümleri
2. Negative Validator — NOT X varsa → assert X not in notes/accords
3. Ranking Validator  — sıralama gerçekten azalan mı?
4. Numeric Validator  — rating > X → assert rating > X
5. No Match Validator — sonuç gerçekten boş mu?
"""

import json
import re
import sys

def load_perfume_db():
    db = {}
    with open("/home/sefasys/Desktop/Perfume-Dataset/perfumes_clean.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            key = f"{p['name'].strip().lower()}::{p['brand'].strip().lower()}"
            if key not in db:
                db[key] = []
            db[key].append(p)
    return db



def parse_answer_perfumes(answer_text):
    """Cevap metninden parfüm isimlerini ve detaylarını çıkar."""
    perfumes = []
    current = None
    for line in answer_text.split("\n"):
        line = line.strip()
        # "1. Aventus by Creed" formatı
        m = re.match(r"^\d+\.\s+(.+?)\s+by\s+(.+)$", line)
        if m:
            if current:
                perfumes.append(current)
            current = {
                "name": m.group(1).strip(),
                "brand": m.group(2).strip(),
                "notes": [],
                "accords": [],
                "rating": None,
                "popularity": None,
                "seasons": [],
            }
        elif current and line.startswith("- "):
            detail = line[2:].strip()
            dl = detail.lower()

            # Notes (Top Notes / Middle Notes / Base Notes / Notes)
            if dl.startswith("top notes:") or dl.startswith("middle notes:") or dl.startswith("base notes:") or dl.startswith("notes:"):
                prefix_end = detail.index(":") + 1
                notes_str = detail[prefix_end:].strip()
                current["notes"].extend([n.strip().lower() for n in notes_str.split(",") if n.strip()])

            # Accords
            elif dl.startswith("accords:"):
                acc_str = detail.split(":", 1)[1].strip()
                current["accords"].extend([a.strip().lower() for a in acc_str.split(",") if a.strip()])

            # Rating
            elif dl.startswith("rating:"):
                rm = re.search(r"([\d.]+)/5", detail)
                if rm:
                    current["rating"] = float(rm.group(1))
                pm = re.search(r"\((\d+)\s*votes?\)", detail)
                if pm:
                    current["popularity"] = int(pm.group(1))

            # Seasons
            elif dl.startswith("seasons:"):
                s_str = detail.split(":", 1)[1].strip()
                current["seasons"].extend([s.strip().lower() for s in s_str.split(",") if s.strip() and s.strip().lower() != "n/a"])

    if current:
        perfumes.append(current)
    return perfumes


def parse_context_names(user_content):
    """[PERFUMES] bloğundan parfüm isimlerini çıkar."""
    names = set()
    m = re.search(r"\[PERFUMES\](.*?)\[/PERFUMES\]", user_content, re.DOTALL)
    if not m:
        return names
    block = m.group(1)
    for line in block.split("\n"):
        line = line.strip()
        # "Name by Brand (Year) — gender" formatı
        pm = re.match(r"^(.+?)\s+by\s+(.+?)(?:\s*\(\d{4}\))?\s*[—–-]\s*(?:male|female|unisex)", line)
        if pm:
            names.add(pm.group(1).strip().lower())
    return names


def extract_query_negatives(user_content):
    """Soru metninden negatif kısıtları çıkar."""
    not_items = set()
    # "without X", "excluding X", "not X", "but without X"
    patterns = [
        r"(?:without|excluding|not)\s+([\w\s]+?)(?:\s+and\s+|\s*,\s*|\s*\.\s*|$)",
    ]
    # [PERFUMES] bloğunu çıkar
    query = re.sub(r"\[PERFUMES\].*?\[/PERFUMES\]", "", user_content, flags=re.DOTALL).strip()

    for pattern in patterns:
        for match in re.finditer(pattern, query, re.IGNORECASE):
            item = match.group(1).strip().lower()
            if item and len(item) < 30:
                not_items.add(item)
    return not_items


def extract_min_rating(user_content):
    """Soru metninden minimum rating değerini çıkar."""
    query = re.sub(r"\[PERFUMES\].*?\[/PERFUMES\]", "", user_content, flags=re.DOTALL).strip()
    m = re.search(r"(?:above|higher than|over|rated above)\s+([\d.]+)", query, re.IGNORECASE)
    if m:
        val_str = m.group(1).rstrip(".")
        try:
            return float(val_str)
        except ValueError:
            return None
    return None


def extract_query_gender(user_content):
    """Soru metninden cinsiyet kısıtını çıkar."""
    allowed = extract_allowed_genders(user_content)
    if len(allowed) == 1:
        return next(iter(allowed))
    return None


def extract_allowed_genders(user_content):
    """Soru metninden izin verilen cinsiyet kümesini çıkar."""
    query = re.sub(r"\[PERFUMES\].*?\[/PERFUMES\]", "", user_content, flags=re.DOTALL).strip()
    query_lower = query.lower()

    has_unisex = re.search(r"\bunisex\b", query_lower)
    has_male = re.search(r"\b(male|men)\b", query_lower)
    has_female = re.search(r"\b(female|women)\b", query_lower)

    if has_male and has_unisex:
        return {"male", "unisex"}
    if has_female and has_unisex:
        return {"female", "unisex"}
    if has_unisex:
        return {"unisex"}
    if has_male:
        return {"male"}
    if has_female:
        return {"female"}
    return set()


def contains_forbidden_term(terms, forbidden):
    joined = " ".join(terms)
    forbidden_low = forbidden.lower()
    return forbidden_low in terms or re.search(rf"\b{re.escape(forbidden_low)}\b", joined)


def run_validation(filepath):
    print(f"Starting L2 Dataset Validation for: {filepath}")

    violations = {
        "rag": 0,
        "negative": 0,
        "ranking": 0,
        "numeric": 0,
        "no_match": 0,
        "gender": 0,
    }

    print("Loading original Perfume DB for strict validation...")
    perfume_db = load_perfume_db()

    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            record = json.loads(line)
            msgs = record["messages"]
            user_content = msgs[1]["content"]
            answer = msgs[2]["content"]

            has_rag = "[PERFUMES]" in user_content
            is_no_match = answer.strip() == "No matching perfumes found."
            parsed = parse_answer_perfumes(answer)

            # ── 1. RAG Validator ──
            if has_rag and not is_no_match and parsed:
                context_names = parse_context_names(user_content)
                for p in parsed:
                    pname = p["name"].lower()
                    if pname not in context_names:
                        print(f"[{idx}] RAG Violation: '{p['name']}' not in context.")
                        violations["rag"] += 1
                        break

            # ── 2. Negative Validator ──
            if not is_no_match and parsed:
                negatives = extract_query_negatives(user_content)
                for neg in negatives:
                    for p in parsed:
                        terms = p["notes"] + p["accords"]
                        if contains_forbidden_term(terms, neg):
                            print(f"[{idx}] Negation Violation: Perfume '{p['name']}' contains excluded term '{neg}' in: {' '.join(terms)}")
                            violations["negative"] += 1
                            break

            # ── 3. Ranking Validator ──
            if len(parsed) >= 2:
                query_lower = re.sub(r"\[PERFUMES\].*?\[/PERFUMES\]", "", user_content, flags=re.DOTALL).lower()
                if "highest rated" in query_lower or "top rated" in query_lower or "best rated" in query_lower:
                    ratings = [p["rating"] for p in parsed if p["rating"] is not None]
                    if len(ratings) >= 2:
                        for i in range(len(ratings) - 1):
                            if ratings[i] < ratings[i + 1]:
                                print(f"[{idx}] Ranking Violation: ratings not descending: {ratings}")
                                violations["ranking"] += 1
                                break

                if "most popular" in query_lower or "most votes" in query_lower:
                    pops = [p["popularity"] for p in parsed if p["popularity"] is not None]
                    if len(pops) >= 2:
                        for i in range(len(pops) - 1):
                            if pops[i] < pops[i + 1]:
                                print(f"[{idx}] Ranking Violation: popularity not descending: {pops}")
                                violations["ranking"] += 1
                                break

            # ── 4. Numeric Validator ──
            if not is_no_match and parsed:
                min_rating = extract_min_rating(user_content)
                if min_rating is not None:
                    for p in parsed:
                        if p["rating"] is not None and p["rating"] < min_rating:
                            print(f"[{idx}] Numeric Violation: '{p['name']}' rating {p['rating']} < {min_rating}")
                            violations["numeric"] += 1
                            break

            # ── 5. No Match Validator ──
            if is_no_match:
                if parsed:
                    print(f"[{idx}] No-Match Violation: answer says 'no match' but parsed {len(parsed)} perfumes.")
                    violations["no_match"] += 1

            # ── 6. Gender Validator ──
            if not is_no_match and parsed:
                allowed_genders = extract_allowed_genders(user_content)
                if allowed_genders:
                    for p in parsed:
                        key = f"{p['name'].strip().lower()}::{p['brand'].strip().lower()}"
                        db_perfumes = perfume_db.get(key)
                        if db_perfumes:
                            matched_gender = False
                            for db_p in db_perfumes:
                                actual_gender = db_p["metadata"].get("gender", "unisex").lower()
                                if actual_gender in allowed_genders:
                                    matched_gender = True
                                    break
                            
                            if not matched_gender:
                                actual_genders = [db_p["metadata"].get("gender", "unisex").lower() for db_p in db_perfumes]
                                print(f"[{idx}] Gender Violation: Query allowed {sorted(allowed_genders)}, but '{p['name']}' is {actual_genders}.")
                                violations["gender"] += 1
                                break

    total = sum(violations.values())
    print(f"\n--- L2 Validation Report ---")
    print(f"Total Records Tested       : {idx}")
    print(f"RAG Violations             : {violations['rag']}")
    print(f"Negative Filter Violations : {violations['negative']}")
    print(f"Ranking Violations         : {violations['ranking']}")
    print(f"Numeric Filter Violations  : {violations['numeric']}")
    print(f"No-Match Violations        : {violations['no_match']}")
    print(f"Gender Violations          : {violations['gender']}")
    print(f"----------------------------")

    if total == 0:
        print("✅ All validations passed!")
    else:
        print(f"⚠️  Total violations: {total}")

    return total


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_l2_dataset.py <dataset.jsonl>")
        sys.exit(1)
    total_violations = run_validation(sys.argv[1])
    sys.exit(1 if total_violations > 0 else 0)
