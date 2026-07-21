import json
import re
import os

RAW_FILE = "/home/sefasys/Desktop/Perfume-Dataset/archive(2)/perfumes.jsonl"
CLEAN_FILE = "/home/sefasys/Desktop/Perfume-Dataset/perfumes_clean.jsonl"

# List of common abbreviations to avoid false sentence split
ABBREVIATIONS = ["mr.", "mrs.", "ms.", "dr.", "st.", "prof.", "gen.", "rep.", "sen.", "col.", "capt.", 
                 "ltd.", "inc.", "co.", "vs.", "eg.", "ie.", "ca.", "approx.", "est.", "min.", "max.", 
                 "vol.", "temp.", "wt.", "oz.", "ml.", "p.m.", "a.m."]

def clean_html(text):
    if not text:
        return ""
    # Remove HTML tags
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text).strip()

def get_first_two_sentences(text):
    cleaned = clean_html(text)
    if not cleaned:
        return ""
    
    # Temporarily replace abbreviations with placeholders to avoid splitting on them
    placeholder_text = cleaned
    for abbr in ABBREVIATIONS:
        # Match case-insensitive and replace period with special character
        placeholder = abbr.replace(".", "@@@")
        # Ensure regex boundaries or exact match
        placeholder_text = re.sub(rf"\b{re.escape(abbr)}", placeholder, placeholder_text, flags=re.IGNORECASE)
    
    # Split sentences on . or ? or ! followed by space
    sentences = re.split(r'(?<=\.|\?|\!)\s+', placeholder_text)
    
    # Restore the periods in the sentences
    final_sentences = []
    for s in sentences:
        s_restored = s.replace("@@@", ".")
        if s_restored.strip():
            final_sentences.append(s_restored.strip())
            
    # Take first 2 sentences and join
    two_sentences = " ".join(final_sentences[:2]).strip()
    
    if not two_sentences:
        return cleaned[:200]
    return two_sentences

def normalize_text(text):
    if not text:
        return ""
    # Lowercase, strip extra spaces, replace double spaces
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    # Replace special characters/hyphens with space or clean form
    text = re.sub(r'[^a-z0-9\s\-\.]', '', text)
    return text

def process_perfume(record):
    name = record.get("name", "Unknown")
    brand = record.get("brand", "Unknown")
    year = record.get("year")
    gender = record.get("gender", "unisex")
    
    # 1. Accords (Strength >= 30)
    accords = record.get("accords", [])
    filtered_accords = [a for a in accords if isinstance(a, dict) and a.get("strength", 0) >= 30]
    # Sort by strength descending
    filtered_accords = sorted(filtered_accords, key=lambda x: x.get("strength", 0), reverse=True)
    
    accords_text_list = [f"{a['name']} ({a['strength']}%)" for a in filtered_accords]
    accords_str = ", ".join(accords_text_list)
    
    # 2. Notes (Tiered or Flat)
    notes = record.get("notes", {})
    tiered = notes.get("tiered", {})
    flat = notes.get("flat", [])
    
    notes_parts = []
    notes_list_metadata = []
    
    has_tiered = any(tiered.get(k) for k in ["top", "middle", "base"] if tiered.get(k))
    has_flat = len(flat) > 0
    
    if has_tiered:
        top = [n.get("name", "") for n in tiered.get("top", []) if n.get("name")]
        middle = [n.get("name", "") for n in tiered.get("middle", []) if n.get("name")]
        base = [n.get("name", "") for n in tiered.get("base", []) if n.get("name")]
        
        tiered_parts = []
        if top:
            tiered_parts.append(f"Top Notes: {', '.join(top)}")
            notes_list_metadata.extend(top)
        if middle:
            tiered_parts.append(f"Middle Notes: {', '.join(middle)}")
            notes_list_metadata.extend(middle)
        if base:
            tiered_parts.append(f"Base Notes: {', '.join(base)}")
            notes_list_metadata.extend(base)
            
        if tiered_parts:
            notes_parts.append(" | ".join(tiered_parts))
    elif has_flat:
        flat_notes = [n.get("name", "") for n in flat if n.get("name")]
        if flat_notes:
            notes_parts.append(f"Notes: {', '.join(flat_notes)}")
            notes_list_metadata.extend(flat_notes)
            
    notes_str = notes_parts[0] if notes_parts else ""
    
    # 3. Performance
    rating_data = record.get("rating", {})
    rating_avg = rating_data.get("average")
    people_votes = record.get("people", 0) or 0
    
    perf_parts = []
    if rating_avg is not None:
        perf_parts.append(f"Rating: {rating_avg:.2f}/5 ({people_votes} votes)")
    
    longevity = record.get("longevity", {})
    lon_avg = longevity.get("average")
    if lon_avg is not None:
        perf_parts.append(f"Longevity: {lon_avg:.2f}/5")
        
    sillage = record.get("sillage", {})
    sil_avg = sillage.get("average")
    if sil_avg is not None:
        perf_parts.append(f"Sillage: {sil_avg:.2f}/4")
        
    price_val = record.get("price_value", {})
    val_avg = price_val.get("average")
    if val_avg is not None:
        perf_parts.append(f"Value: {val_avg:.2f}/5")
        
    perf_str = " | ".join(perf_parts)
    
    # 4. Seasons and Times (Threshold >= 20%)
    seasons = record.get("seasons", {})
    total_season_votes = sum(seasons.values())
    best_seasons = []
    if total_season_votes > 0:
        for s, votes in seasons.items():
            if votes / total_season_votes >= 0.20:
                best_seasons.append(s)
                
    daypart = record.get("daypart", {})
    total_time_votes = sum(daypart.values())
    time_profile = []
    if total_time_votes > 0:
        for t, votes in daypart.items():
            if votes / total_time_votes >= 0.20:
                time_profile.append(t)
                
    seasons_str = f"Best Seasons: {', '.join(best_seasons).title()}" if best_seasons else ""
    time_str = f"Time: {', '.join(time_profile).title()}" if time_profile else ""
    
    seasons_and_time = []
    if seasons_str:
        seasons_and_time.append(seasons_str)
    if time_str:
        seasons_and_time.append(time_str)
    seasons_time_str = " | ".join(seasons_and_time)
    
    # 5. Perfumers
    perfumers = [p.get("name", "") for p in record.get("perfumers", []) if p.get("name")]
    perfumer_str = f"Perfumer: {', '.join(perfumers)}" if perfumers else ""
    
    # 6. Description
    desc_str = get_first_two_sentences(record.get("description", ""))
    
    # Assemble Card Text
    card_parts = []
    year_str = f" ({year})" if year else ""
    card_parts.append(f"{name} by {brand}{year_str} — {gender}")
    
    if accords_str:
        card_parts.append(f"Accords: {accords_str}")
    if notes_str:
        card_parts.append(notes_str)
    if perf_str:
        card_parts.append(perf_str)
    if seasons_time_str:
        card_parts.append(seasons_time_str)
    if perfumer_str:
        card_parts.append(perfumer_str)
    if desc_str:
        card_parts.append(f"Description: {desc_str}")
        
    card_text = "\n".join(card_parts)
    
    # 7. Metadata lists normalization
    normalized_notes = list(set([normalize_text(n) for n in notes_list_metadata if n]))
    normalized_accords = list(set([normalize_text(a.get("name", "")) for a in accords if a.get("name")]))
    
    metadata = {
        "brand": brand,
        "gender": gender,
        "year": int(year) if year else None,
        "rating": float(rating_avg) if rating_avg is not None else None,
        "popularity": int(people_votes),
        "best_seasons": best_seasons,
        "time_profile": time_profile,
        "notes_list": normalized_notes,
        "accords_list": normalized_accords
    }
    
    return {
        "id": record.get("id"),
        "slug": record.get("slug"),
        "name": name,
        "brand": brand,
        "card_text": card_text,
        "metadata": metadata,
        "similar": record.get("similar", {})
    }

def clean_dataset():
    if not os.path.exists(RAW_FILE):
        print(f"Error: Raw file not found at {RAW_FILE}")
        return

    print("Cleaning and preparing dataset...")
    processed_count = 0
    
    with open(RAW_FILE, "r", encoding="utf-8") as f_in, \
         open(CLEAN_FILE, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            try:
                record = json.loads(line)
            except Exception as e:
                print(f"JSON decode error: {e}")
                continue
                
            # If basic identifiers are missing, skip
            if not record.get("name") or not record.get("brand"):
                continue
                
            processed = process_perfume(record)
            f_out.write(json.dumps(processed, ensure_ascii=False) + "\n")
            
            processed_count += 1
            if processed_count % 25000 == 0:
                print(f"Processed and cleaned {processed_count} perfumes...")
                
    print(f"Dataset cleaning complete! Saved {processed_count} records to {CLEAN_FILE}")

if __name__ == "__main__":
    clean_dataset()
