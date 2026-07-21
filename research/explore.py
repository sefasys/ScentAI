import json
from collections import Counter
import os

RAW_FILE = "/home/sefasys/Desktop/Perfume-Dataset/archive(2)/perfumes.jsonl"

def explore_dataset():
    if not os.path.exists(RAW_FILE):
        print(f"Error: Raw file not found at {RAW_FILE}")
        return

    print("Analyzing dataset. Please wait...")
    
    total_records = 0
    missing_fields = {
        "name": 0,
        "brand": 0,
        "year": 0,
        "gender": 0,
        "description": 0,
        "accords": 0,
        "notes": 0,
        "rating": 0,
        "people": 0
    }
    
    notes_types = {
        "tiered": 0,
        "flat": 0,
        "empty": 0
    }
    
    all_accords = Counter()
    all_notes = Counter()
    brand_counts = Counter()
    gender_counts = Counter()
    
    # Read in chunks to prevent memory issues
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        for line in f:
            total_records += 1
            try:
                record = json.loads(line)
            except Exception as e:
                print(f"JSON decode error at line {total_records}: {e}")
                continue
            
            # Check basic fields
            for field in missing_fields.keys():
                val = record.get(field)
                if val is None or val == "" or val == []:
                    missing_fields[field] += 1
            
            # Analyze notes
            notes = record.get("notes", {})
            tiered = notes.get("tiered", {})
            flat = notes.get("flat", [])
            
            has_tiered = any(tiered.get(k) for k in ["top", "middle", "base"] if tiered.get(k))
            has_flat = len(flat) > 0
            
            if has_tiered:
                notes_types["tiered"] += 1
            elif has_flat:
                notes_types["flat"] += 1
            else:
                notes_types["empty"] += 1
                
            # Collect accords
            accords = record.get("accords", [])
            for acc in accords:
                if isinstance(acc, dict) and "name" in acc:
                    all_accords[acc["name"].lower()] += 1
            
            # Collect notes names for top frequency check
            if has_tiered:
                for level in ["top", "middle", "base"]:
                    for n in tiered.get(level, []):
                        if isinstance(n, dict) and "name" in n:
                            all_notes[n["name"].lower()] += 1
            elif has_flat:
                for n in flat:
                    if isinstance(n, dict) and "name" in n:
                        all_notes[n["name"].lower()] += 1
                        
            # Collect brand & gender
            brand = record.get("brand")
            if brand:
                brand_counts[brand] += 1
            gender = record.get("gender")
            if gender:
                gender_counts[gender] += 1
                
            # Show progress
            if total_records % 25000 == 0:
                print(f"Processed {total_records} records...")

    # Print Results
    print("\n" + "="*50)
    print("📊 DATASET ANALYSIS REPORT")
    print("="*50)
    print(f"Total Records Analyzed: {total_records}")
    
    print("\n❌ Missing/Null Fields:")
    for field, count in missing_fields.items():
        pct = (count / total_records) * 100
        print(f"  - {field}: {count} records ({pct:.2f}%)")
        
    print("\n🎼 Notes Structure:")
    for ntype, count in notes_types.items():
        pct = (count / total_records) * 100
        print(f"  - {ntype}: {count} ({pct:.2f}%)")
        
    print("\n👫 Gender Distribution:")
    for g, count in gender_counts.items():
        pct = (count / total_records) * 100
        print(f"  - {g}: {count} ({pct:.2f}%)")

    print(f"\n🏷️ Total Unique Brands: {len(brand_counts)}")
    print("Top 10 Brands by count:")
    for b, count in brand_counts.most_common(10):
        print(f"  - {b}: {count}")

    print(f"\n🌿 Total Unique Accords: {len(all_accords)}")
    print("Top 15 Accords:")
    print(", ".join([f"{a} ({c})" for a, c in all_accords.most_common(15)]))

    print(f"\n🌸 Total Unique Notes: {len(all_notes)}")
    print("Top 15 Notes:")
    print(", ".join([f"{n} ({c})" for n, c in all_notes.most_common(15)]))
    print("="*50)

if __name__ == "__main__":
    explore_dataset()
