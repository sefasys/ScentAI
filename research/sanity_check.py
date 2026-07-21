import json
import random
import os
import sys

CLEAN_FILE = "/home/sefasys/Desktop/Perfume-Dataset/perfumes_clean.jsonl"

def run_sanity_check():
    if not os.path.exists(CLEAN_FILE):
        print(f"Error: Cleaned dataset not found at {CLEAN_FILE}")
        return

    print("Loading cleaned dataset...")
    all_perfumes = []
    with open(CLEAN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            all_perfumes.append(json.loads(line))
            
    print(f"Loaded {len(all_perfumes)} perfumes.")
    
    # Select 1000 random perfumes for the sanity check
    # We want to make sure Aventus is in it if possible, to verify the pineapple woody check
    sampled = []
    aventus_found = False
    for p in all_perfumes:
        if p.get("name") == "Aventus" and p.get("brand") == "Creed":
            sampled.append(p)
            aventus_found = True
            break
            
    remaining_needed = 1000 - len(sampled)
    sampled.extend(random.sample(all_perfumes, remaining_needed))
    random.shuffle(sampled)
    
    print(f"Selected {len(sampled)} perfumes for testing. (Aventus included: {aventus_found})")
    
    # Try importing chromadb and sentence_transformers
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Installing required libraries: chromadb, sentence-transformers...")
        # Since we run in user environment, let's suggest installing them or we can run pip install
        # Let's exit and let the wrapper script handle or install
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "chromadb", "sentence-transformers"])
        import chromadb
        from sentence_transformers import SentenceTransformer

    print("Loading BAAI/bge-m3 embedding model...")
    # Using cpu or gpu automatically
    model = SentenceTransformer("BAAI/bge-m3")
    
    print("Initializing in-memory ChromaDB...")
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="sanity_check",
        metadata={"hnsw:space": "cosine"}
    )
    
    print("Embedding cards and inserting to DB...")
    ids = [str(p["id"]) for p in sampled]
    documents = [p["card_text"] for p in sampled]
    
    # We can pass metadata directly. Need to convert list types in metadata to strings/JSON strings 
    # because ChromaDB metadata only supports str, int, float, bool.
    metadatas = []
    for p in sampled:
        meta = p["metadata"].copy()
        # Convert lists to comma-separated strings
        meta["best_seasons"] = ", ".join(meta["best_seasons"])
        meta["time_profile"] = ", ".join(meta["time_profile"])
        meta["notes_list"] = ", ".join(meta["notes_list"][:30]) # cap length
        meta["accords_list"] = ", ".join(meta["accords_list"])
        # Handle None values by converting to empty string or filtering out
        clean_meta = {}
        for k, v in meta.items():
            if v is not None:
                clean_meta[k] = v
        metadatas.append(clean_meta)

    # Embed documents
    embeddings = model.encode(documents, batch_size=32, show_progress_bar=True).tolist()
    
    # Add to collection
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    print("ChromaDB sanity check database is ready!")
    
    # Test Queries
    queries = [
        "pineapple woody fragrance",
        "rose perfume",
        "summer citrus scent",
        "sweet vanilla perfume",
        "old school masculine fougere"
    ]
    
    print("\n" + "="*50)
    print("🔍 RUNNING RETRIEVAL SANITY CHECK")
    print("="*50)
    
    for q in queries:
        print(f"\nQuery: '{q}'")
        q_emb = model.encode([q]).tolist()
        results = collection.query(
            query_embeddings=q_emb,
            n_results=3
        )
        
        # Display top 3 results
        for idx, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            brand = meta.get("brand", "Unknown")
            name = doc.split(" by ")[0] # extract name roughly
            print(f"  [{idx+1}] {brand} - {name}")
            # print first two lines of the card
            card_lines = doc.split("\n")
            print(f"      Card: {card_lines[0]}")
            if len(card_lines) > 1:
                print(f"      {card_lines[1]}")
    
    print("="*50)

if __name__ == "__main__":
    run_sanity_check()
