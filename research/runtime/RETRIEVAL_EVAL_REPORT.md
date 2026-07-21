# ScentAI Retrieval Evaluation

## Current Runtime Stack

```text
ChromaDB collection : scentai_perfumes
Vector DB           : chroma_db_bge_m3/
Runtime catalog     : scentai_catalog.sqlite3 (metrics + 692K similarity edges)
Embedding model     : BAAI/bge-m3
Default fetch_k     : 160 for evaluation, 80-120 for runtime
Default top_k       : 8-12
Default popularity  : >= 100 unless niche mode is detected
Optional reranker   : BAAI/bge-reranker-base
```

## Pipeline

```text
user query
→ deterministic query analyzer
→ intent route: general, brand-constrained, or reference-similarity
→ metadata filter: gender, season, time, min_rating, min_popularity
→ BGE-M3 vector retrieval
→ for similarity: exact reference resolution + document-neighbor retrieval
→ relaxed fallback if strict pool is too small
→ negative note/accord/entity filter
→ general score: semantic + popularity + rating + intent hits
→ similarity score: semantic + accords + notes + season + time + gender + quality
→ optional CrossEncoder reranker
→ brand dedup
→ final context candidates
```

## Evaluation Set

The fixed evaluation set lives at:

```text
research/runtime/retrieval_eval_cases.json
```

Coverage:

- English and Turkish queries
- fresh/citrus/summer
- warm spicy/winter
- vanilla/date night
- clean office
- masculine fougere
- Aventus-like reference retrieval
- dynamic Bleu de Chanel and Black Orchid reference retrieval
- positive and negative brand constraints, including spelling-tolerant exclusions
- negative constraints such as `without vanilla`, `less smoky`, `not too sweet`
- niche mode
- gender, season, and time metadata filters

## Latest Result

Command:

```bash
python research/evaluate_retrieval.py --top-k 8 --fetch-k 120
```

Result:

```text
Cases            : 20
Passed           : 20/20 (100.0%)
Avg hit count    : 7.60/8
Avg violations   : 0.00
Avg popularity   : 3087.7
```

## Notes

- BGE-M3 retrieval is strong enough for both English and Turkish semantic queries.
- Raw vector search can surface obscure literal matches; the popularity threshold and score blend are necessary.
- `popularity >= 100` is the default for normal recommendation mode.
- Niche mode lowers the popularity constraint.
- Negative filtering is strict after retrieval, so disliked notes/accords are excluded from final candidates.
- Reference queries resolve the source perfume dynamically; the exact source card is removed before generation.
- Similarity allows at most one same-brand flanker; `dupe`/`alternative` requests remove the reference brand entirely.
- Reranker support is implemented but optional; enable it for deployment or higher-quality evaluation.

## Useful Commands

```bash
python research/test_retrieval.py --top-k 8 --fetch-k 100
```

```bash
python research/evaluate_retrieval.py --top-k 8 --fetch-k 160
```

With optional reranker:

```bash
python research/evaluate_retrieval.py \
  --top-k 8 \
  --fetch-k 160 \
  --reranker BAAI/bge-reranker-base
```
