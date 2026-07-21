# L3 Plan

## Purpose

L3 is a semantic translation dataset.

It teaches the model to turn casual, vague, preference-based user language into perfume matches. It should not teach long explanations, reasoning chains, or detailed factual filtering.

## Output Format

L3 answers should be list-only:

```text
1. Perfume Name by Brand
2. Perfume Name by Brand
3. Perfume Name by Brand
```

No intro, no explanation, no "why this matches" text.

## Boundary With L2

L2 is strict database filtering.

- "List male perfumes" means `gender == male`.
- "List vanilla perfumes for winter" means exact explicit filters.
- L2 does not infer vibe, mood, occasion, or broaden gender.

L3 is semantic recommendation matching.

- "something for men" can include `male` and `unisex`.
- "cozy winter scent" maps to likely accords/seasons.
- "not too sweet" is a preference signal, not always a strict `not sweet` filter.

## Gender Rules

Use L3-specific gender behavior:

- "for men" / "masculine" / "my boyfriend" -> `male + unisex`
- "for women" / "feminine" / "my girlfriend" -> `female + unisex`
- "unisex" -> prefer `unisex`
- no gender mentioned -> no gender constraint

This is intentionally different from L2.

## Hard vs Soft Constraints

Hard constraints:

- explicit "no X"
- explicit "without X"
- explicit gender target
- explicit season/time when clearly stated
- RAG context-only rule

Soft constraints:

- cozy
- clean
- fresh
- dark
- bright
- expensive-smelling
- office-safe
- date night
- youthful
- mature
- creamy
- elegant
- not too sweet
- not cloying

Soft constraints should contribute to score, not behave like exact filters unless the query is explicit.

## Categories

Initial L3 categories:

```text
casual_vibe
occasion
likes_dislikes
negative_preference
reference_similarity
messy_query
conceptual_contradiction
```

Suggested default ratios, not final:

```text
casual_vibe              22%
occasion                 14%
likes_dislikes           16%
negative_preference      12%
reference_similarity     14%
messy_query              14%
conceptual_contradiction 8%
```

Counts should stay configurable through CLI:

```bash
python research/generate_l3.py --total 100 --output training_L3_v2.jsonl
```

and:

```bash
python research/generate_l3.py \
  --category-counts casual_vibe=100,occasion=50,likes_dislikes=80 \
  --output training_L3_v2.jsonl
```

## RAG Rules

If context exists, the answer must only contain perfumes from context.

Context should contain:

- strong matches
- hard negatives
- random fillers

Final answer should be selected from context after scoring.

## LLM Usage

Use an external LLM only to generate natural user queries.

Answers must always be generated programmatically.

LLM prompt should receive hidden structured criteria and category, then output only the user query.

Fallback without API key should use deterministic templates, not dummy text.

Current production preference:

- Groq/OpenAI-compatible endpoint for normal API generation.
- `--fallback-policy fail` for real production runs, so hidden template fallback does not silently enter the dataset.
- Gemini can stay as an optional provider, but it has been unstable for quota/high-demand errors.

Durability rules for long runs:

- Write JSONL records immediately, not only after the full run finishes.
- Flush and fsync every `--checkpoint-every` records.
- Default checkpoint interval: 100 records.
- Use `--resume` to continue an interrupted run from the existing output file.
- Use `--overwrite` only when intentionally starting a fresh output.

Later manual/web LLM method:

- Export query-generation jobs as small structured prompts.
- Paste batches into a web LLM manually if API quotas become painful.
- Import the returned user queries back into the generator.
- Keep answers, RAG context, validation, and final JSONL assembly programmatic.
- Treat this as a separate enrichment pipeline after the normal Groq production path is stable.

## Planned Files

```text
research/core/semantic.py
research/core/llm_query.py
research/generators/l3.py
research/generate_l3.py
```

Possible validator later:

```text
research/validators/l3.py
```

## Validation Targets

Minimum checks:

- JSONL parse succeeds.
- Default training output contains only `messages`.
- Debug mode may include `_meta`.
- RAG answers are subset of context.
- Explicit negative preferences are not violated.
- L3 gender behavior follows allowed gender sets.
- No answer contains intro/explanation text.

## Query Diversity Targets

L3 queries should sound naturally varied, but not artificially unique.

Target distribution:

- 20-30% classic request phrasing such as "I'm looking for...", "I want...", "Can you suggest..."
- 20-25% short/direct requests such as "fresh summer scent for men"
- 15-20% casual vague vibe language such as "something cozy but not too sweet"
- 10-15% messy/fragmented queries
- 10-15% reference-based similarity queries
- 5-10% longer personal-context queries

Quality targets:

- Most common opening phrase should stay at or below about 30%.
- Top 5 opening phrases together should stay at or below about 65%.
- Exact duplicate queries should be 0.
- Near-template repetition should be low inside each category.
- "I'm looking for..." is acceptable, but should usually stay around 20-30%, not dominate the dataset.

## Open Decisions

- Final L3 total count.
- Final category ratios.
- Whether `not too sweet` should be a soft penalty or hard exclusion in some templates.
- Whether to include 3 results always, or allow 1-5 depending on context strength.
