# Main Perfume Dataset Audit

Source audited: `perfumes_clean.jsonl` (276 MB)

## Executive Assessment

This is not merely a perfume-card dataset. It contains three complementary recommendation signals:

1. Structured product facts: identity, gender, year, accords, notes, seasons, time, rating, longevity, sillage, and value.
2. Olfactory similarity: `reminds_me_of` edges with positive and negative community votes.
3. Taste affinity: `also_liked` edges representing collaborative user preference rather than literal smell similarity.

The source is strong enough for a hybrid recommendation engine. The LLM should explain already-ranked grounded candidates; it should not be the primary ranker.

## Inventory

| Signal | Coverage / volume |
|---|---:|
| Perfume records | 131,930 |
| Unique IDs | 131,930 |
| Accord lists | 129,161 (97.9%) |
| Accord percentages in card text | 129,161 (97.9%) |
| Note lists | 129,388 (98.1%) |
| Top/middle/base note pyramids | about 92,900 records (70.4%) |
| Year | 107,797 (81.7%) |
| Best seasons | 111,351 (84.4%) |
| Time profile | 109,559 (83.0%) |
| Positive longevity | 108,646 (82.4%) |
| Positive sillage | 109,766 (83.2%) |
| Positive value score | 101,051 (76.6%) |
| Perfumer | 49,213 (37.3%) |
| Description | 131,930 (100%) |
| `reminds_me_of` edges | 692,729 |
| `also_liked` edges | 1,645,055 |
| Community similarity votes | 12,371,989 |

Only three similarity targets are absent from the catalog; all `also_liked` targets resolve. Graph referential integrity is therefore excellent.

## Quality Risks

### Rating confidence

- Median vote count is 11; 16,578 records have zero votes.
- 8,410 records show a perfect 5.0 rating.
- 8,379 of those perfect ratings have fewer than ten votes.
- Raw rating sort therefore promotes tiny-sample records and must not be used directly.

Runtime mitigation: rating ranking now uses a Bayesian estimate with the catalog vote-weighted mean (4.01) and 50 prior votes.

### Duplicate products

- 1,462 non-empty slug groups are duplicated, accounting for 1,510 extra rows.
- 1,425 normalized name/brand groups are duplicated, accounting for 1,455 extra rows.
- Examples include Acqua di Gio, Clinique Happy, Samsara EDP, and Mitsouko EDT.

Runtime mitigation: recommendation lists now deduplicate normalized `name + brand`. A future catalog build should maintain a canonical-ID/alias table so graph votes from duplicate nodes can be merged.

### Graph confidence and truncation

- 198,997 similarity edges (28.7%) have at least ten votes.
- 411,137 directed similarity edges have a reverse edge (59.3%).
- 13,849 `reminds_me_of` lists and 77,845 `also_liked` lists hit the 20-item source cap.
- The graph therefore represents top-neighbor samples, not complete preference histories.

Use vote confidence and reciprocal evidence. Do not interpret a missing edge as a negative label.

### Field anomalies

Three launch years are suspicious: 20, 1070, and 1533. Values should be retained in source provenance but excluded from ordinary modern-release filtering until reviewed.

## Information Currently Lost By The Compact Catalog

The first runtime catalog intentionally flattened the source. It does not yet preserve:

- Accord strength percentages and rank order.
- Top/middle/base note layer and note rank.
- The 1.65M `also_liked` graph.
- Perfumer and description fields.
- Canonical mappings for duplicate records.

These are the highest-value additions for catalog v2.

## Recommended Hybrid Retrieval

Candidate generation should run several channels and fuse them after hard filters:

1. Semantic channel: BGE-M3 over grounded perfume cards.
2. Structural channel: weighted accord similarity plus layer-aware note similarity.
3. Smell graph channel: confidence-weighted `reminds_me_of` neighbors.
4. Taste graph channel: normalized `also_liked` neighbors or personalized PageRank from multiple liked/owned perfumes.
5. Quality channel: Bayesian rating, popularity confidence, longevity, sillage, and value.

The query router should select weights by intent:

| User intent | Dominant signal |
|---|---|
| `smells like X`, `dupe for X` | `reminds_me_of` + structural similarity |
| `if I like X, what else?` | `also_liked` + semantic preference similarity |
| `between X and Y` | balanced minimum similarity to both references |
| `what is missing from my collection?` | collection coverage + multi-seed taste graph |
| factual filters and rankings | deterministic structured metadata |

After fusion, apply diversity-aware reranking so one brand, clone family, or duplicate product cannot dominate the list. The Gemma model then receives only the final grounded cards and writes the explanation.

## Training Value Beyond SFT

The graph can produce higher-quality ranking supervision:

- Positive pairs: high-confidence, positive-net `reminds_me_of` edges.
- Hard negatives: high-downvote similarity edges and structurally close but community-rejected pairs.
- Taste positives: `also_liked` edges, kept separate from literal similarity labels.
- Pairwise ranking: high-confidence edge over a low-confidence or rejected edge for the same source perfume.
- Collection tasks: multi-seed graph recommendations with owned-item exclusion.

This supports a reranker or preference-optimization stage. Mixing `also_liked` and `reminds_me_of` into one undifferentiated SFT label would teach the wrong semantics.

## Implementation Order

1. Canonicalize duplicate products and merge graph evidence through aliases.
2. Build catalog v2 with accord weights, note layers, and `also_liked` edges.
3. Add intent-specific graph retrieval and reciprocal-rank fusion.
4. Add graph-aware offline evaluation: Recall@K, NDCG@K, catalog coverage, novelty, and diversity.
5. Generate pairwise ranking data and train/evaluate a reranker before considering another large SFT run.

Before public redistribution or deployment, keep source provenance and publication rights documented separately from the technical quality assessment.
