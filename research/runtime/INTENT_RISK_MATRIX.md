# ScentAI Runtime Intent Risk Matrix

This matrix tracks query types that need behavior beyond generic semantic retrieval.

Gemma is the primary intent/constraint planner for every query. Planner fields carry evidence quotes that are checked against the user text. The deterministic analyzer remains a safety fallback, while catalog lookup, hard filters, and output validation enforce the model-understood request.

| Intent | Example | Current status | Required runtime behavior |
|---|---|---|---|
| General recommendation | `fresh summer scent for men` | Supported | Intent retrieval + metadata filters |
| Single-perfume profile | `Aventus nasıl bir parfüm; nerede kullanılır?` | Supported | Resolve one exact card, then let the model explain character, wear context, practical performance, versatility, and tradeoffs without outside facts |
| Positive brand constraint | `Versace'den erkek parfümü öner` | Supported | Resolve brand, hard-filter brand, disable cross-brand dedup |
| Negative brand/perfume | `avoid Tom Ford` | Supported | Fuzzy entity resolution + pre-generation hard filter |
| Reference similarity | `something like Aventus` | Supported | Resolve exact reference + document similarity + structured reranking; max one same-brand flanker |
| Dupe/alternative | `a dupe for Sauvage` | Supported | Similarity route + remove the exact reference and its brand |
| Similarity with differences | `like Aventus but less smoky` | Supported | Similarity route + negative trait hard filter |
| Exact database lookup | `show all recorded fields for X` | Supported | Deterministic field copy; bypass generation |
| Contradictory constraints | `sweet vanilla without sweet or vanilla` | Supported | Detect contradiction and request clarification before retrieval |
| Multi-perfume comparison | `X mi Y mi bana daha uygun?` | Supported | Regex is only a fast path; model planner recognizes free-form intent, exact cards ground the model-written comparison |
| Superlative/ranking | `highest-rated summer Versace` | Supported | Explicit metadata sort after hard filters |
| Year/new release | `Versace releases after 2020` | Supported | Parse year bounds and filter `year` metadata |
| Performance request | `long-lasting with strong sillage` | Supported | Runtime catalog provides longevity/sillage for all 131,930 records |
| Budget/current price | `under $100` | Unsupported | Add a timestamped price source; never infer price from the LLM |
| Availability/discontinued | `still available in Turkey?` | Unsupported | Add a current commerce/availability source |
| Exact recommendation count | `give me exactly two` | Supported | Parse count and reject/regenerate incorrect output cardinality |
| Collection-aware advice | `My collection: X, Y. What is missing?` | Supported for explicit lists | Resolve owned bottles, find the least-covered broad wear profile, exclude owned bottles |
| Multiple references | `between Aventus and Hacivat` | Supported | Resolve both references and rank candidates by balanced hybrid similarity to both |
| Layering | `what can I layer with X?` | Unsupported | Requires explicit layering evidence or a dedicated curated dataset |
| Allergy/safety | `safe for asthma` | Must refuse medical assurance | Provide a safety disclaimer and avoid unsupported claims |
| Subjective social claims | `most complimented` | Unsupported | Requires a dedicated evidence source; do not treat ratings as compliments |

Quality priorities after the main intent coverage:

1. Expand automated checks for consultative depth without confusing conservative interpretation with hallucination.
2. Measure single-profile, recommendation, similarity, comparison, and collection quality separately.
3. Add a preference stage only after collecting real model failures and accepted alternatives.
4. Persist collection state outside a single request when an application user-profile store is added.
