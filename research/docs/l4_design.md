# L4 Plan

## Purpose

L4 is a grounded reasoning dataset.

It teaches the model to recommend perfumes with short, personalized explanations. L4 should not only answer "what matches?" but also explain "why this matches this user/scenario" using facts present in the perfume cards.

L4 should stay controlled:

- perfume selection is programmatic
- user query generation may use an external LLM
- explanation generation may use an external LLM later, but must be grounded in evidence
- answers must not invent notes, accords, prices, similarity claims, or performance claims

## Boundary With Earlier Levels

L1 is factual database answering.

L2 is strict explicit filtering.

L3 is semantic recommendation matching with list-only output.

L4 is explanatory recommendation:

- interprets user context
- selects suitable perfumes
- explains each recommendation briefly
- makes a best-pick decision when appropriate

## Boundary With L5

L4 is single-turn personalization.

Allowed in L4:

```text
I own Aventus and Terre d'Hermes. I want something warmer for winter nights.
```

The user's preferences are inside the current query.

Not L4; this belongs to L5:

```text
[USER PROFILE]
Likes: Aventus, woody, citrus
Dislikes: sweet, powdery

User: Recommend something for winter.
```

L5 should teach persistent preference memory, profile usage, and multi-turn personalization.

## Output Format

Default L4 answer format:

```text
Based on your preferences, I would prioritize [short reason].

1. Perfume Name by Brand
Why: [1-2 grounded sentences]

2. Perfume Name by Brand
Why: [1-2 grounded sentences]

3. Perfume Name by Brand
Why: [1-2 grounded sentences]

Best pick: Perfume Name by Brand
```

Rules:

- keep explanations concise
- usually recommend 2-3 perfumes
- include `Best pick:` when the scenario asks for a decision or ranking
- no markdown tables
- no long essays
- no chain-of-thought
- no claims that cannot be traced to perfume cards or the user query

## Grounding Rules

If `[PERFUMES]` context exists:

- every recommended perfume must be from context
- every named note/accord/season/time/gender/rating fact must exist in that perfume card
- no outside pricing claims
- no "compliment guaranteed" claims
- no unsupported performance claims beyond card fields such as longevity/sillage when present

If no context exists:

- answer can use selected perfumes from the local dataset
- explanation should still rely on metadata fields
- no fabricated external knowledge

## Categories

Initial L4 categories:

```text
persona_upgrade
occasion_with_tradeoffs
compare_and_decide
avoidance_reasoning
style_translation
collection_gap
no_strong_match
```

Suggested default ratios:

```text
persona_upgrade          18%
occasion_with_tradeoffs  18%
compare_and_decide       14%
avoidance_reasoning      14%
style_translation        16%
collection_gap           15%
no_strong_match           5%
```

## Category Definitions

### persona_upgrade

The user has a current scent, style, or daily habit and wants an upgrade, variation, or more special version.

Example:

```text
I wear Bleu de Chanel to work, but I want something more special for winter dinners.
```

Expected answer:

- acknowledges the current style
- moves toward richer/deeper/more formal options
- explains continuity and difference

### occasion_with_tradeoffs

The user has an occasion plus constraints that need balancing.

Example:

```text
I need a summer office scent that still feels confident, not too casual.
```

Expected answer:

- explains the tradeoff
- avoids over-projecting/heavy picks when the context says office/summer
- recommends options that balance the constraints

### compare_and_decide

The user asks the assistant to choose between options.

Example:

```text
Which of these would be safer for a wedding: X, Y, or Z?
```

Expected answer:

- compares based on context fields
- chooses a best pick
- may mention why the other options are less ideal

### avoidance_reasoning

The user has explicit dislikes or risks to avoid.

Example:

```text
I hate overly sweet vanilla scents but want something warm for winter.
```

Expected answer:

- respects explicit avoid terms as hard constraints where possible
- explains why selected options reduce that risk
- does not recommend perfumes dominated by avoided accords/notes

### style_translation

The user gives an abstract aesthetic or social signal.

Example:

```text
I want something expensive-smelling and understated, not loud.
```

Expected answer:

- translates style words into scent evidence
- stays grounded in accords, notes, season/time, and rating/popularity fields

### collection_gap

The user lists a small current collection and asks what role to add next.

Example:

```text
I own Aventus, Terre d'Hermes, and The One. What should I add for cold weather?
```

Expected answer:

- identifies the missing role from the current collection
- recommends options that fill that role
- stays single-turn; no persistent memory

### no_strong_match

The context does not contain a clearly good match.

Example:

```text
From these options, do any fit a fresh aquatic summer scent without sweetness?
```

Expected answer:

- says no option is a perfect match
- gives the closest option if useful
- explains the mismatch briefly

Keep this category small, around 5%.

## RAG Ratio

Recommended default:

```text
RAG: 85-90%
No RAG: 10-15%
```

L4 should be more RAG-heavy than L3 because explanations are more hallucination-prone.

## LLM Usage

Recommended production approach:

1. Programmatically choose scenario/profile and target perfumes.
2. Programmatically build evidence for each target perfume.
3. Use an external LLM for natural user query generation.
4. Initially generate explanations programmatically for maximum grounding.
5. Later optionally let the LLM rewrite explanations, but only from strict evidence packs.

For production runs:

- prefer Groq/OpenAI-compatible provider
- use `--fallback-policy fail` when LLM-generated text is required
- checkpoint every 100 records
- support resume

## Validation Targets

Minimum checks:

- JSONL parse succeeds.
- Default training output contains only `messages`.
- Debug mode may include `_meta`.
- RAG answers recommend only perfumes from context.
- All `Why:` perfume names appear in the answer list.
- `Best pick:` is one of the recommended perfumes.
- Explicit avoid terms are not violated by dominant accords/notes.
- No answer contains unsupported price claims.
- No answer contains unsupported absolute claims such as guaranteed compliments.
- No answer uses perfumes outside the selected answer IDs in debug mode.
- No internal generator labels such as `winter_evening`, `date_night`, or `gym_after` should appear in final answers.
- No single explanation phrase should dominate the dataset.
- User-facing answers should avoid dataset/process words such as "metadata" unless genuinely necessary.

## Answer Quality Targets

L4 answers should feel like concise expert recommendations, not rigid template output.

Targets:

- category-specific intros instead of one universal opening
- varied `Why:` sentence openings
- natural public labels such as "warm winter evenings", never internal labels such as `winter_evening`
- grounded evidence from accords, notes, season/time, rating, and explicit avoid terms
- no invented price, compliment, or performance claims
- no chain-of-thought or long essay structure
- usually 2-3 recommendations plus `Best pick:`

## Production Gate

Before main L4 production, run at least a 300-record debug batch.

Required gates:

```text
validator errors                       = 0
fallback records                       = 0
exact duplicate queries                <= 0.5%
most common 4-word query opening       <= 10%
top 5 four-word query openings         <= 35%
top Why opening                        <= 30%
top 3 Why openings combined            <= 65%
internal label leaks                   = 0
unsupported claims                     = 0
manual sample quality                  >= 8/10
```

If these pass, L4 is eligible for main dataset production.

## Query Diversity Targets

L4 queries should be richer than L3 queries.

Target mix:

- 20-25% current-scent upgrade requests
- 20-25% occasion plus tradeoff requests
- 10-15% direct comparison requests
- 10-15% dislike/avoidance-heavy requests
- 10-15% abstract style requests
- 10-15% small collection gap requests
- about 5% no-strong-match requests

Quality targets:

- avoid overusing "I'm looking for..."
- include realistic user context, but keep most queries under 70 words
- exact duplicate queries should be 0
- persona details should feel useful, not decorative

## Open Decisions

- Final L4 total count.
- Whether L4 explanations should stay fully programmatic or be LLM-rewritten after validation.
- Whether every answer should include exactly 3 recommendations.
- Whether `Best pick:` should appear in all categories or only decision-heavy categories.
- How much comparison detail to include for rejected options in `compare_and_decide`.
