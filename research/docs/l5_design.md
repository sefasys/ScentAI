# L5 Plan

## Purpose

L5 is a preference-aware personalization dataset.

It teaches the model to use a persistent user profile or preference memory when answering a new perfume request.

L5 should answer:

```text
Given what we already know about this user, how should the recommendation change?
```

## Boundary With L4

L4 is single-turn personalization.

The user puts preferences inside the current query:

```text
I own Aventus and Terre d'Hermes. I want something warmer for winter nights.
```

L5 is profile-aware personalization.

Preferences are stored separately from the current query:

```text
[USER PROFILE]
Likes: Aventus by Creed; woody, citrus
Dislikes: sweet, powdery
Previously recommended: Bleu de Chanel by Chanel
[/USER PROFILE]

User: Recommend something for winter.
```

The model must use the profile, but only when relevant.

## Core Behaviors

L5 should teach the model to:

- use liked perfumes, liked notes, and liked accords as positive signals
- respect disliked notes/accords/perfumes as negative signals
- avoid previously recommended perfumes when the user asks for something new
- handle empty or weak profiles conservatively
- explain how the profile affected the recommendation
- not overfit to profile when the current query clearly asks for something different
- admit when the profile and current query conflict

## Profile Format

Use a structured block before `[PERFUMES]`:

```text
[USER PROFILE]
Liked perfumes: Aventus by Creed; Terre d'Hermes by Hermès
Disliked perfumes: Sauvage by Dior
Liked notes: bergamot; vetiver; cedar
Disliked notes: vanilla; coconut
Liked accords: woody; citrus; aromatic
Disliked accords: sweet; powdery
Preferred gender: male + unisex
Preferred seasons: spring; summer
Preferred times: day
Previously recommended: Bleu de Chanel by Chanel
Profile confidence: medium
[/USER PROFILE]
```

Rules:

- include only fields that matter for the sample
- empty profiles should be explicit:

```text
[USER PROFILE]
No stable preferences recorded yet.
[/USER PROFILE]
```

## Output Format

Default L5 answer format:

```text
Using your profile, I would prioritize [short profile-aware reason].

1. Perfume Name by Brand
Why: [grounded reason connecting profile + current request]

2. Perfume Name by Brand
Why: [grounded reason connecting profile + current request]

3. Perfume Name by Brand
Why: [grounded reason connecting profile + current request]

Best pick: Perfume Name by Brand
```

Rules:

- concise expert tone
- no long essays
- no chain-of-thought
- no unsupported price or compliment claims
- if profile is empty, do not pretend to know user taste
- if profile conflicts with query, mention the tradeoff briefly

## Categories

Initial L5 categories:

```text
empty_profile
profile_likes
profile_dislikes
profile_likes_and_dislikes
avoid_previous_recommendations
profile_query_conflict
profile_update_request
low_confidence_profile
```

Suggested default ratios:

```text
empty_profile                    10%
profile_likes                    18%
profile_dislikes                 16%
profile_likes_and_dislikes       20%
avoid_previous_recommendations   12%
profile_query_conflict           12%
profile_update_request            6%
low_confidence_profile            6%
```

## Category Definitions

### empty_profile

The profile has no stable preferences.

Expected behavior:

- do not claim personalization
- answer based on current query and context
- optionally say the profile does not add much yet

### profile_likes

The profile contains positive taste signals.

Expected behavior:

- recommend perfumes matching liked accords/notes/perfumes
- mention the profile influence briefly

### profile_dislikes

The profile contains strong dislikes.

Expected behavior:

- avoid disliked notes/accords/perfumes
- explain that avoidance without overemphasizing it

### profile_likes_and_dislikes

The profile has both positive and negative signals.

Expected behavior:

- satisfy current query
- boost likes
- avoid dislikes
- explain both sides concisely

### avoid_previous_recommendations

The profile contains previously recommended perfumes.

Expected behavior:

- do not repeat previously recommended perfumes
- recommend alternatives that still fit taste/current request

### profile_query_conflict

The profile and current query pull in different directions.

Example:

```text
Profile dislikes sweet scents.
User asks for a cozy vanilla winter scent.
```

Expected behavior:

- acknowledge the tradeoff
- choose options that satisfy the current query while minimizing conflict
- do not ignore either side

### profile_update_request

The user asks to update or use a changed preference.

Example:

```text
I used to avoid rose, but lately I want to try a soft rose scent.
```

Expected behavior:

- respect the current message over old profile when explicit
- avoid treating old dislike as permanent

### low_confidence_profile

The profile is weak, sparse, or based on one signal.

Expected behavior:

- use it softly
- avoid overconfident claims
- keep recommendations mostly grounded in the current request

## RAG Rules

Recommended default:

```text
RAG: 90%
No RAG: 10%
```

If `[PERFUMES]` context exists:

- answer only from context
- profile may influence ranking within context
- profile must not justify choosing a perfume outside context

## Selection Strategy

Programmatic selection should combine:

```text
current query semantic profile
+ liked profile signals
- disliked profile signals
- previously recommended IDs
```

Suggested weighting:

```text
current query hard constraints       highest priority
explicit dislikes                    hard exclusion where possible
previously recommended               hard exclusion in avoid_previous category
profile likes                        soft boost
low-confidence likes                 small soft boost
profile/query conflict               current query wins, but safer compromise preferred
```

## LLM Usage

Use external LLM only for natural user query generation.

Answers should initially remain programmatic and evidence-grounded, like L4.

Later optional upgrade:

- LLM rewrite from strict evidence pack
- validate against profile/context facts

## Validation Targets

Minimum checks:

- JSONL parse succeeds
- default output contains only `messages`
- debug mode may include `_meta`
- RAG answers recommend only perfumes from context
- answer IDs do not include `previously_recommended_ids` for avoid-previous samples
- answer IDs do not include disliked perfume IDs
- explicit disliked notes/accords are not present in selected perfumes where hard-avoid applies
- empty profile answers do not claim strong personalization
- profile/query conflict answers mention the tradeoff
- `Best pick:` is one of the recommended perfumes

## Production Gate

Before main L5 production, run a 300-record debug batch.

Required gates:

```text
validator errors                       = 0
fallback records                       = 0
exact duplicate queries                <= 0.5%
most common 4-word query opening       <= 10%
top 5 four-word query openings         <= 35%
profile misuse violations              = 0
previous recommendation repeats         = 0
manual sample quality                  >= 8/10
```

## Open Decisions

- How verbose should profile-aware explanations be?
- Should L5 always include `Best pick:`?
- Should profile update samples output only a recommendation, or also acknowledge the profile change?
- How often should empty/low-confidence profiles appear in final data?
