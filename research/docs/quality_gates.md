# Quality Test Plan

## Purpose

This plan defines the quality gate before main dataset production.

The goal is to avoid producing thousands of records with hidden repetition, weak wording, fallback queries, RAG violations, or unsupported claims.

## Scope

Run this plan before main production for:

```text
L3 semantic list recommendations
L4 grounded reasoning recommendations
L5 preference/profile-aware recommendations
```

L1 and L2 are mostly deterministic, so they need validator and spot-checking, but not provider quality testing.

## Test Batch Sizes

Recommended provider quality test:

```text
L3: 300 records
L4: 300 records
L5: 300 records
```

These batches should use:

```text
--include-debug-meta
--fallback-policy fail
--checkpoint-every 100
```

## Provider Pool

Use:

```text
research/provider_pool.json
```

Default quality mix:

```text
Llama 3.3 70B     60%
GPT-4.1 mini      40%
```

Qwen is intentionally not part of the main OpenRouter quality pool. It can be used for free/Groq experiments, but the Qwen preflight showed enough style repetition and model-specific handling that it should not carry main production quality.

The exact observed distribution can vary slightly because provider selection is weighted per request.

## Provider Smoke Rule

Before running any 250+ record batch with a new provider or model, run a small smoke test first.

Minimum smoke target:

```text
L3: 20 records
L4: 20 records
L5: 20 records
```

For provider-pool production, also run at least one direct smoke test per model in the pool before using weighted routing.

The smoke test must pass:

```text
validator errors        = 0
duplicate queries       = 0
reasoning leaks         = 0
empty model outputs     = 0
fallback records        = 0
obvious prompt leakage  = 0
```

Reasoning leaks include output such as:

```text
<think>...</think>
think okay...
let me tackle this...
the user wants...
```

Do not start a 250/300-record preflight until the smoke output has been inspected with both:

```bash
python research/validators/<level>.py <file>
python research/validators/query_quality.py <file>
```

If a model needs special handling, add it to `research/core/llm_query.py` before rerunning the smoke test. For example, Groq `qwen/qwen3-32b` needs `/no_think` and empty think-tag cleanup.

Current OpenRouter smoke order:

```text
1. meta-llama/llama-3.3-70b-instruct
2. openai/gpt-4.1-mini
```

## Commands

### L3 Test

```bash
python research/generate_l3.py \
  --total 300 \
  --query-provider pool \
  --provider-pool research/provider_pool.json \
  --fallback-policy fail \
  --checkpoint-every 100 \
  --overwrite \
  --include-debug-meta \
  --output /tmp/l3_quality_gate_300.jsonl
```

Validate:

```bash
python research/validators/l3.py /tmp/l3_quality_gate_300.jsonl
python research/validators/query_quality.py /tmp/l3_quality_gate_300.jsonl
```

### L4 Test

```bash
python research/generate_l4.py \
  --total 300 \
  --query-provider pool \
  --provider-pool research/provider_pool.json \
  --fallback-policy fail \
  --checkpoint-every 100 \
  --overwrite \
  --include-debug-meta \
  --output /tmp/l4_quality_gate_300.jsonl
```

Validate:

```bash
python research/validators/l4.py /tmp/l4_quality_gate_300.jsonl
python research/validators/query_quality.py /tmp/l4_quality_gate_300.jsonl
```

### L5 Test

Add after L5 generator and validator exist.

Expected shape:

```bash
python research/generate_l5.py \
  --total 300 \
  --query-provider pool \
  --provider-pool research/provider_pool.json \
  --fallback-policy fail \
  --checkpoint-every 100 \
  --overwrite \
  --include-debug-meta \
  --output /tmp/l5_quality_gate_300.jsonl
```

Expected validation:

```bash
python research/validators/l5.py /tmp/l5_quality_gate_300.jsonl
python research/validators/query_quality.py /tmp/l5_quality_gate_300.jsonl
```

## Hard Gates

These must be zero:

```text
JSON parse errors
format errors
RAG subset violations
fallback records
internal generator label leaks
unsupported claims
```

Level-specific hard gates:

```text
L3 answer list-format violations      = 0
L3 gender violations                  = 0
L3 negative preference violations     = 0
L4 best_pick violations               = 0
L4 repetitive banned phrase violations= 0
L5 profile contradiction violations   = 0
L5 profile-memory misuse              = 0
```

## Query Diversity Gates

For each 300-record test:

```text
exact duplicate queries              <= 1
most common 3-word opening           <= 45
most common 4-word opening           <= 30
top 5 four-word openings combined    <= 105
"I'm looking for" family             <= 75
queries under 8 words                = 0
queries over 70 words                <= 6
```

If a single provider causes most repetition, reduce its pool weight and rerun the test.

## L4 Answer Gates

For L4 300-record test:

```text
top Why opening                       <= 90
top 3 Why openings combined           <= 195
answers with weak/no evidence Why     <= 15
answers with awkward grammar          <= 9
manual sample quality                 >= 8/10
```

Manual review sample:

```text
30 records total
5 records from each major L4 category if possible
at least 5 records from each provider source if possible
```

Manual scoring:

```text
10 = natural, grounded, useful
8  = good enough for training
6  = technically valid but too stiff/repetitive
4  = weak, awkward, or misleading
0  = invalid or unsafe for training
```

Main production requires average manual score:

```text
>= 8/10
```

## L3 Answer Gates

For L3 300-record test:

```text
answers must be list-only
no explanations
no intro text
RAG answers only from context
manual sample quality >= 8/10
```

Manual review sample:

```text
30 records total
cover all L3 categories
include at least 10 RAG records
include at least 5 no/low-context records if present
```

## Provider Review

After query quality analysis, inspect distribution by source:

```text
pool:openrouter_llama33_70b
pool:openrouter_gpt41_mini
```

For each source, check:

```text
duplicate count
average query length
top opening phrases
category coverage
manual naturalness
grammar issues
hidden-field leakage
```

Provider action rules:

```text
excellent quality and low repetition -> keep or increase weight
good but repetitive                  -> reduce weight
grammar/format issues                -> reduce heavily or remove
hidden-field leakage                 -> remove until prompt is fixed
```

## Pass/Fail Decision

Pass:

```text
all hard gates pass
query diversity gates pass or are very close
manual quality average >= 8/10
no provider shows severe failure
```

Conditional pass:

```text
all hard gates pass
minor repetition issue exists
adjust provider weights before main production
```

Fail:

```text
any hard gate fails
manual quality average < 8/10
one provider has repeated grammar/format failure
```

## Main Production Approval

Only after the relevant quality tests pass:

```text
L3 main production: 7,000 records
L4 main production: 10,000 records
L5 main production: 5,000 records
```

Production should still use:

```text
--fallback-policy fail
--checkpoint-every 100
--resume when interrupted
```
