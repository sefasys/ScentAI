

# Provider Pool Plan

## Goal

Use multiple paid/API providers for L3-L5 query generation without sacrificing quality.

The LLM should generate only user queries. Perfume selection, answers, RAG context, and validation remain programmatic.

## Quality-First Budget

Working budget:

```text
L3 + L4 + L5 query generation: about $2-5
```

This is enough because the model is not writing long answers. It only writes short user requests.

## Preferred Model Mix

Quality-first mix:

```text
Llama 3.3 70B     60%
GPT-4.1 mini      40%
```

Reasoning:

- Llama 3.3 70B is the main natural-language quality backbone.
- GPT-4.1 mini adds strong instruction following and phrasing variety.
- Qwen3 32B is not in the main quality pool. It worked after `/no_think`, but showed enough style repetition and provider-specific risk that it should stay experimental/free-preflight only.
- Gemini is not a default provider because it was unstable for quota/503 errors and is more expensive for output tokens.

## Smoke Before Batch Production

Never run a 250+ record batch with a new provider/model before a small direct smoke test.

Required direct smoke before weighted routing:

```text
L3: 20 records per model
L4: 20 records per model
L5: 20 records per model
```

The smoke must be checked with the level validator and `query_quality.py`.

Hard smoke blockers:

```text
reasoning leakage such as <think> or "think okay"
empty model outputs
prompt analysis instead of user query
fallback records
validator errors
obvious repetitive opening behavior
```

Only after each model passes smoke should it be included in the provider pool.

Model-specific handling should live in `research/core/llm_query.py`. For example, Groq `qwen/qwen3-32b` needs `/no_think`, while other providers may need different handling or no special handling at all.

### OpenRouter Direct Smoke Commands

Set the API key first:

```bash
export OPENROUTER_API_KEY="..."
```

Llama 3.3 70B smoke:

```bash
python research/generate_l3.py --total 20 --query-provider compat --compat-base-url https://openrouter.ai/api/v1 --compat-api-key-env OPENROUTER_API_KEY --compat-model meta-llama/llama-3.3-70b-instruct --fallback-policy fail --checkpoint-every 10 --overwrite --include-debug-meta --output /tmp/l3_openrouter_llama_smoke_20.jsonl
python research/generate_l4.py --total 20 --query-provider compat --compat-base-url https://openrouter.ai/api/v1 --compat-api-key-env OPENROUTER_API_KEY --compat-model meta-llama/llama-3.3-70b-instruct --fallback-policy fail --checkpoint-every 10 --overwrite --include-debug-meta --output /tmp/l4_openrouter_llama_smoke_20.jsonl
python research/generate_l5.py --total 20 --query-provider compat --compat-base-url https://openrouter.ai/api/v1 --compat-api-key-env OPENROUTER_API_KEY --compat-model meta-llama/llama-3.3-70b-instruct --fallback-policy fail --checkpoint-every 10 --overwrite --include-debug-meta --output /tmp/l5_openrouter_llama_smoke_20.jsonl
```

GPT-4.1 mini smoke:

```bash
python research/generate_l3.py --total 20 --query-provider compat --compat-base-url https://openrouter.ai/api/v1 --compat-api-key-env OPENROUTER_API_KEY --compat-model openai/gpt-4.1-mini --fallback-policy fail --checkpoint-every 10 --overwrite --include-debug-meta --output /tmp/l3_openrouter_gpt41mini_smoke_20.jsonl
python research/generate_l4.py --total 20 --query-provider compat --compat-base-url https://openrouter.ai/api/v1 --compat-api-key-env OPENROUTER_API_KEY --compat-model openai/gpt-4.1-mini --fallback-policy fail --checkpoint-every 10 --overwrite --include-debug-meta --output /tmp/l4_openrouter_gpt41mini_smoke_20.jsonl
python research/generate_l5.py --total 20 --query-provider compat --compat-base-url https://openrouter.ai/api/v1 --compat-api-key-env OPENROUTER_API_KEY --compat-model openai/gpt-4.1-mini --fallback-policy fail --checkpoint-every 10 --overwrite --include-debug-meta --output /tmp/l5_openrouter_gpt41mini_smoke_20.jsonl
```

After each smoke, run the level validator and `query_quality.py`.

## Test Before Main Production

Before full L3/L4/L5 production, run a balanced provider test:

```text
L3: 300 records
L4: 300 records
L5: 300 records
```

Suggested split:

```text
Llama 3.3 70B: about 60%
GPT-4.1 mini : about 40%
```

Then inspect:

- exact duplicate queries
- most common opening phrases
- top 5 opening phrase concentration
- model/source distribution
- category distribution
- query length min/avg/max
- template-like phrases such as "I'm looking for"
- weird grammar, truncation, hidden field leakage, or reasoning leakage

## Production Gate Criteria

Do not start main dataset production until the provider test passes these gates.

### Hard Gates

These must be zero:

```text
validator json errors
validator format errors
RAG subset violations
unsupported factual claims
internal generator label leaks
fallback records in final/debug provider tests
```

For L3 specifically:

```text
answer format violations = 0
gender rule violations = 0
negative preference violations = 0
```

For L4 specifically:

```text
best_pick violations = 0
unsupported price/compliment claims = 0
repetitive banned phrase violations = 0
```

### Query Diversity Gates

Target thresholds:

```text
exact duplicate queries              <= 0.5%
most common 3-word opening           <= 15%
most common 4-word opening           <= 10%
top 5 four-word openings combined    <= 35%
"I'm looking for" family             <= 25%
queries under 8 words                = 0
queries over 70 words                <= 2%
```

For a 300-record test:

```text
exact duplicates: ideally 0, maximum 1
most common 3-word opening: <= 45 records
most common 4-word opening: <= 30 records
top 5 four-word openings combined: <= 105 records
```

### L4 Answer Quality Gates

Target thresholds:

```text
top Why opening                       <= 30%
top 3 Why openings combined           <= 65%
answers with weak/no evidence Why     <= 5%
answers with awkward grammar          <= 3%
manual sample quality                 >= 8/10
```

Manual sample quality means:

- explanations are grounded in visible perfume card facts
- answer sounds like a concise expert, not a rigid template
- no invented prices, popularity claims, or guaranteed effects
- no internal labels such as `winter_evening` or `gym_after`
- no obvious mismatch between user query and recommended scent role

### Decision Rule

If all hard gates pass and soft quality gates are close:

```text
Proceed to main production.
```

If hard gates pass but query/answer diversity is weak:

```text
Adjust provider weights or prompt wording, then rerun the 300-record test.
```

If any hard gate fails:

```text
Fix generator/validator before producing more data.
```

## Production Rules

- Use `--fallback-policy fail`.
- Keep checkpoint/resume enabled.
- Keep `_meta.query_source` in debug batches.
- Do not merge debug metadata into final training data unless intentionally needed.
- If one model shows repetitive phrasing, reduce its pool weight before main production.

## Recommended Test Commands

L3:

```bash
python research/generate_l3.py \
  --total 300 \
  --query-provider pool \
  --provider-pool research/provider_pool.json \
  --fallback-policy fail \
  --checkpoint-every 100 \
  --overwrite \
  --include-debug-meta \
  --output /tmp/l3_pool_300_debug.jsonl
```

L4:

```bash
python research/generate_l4.py \
  --total 300 \
  --query-provider pool \
  --provider-pool research/provider_pool.json \
  --fallback-policy fail \
  --checkpoint-every 100 \
  --overwrite \
  --include-debug-meta \
  --output /tmp/l4_pool_300_debug.jsonl
```

L5:

```bash
python research/generate_l5.py \
  --total 300 \
  --query-provider pool \
  --provider-pool research/provider_pool.json \
  --fallback-policy fail \
  --checkpoint-every 100 \
  --overwrite \
  --include-debug-meta \
  --output /tmp/l5_pool_300_debug.jsonl
```

Quality analysis:

```bash
python research/validators/query_quality.py /tmp/l3_pool_300_debug.jsonl
python research/validators/query_quality.py /tmp/l4_pool_300_debug.jsonl
python research/validators/query_quality.py /tmp/l5_pool_300_debug.jsonl
```

## Main Production Direction

If the L3/L4/L5 provider test looks good:

```text
L3: 7,000
L4: 10,000
L5: 5,000
```

Use the same provider pool, then adjust weights based on the query quality report.
