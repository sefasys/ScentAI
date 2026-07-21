# Preflight Run Log

## 2026-06-27 - Groq L3 Preflight Attempt

Command target:

```text
L3: 300 records
Provider: Groq, llama-3.3-70b-versatile
Fallback policy: fail
Output: /tmp/l3_groq_preflight_300.jsonl
```

Result:

```text
Generated records: 271/300
Stopped because Groq TPD reached ~100K tokens.
```

Validator result on 271 records:

```text
json        : 0
list_format : 0
rag_subset  : 0
gender      : 0
negative    : 0
```

Query quality result on 271 records:

```text
duplicate queries         : 0
query words               : min 9 / avg 29.0 / max 76
top 3-word opening        : 40x "lately i've been"
top 4-word opening        : 31x "lately i've been craving"
template-like hits        : 3x "I'm looking for", 1x "I want something"
```

Gate status:

```text
Hard validator gates: pass
Duplicate gate: pass
Most common 3-word opening gate: pass
Most common 4-word opening gate: borderline fail
```

Action taken:

```text
Added "Lately I have been craving" / "Lately I've been craving" as a discouraged opening in query generation and query quality checks.
```

Next step:

```text
Rerun L3 300 preflight after Groq quota refresh, or switch the full quality gate to OpenRouter.
Then run L4 300 and L5 300 with the same validator + query_quality checks.
```

## 2026-06-27 - Groq Qwen No-Think Preflight

Model:

```text
Groq qwen/qwen3-32b
Special handling: append /no_think and strip empty <think></think> tags.
```

Important lesson:

```text
Do not start 250+ records with a new provider/model before a direct smoke test.
The first Qwen 300 run leaked reasoning as <think>...</think> and was rejected.
After /no_think, reasoning leakage disappeared.
```

### L3

```text
Output: /tmp/l3_groq_qwen_nothink_preflight_300.jsonl
Generated records: 300/300
RAG ratio: 221/300
```

Validator:

```text
json        : 0
list_format : 0
rag_subset  : 0
gender      : 0
negative    : 0
```

Query quality:

```text
duplicate queries      : 0
query words            : min 9 / avg 23.2 / max 65
reasoning leakage      : 0
top 4-word opening     : 29x "lately i ve been"
status                 : pass, but close to diversity limit
```

### L4

```text
Output: /tmp/l4_groq_qwen_nothink_preflight_250.jsonl
Generated records: 250/250
RAG ratio: 226/250
```

Validator:

```text
json                  : 0
format                : 0
rag_subset            : 0
best_pick             : 0
unsupported_claim     : 0
internal_label        : 0
repetitive_phrase     : 0
negative              : 0
```

Query quality:

```text
duplicate queries      : 0
query words            : min 10 / avg 23.8 / max 65
reasoning leakage      : 0
top 4-word opening     : 27x "lately i ve been"
status                 : structural pass, diversity warning
```

Action after L4:

```text
Keep "lately I have been" as a prompt-level discouraged opening.
Do not hard-reject diversity-only openings during generation; let query_quality report them.
```

### L5

```text
Output: /tmp/l5_groq_qwen_nothink_preflight_250.jsonl
Generated records: 90/250
Stopped because Groq qwen/qwen3-32b reached RPD 1000.
```

Validator on 90 records:

```text
json                  : 0
format                : 0
profile_block         : 0
rag_subset            : 0
best_pick             : 0
unsupported_claim     : 0
disliked_perfume      : 0
previous_repeat       : 0
negative              : 0
empty_profile_misuse  : 0
conflict              : 0
```

Query quality on 90 records:

```text
duplicate queries      : 0
query words            : min 12 / avg 23.3 / max 37
reasoning leakage      : 0
top 4-word opening     : 8x "lately i've been drawn"
status                 : partial structural pass
```

Next step:

```text
Resume L5 after Groq RPD refresh, or complete remaining L5 quality gate with OpenRouter.
For OpenRouter, run 20-record direct smoke per model before any 250+ batch.
```

## 2026-06-27 - OpenRouter Provider Configuration

Decision:

```text
Qwen removed from the main OpenRouter quality pool.
Qwen remains experimental/free-preflight only.
```

Active provider pool:

```text
File: research/provider_pool.json

openrouter_llama33_70b : 60%
openrouter_gpt41_mini  : 40%
```

Required next step before OpenRouter preflight:

```text
Run direct 20-record smoke tests for each model:

1. meta-llama/llama-3.3-70b-instruct
2. openai/gpt-4.1-mini

For each model:
L3: 20
L4: 20
L5: 20
```

Only after all direct smoke tests pass should the weighted provider-pool preflight start:

```text
L3: 300
L4: 300
L5: 300
```

## 2026-06-27 - OpenRouter Direct Smoke Results

API key:

```text
Loaded from openrouter_api.txt. Key was not printed.
```

### Llama 3.3 70B

Model:

```text
meta-llama/llama-3.3-70b-instruct
```

Smoke outputs:

```text
L3: /tmp/l3_openrouter_llama_smoke_20.jsonl
L4: /tmp/l4_openrouter_llama_smoke_20.jsonl
L5: /tmp/l5_openrouter_llama_smoke_20.jsonl
```

Results:

```text
L3 validator errors: 0
L4 validator errors: 0
L5 validator errors: 0
duplicate queries  : 0 for all three smoke files
reasoning leakage  : 0 for all three smoke files
fallback records   : 0
```

Query style notes:

```text
L3 avg words: 27.8
L4 avg words: 30.8
L5 avg words: 29.2
Natural, varied, slightly richer than GPT-4.1 mini.
```

Decision:

```text
Pass. Keep as main provider.
```

### GPT-4.1 Mini

Model:

```text
openai/gpt-4.1-mini
```

Smoke outputs:

```text
L3: /tmp/l3_openrouter_gpt41mini_smoke_20.jsonl
L4: /tmp/l4_openrouter_gpt41mini_smoke_20.jsonl
L5: /tmp/l5_openrouter_gpt41mini_smoke_20.jsonl
```

Results:

```text
L3 validator errors: 0
L4 validator errors: 0
L5 validator errors: 0
duplicate queries  : 0 for all three smoke files
reasoning leakage  : 0 for all three smoke files
fallback records   : 0
```

Query style notes:

```text
L3 avg words: 20.1
L4 avg words: 23.6
L5 avg words: 19.3
Clean, compact, strong instruction following.
```

Decision:

```text
Pass. Keep as 40% provider.
```

Next step:

```text
Run weighted provider-pool preflight:

L3: 300
L4: 300
L5: 300

Provider pool:
Llama 3.3 70B 60%
GPT-4.1 mini  40%
```

## 2026-06-27 - DeepSeek V4 Flash Candidate Smoke

Model:

```text
deepseek/deepseek-v4-flash
```

Important handling:

```text
OpenRouter returned reasoning-only output when no reasoning control was set.
`research/core/llm_query.py` now sends:

reasoning: {"enabled": false}

for deepseek/deepseek-v4-flash on OpenRouter.
```

Smoke outputs:

```text
L3: /tmp/l3_openrouter_deepseekv4flash_smoke_20.jsonl
L4: /tmp/l4_openrouter_deepseekv4flash_smoke_20.jsonl
L5: /tmp/l5_openrouter_deepseekv4flash_smoke_20.jsonl
```

Results:

```text
L3 validator errors: 0
L4 validator errors: 0
L5 validator errors: 0
duplicate queries  : 0 for all three smoke files
reasoning leakage  : 0 for all three smoke files
fallback records   : 0
```

Query style notes:

```text
L3 avg words: 22.7
L4 avg words: 26.3
L5 avg words: 22.6
Clean and varied in the 20-record smoke.
```

Decision:

```text
Candidate passed smoke.
Do not replace Llama/GPT with it yet.
If included, start with a small weight such as 10-15% and verify in provider-pool preflight.
```

## 2026-06-27 - OpenRouter 60/40 Provider-Pool Preflight

Provider pool:

```text
openrouter_llama33_70b : 60%
openrouter_gpt41_mini  : 40%
DeepSeek               : not included
Qwen                   : not included
```

### L3

```text
Output: /tmp/l3_openrouter_pool_preflight_300.jsonl
Generated records: 300/300
RAG ratio: 221/300
Provider split: Llama 176 / GPT-4.1 mini 124
```

Validator:

```text
json        : 0
list_format : 0
rag_subset  : 0
gender      : 0
negative    : 0
```

Query quality:

```text
duplicate queries      : 0
query words            : min 9 / avg 25.9 / max 72
reasoning leakage      : 0
top 4-word opening     : 10x "i usually go for"
status                 : pass
```

### L4

```text
Output: /tmp/l4_openrouter_pool_preflight_300.jsonl
Generated records: 300/300
RAG ratio: 269/300
Provider split: Llama 171 / GPT-4.1 mini 129
```

Validator:

```text
json                  : 0
format                : 0
rag_subset            : 0
best_pick             : 0
unsupported_claim     : 0
internal_label        : 0
repetitive_phrase     : 0
negative              : 0
```

Query quality:

```text
duplicate queries      : 0
query words            : min 9 / avg 27.3 / max 58
reasoning leakage      : 0
top 4-word opening     : 11x "avoiding anything too overpowering"
status                 : pass
```

### L5

```text
Output: /tmp/l5_openrouter_pool_preflight_300.jsonl
Generated records: 300/300
RAG ratio: 269/300
Provider split: Llama 174 / GPT-4.1 mini 126
```

Note:

```text
GPT-4.1 mini returned one transient 500 during generation.
The provider pool recovered by using the other provider.
No fallback records were produced.
```

Validator:

```text
json                  : 0
format                : 0
profile_block         : 0
rag_subset            : 0
best_pick             : 0
unsupported_claim     : 0
disliked_perfume      : 0
previous_repeat       : 0
negative              : 0
empty_profile_misuse  : 0
conflict              : 0
```

Query quality:

```text
duplicate queries      : 0
query words            : min 8 / avg 23.7 / max 48
reasoning leakage      : 0
top 4-word opening     : 8x "lately i've developed a"
status                 : pass
```

Decision:

```text
OpenRouter 60/40 pool passed preflight.
Eligible for main dataset production.
```
