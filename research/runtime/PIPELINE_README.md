# ScentAI Runtime Pipeline

The runtime path is:

1. Gemma plans the free-form query into an evidence-bearing JSON intent and constraints.
   This covers recommendation, single-perfume profile, similarity, alternative, preference, comparison, exact lookup, collection, ranking, safety boundaries, counts, brands, years, and positive/negative preferences.
2. Validate planner evidence against the exact user text.
   The former regex analyzer remains a safety fallback, not the primary understanding layer.
3. Retrieve candidates from `chroma_db_bge_m3` with BGE-M3.
   Reference-similarity queries resolve one or two exact source perfumes, then combine community votes, semantic retrieval, and structured similarity.
4. Apply metadata constraints and hard exclusions before generation.
5. Build grounded perfume cards and native Gemma 4 chat messages.
6. Generate with the pilot LoRA adapter.
7. Turn card evidence into consultant-style character, wear-context, performance, and tradeoff explanations.
8. Validate perfume names, card facts, strict filters, excluded entities, and mechanical catalog-answer patterns.
9. Retry once after a validation failure; use a grounded explanatory fallback if the retry also fails.
10. Route exact database-field requests to deterministic field copying after model intent recognition.

Natural questions about one perfume use a model-written grounded profile. Explicit requests for exact,
verbatim, or raw database fields remain deterministic.

## Colab

Upload these three files to Drive:

- `colab_scentai_full_pipeline.ipynb`: open this in Colab and run all cells.
- `scentai_runtime_bundle.zip`: place it at `MyDrive/Perfume-Dataset/scentai_runtime_bundle.zip`.
- `scentai_catalog.zip`: place it at `MyDrive/Perfume-Dataset/scentai_catalog.zip`; the notebook extracts it automatically.

The notebook expects these existing assets:

- `MyDrive/Perfume-Dataset/chroma_db_bge_m3`
- `MyDrive/Perfume-Dataset/scentai_catalog.sqlite3`
- `MyDrive/Perfume-Dataset/models/scentai-gemma-4-12b-it-pilot-fastmodel-lora/best_lora_adapter`

Change only `my_prompt` in the final cell for interactive use. Put exclusions in the same natural-language query. Results are appended to:

`MyDrive/Perfume-Dataset/runs/interactive_pipeline_results.jsonl`

On GPUs with at least 35 GB VRAM (including an A100 40 GB), the notebook loads Gemma 4 in BF16 for
faster, unquantized inference. Smaller GPUs automatically use 4-bit NF4. Each request prints planner,
answer, and total elapsed time so retries and generation bottlenecks are visible.

## vLLM latency benchmark

`colab_scentai_vllm.ipynb` is an isolated A100 benchmark for the same Gemma 4 12B
base model, standard rank-16 pilot LoRA, retrieval stack, prompts, and validators.
It does not merge, quantize, retrain, or replace the model. Do not install vLLM
inside the working Transformers notebook session; open the vLLM notebook in a
fresh Colab runtime and use `Run all`.

The notebook deliberately uses two isolated Python environments. Chroma, BGE-M3,
NumPy, and SciPy remain in the Colab kernel. vLLM and its CUDA 12.9 PyTorch build
are installed in `/content/vllm_env` and run as a localhost-only API server. This
prevents vLLM's compiled dependencies from replacing or corrupting retrieval
dependencies in the notebook kernel.

The optimized server uses one BF16 vLLM engine. Planner requests select the base
model, while answer requests select the existing adapter through vLLM's LoRA model
name. Prompts are rendered and tokenized by the saved Gemma 4 tokenizer before
their token IDs are sent to the local server, preserving the proven chat-template
behavior. Planner JSON is constrained by a schema, ordinary answers are capped
separately from comparisons, and CUDA/grammar/LoRA warm-up is excluded from
benchmark timings.

Upload the current `scentai_runtime_bundle.zip` alongside the existing assets before
running the notebook. The five-query same-prompt benchmark writes detailed calls to
`runs/interactive_pipeline_vllm_results.jsonl` and its aggregate report to
`runs/vllm_benchmark_report.json`. The report includes mean, median, p90, output
tokens per second, speedup against the measured Transformers baseline, planner parse
errors, retries, template fallbacks, grounding pass rate, and length-limited calls.

## Runtime Modules

- `query_analyzer.py`: intent and hard-constraint extraction.
- `rag.py`: Chroma retrieval, filtering, scoring, and grounded cards.
- `prompts.py`: native chat messages and strict-filter instructions.
- `model_pipeline.py`: orchestration, retry, validation, and safe fallback.
- `exact_lookup.py`: deterministic database-field responses.
- `grounding_checker.py`: output checks against the provided cards.
- `INTENT_RISK_MATRIX.md`: supported, partial, and unsupported user-query families.
- `catalog.py`: exact identity resolution, performance metrics, and community similarity graph.

## Supported Query Shapes

- `Something like Aventus but less smoky.`
- `A dupe for Aventus.`
- `Something between Aventus and Hacivat.`
- `Club de Nuit Man ile Adidas Team Five arasında kaldım; hangisi hangi ortamda daha mantıklı?`
- `My collection: Aventus, Bleu de Chanel. What is missing from my collection?`
- `Recommend exactly two highest-rated Versace perfumes after 2020.`

Collection parsing is intentionally explicit. Persistent user collections belong in an application profile store; the notebook does not silently infer ownership from ordinary prose.
