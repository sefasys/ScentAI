# ScentAI Evaluation Suite v2

This is the quality gate before the full fine-tune run.

The eval suite does not train the model and does not change model outputs. It gives a repeatable way to compare pilot, full, and checkpoint variants on the same cases.

## What v2 Measures

- context-only behavior
- unsupported perfume mentions
- strict negative filters
- forbidden perfume recommendations
- unsupported note claims
- exact field-copy drift for database lookup answers
- minimum context perfume mention
- rough expected-overlap with held-out answers
- category, mode, difficulty, and tag-level failure patterns

## Files

- `build_eval_set.py`: builds the fixed stratified eval set.
- `colab_eval_runner_cell.py`: Colab generation cell, with resume and per-answer flushing.
- `score_eval_outputs.py`: scores generated answers and writes JSON/Markdown reports.
- `build_human_review_pack.py`: creates a compact failure/near-miss review pack.

## Build Eval Set

Run once locally:

```bash
python -m research.evaluation.build_eval_set \
  --input train_set/finetune/gemma/full_validation.jsonl \
  --output train_set/eval/scentai_eval_v2.jsonl \
  --max-cases 160
```

Upload/sync these to Drive:

```text
train_set/eval/scentai_eval_v2.jsonl
train_set/eval/scentai_eval_v2.manifest.json
```

## Generate Outputs In Colab

After the LoRA/model is loaded, run the notebook cell:

```text
COPY/UPDATE EVAL CELL: scentai-eval-runner
```

It writes:

```text
OUTPUT_DIR/scentai_eval_v2_outputs.jsonl
OUTPUT_DIR/scentai_eval_v2_outputs.metadata.json
```

The runner resumes automatically if the output file already has completed case IDs.

## Score Outputs Locally

```bash
python -m research.evaluation.score_eval_outputs \
  --eval-set train_set/eval/scentai_eval_v2.jsonl \
  --outputs path/to/scentai_eval_v2_outputs.jsonl \
  --report train_set/eval/pilot_eval_v2_report.json
```

Then build a review pack:

```bash
python -m research.evaluation.build_human_review_pack \
  --report train_set/eval/pilot_eval_v2_report.json \
  --output train_set/eval/pilot_eval_v2_human_review.md
```

## Suggested Pre-Full Gates

Use these before spending time on the full run:

- overall `pass_rate >= 0.90`
- `strict_filter_pass_rate >= 0.95`
- `field_copy_pass_rate >= 0.90`
- `context_only_rate >= 0.98`
- `minimum_context_mention_rate >= 0.95`

If the model fails only database lookup but passes recommendation categories, we can either:

- improve prompt/database lookup mode and re-evaluate, or
- increase exact-copy examples in the fine-tune data before full run.

## Suggested Post-Full Gates

- overall `pass_rate >= 0.95`
- `strict_filter_pass_rate >= 0.97`
- `field_copy_pass_rate >= 0.95`
- manually inspect 30-50 cases from the human review pack

## Important Limitations

This deterministic eval still does not fully measure:

- subjective recommendation quality
- nuanced ranking quality
- tone/style preferences beyond obvious format issues
- whether the answer is the best possible perfume advice

For final model selection, use this suite plus a small human or LLM-judge review.

