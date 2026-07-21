# ScentAI Final Evaluation v1

This is the release-quality evaluation for the complete Stage 3 system:

`query -> planner -> retrieval -> grounded advisor -> validator -> fallback`

It evaluates the deployed pipeline, not an isolated base model or LoRA adapter.

## Fixed Set

`final_eval_v1.jsonl` contains 120 deterministic cases:

| Category | Cases |
|---|---:|
| Recommendation | 20 |
| Perfume profile | 15 |
| Comparison | 15 |
| Similarity/alternative | 15 |
| Hard filters | 15 |
| Entity resolution | 10 |
| Stateful conversation | 10 |
| Unsupported requests | 10 |
| Multilingual/noisy input | 10 |

The split is 61 English and 59 Turkish cases. Expected perfume labels are checked
against `scentai_catalog.sqlite3` when the set is built.

Rebuild and validate locally with:

```bash
python -m research.evaluation.build_final_eval_v1
python -m unittest research.evaluation.test_final_eval -v
```

## Run In Colab

Use the generated notebook:

```text
notebooks/full_pipeline_colab.ipynb
```

The `Final evaluation v1 - 120 frozen cases` cell is self-contained. The case set
and runner are embedded by `tools/notebook_builders/build_pipeline_notebook.py`; no separate eval
upload is required.

For a ten-case runner smoke test, set:

```python
FINAL_EVAL_LIMIT = 10
```

For the release run, keep:

```python
FINAL_EVAL_LIMIT = None
FINAL_EVAL_RESUME = True
```

Each result is flushed and fsynced to Drive immediately. After a disconnect,
rerunning the cell skips completed IDs and restores multi-turn conversation state.

## Outputs

The notebook writes:

```text
MyDrive/Perfume-Dataset/runs/final_evaluation/
├── final_eval_v1.jsonl
├── final_eval_outputs.jsonl
├── final_eval_summary.json
├── final_eval_human_review.csv
└── final_eval_metadata.json
```

`final_eval_human_review.csv` contains all automatic failures first, followed by a
stratified sample up to 40 rows. Fill the four 1-5 columns manually:

- grounding
- technical accuracy
- advisor value
- naturalness

## Automatic Gates

- overall pass rate >= 95%
- response-language pass rate = 100%
- requested-count pass rate = 100%
- hard-filter pass rate = 100%
- entity-resolution pass rate = 100%
- unsupported-route pass rate = 100%
- conversation no-repeat pass rate = 100%
- performance-calibration pass rate = 100%
- first-attempt generation rate >= 90%
- fallback rate <= 5%
- p50 latency <= 12 seconds
- p95 latency <= 20 seconds

An automatic pass is necessary but not sufficient. The release decision also uses
the completed human-review CSV and the later 30-case ablation run.

## Resume And Fresh Runs

Resume is safe only when the model, adapter, catalog, and orchestrator are unchanged.
To evaluate a changed pipeline from scratch, archive or rename the existing
`runs/final_evaluation` directory before rerunning the same pipeline
revision. A new pipeline revision should use a new sibling output directory so
old and new reports remain directly comparable.
