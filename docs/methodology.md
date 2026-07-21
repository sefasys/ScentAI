# Methodology

## Training Curriculum

ScentAI uses five synthetic-data levels rather than one undifferentiated instruction set.

| Level | Purpose | Main failure controlled |
| --- | --- | --- |
| L1 | factual cards and direct answers | invented catalog facts |
| L2 | filters, rankings, comparisons, no-match cases | ignored constraints |
| L3 | varied natural-language requests | brittle keyword routing |
| L4 | explanations, tradeoffs, and wear context | database-like prose |
| L5 | preferences and follow-up state | repetitive or contradictory personalization |

The target mix is 4,000 L1, 6,000 L2, 7,000 L3, 10,000 L4, and 5,000 L5 records. L4 is intentionally the largest because persuasive grounded explanation is the primary product behavior.

## Generation Controls

- Deterministic seeds vary between chunks.
- Streaming JSONL checkpoints prevent long API runs from being lost.
- Final L3-L5 generation fails rather than silently falling back to hardcoded language.
- Provider-specific prompting lives behind one compatibility layer.
- Query-opening and phrase-frequency audits detect template collapse.
- Category-level validators run before chunks enter the final set.

## Fine-Tuning

The selected model is Gemma 4 12B instruction-tuned. Training uses QLoRA with response-only loss masking and manual adapter checkpoints suitable for interruptible Colab sessions. A full training notebook is provided at [`notebooks/train_gemma4_lora_colab.ipynb`](../notebooks/train_gemma4_lora_colab.ipynb).

The adapter is not treated as a database. Retrieval supplies facts at inference time; fine-tuning shapes response behavior, grounding discipline, and consultant tone.

## Evaluation Philosophy

Loss alone was not used as the release criterion. The final evaluation checks observable behavior:

- unsupported perfume mentions;
- strict-filter violations;
- requested recommendation counts;
- entity resolution;
- multilingual response consistency;
- multi-turn non-repetition;
- longevity and sillage calibration;
- unsupported request handling;
- fallback and retry rates.

Human review remains important because a response can be technically grounded while still sounding repetitive, mechanical, or unhelpful.

