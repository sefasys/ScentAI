---
base_model: google/gemma-4-12B-it
library_name: peft
pipeline_tag: text-generation
license: cc-by-nc-sa-4.0
language:
- en
- tr
datasets:
- sefasoysal/scentai-32k-grounded-perfume-conversations
tags:
- peft
- lora
- gemma4
- transformers
- perfume
- recommendation
- retrieval-augmented-generation
---

# ScentAI Gemma 4 12B LoRA

ScentAI is a rank-16 LoRA adapter for `google/gemma-4-12B-it`, trained on the
[ScentAI 32K Grounded Perfume Conversations](https://www.kaggle.com/datasets/sefasoysal/scentai-32k-grounded-perfume-conversations)
curriculum. It adapts Gemma 4 toward evidence-grounded perfume consultation, including factual
profiles, hard constraints, semantic recommendations, comparisons, and preference-aware responses.

The complete retrieval, orchestration, validation, evaluation, and deployment source is available
in the [ScentAI GitHub repository](https://github.com/sefasys/ScentAI).

This repository contains only the PEFT adapter. It does not duplicate the approximately 24 GB base
model, the 131,930-item perfume catalog, the BGE-M3 index, or the ScentAI validation pipeline.

## Important Scope

The complete ScentAI application combines:

1. canonical perfume-name resolution;
2. BGE-M3 semantic retrieval and structured filtering;
3. the Gemma 4 base model and this LoRA adapter;
4. deterministic grounding, hard-filter, language, and response validators.

The adapter alone has no live perfume database. For reliable recommendations, provide retrieved
perfume evidence in the prompt and validate generated names and claims. The published pipeline-level
evaluation should not be interpreted as an isolated benchmark of the adapter weights.

## Adapter Configuration

| Field | Value |
| --- | --- |
| Base model | `google/gemma-4-12B-it` |
| PEFT method | LoRA |
| Rank | 16 |
| Alpha | 32 |
| Dropout during training | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| DoRA | No |
| Training data | 30,400 train / 1,600 validation examples |
| Primary training language | English |

## Loading With Transformers And PEFT

Install current Transformers, PEFT, Accelerate, and PyTorch versions that support Gemma 4:

```bash
pip install -U transformers peft accelerate torch
```

Load the official processor from the base model and attach the adapter explicitly:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

BASE_MODEL = "google/gemma-4-12B-it"
ADAPTER_MODEL = "sefasoysal/scentai-gemma-4-12b-it-lora"

processor = AutoProcessor.from_pretrained(BASE_MODEL)
base_model = AutoModelForMultimodalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
model.eval()

messages = [
    {
        "role": "system",
        "content": (
            "You are ScentAI, an expert perfume consultant. Use only the supplied "
            "perfume evidence for product facts and recommendations."
        ),
    },
    {
        "role": "user",
        "content": """[PERFUMES]
Versace Pour Homme by Versace
Accords: floral, musky, fresh spicy, citrus, aromatic, green, fresh
Best seasons: spring, summer | Time: day

Prada L'Homme by Prada
Accords: iris, powdery, clean, woody, amber
Best seasons: spring, summer, autumn | Time: day

[QUERY]
Recommend a clean daytime office fragrance and explain the trade-off.""",
    },
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=False,
).to(model.device)

prompt_length = inputs["input_ids"].shape[-1]
with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=260,
        do_sample=False,
    )

raw_response = processor.decode(
    output_ids[0][prompt_length:],
    skip_special_tokens=False,
)
response = processor.parse_response(raw_response)
print(response.get("text", response) if isinstance(response, dict) else response)
```

The base model is large. Use suitable accelerator memory or a Transformers-supported quantization
configuration. Quantization changes memory requirements but does not remove the need to download or
otherwise access the base weights.

## Training Data And Provenance

The adapter was trained on ScentAI 32K, a synthetic instruction-tuning corpus derived from structured
perfume evidence. That evidence originates from Le Decanteur's
[Fragrantica Perfumes: Ratings, Notes, Votes & More](https://www.kaggle.com/datasets/ledecanteur/fragrantica-perfumes)
dataset. See the ScentAI dataset card for construction details, duplicate statistics, intended uses,
and limitations.

## Evaluation

The frozen ScentAI V4 application suite contains 120 English and Turkish cases covering grounding,
hard filters, entity resolution, language behavior, conversation continuity, and performance
calibration. The full application passed all functional cases. This result includes retrieval,
validation, and fallback behavior and is not a standalone adapter score.

## Limitations

- The adapter can hallucinate when used without retrieved context and output validation.
- Ratings, longevity, and sillage values are community-derived snapshots, not laboratory facts.
- The training corpus is primarily English; Turkish behavior also relies substantially on the base model.
- The adapter does not provide live prices, stock, reformulation status, or medical guidance.
- The data and model can reproduce popularity and coverage biases from the source catalog.
- This is an experimental research release, not a commercial fragrance recommendation service.

## License

The adapter is released under CC BY-NC-SA 4.0 as a conservative continuation of the training
dataset's attribution, non-commercial, and share-alike conditions. The base model is distributed
separately under its own license. Product names and trademarks remain the property of their owners.

ScentAI is independent and is not affiliated with or endorsed by Google, Hugging Face, Fragrantica,
Le Decanteur, or any perfume house.
