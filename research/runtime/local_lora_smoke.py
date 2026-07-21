from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma4UnifiedForConditionalGeneration,
)


DEFAULT_ADAPTER = (
    "models/scentai-gemma-4-12b-it-pilot-fastmodel-lora/best_lora_adapter"
)
DEFAULT_MODEL = "google/gemma-4-12B-it"


def build_messages(prompt: str) -> list[dict[str, str]]:
    system = (
        "You are ScentAI, an expert perfume consultant. "
        "Only recommend perfumes from the provided context. "
        "Never invent perfume facts."
    )
    context = """[PERFUMES]
Aventus Cologne by Creed — male
Accords: musky, leather, citrus, fresh spicy, aromatic, smoky, woody, powdery
Rating: 4.34/5 (3355 votes)
Best Seasons: Spring, Summer | Time: Day

Versace Pour Homme by Versace — male
Accords: floral, musky, fresh spicy, citrus, rose, aromatic, green, fresh
Rating: 4.27/5 (21534 votes)
Best Seasons: Spring, Summer | Time: Day

Prada L'Homme by Prada — male
Accords: iris, powdery, clean, woody, amber, citrus
Rating: 4.33/5 (9602 votes)
Best Seasons: Spring, Fall | Time: Day

Vanilla Woods by The 7 Virtues — female
Accords: vanilla, sweet, woody, caramel
Rating: 4.05/5 (4421 votes)
Best Seasons: Fall, Winter | Time: Night
[/PERFUMES]"""
    user = f"{context}\n\nUser request: {prompt}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def decode(processor, token_ids) -> str:
    raw = processor.decode(token_ids, skip_special_tokens=False)
    if hasattr(processor, "parse_response"):
        try:
            parsed = processor.parse_response(raw)
            if isinstance(parsed, dict):
                return str(parsed.get("text") or parsed.get("content") or parsed)
            if parsed is not None:
                return str(parsed)
        except Exception:
            pass
    return (
        raw.replace("<turn|>", "")
        .replace("<eos>", "")
        .replace("<end_of_turn>", "")
        .strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--max-new-tokens", type=int, default=140)
    parser.add_argument("--gpu-memory", default="7GiB")
    parser.add_argument(
        "--prompt",
        action="append",
        default=[
            "Recommend a fresh citrus summer cologne for men.",
            "I want something like Aventus but less smoky.",
            "Recommend a clean office scent without vanilla.",
        ],
    )
    args = parser.parse_args()

    adapter_dir = Path(args.adapter)
    if not adapter_dir.exists():
        raise FileNotFoundError(adapter_dir)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(str(adapter_dir))
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=quant,
        device_map="auto",
        max_memory={0: args.gpu_memory, "cpu": "32GiB"},
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()

    for prompt in args.prompt:
        messages = build_messages(prompt)
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=getattr(processor, "pad_token_id", None),
            )
        new_tokens = outputs[0][input_len:]
        print("\n" + "=" * 100)
        print(prompt)
        print("-" * 100)
        print(decode(processor, new_tokens))


if __name__ == "__main__":
    main()
