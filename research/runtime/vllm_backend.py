from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


RUNTIME_API_VERSION = 2
PLANNER_MARKER = "intent planner for a grounded perfume assistant"

PLANNER_INTENTS = [
    "recommendation",
    "similarity",
    "alternative",
    "preference_recommendation",
    "comparison",
    "perfume_profile",
    "exact_lookup",
    "collection_gap",
    "ranking",
    "unsupported_price",
    "unsupported_availability",
    "unsupported_medical",
    "unsupported_social_claim",
    "unsupported_layering",
]


def _evidence_value(value_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "value": value_schema,
            "evidence": {"type": "string", "minLength": 1},
        },
        "required": ["value", "evidence"],
        "additionalProperties": False,
    }


STRING_EVIDENCE = _evidence_value({"type": "string", "minLength": 1})
INTEGER_EVIDENCE = _evidence_value({"type": "integer"})

PLANNER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": PLANNER_INTENTS},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "perfumes": {"type": "array", "items": STRING_EVIDENCE, "maxItems": 8},
        "requested_brand": STRING_EVIDENCE,
        "gender": _evidence_value({"type": "string", "enum": ["male", "female", "unisex"]}),
        "season": _evidence_value(
            {"type": "string", "enum": ["spring", "summer", "autumn", "winter"]}
        ),
        "time_profile": _evidence_value({"type": "string", "enum": ["day", "night"]}),
        "wanted_accords": {"type": "array", "items": STRING_EVIDENCE, "maxItems": 12},
        "wanted_notes": {"type": "array", "items": STRING_EVIDENCE, "maxItems": 12},
        "excluded_accords": {"type": "array", "items": STRING_EVIDENCE, "maxItems": 12},
        "excluded_notes": {"type": "array", "items": STRING_EVIDENCE, "maxItems": 12},
        "excluded_entities": {"type": "array", "items": STRING_EVIDENCE, "maxItems": 12},
        "owned_perfumes": {"type": "array", "items": STRING_EVIDENCE, "maxItems": 20},
        "requested_count": INTEGER_EVIDENCE,
        "sort_by": _evidence_value(
            {
                "type": "string",
                "enum": ["rating", "popularity", "year", "longevity", "sillage", "value_score"],
            }
        ),
        "year_min": INTEGER_EVIDENCE,
        "year_max": INTEGER_EVIDENCE,
    },
    "required": ["intent", "confidence"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class GenerationMetric:
    role: str
    elapsed_seconds: float
    prompt_tokens: int
    generated_tokens: int
    tokens_per_second: float
    used_lora: bool
    max_tokens: int
    finish_reason: str | None


def is_planner_messages(messages: list[dict[str, str]]) -> bool:
    return bool(
        messages
        and PLANNER_MARKER in str(messages[0].get("content") or "").lower()
    )


def is_comparison_messages(messages: list[dict[str, str]]) -> bool:
    if not messages:
        return False
    system = str(messages[0].get("content") or "").lower()
    return "careful perfume comparison assistant" in system


def answer_token_budget(
    messages: list[dict[str, str]],
    requested_max_tokens: int,
    *,
    answer_max_tokens: int = 280,
    comparison_max_tokens: int = 380,
) -> int:
    configured = comparison_max_tokens if is_comparison_messages(messages) else answer_max_tokens
    return min(max(int(requested_max_tokens), 1), configured)


def validate_adapter_config(adapter_dir: Path | str, *, max_lora_rank: int) -> dict[str, Any]:
    import json

    adapter_path = Path(adapter_dir)
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"LoRA adapter config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rank = int(config.get("r") or 0)
    if rank <= 0:
        raise ValueError(f"Invalid LoRA rank in {config_path}: {rank}")
    if rank > max_lora_rank:
        raise ValueError(f"Adapter rank {rank} exceeds max_lora_rank={max_lora_rank}")
    if config.get("use_dora"):
        raise ValueError("This vLLM path expects a standard LoRA adapter, but use_dora=true.")
    return config


class VLLMHTTPMessageGenerator:
    """Call an isolated vLLM OpenAI-compatible server from the retrieval process."""

    def __init__(
        self,
        tokenizer,
        *,
        base_url: str,
        base_model_name: str,
        adapter_model_name: str = "scentai",
        planner_max_tokens: int = 192,
        answer_max_tokens: int = 280,
        comparison_max_tokens: int = 380,
        repetition_penalty: float = 1.08,
        request_timeout_seconds: float = 600.0,
        session=None,
    ) -> None:
        import requests

        self.tokenizer = tokenizer
        self.base_url = base_url.rstrip("/")
        self.completions_url = f"{self.base_url}/v1/completions"
        self.base_model_name = base_model_name
        self.adapter_model_name = adapter_model_name
        self.planner_max_tokens = planner_max_tokens
        self.answer_max_tokens = answer_max_tokens
        self.comparison_max_tokens = comparison_max_tokens
        self.repetition_penalty = repetition_penalty
        self.request_timeout_seconds = request_timeout_seconds
        self.session = session or requests.Session()
        self.metrics: list[GenerationMetric] = []

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        if not isinstance(prompt, str) or not prompt.strip():
            raise TypeError(f"Expected a rendered chat string, got {type(prompt).__name__}")
        return prompt

    def _encode_prompt(self, prompt: str) -> list[int]:
        encoded = self.tokenizer(prompt, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return list(input_ids)

    def _post_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.session.post(
                self.completions_url,
                json=payload,
                timeout=self.request_timeout_seconds,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not reach the local vLLM server at {self.completions_url}: {exc}"
            ) from exc

        if not response.ok:
            try:
                detail = response.json()
            except Exception:
                detail = response.text[:2000]
            raise RuntimeError(
                f"vLLM completion request failed with HTTP {response.status_code}: {detail}"
            )

        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"vLLM returned no completion choices: {body}")
        return body

    def __call__(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        planner_mode = is_planner_messages(messages)
        prompt = self._render_prompt(messages)
        prompt_token_ids = self._encode_prompt(prompt)
        prompt_tokens = len(prompt_token_ids)

        if planner_mode:
            budget = self.planner_max_tokens
            model_name = self.base_model_name
            role = "planner"
            payload: dict[str, Any] = {
                "model": model_name,
                "prompt": prompt_token_ids,
                "temperature": 0.0,
                "max_tokens": budget,
                "structured_outputs": {"json": PLANNER_JSON_SCHEMA},
                "stream": False,
            }
        else:
            budget = answer_token_budget(
                messages,
                max_new_tokens,
                answer_max_tokens=self.answer_max_tokens,
                comparison_max_tokens=self.comparison_max_tokens,
            )
            model_name = self.adapter_model_name
            role = "comparison" if is_comparison_messages(messages) else "answer"
            payload = {
                "model": model_name,
                "prompt": prompt_token_ids,
                "temperature": 0.0,
                "max_tokens": budget,
                "repetition_penalty": self.repetition_penalty,
                "stream": False,
            }

        started = time.perf_counter()
        response_body = self._post_completion(payload)
        elapsed = time.perf_counter() - started
        completion = response_body["choices"][0]
        completion_text = str(completion.get("text") or "").strip()
        usage = response_body.get("usage") or {}
        generated_tokens = int(
            usage.get("completion_tokens")
            or len(self.tokenizer(completion_text, add_special_tokens=False)["input_ids"])
        )
        metric = GenerationMetric(
            role=role,
            elapsed_seconds=round(elapsed, 4),
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            tokens_per_second=round(generated_tokens / elapsed, 3) if elapsed else 0.0,
            used_lora=not planner_mode,
            max_tokens=budget,
            finish_reason=completion.get("finish_reason"),
        )
        self.metrics.append(metric)
        print(
            f"[vllm-http:{role}] {elapsed:.2f}s | prompt_tokens={prompt_tokens} | "
            f"generated_tokens={generated_tokens} | {metric.tokens_per_second:.1f} tok/s | "
            f"finish={metric.finish_reason}"
        )
        return completion_text

    def clear_metrics(self) -> None:
        self.metrics.clear()

    def metrics_as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(metric) for metric in self.metrics]
