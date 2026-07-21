from __future__ import annotations

import importlib.util
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scentai_deploy.config import ApiSettings, RetrievalSettings
from scentai_deploy.runtime import RuntimeBundle
from scentai_deploy.sessions import InMemorySessionRegistry, UnknownSessionError


ROOT = Path(__file__).resolve().parents[2]


def make_test_adapter(tmp_path: Path) -> Path:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({
            "base_model_name_or_path": "google/gemma-4-12B-it",
            "r": 16,
            "use_dora": False,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        }),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"test-adapter")
    return adapter


class FakePipeline:
    def run(self, query, *, conversation_context=None):
        action = "more_options" if conversation_context else "new_request"
        return {
            "query": query,
            "route": "llm_grounded",
            "response_language": "en",
            "plan": {"conversation_action": action},
            "answer": "1. Test Perfume by Test Brand\nGrounded answer.",
            "candidates": [
                {
                    "perfume_id": 1,
                    "name": "Test Perfume",
                    "brand": "Test Brand",
                    "label": "Test Perfume by Test Brand",
                }
            ],
            "validation": {
                "pass": True,
                "reasons": [],
                "mentioned_candidates": ["Test Perfume by Test Brand"],
            },
            "generation_attempts": 1,
            "timings": {"total_seconds": 0.01},
        }


class FakeModelHttp:
    def get(self, path):
        assert path == "/v1/models"
        return {
            "data": [
                {"id": "google/gemma-4-12B-it"},
                {"id": "scentai"},
            ]
        }


class FakeRetrievalClient:
    def health(self):
        return {"status": "ok", "count": 131930}


def test_default_settings_preserve_v4_model_routing(monkeypatch):
    for key in (
        "SCENTAI_MODEL_URL",
        "SCENTAI_RETRIEVAL_URL",
        "SCENTAI_BASE_MODEL",
        "SCENTAI_LORA_NAME",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = ApiSettings.from_env()
    assert settings.base_model_name == "google/gemma-4-12B-it"
    assert settings.lora_model_name == "scentai"
    assert settings.model_base_url == "http://model:8010"
    assert settings.retrieval_base_url == "http://retrieval:8020"


def test_retrieval_settings_require_both_databases(tmp_path, monkeypatch):
    chroma = tmp_path / "chroma"
    chroma.mkdir()
    catalog = tmp_path / "catalog.sqlite3"
    monkeypatch.setenv("SCENTAI_CHROMA_DIR", str(chroma))
    monkeypatch.setenv("SCENTAI_CATALOG_PATH", str(catalog))
    monkeypatch.setenv("SCENTAI_RETRIEVAL_MAX_CONCURRENCY", "1")
    settings = RetrievalSettings.from_env()
    with pytest.raises(FileNotFoundError):
        settings.validate_artifacts()
    (chroma / "chroma.sqlite3").touch()
    catalog.touch()
    settings.validate_artifacts()


def test_session_registry_supports_follow_up_and_rejects_unknown_ids():
    registry = InMemorySessionRegistry(FakePipeline(), ttl_seconds=3600, max_sessions=10)
    session_id, first = registry.run("first")
    same_id, second = registry.run("more", session_id)
    assert same_id == session_id
    assert first["plan"]["conversation_action"] == "new_request"
    assert second["plan"]["conversation_action"] == "more_options"
    with pytest.raises(UnknownSessionError):
        registry.run("unknown", str(uuid.uuid4()))
    assert registry.delete(session_id)


def test_runtime_readiness_requires_base_and_lora_models():
    settings = ApiSettings.from_env()
    runtime = RuntimeBundle(
        settings=settings,
        model_http=FakeModelHttp(),
        retrieval_client=FakeRetrievalClient(),
        pipeline=FakePipeline(),
        sessions=InMemorySessionRegistry(FakePipeline(), ttl_seconds=3600, max_sessions=10),
        executor=ThreadPoolExecutor(max_workers=1),
    )
    try:
        report = runtime.readiness()
        assert report["ready"]
        assert report["checks"]["model"]["served_models"] == [
            "google/gemma-4-12B-it",
            "scentai",
        ]
    finally:
        runtime.close()


def test_model_entrypoint_accepts_the_known_adapter_shape(tmp_path):
    path = ROOT / "deploy" / "model_server" / "entrypoint.py"
    spec = importlib.util.spec_from_file_location("scentai_model_entrypoint", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    adapter = make_test_adapter(tmp_path)
    config = module.validate_adapter(adapter, "google/gemma-4-12B-it", 16)
    assert config["r"] == 16
    assert set(config["target_modules"]) == module.EXPECTED_TARGETS


def test_model_entrypoint_builds_the_frozen_vllm_command(monkeypatch, tmp_path):
    path = ROOT / "deploy" / "model_server" / "entrypoint.py"
    spec = importlib.util.spec_from_file_location("scentai_model_command", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    adapter = make_test_adapter(tmp_path)
    monkeypatch.setenv("SCENTAI_ADAPTER_DIR", str(adapter))
    monkeypatch.setenv("SCENTAI_GPU_MEMORY_UTILIZATION", "0.65")
    captured = {}

    def capture_exec(program, args):
        captured["program"] = program
        captured["args"] = args
        raise SystemExit(0)

    monkeypatch.setattr(module.os, "execvp", capture_exec)
    with pytest.raises(SystemExit):
        module.main()
    args = captured["args"]
    assert captured["program"] == "vllm"
    assert args[:3] == ["vllm", "serve", "google/gemma-4-12B-it"]
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.65"
    assert args[args.index("--max-lora-rank") + 1] == "16"
    descriptor = json.loads(args[args.index("--lora-modules") + 1])
    assert descriptor == {
        "name": "scentai",
        "path": str(adapter),
        "base_model_name": "google/gemma-4-12B-it",
    }


def test_frozen_eval_has_no_functional_failures():
    summary = json.loads(
        (ROOT / "evaluation" / "final_eval_summary.json").read_text(encoding="utf-8")
    )
    assert summary["case_count"] == summary["pass_count"] == 120
    assert summary["metrics"]["hard_filter_pass_rate"] == 1.0
    assert summary["metrics"]["entity_resolution_pass_rate"] == 1.0
    assert summary["metrics"]["performance_calibration_pass_rate"] == 1.0
