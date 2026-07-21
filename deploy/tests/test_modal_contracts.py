from __future__ import annotations

import json
from pathlib import Path

import pytest

from scentai_deploy.modal_bridge import (
    EXPECTED_ADAPTER_TARGETS,
    ModalJsonClient,
    dispatch_retrieval,
    validate_modal_artifacts,
)
from scentai_deploy.http_smoke import CASES as HTTP_SMOKE_CASES, run_http_smoke
from scentai_deploy.modal_regression import load_cases, rescore_report, score_case


ROOT = Path(__file__).resolve().parents[2]


class RemoteCall:
    def __init__(self, function):
        self.function = function

    def remote(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class FakeWorker:
    def __init__(self, function):
        self.request = RemoteCall(function)


class FakeEngine:
    def health(self):
        return {"status": "ok", "count": 10}

    def search(self, payload):
        return {"results": [payload]}

    def resolve(self, payload):
        return {"resolved": payload}

    def similar(self, payload):
        return {"reference": payload, "results": []}


def test_modal_json_client_unwraps_success_and_errors():
    client = ModalJsonClient(
        FakeWorker(lambda method, path, payload: {"status_code": 200, "body": {"path": path}})
    )
    assert client.get("/health") == {"path": "/health"}
    failing = ModalJsonClient(
        FakeWorker(lambda method, path, payload: {"status_code": 503, "body": {"detail": "cold"}})
    )
    with pytest.raises(RuntimeError, match="503"):
        failing.get("/v1/models")


def test_retrieval_dispatch_preserves_stage5_routes():
    engine = FakeEngine()
    assert dispatch_retrieval(engine, "GET", "/health", None)["body"]["status"] == "ok"
    assert dispatch_retrieval(engine, "POST", "/search", {"query": "iris"})["body"] == {
        "results": [{"query": "iris"}]
    }
    assert dispatch_retrieval(engine, "GET", "/unknown", None)["status_code"] == 404


def test_modal_artifact_validation_checks_release_shape(tmp_path):
    model_root = tmp_path / "models"
    adapter = model_root / "scentai"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "google/gemma-4-12B-it",
                "r": 16,
                "use_dora": False,
                "target_modules": sorted(EXPECTED_ADAPTER_TARGETS),
            }
        ),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    data_root = tmp_path / "data"
    chroma = data_root / "chroma_db_bge_m3"
    chroma.mkdir(parents=True)
    (chroma / "chroma.sqlite3").write_bytes(b"chroma")
    (data_root / "scentai_catalog.sqlite3").write_bytes(b"catalog")
    report = validate_modal_artifacts(model_root, data_root)
    assert report["status"] == "ok"
    assert report["adapter"]["rank"] == 16


def test_modal_smoke_selection_covers_multiple_categories():
    path = ROOT / "evaluation" / "final_eval_v1.jsonl"
    cases = load_cases(path, 12)
    assert len(cases) == 12
    assert len({case["category"] for case in cases}) >= 8
    assert [case["turn"] for case in cases if case.get("session_id") == "conv_vanilla"] == [1, 2]
    date_case = next(case for case in cases if case["id"] == "fev1_rec_002")
    assert date_case["expected"]["required_terms"] == ["vanilla", "warm spicy"]


def test_modal_regression_score_checks_frozen_contract():
    case = {
        "expected": {
            "response_language": "en",
            "intent_in": ["recommendation"],
            "requested_count": 3,
            "required_terms": ["vanilla"],
            "excluded_terms": [],
        }
    }
    result = {
        "response_language": "en",
        "plan": {
            "intent": "recommendation",
            "requested_count": 3,
            "required_terms": ["vanilla"],
            "excluded_terms": [],
        },
        "validation": {"pass": True, "mentioned_candidates": []},
        "candidates": [],
    }
    assert score_case(case, result)["pass"]
    result["plan"]["required_terms"] = ["vanilla", "warm spicy"]
    assert score_case(case, result)["pass"]
    result["plan"]["required_terms"] = ["warm spicy"]
    assert "required_terms_mismatch" in score_case(case, result)["failures"]
    result["plan"]["required_terms"] = ["vanilla"]
    result["plan"]["excluded_terms"] = ["vanilla"]
    assert "excluded_terms_mismatch" in score_case(case, result)["failures"]


def test_modal_regression_score_accepts_equivalent_follow_up_actions():
    case = {
        "expected": {
            "response_language": "tr",
            "intent_in": ["recommendation"],
            "conversation_action": "refine_previous",
        }
    }
    result = {
        "response_language": "tr",
        "plan": {
            "intent": "recommendation",
            "conversation_action": "more_options",
            "inherited_previous_constraints": True,
        },
        "validation": {"pass": True, "mentioned_candidates": []},
        "candidates": [],
    }

    assert score_case(case, result)["pass"]


def test_modal_regression_score_rejects_new_request_for_follow_up():
    case = {
        "expected": {
            "response_language": "en",
            "intent_in": ["recommendation"],
            "conversation_action": "more_options",
        }
    }
    result = {
        "response_language": "en",
        "plan": {"intent": "recommendation", "conversation_action": "new_request"},
        "validation": {"pass": True, "mentioned_candidates": []},
        "candidates": [],
    }

    score = score_case(case, result)
    assert not score["pass"]
    assert score["failures"] == ["conversation_action_mismatch"]


def test_http_smoke_scores_the_public_contract_without_exposing_the_key():
    calls = []

    def requester(url, *, api_key=None, payload=None):
        calls.append({"url": url, "api_key": api_key, "payload": payload})
        if url.endswith("/health/live"):
            return {"status": "ok"}
        case = HTTP_SMOKE_CASES[len(calls) - 2]
        labels = [case.get("contains")] if case.get("contains") else []
        return {
            "validation_passed": True,
            "language": case["language"],
            "route": case.get("route", "llm_grounded"),
            "recommendations": [
                {"label": label or f"Perfume {index}"}
                for index, label in enumerate(labels or [None] * case.get("count", 1))
            ],
        }

    report = run_http_smoke("https://example.modal.run/", "secret-value", requester=requester)
    assert report["pass"]
    assert len(report["outputs"]) == len(HTTP_SMOKE_CASES)
    assert all(call["api_key"] == "secret-value" for call in calls[1:])


def test_saved_modal_outputs_can_be_rescored_after_contract_correction(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "case-1",
                "category": "recommendation",
                "query": "warm vanilla",
                "expected": {
                    "response_language": "en",
                    "intent_in": ["recommendation"],
                    "required_terms": ["vanilla", "warm spicy"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = {
        "response_language": "en",
        "plan": {"intent": "recommendation", "required_terms": ["warm spicy", "vanilla"]},
        "validation": {"pass": True, "mentioned_candidates": []},
        "candidates": [],
    }
    report = {
        "outputs": [
            {
                "id": "case-1",
                "category": "recommendation",
                "expected": {"required_terms": ["vanilla", "spicy"]},
                "score": {"pass": False},
                "result": result,
            }
        ]
    }
    rescored = rescore_report(report, cases_path)
    assert rescored["pass_count"] == 1
    assert rescored["failure_count"] == 0
    assert rescored["outputs"][0]["expected"]["required_terms"] == ["vanilla", "warm spicy"]
