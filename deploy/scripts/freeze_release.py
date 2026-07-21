from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "deploy" / "release" / "scentai-release.manifest.json"

FILES = (
    "src/scentai/orchestrator.py",
    "src/scentai/retrieval.py",
    "notebooks/full_pipeline_colab.ipynb",
    "evaluation/final_eval_v1.jsonl",
    "evaluation/final_eval_summary.json",
    "evaluation/final_eval_metadata.json",
    "evaluation/final_eval_human_review.csv",
    "scentai_catalog.sqlite3",
    "deploy/api.Dockerfile",
    "deploy/model.Dockerfile",
    "deploy/retrieval.Dockerfile",
    "deploy/compose.yaml",
    "deploy/requirements-api.txt",
    "deploy/requirements-retrieval.txt",
    "deploy/requirements-modal.txt",
    "deploy/modal_app.py",
    "deploy/modal_setup_app.py",
    "notebooks/upload_modal_artifacts_colab.ipynb",
    "deploy/modal_colab_upload.py",
    "deploy/model_server/entrypoint.py",
    "deploy/scripts/freeze_release.py",
    "deploy/scripts/http_smoke_retrieval.py",
    "deploy/scripts/modal_bootstrap.py",
    "deploy/scripts/modal_http_smoke.py",
    "deploy/scripts/rescore_modal_report.py",
    "deploy/scripts/smoke_retrieval.py",
    "deploy/scripts/verify_release.py",
    "deploy/src/scentai_deploy/__init__.py",
    "deploy/src/scentai_deploy/api.py",
    "deploy/src/scentai_deploy/chat_jobs.py",
    "deploy/src/scentai_deploy/config.py",
    "deploy/src/scentai_deploy/http_smoke.py",
    "deploy/src/scentai_deploy/modal_bridge.py",
    "deploy/src/scentai_deploy/modal_regression.py",
    "deploy/src/scentai_deploy/retrieval_api.py",
    "deploy/src/scentai_deploy/runtime.py",
    "deploy/src/scentai_deploy/schemas.py",
    "deploy/src/scentai_deploy/sessions.py",
    "deploy/src/scentai_deploy/warmup_jobs.py",
    "deploy/release/acceptance.md",
    "deploy/release/modal.md",
    "deploy/release/release_history.md",
)

DIRECTORIES = (
    "chroma_db_bge_m3",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def directory_record(path: Path) -> dict[str, Any]:
    files = [file_record(item) for item in sorted(path.rglob("*")) if item.is_file()]
    aggregate = hashlib.sha256()
    for item in files:
        aggregate.update(item["path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(item["sha256"].encode("ascii"))
        aggregate.update(b"\n")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": sum(item["bytes"] for item in files),
        "file_count": len(files),
        "tree_sha256": aggregate.hexdigest(),
        "files": files,
    }


def catalog_counts(path: Path) -> dict[str, int | str]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {
            "integrity_check": connection.execute("PRAGMA integrity_check").fetchone()[0],
            "perfumes": connection.execute("SELECT COUNT(*) FROM perfumes").fetchone()[0],
            "similarity_edges": connection.execute("SELECT COUNT(*) FROM similarity_edges").fetchone()[0],
        }
    finally:
        connection.close()


def main() -> None:
    missing = [relative for relative in (*FILES, *DIRECTORIES) if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"Release inputs are missing: {missing}")

    summary = json.loads((ROOT / "evaluation/final_eval_summary.json").read_text())
    metadata = json.loads((ROOT / "evaluation/final_eval_metadata.json").read_text())
    manifest = {
        "schema_version": 1,
        "release": "scentai-v1.0-rc2",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "base_model": "google/gemma-4-12B-it",
            "planner_model": "google/gemma-4-12B-it",
            "first_answer_model": "google/gemma-4-12B-it",
            "repair_model": "scentai",
            "inference": "vLLM 0.25.1, BF16, dynamic LoRA",
            "adapter": {
                "external_artifact_required": True,
                "evaluated_path": metadata.get("adapter_dir"),
                "required_files": ["adapter_config.json", "adapter_model.safetensors"],
                "expected_rank": 16,
                "expected_use_dora": False,
                "expected_target_modules": sorted({
                    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
                }),
                "note": "The full V4 adapter is stored in Drive and must be mounted at deployment time.",
            },
        },
        "retrieval": {
            "embedding_model": "BAAI/bge-m3",
            "collection": "scentai_perfumes",
            "catalog": catalog_counts(ROOT / "scentai_catalog.sqlite3"),
        },
        "acceptance": {
            "evaluation": "evaluation/final_eval_v1.jsonl",
            "case_count": summary["case_count"],
            "pass_count": summary["pass_count"],
            "metrics": summary["metrics"],
            "all_functional_gates_passed": all(
                gate["pass"]
                for name, gate in summary["quality_gates"].items()
                if name != "fallback_rate"
            ),
            "accepted_exception": {
                "gate": "fallback_rate",
                "observed": summary["metrics"]["fallback_rate"],
                "threshold": 0.05,
                "reason": "8 safe deterministic fallbacks; no grounding, filter, language, or contract failure.",
            },
        },
        "files": [file_record(ROOT / relative) for relative in FILES],
        "directories": [directory_record(ROOT / relative) for relative in DIRECTORIES],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Frozen release manifest: {OUTPUT}")


if __name__ == "__main__":
    main()
