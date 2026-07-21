from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "deploy" / "release" / "scentai-release.manifest.json"
EXPECTED_TARGETS = {
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_adapter(path: Path, model_name: str) -> None:
    config_path = path / "adapter_config.json"
    weights_path = path / "adapter_model.safetensors"
    assert config_path.is_file(), f"Missing {config_path}"
    assert weights_path.is_file() and weights_path.stat().st_size > 0, f"Missing {weights_path}"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config.get("base_model_name_or_path") == model_name, config
    assert int(config.get("r") or 0) == 16, config
    assert not config.get("use_dora", False), config
    assert set(config.get("target_modules") or []) == EXPECTED_TARGETS, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen ScentAI deployment artifacts")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--adapter", type=Path, help="Full V4 adapter directory")
    parser.add_argument("--fast", action="store_true", help="Skip large-file content hashes")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = ROOT / item["path"]
        assert path.is_file(), f"Missing {path}"
        assert path.stat().st_size == item["bytes"], f"Size mismatch: {path}"
        if not args.fast:
            assert sha256_file(path) == item["sha256"], f"Hash mismatch: {path}"

    for directory in manifest["directories"]:
        for item in directory["files"]:
            path = ROOT / item["path"]
            assert path.is_file(), f"Missing {path}"
            assert path.stat().st_size == item["bytes"], f"Size mismatch: {path}"
            if not args.fast:
                assert sha256_file(path) == item["sha256"], f"Hash mismatch: {path}"

    summary = json.loads((ROOT / "evaluation/final_eval_summary.json").read_text())
    assert summary["case_count"] == summary["pass_count"] == 120, summary
    for metric in (
        "overall_pass_rate", "language_pass_rate", "requested_count_pass_rate",
        "hard_filter_pass_rate", "entity_resolution_pass_rate",
        "conversation_no_repeat_pass_rate", "performance_calibration_pass_rate",
    ):
        assert summary["metrics"][metric] == 1.0, (metric, summary["metrics"][metric])

    catalog = ROOT / "scentai_catalog.sqlite3"
    connection = sqlite3.connect(f"file:{catalog}?mode=ro", uri=True)
    try:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("SELECT COUNT(*) FROM perfumes").fetchone()[0] == 131_930
        assert connection.execute("SELECT COUNT(*) FROM similarity_edges").fetchone()[0] == 692_729
    finally:
        connection.close()

    if args.adapter:
        verify_adapter(args.adapter, manifest["model"]["base_model"])
        adapter_status = str(args.adapter)
    else:
        adapter_status = "external adapter not supplied; startup validation remains mandatory"
    print(json.dumps({"status": "ok", "release": manifest["release"], "adapter": adapter_status}, indent=2))


if __name__ == "__main__":
    main()
