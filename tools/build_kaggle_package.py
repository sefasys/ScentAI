from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DOCS = REPO_ROOT / "dataset"
DATA_FILES = {
    "scentai_train_gemma.jsonl.gz": ("train_set/finetune/gemma/full_train.jsonl", 30_400, "model"),
    "scentai_validation_gemma.jsonl.gz": ("train_set/finetune/gemma/full_validation.jsonl", 1_600, "model"),
    "scentai_full_openai.jsonl.gz": ("train_set/final/training_main_32000_openai.jsonl", 32_000, "assistant"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl_gzip(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def compress_jsonl(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    with source.open("rb") as input_handle, gzip.open(temporary, "wb", compresslevel=9) as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=8 * 1024 * 1024)
    temporary.replace(destination)


def build_metadata(owner: str) -> dict[str, Any]:
    return {
        "title": "ScentAI 32K Grounded Perfume Conversations",
        "id": f"{owner}/scentai-32k-grounded-perfume-conversations",
        "subtitle": "Synthetic L1-L5 chat data for grounded perfume recommendation and preference-aware fine-tuning",
        "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the upload-ready ScentAI Kaggle dataset")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Project containing train_set/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", required=True, help="Kaggle username/owner slug")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(f"Output is not empty: {output}. Pass --force to replace package files.")
    output.mkdir(parents=True, exist_ok=True)

    records = []
    for output_name, (relative_source, expected_count, assistant_role) in DATA_FILES.items():
        source = dataset_root / relative_source
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / output_name
        print(f"Compressing {source.name} -> {destination.name}")
        compress_jsonl(source, destination)
        actual_count = count_jsonl_gzip(destination)
        if actual_count != expected_count:
            raise ValueError(f"Expected {expected_count} records in {source}, found {actual_count}")
        records.append({
            "path": destination.name,
            "records": actual_count,
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "assistant_role": assistant_role,
            "source_export": relative_source,
        })

    for source_name, destination_name in (
        ("README.md", "DATASET_CARD.md"),
        ("ATTRIBUTION.md", "ATTRIBUTION.md"),
        ("LICENSE.md", "LICENSE.md"),
        ("statistics.json", "statistics.json"),
    ):
        source = DATASET_DOCS / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, output / destination_name)

    metadata = build_metadata(args.owner)
    (output / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "dataset": metadata["title"],
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": "CC-BY-NC-SA-4.0",
        "semantic_record_count": 32_000,
        "note": "Gemma train+validation and OpenAI full are role variants of the same corpus.",
        "files": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "upload_bytes": sum(record["bytes"] for record in records),
        "files": [record["path"] for record in records],
    }, indent=2))


if __name__ == "__main__":
    main()

