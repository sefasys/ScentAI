from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy" / "src"))

from scentai_deploy.modal_regression import rescore_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore saved Modal outputs against corrected contracts")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=ROOT / "evaluation/final_eval_v1.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    rescored = rescore_report(report, args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rescored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "case_count": rescored["case_count"],
                "pass_count": rescored["pass_count"],
                "failure_count": rescored["failure_count"],
            },
            indent=2,
        )
    )
    print("Saved:", args.output)
    if rescored["failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
