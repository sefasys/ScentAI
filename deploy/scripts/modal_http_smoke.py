from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy" / "src"))

from scentai_deploy.http_smoke import run_http_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the deployed ScentAI Modal API")
    parser.add_argument("--url", required=True, help="Modal web URL without a trailing slash")
    parser.add_argument("--output", type=Path, default=Path("deploy/reports/modal_http_smoke.json"))
    args = parser.parse_args()
    api_key = os.environ.get("SCENTAI_API_KEY") or getpass.getpass("ScentAI API key: ").strip()
    if not api_key:
        raise ValueError("SCENTAI_API_KEY is required")
    report = run_http_smoke(args.url, api_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", args.output)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
