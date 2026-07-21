from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT_SRC = ROOT / "deploy" / "src"
PROJECT_SRC = ROOT / "src"
for path in (PROJECT_SRC, ROOT, DEPLOYMENT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture
def anyio_backend():
    return "asyncio"
