from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = int(os.environ.get(name, default))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _csv_env(name: str, default: str = "") -> tuple[str, ...]:
    raw = os.environ.get(name, default)
    return tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))


@dataclass(frozen=True)
class ApiSettings:
    model_base_url: str
    retrieval_base_url: str
    base_model_name: str
    lora_model_name: str
    model_timeout_seconds: int
    retrieval_timeout_seconds: int
    session_ttl_seconds: int
    max_sessions: int
    request_worker_threads: int
    api_key: str | None
    cors_origins: tuple[str, ...]
    expose_debug: bool

    @classmethod
    def from_env(cls) -> "ApiSettings":
        return cls(
            model_base_url=os.environ.get("SCENTAI_MODEL_URL", "http://model:8010").rstrip("/"),
            retrieval_base_url=os.environ.get("SCENTAI_RETRIEVAL_URL", "http://retrieval:8020").rstrip("/"),
            base_model_name=os.environ.get("SCENTAI_BASE_MODEL", "google/gemma-4-12B-it").strip(),
            lora_model_name=os.environ.get("SCENTAI_LORA_NAME", "scentai").strip(),
            model_timeout_seconds=_env_int("SCENTAI_MODEL_TIMEOUT_SECONDS", 180),
            retrieval_timeout_seconds=_env_int("SCENTAI_RETRIEVAL_TIMEOUT_SECONDS", 45),
            session_ttl_seconds=_env_int("SCENTAI_SESSION_TTL_SECONDS", 3600, minimum=60),
            max_sessions=_env_int("SCENTAI_MAX_SESSIONS", 1000),
            request_worker_threads=_env_int("SCENTAI_REQUEST_WORKER_THREADS", 16),
            api_key=(os.environ.get("SCENTAI_API_KEY") or "").strip() or None,
            cors_origins=_csv_env("SCENTAI_CORS_ORIGINS"),
            expose_debug=_env_bool("SCENTAI_EXPOSE_DEBUG", False),
        )


@dataclass(frozen=True)
class RetrievalSettings:
    chroma_dir: Path
    catalog_path: Path
    max_concurrency: int

    @classmethod
    def from_env(cls) -> "RetrievalSettings":
        return cls(
            chroma_dir=Path(os.environ.get("SCENTAI_CHROMA_DIR", "/data/chroma_db_bge_m3")),
            catalog_path=Path(os.environ.get("SCENTAI_CATALOG_PATH", "/data/scentai_catalog.sqlite3")),
            max_concurrency=_env_int("SCENTAI_RETRIEVAL_MAX_CONCURRENCY", 2),
        )

    def validate_artifacts(self) -> None:
        chroma_sqlite = self.chroma_dir / "chroma.sqlite3"
        if not chroma_sqlite.is_file():
            raise FileNotFoundError(f"Missing Chroma database: {chroma_sqlite}")
        if not self.catalog_path.is_file():
            raise FileNotFoundError(f"Missing catalog database: {self.catalog_path}")
        if self.max_concurrency != 1:
            raise ValueError(
                "SCENTAI_RETRIEVAL_MAX_CONCURRENCY must be 1; BGE-M3 is kept on one dedicated worker thread"
            )
