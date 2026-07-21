from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal


WarmupState = Literal["queued", "running", "succeeded", "failed"]


@dataclass
class WarmupJob:
    job_id: str
    status: WarmupState
    created_at: float
    updated_at: float
    report: dict[str, Any] | None = None
    error: str | None = None


class WarmupJobRegistry:
    """Share one active model warm-up across browser clients in one API replica."""

    def __init__(self, *, ttl_seconds: int = 1800) -> None:
        self.ttl_seconds = ttl_seconds
        self._jobs: dict[str, WarmupJob] = {}
        self._lock = threading.RLock()

    def create_or_active(self) -> tuple[WarmupJob, bool]:
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            active = next(
                (job for job in self._jobs.values() if job.status in {"queued", "running"}),
                None,
            )
            if active is not None:
                return active, False
            job = WarmupJob(str(uuid.uuid4()), "queued", now, now)
            self._jobs[job.job_id] = job
            return job, True

    def get(self, job_id: str) -> WarmupJob | None:
        try:
            normalized = str(uuid.UUID(str(job_id)))
        except (TypeError, ValueError, AttributeError):
            return None
        with self._lock:
            self._purge(time.monotonic())
            return self._jobs.get(normalized)

    def mark_running(self, job_id: str) -> None:
        self._update(job_id, status="running")

    def succeed(self, job_id: str, report: dict[str, Any]) -> None:
        self._update(job_id, status="succeeded", report=report)

    def fail(self, job_id: str, error: str, report: dict[str, Any] | None = None) -> None:
        self._update(job_id, status="failed", error=error, report=report)

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in values.items():
                setattr(job, key, value)
            job.updated_at = time.monotonic()

    def _purge(self, now: float) -> None:
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if now - job.updated_at > self.ttl_seconds
        ]
        for job_id in expired:
            self._jobs.pop(job_id, None)
