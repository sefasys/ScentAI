from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Literal

from .schemas import ChatResponse


JobState = Literal["queued", "running", "succeeded", "failed"]


@dataclass
class ChatJob:
    job_id: str
    status: JobState
    created_at: float
    updated_at: float
    response: ChatResponse | None = None
    error: str | None = None
    error_status: int | None = None


class ChatJobRegistry:
    """Small thread-safe result store for browser polling within one API replica."""

    def __init__(self, *, ttl_seconds: int = 3600, max_jobs: int = 2000) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_jobs = max_jobs
        self._jobs: dict[str, ChatJob] = {}
        self._lock = threading.RLock()

    def create(self) -> ChatJob:
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            if len(self._jobs) >= self.max_jobs:
                oldest = min(self._jobs, key=lambda key: self._jobs[key].updated_at)
                self._jobs.pop(oldest, None)
            job = ChatJob(str(uuid.uuid4()), "queued", now, now)
            self._jobs[job.job_id] = job
            return job

    def get(self, job_id: str) -> ChatJob | None:
        try:
            normalized = str(uuid.UUID(str(job_id)))
        except (TypeError, ValueError, AttributeError):
            return None
        with self._lock:
            self._purge(time.monotonic())
            return self._jobs.get(normalized)

    def mark_running(self, job_id: str) -> None:
        self._update(job_id, status="running")

    def succeed(self, job_id: str, response: ChatResponse) -> None:
        self._update(job_id, status="succeeded", response=response)

    def fail(self, job_id: str, error: str, error_status: int = 500) -> None:
        self._update(job_id, status="failed", error=error, error_status=error_status)

    def _update(self, job_id: str, **values) -> None:
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
