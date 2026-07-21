from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from scentai.orchestrator import ScentAISession


class UnknownSessionError(LookupError):
    pass


@dataclass
class _SessionEntry:
    session: ScentAISession
    last_seen: float
    lock: threading.RLock = field(default_factory=threading.RLock)
    active_requests: int = 0


class InMemorySessionRegistry:
    """Bounded, thread-safe session state for a single API replica."""

    def __init__(self, pipeline: Any, *, ttl_seconds: int, max_sessions: int) -> None:
        self.pipeline = pipeline
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._entries: dict[str, _SessionEntry] = {}
        self._lock = threading.RLock()

    def run(self, query: str, session_id: str | None = None) -> tuple[str, dict[str, Any]]:
        now = time.monotonic()
        with self._lock:
            self._purge_expired(now)
            if session_id is None:
                session_id = str(uuid.uuid4())
                self._ensure_capacity()
                entry = _SessionEntry(ScentAISession(self.pipeline), now)
                self._entries[session_id] = entry
            else:
                session_id = self._normalize_session_id(session_id)
                entry = self._entries.get(session_id)
                if entry is None:
                    raise UnknownSessionError("Session was not found or has expired")
                entry.last_seen = now
            entry.active_requests += 1

        try:
            with entry.lock:
                result = entry.session.run(query)
        finally:
            with self._lock:
                entry.active_requests -= 1
                entry.last_seen = time.monotonic()
        return session_id, result

    def delete(self, session_id: str) -> bool:
        normalized = self._normalize_session_id(session_id)
        with self._lock:
            return self._entries.pop(normalized, None) is not None

    def count(self) -> int:
        with self._lock:
            self._purge_expired(time.monotonic())
            return len(self._entries)

    @staticmethod
    def _normalize_session_id(session_id: str) -> str:
        try:
            value = uuid.UUID(str(session_id))
        except (TypeError, ValueError, AttributeError) as exc:
            raise UnknownSessionError("Session ID is invalid") from exc
        return str(value)

    def _purge_expired(self, now: float) -> None:
        expired = [
            session_id
            for session_id, entry in self._entries.items()
            if now - entry.last_seen > self.ttl_seconds and entry.active_requests == 0
        ]
        for session_id in expired:
            self._entries.pop(session_id, None)

    def _ensure_capacity(self) -> None:
        if len(self._entries) < self.max_sessions:
            return
        idle = [key for key, entry in self._entries.items() if entry.active_requests == 0]
        if not idle:
            raise RuntimeError("Session capacity is temporarily exhausted")
        oldest_id = min(idle, key=lambda key: self._entries[key].last_seen)
        self._entries.pop(oldest_id, None)
