from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from scentai.orchestrator import (
    JsonHttpClient,
    RetrievalClient,
    ScentAIOrchestrator,
    VLLMClient,
)

from .config import ApiSettings
from .sessions import InMemorySessionRegistry


@dataclass
class RuntimeBundle:
    settings: ApiSettings
    model_http: JsonHttpClient
    retrieval_client: RetrievalClient
    pipeline: ScentAIOrchestrator
    sessions: InMemorySessionRegistry
    executor: ThreadPoolExecutor

    @classmethod
    def create(cls, settings: ApiSettings) -> "RuntimeBundle":
        model_http = JsonHttpClient(
            settings.model_base_url,
            timeout=settings.model_timeout_seconds,
        )
        retrieval_client = RetrievalClient(
            JsonHttpClient(
                settings.retrieval_base_url,
                timeout=settings.retrieval_timeout_seconds,
            )
        )
        return cls.create_with_clients(settings, model_http, retrieval_client)

    @classmethod
    def create_with_clients(
        cls,
        settings: ApiSettings,
        model_http: Any,
        retrieval_client: RetrievalClient,
    ) -> "RuntimeBundle":
        """Build the frozen V4 runtime over HTTP or an internal RPC transport."""
        pipeline = ScentAIOrchestrator(
            VLLMClient(model_http),
            retrieval_client,
            planner_model=settings.base_model_name,
            answer_model=settings.base_model_name,
            repair_answer_model=settings.lora_model_name,
        )
        sessions = InMemorySessionRegistry(
            pipeline,
            ttl_seconds=settings.session_ttl_seconds,
            max_sessions=settings.max_sessions,
        )
        executor = ThreadPoolExecutor(
            max_workers=settings.request_worker_threads,
            thread_name_prefix="scentai-request",
        )
        return cls(settings, model_http, retrieval_client, pipeline, sessions, executor)

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)

    def readiness(self) -> dict[str, Any]:
        checks: dict[str, Any] = {}
        ready = True
        try:
            retrieval = self.retrieval_client.health()
            retrieval_ok = retrieval.get("status") == "ok"
            checks["retrieval"] = {"ready": retrieval_ok, "details": retrieval}
            ready = ready and retrieval_ok
        except Exception as exc:
            checks["retrieval"] = {"ready": False, "error": type(exc).__name__}
            ready = False

        try:
            models = self.model_http.get("/v1/models").get("data", [])
            model_ids = sorted(str(item.get("id")) for item in models if item.get("id"))
            required = {self.settings.base_model_name, self.settings.lora_model_name}
            model_ok = required.issubset(model_ids)
            checks["model"] = {
                "ready": model_ok,
                "required_models": sorted(required),
                "served_models": model_ids,
            }
            ready = ready and model_ok
        except Exception as exc:
            checks["model"] = {"ready": False, "error": type(exc).__name__}
            ready = False
        return {"ready": ready, "checks": checks, "sessions": self.sessions.count()}
