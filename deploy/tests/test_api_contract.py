from __future__ import annotations

import asyncio

import pytest
from httpx2 import ASGITransport, AsyncClient

import scentai_deploy.api as api_module
from scentai_deploy.config import ApiSettings
from scentai_deploy.sessions import InMemorySessionRegistry
from scentai_deploy.warmup_jobs import WarmupJobRegistry

from .test_deployment_contracts import FakeModelHttp, FakePipeline, FakeRetrievalClient


class FakeRuntime:
    def __init__(self):
        self.settings = ApiSettings.from_env()
        self.model_http = FakeModelHttp()
        self.retrieval_client = FakeRetrievalClient()
        self.pipeline = FakePipeline()
        self.sessions = InMemorySessionRegistry(self.pipeline, ttl_seconds=3600, max_sessions=10)

    def readiness(self):
        return {"ready": True, "checks": {}, "sessions": self.sessions.count()}


async def immediate_call(runtime, function, *args):
    """Exercise ASGI contracts without the managed sandbox's executor shutdown issue."""
    return function(*args)


@pytest.mark.anyio
async def test_chat_and_follow_up_contract(monkeypatch):
    runtime = FakeRuntime()
    api_module.app.state.runtime = runtime
    monkeypatch.setattr(api_module, "_submit", immediate_call)
    async with AsyncClient(
        transport=ASGITransport(app=api_module.app),
        base_url="http://testserver",
    ) as client:
        first = await client.post("/v1/chat", json={"query": "Recommend one perfume."})
        assert first.status_code == 200, first.text
        body = first.json()
        assert body["validation_passed"] is True
        assert body["recommendations"][0]["perfume_id"] == 1
        second = await client.post(
            "/v1/chat",
            json={"query": "Another one.", "session_id": body["session_id"]},
        )
        assert second.status_code == 200, second.text
        assert second.json()["session_id"] == body["session_id"]
        deleted = await client.delete(f"/v1/sessions/{body['session_id']}")
        assert deleted.status_code == 204


@pytest.mark.anyio
async def test_unknown_session_is_404(monkeypatch):
    runtime = FakeRuntime()
    api_module.app.state.runtime = runtime
    monkeypatch.setattr(api_module, "_submit", immediate_call)
    async with AsyncClient(
        transport=ASGITransport(app=api_module.app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/v1/chat",
            json={"query": "Hello", "session_id": "11111111-1111-1111-1111-111111111111"},
        )
        assert response.status_code == 404


@pytest.mark.anyio
async def test_polled_chat_job_returns_without_holding_the_submit_request(monkeypatch):
    runtime = FakeRuntime()
    api_module.app.state.runtime = runtime
    api_module.app.state.chat_jobs = api_module.ChatJobRegistry()
    api_module.app.state.chat_job_tasks = set()
    monkeypatch.setattr(api_module, "_submit", immediate_call)
    async with AsyncClient(
        transport=ASGITransport(app=api_module.app),
        base_url="http://testserver",
    ) as client:
        accepted = await client.post("/v1/chat/jobs", json={"query": "Recommend one perfume."})
        assert accepted.status_code == 202, accepted.text
        accepted_body = accepted.json()
        assert accepted_body["status"] == "queued"

        for _ in range(10):
            await asyncio.sleep(0)
            result = await client.get(f"/v1/chat/jobs/{accepted_body['job_id']}")
            assert result.status_code == 200
            if result.json()["status"] == "succeeded":
                break
        else:
            pytest.fail("Chat job did not finish")

        body = result.json()
        assert body["response"]["validation_passed"] is True
        assert body["response"]["session_id"]


@pytest.mark.anyio
async def test_polled_chat_job_preserves_unknown_session_failure(monkeypatch):
    runtime = FakeRuntime()
    api_module.app.state.runtime = runtime
    api_module.app.state.chat_jobs = api_module.ChatJobRegistry()
    api_module.app.state.chat_job_tasks = set()
    monkeypatch.setattr(api_module, "_submit", immediate_call)
    async with AsyncClient(
        transport=ASGITransport(app=api_module.app),
        base_url="http://testserver",
    ) as client:
        accepted = await client.post(
            "/v1/chat/jobs",
            json={"query": "Hello", "session_id": "11111111-1111-1111-1111-111111111111"},
        )
        job_id = accepted.json()["job_id"]
        for _ in range(10):
            await asyncio.sleep(0)
            result = await client.get(f"/v1/chat/jobs/{job_id}")
            if result.json()["status"] == "failed":
                break
        assert result.json()["error_status"] == 404


@pytest.mark.anyio
async def test_warmup_job_is_shared_and_reports_readiness(monkeypatch):
    runtime = FakeRuntime()
    api_module.app.state.runtime = runtime
    api_module.app.state.warmup_jobs = WarmupJobRegistry()
    api_module.app.state.background_tasks = set()
    release = asyncio.Event()
    readiness_calls = 0

    async def delayed_readiness(runtime_arg, function, *args):
        nonlocal readiness_calls
        if function == runtime.readiness:
            readiness_calls += 1
            await release.wait()
        return function(*args)

    monkeypatch.setattr(api_module, "_submit", delayed_readiness)
    async with AsyncClient(
        transport=ASGITransport(app=api_module.app),
        base_url="http://testserver",
    ) as client:
        first = await client.post("/v1/runtime/warmup/jobs")
        second = await client.post("/v1/runtime/warmup/jobs")
        assert first.status_code == 202
        assert second.status_code == 202
        assert first.json()["job_id"] == second.json()["job_id"]

        release.set()
        for _ in range(10):
            await asyncio.sleep(0)
            result = await client.get(
                f"/v1/runtime/warmup/jobs/{first.json()['job_id']}"
            )
            if result.json()["status"] == "succeeded":
                break
        else:
            pytest.fail("Warm-up job did not finish")

        assert result.json()["ready"] is True
        assert readiness_calls == 1
