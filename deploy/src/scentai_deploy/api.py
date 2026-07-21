from __future__ import annotations

import asyncio
import hmac
import logging
import uuid
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

from .chat_jobs import ChatJobRegistry
from .config import ApiSettings
from .runtime import RuntimeBundle
from .schemas import (
    CandidateSummary,
    ChatJobAccepted,
    ChatJobStatus,
    ChatRequest,
    ChatResponse,
    WarmupJobAccepted,
    WarmupJobStatus,
)
from .sessions import UnknownSessionError
from .warmup_jobs import WarmupJobRegistry


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = ApiSettings.from_env()
    runtime_factory = getattr(app.state, "runtime_factory", RuntimeBundle.create)
    runtime = runtime_factory(settings)
    app.state.runtime = runtime
    try:
        yield
    finally:
        runtime.close()


app = FastAPI(
    title="ScentAI API",
    version="1.0.0-rc1",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

_cors_settings = ApiSettings.from_env()
if _cors_settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(_cors_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    )


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", "").strip()[:64] or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


def _runtime(request: Request) -> RuntimeBundle:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service is starting")
    return runtime


def _chat_jobs(request: Request) -> ChatJobRegistry:
    jobs = getattr(request.app.state, "chat_jobs", None)
    if jobs is None:
        jobs = ChatJobRegistry()
        request.app.state.chat_jobs = jobs
    return jobs


def _warmup_jobs(request: Request) -> WarmupJobRegistry:
    jobs = getattr(request.app.state, "warmup_jobs", None)
    if jobs is None:
        jobs = WarmupJobRegistry()
        request.app.state.warmup_jobs = jobs
    return jobs


def _track_background_task(app: FastAPI, task: asyncio.Task) -> None:
    tasks = getattr(app.state, "background_tasks", None)
    if tasks is None:
        tasks = set()
        app.state.background_tasks = tasks
    tasks.add(task)
    task.add_done_callback(tasks.discard)


def _authorize(runtime: RuntimeBundle, supplied_key: str | None) -> None:
    expected = runtime.settings.api_key
    if expected and not (supplied_key and hmac.compare_digest(expected, supplied_key)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


async def _submit(runtime: RuntimeBundle, function, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(runtime.executor, partial(function, *args))


def _build_chat_response(
    request_id: str,
    payload: ChatRequest,
    runtime: RuntimeBundle,
    session_id: str,
    result: dict,
) -> ChatResponse:
    if result.get("route") == "validated_template_fallback":
        failure_reasons = [
            failure.get("reasons", [])
            for failure in result.get("generation_failures", [])
        ]
        logger.warning(
            "Validated answer fallback used: intent=%s failure_reasons=%s",
            result.get("plan", {}).get("intent"),
            failure_reasons,
            extra={
                "request_id": request_id,
                "intent": result.get("plan", {}).get("intent"),
                "failure_reasons": failure_reasons,
            },
        )
    mentioned = set(result.get("validation", {}).get("mentioned_candidates", []))
    candidates = [
        CandidateSummary(
            perfume_id=int(candidate["perfume_id"]),
            label=str(candidate["label"]),
            name=str(candidate.get("name") or ""),
            brand=str(candidate.get("brand") or ""),
        )
        for candidate in result.get("candidates", [])
        if candidate.get("label") in mentioned
    ]
    allow_debug = payload.debug and runtime.settings.expose_debug
    debug = None
    if allow_debug:
        debug = {
            "plan": result.get("plan"),
            "retrieval": result.get("retrieval"),
            "validation": result.get("validation"),
            "generation_failures": result.get("generation_failures", []),
        }
    return ChatResponse(
        request_id=request_id,
        session_id=session_id,
        answer=str(result.get("answer") or ""),
        route=str(result.get("route") or "unknown"),
        language=str(result.get("response_language") or "en"),
        recommendations=candidates,
        validation_passed=bool(result.get("validation", {}).get("pass")),
        generation_attempts=int(result.get("generation_attempts") or 0),
        total_seconds=float(result.get("timings", {}).get("total_seconds") or 0.0),
        debug=debug,
    )


async def _execute_chat(
    runtime: RuntimeBundle,
    payload: ChatRequest,
    request_id: str,
) -> ChatResponse:
    readiness = await _submit(runtime, runtime.readiness)
    if not readiness["ready"]:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Dependencies are not ready")

    try:
        session_id, result = await _submit(
            runtime,
            runtime.sessions.run,
            payload.query,
            payload.session_id,
        )
    except UnknownSessionError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if result.get("route") == "retrieval_error":
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Retrieval failed")
    return _build_chat_response(request_id, payload, runtime, session_id, result)


async def _run_chat_job(
    jobs: ChatJobRegistry,
    job_id: str,
    runtime: RuntimeBundle,
    payload: ChatRequest,
    request_id: str,
) -> None:
    jobs.mark_running(job_id)
    try:
        response = await _execute_chat(runtime, payload, request_id)
    except HTTPException as exc:
        jobs.fail(job_id, str(exc.detail), exc.status_code)
    except Exception:
        logger.exception("Unhandled chat job failure", extra={"job_id": job_id})
        jobs.fail(job_id, "ScentAI could not complete this request", 500)
    else:
        jobs.succeed(job_id, response)


async def _run_warmup_job(
    jobs: WarmupJobRegistry,
    job_id: str,
    runtime: RuntimeBundle,
) -> None:
    jobs.mark_running(job_id)
    try:
        report = await _submit(runtime, runtime.readiness)
    except Exception:
        logger.exception("Unhandled runtime warm-up failure", extra={"job_id": job_id})
        jobs.fail(job_id, "ScentAI could not start the inference runtime")
        return
    if report.get("ready"):
        jobs.succeed(job_id, report)
    else:
        jobs.fail(job_id, "One or more ScentAI dependencies are not ready", report)


@app.get("/health/live")
async def health_live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def health_ready(request: Request, response: Response):
    runtime = _runtime(request)
    report = await _submit(runtime, runtime.readiness)
    if not report["ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report


@app.post(
    "/v1/runtime/warmup/jobs",
    response_model=WarmupJobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_warmup_job(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> WarmupJobAccepted:
    runtime = _runtime(request)
    _authorize(runtime, x_api_key)
    job, created = _warmup_jobs(request).create_or_active()
    if created:
        _track_background_task(
            request.app,
            asyncio.create_task(_run_warmup_job(_warmup_jobs(request), job.job_id, runtime)),
        )
    return WarmupJobAccepted(job_id=job.job_id, status=job.status)


@app.get("/v1/runtime/warmup/jobs/{job_id}", response_model=WarmupJobStatus)
async def get_warmup_job(
    job_id: str,
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> WarmupJobStatus:
    runtime = _runtime(request)
    _authorize(runtime, x_api_key)
    job = _warmup_jobs(request).get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Warm-up job was not found or has expired")
    return WarmupJobStatus(
        job_id=job.job_id,
        status=job.status,
        ready=job.status == "succeeded",
        report=job.report,
        error=job.error,
    )


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> ChatResponse:
    runtime = _runtime(request)
    _authorize(runtime, x_api_key)
    return await _execute_chat(runtime, payload, request.state.request_id)


@app.post("/v1/chat/jobs", response_model=ChatJobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def create_chat_job(
    payload: ChatRequest,
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> ChatJobAccepted:
    runtime = _runtime(request)
    _authorize(runtime, x_api_key)
    jobs = _chat_jobs(request)
    job = jobs.create()
    task = asyncio.create_task(
        _run_chat_job(jobs, job.job_id, runtime, payload, request.state.request_id)
    )
    _track_background_task(request.app, task)
    return ChatJobAccepted(job_id=job.job_id)


@app.get("/v1/chat/jobs/{job_id}", response_model=ChatJobStatus)
async def get_chat_job(
    job_id: str,
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> ChatJobStatus:
    runtime = _runtime(request)
    _authorize(runtime, x_api_key)
    job = _chat_jobs(request).get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat job was not found or has expired")
    return ChatJobStatus(
        job_id=job.job_id,
        status=job.status,
        response=job.response,
        error=job.error,
        error_status=job.error_status,
    )


@app.delete("/v1/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> Response:
    runtime = _runtime(request)
    _authorize(runtime, x_api_key)
    try:
        removed = runtime.sessions.delete(session_id)
    except UnknownSessionError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
