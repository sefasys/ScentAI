from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request, Response, status

from scentai.retrieval import RetrievalEngine

from .config import RetrievalSettings
from .schemas import ResolveRequest, SearchRequest, SimilarRequest


class RetrievalRuntime:
    def __init__(self, settings: RetrievalSettings) -> None:
        settings.validate_artifacts()
        self.settings = settings
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="scentai-retrieval",
        )
        # Construct and use torch/SentenceTransformer on the same long-lived thread.
        self.engine = self.executor.submit(
            RetrievalEngine,
            settings.chroma_dir,
            settings.catalog_path,
        ).result()

    def call(self, operation: Callable[[dict[str, Any]], dict[str, Any]], payload: dict[str, Any]):
        return operation(payload)

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = RetrievalSettings.from_env()
    runtime = RetrievalRuntime(settings)
    app.state.runtime = runtime
    try:
        yield
    finally:
        runtime.close()


app = FastAPI(
    title="ScentAI Retrieval API",
    version="1.0.0-rc1",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


def _runtime(request: Request) -> RetrievalRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service is starting")
    return runtime


async def _execute(runtime: RetrievalRuntime, operation, payload: dict[str, Any]):
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            runtime.executor,
            partial(runtime.call, operation, payload),
        )
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.get("/health/live")
async def health_live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
@app.get("/health/ready")
async def health_ready(request: Request, response: Response):
    runtime = _runtime(request)
    loop = asyncio.get_running_loop()
    report = await loop.run_in_executor(runtime.executor, runtime.engine.health)
    if report.get("status") != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report


@app.post("/search")
async def search(payload: SearchRequest, request: Request):
    runtime = _runtime(request)
    return await _execute(runtime, runtime.engine.search, payload.model_dump())


@app.post("/resolve")
async def resolve(payload: ResolveRequest, request: Request):
    runtime = _runtime(request)
    return await _execute(runtime, runtime.engine.resolve, payload.model_dump())


@app.post("/similar")
async def similar(payload: SimilarRequest, request: Request):
    runtime = _runtime(request)
    return await _execute(runtime, runtime.engine.similar, payload.model_dump())
