from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64)
    debug: bool = False

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned


class CandidateSummary(BaseModel):
    perfume_id: int
    label: str
    name: str
    brand: str


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    answer: str
    route: str
    language: str
    recommendations: list[CandidateSummary]
    validation_passed: bool
    generation_attempts: int
    total_seconds: float
    debug: dict[str, Any] | None = None


class ChatJobAccepted(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"
    poll_after_ms: int = 1500


class ChatJobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    response: ChatResponse | None = None
    error: str | None = None
    error_status: int | None = None
    poll_after_ms: int = 1500


class WarmupJobAccepted(BaseModel):
    job_id: str
    status: Literal["queued", "running"]
    poll_after_ms: int = 2000


class WarmupJobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    ready: bool = False
    report: dict[str, Any] | None = None
    error: str | None = None
    poll_after_ms: int = 2000


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)
    filters: dict[str, Any] = Field(default_factory=dict)
    wanted_terms: list[str] = Field(default_factory=list, max_length=30)
    required_terms: list[str] = Field(default_factory=list, max_length=30)
    exclude_terms: list[str] = Field(default_factory=list, max_length=30)
    exclude_ids: list[int] = Field(default_factory=list, max_length=200)
    discovery_mode: str = "balanced"


class ResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hint: str = Field(min_length=1, max_length=300)


class SimilarRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hint: str = Field(min_length=1, max_length=300)
    top_k: int = Field(default=10, ge=1, le=50)
    required_terms: list[str] = Field(default_factory=list, max_length=30)
    exclude_terms: list[str] = Field(default_factory=list, max_length=30)
    exclude_ids: list[int] = Field(default_factory=list, max_length=200)
