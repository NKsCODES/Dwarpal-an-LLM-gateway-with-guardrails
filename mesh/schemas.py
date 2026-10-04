"""Pydantic models for the HTTP surface."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

RequestStatus = Literal["completed", "blocked", "pending_approval", "rejected"]
DecisionAction = Literal["approve", "edit", "reject"]


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=16_000, description="End-user prompt.")
    user_id: str = Field(default="anonymous", max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class GuardrailInfo(BaseModel):
    allowed: bool
    risk_score: float
    categories: list[str] = Field(default_factory=list)
    reason: str | None = None


class RoutingInfo(BaseModel):
    category: str
    tier: str
    model: str
    complexity: float
    confidence: float
    rationale: str


class UsageInfo(BaseModel):
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    baseline_cost_usd: float
    saved_usd: float
    latency_ms: float
    tokens_per_sec: float
    fallback_used: bool = False


class EvalInfo(BaseModel):
    faithfulness: float
    relevance: float
    passed: bool
    evaluator: str


class ChatResponse(BaseModel):
    request_id: str
    thread_id: str
    status: RequestStatus
    answer: str | None = None
    guardrail: GuardrailInfo | None = None
    routing: RoutingInfo | None = None
    usage: UsageInfo | None = None
    evaluation: EvalInfo | None = None
    approval_id: str | None = None
    intercept_reasons: list[str] = Field(default_factory=list)
    reviewed_by: str | None = None
    trace_id: str | None = None


class ApprovalDecision(BaseModel):
    action: DecisionAction
    edited_response: str | None = Field(default=None, max_length=32_000)
    reviewer: str = Field(default="admin", max_length=128)
    note: str | None = Field(default=None, max_length=2_000)

    @field_validator("edited_response")
    @classmethod
    def _strip(cls, value: str | None) -> str | None:
        return value.strip() if value else value


class ApprovalView(BaseModel):
    approval_id: str
    request_id: str
    thread_id: str
    status: Literal["pending", "approved", "edited", "rejected"]
    created_at: float
    decided_at: float | None = None
    user_id: str
    prompt: str
    draft_answer: str
    final_answer: str | None = None
    reasons: list[str]
    faithfulness: float | None = None
    relevance: float | None = None
    model: str | None = None
    tier: str | None = None
    category: str | None = None
    contexts: list[dict[str, Any]] = Field(default_factory=list)
    reviewer: str | None = None
    note: str | None = None
    trace_id: str | None = None
