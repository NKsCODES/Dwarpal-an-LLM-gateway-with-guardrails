"""Central configuration.

Everything is driven by environment variables so the same code runs in a local
mock setup, in CI and behind real providers. Nothing here performs I/O.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if not raw:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class ModelTier:
    """One routing tier: a priced model plus the fallbacks tried after it."""

    name: str
    alias: str
    provider_model: str
    fallbacks: tuple[str, ...]
    input_cost_per_m: float
    output_cost_per_m: float
    ttft_ms: float
    tokens_per_sec: float

    @property
    def input_cost_per_token(self) -> float:
        return self.input_cost_per_m / 1_000_000

    @property
    def output_cost_per_token(self) -> float:
        return self.output_cost_per_m / 1_000_000

    def cost(self, input_tokens: float, output_tokens: float) -> float:
        return input_tokens * self.input_cost_per_token + output_tokens * self.output_cost_per_token


def default_premium() -> ModelTier:
    return ModelTier(
        name="premium",
        alias="mesh-premium",
        provider_model=os.getenv("PREMIUM_MODEL", "gpt-4o"),
        fallbacks=_env_list("PREMIUM_FALLBACKS", ("anthropic/claude-sonnet-4-5",)),
        input_cost_per_m=_env_float("PREMIUM_INPUT_COST_PER_M", 5.00),
        output_cost_per_m=_env_float("PREMIUM_OUTPUT_COST_PER_M", 15.00),
        ttft_ms=_env_float("PREMIUM_TTFT_MS", 500.0),
        tokens_per_sec=_env_float("PREMIUM_TOKENS_PER_SEC", 80.0),
    )


def default_standard() -> ModelTier:
    return ModelTier(
        name="standard",
        alias="mesh-standard",
        provider_model=os.getenv("STANDARD_MODEL", "groq/llama-3.1-8b-instant"),
        fallbacks=_env_list("STANDARD_FALLBACKS", ("anthropic/claude-haiku-4-5",)),
        input_cost_per_m=_env_float("STANDARD_INPUT_COST_PER_M", 0.15),
        output_cost_per_m=_env_float("STANDARD_OUTPUT_COST_PER_M", 0.60),
        ttft_ms=_env_float("STANDARD_TTFT_MS", 200.0),
        tokens_per_sec=_env_float("STANDARD_TOKENS_PER_SEC", 200.0),
    )


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the gateway."""

    service_name: str = "ai-gateway-mesh"
    environment: str = field(default_factory=lambda: os.getenv("GATEWAY_ENV", "local"))

    # LLM access. Mock mode is the default so nothing is spent by accident.
    mock_llm: bool = field(default_factory=lambda: _env_bool("GATEWAY_MOCK_LLM", True))
    mock_latency_scale: float = field(default_factory=lambda: _env_float("MOCK_LATENCY_SCALE", 1.0))
    mock_fail_models: tuple[str, ...] = field(default_factory=lambda: _env_list("MOCK_FAIL_MODELS", ()))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 30.0))
    max_output_tokens: int = 1024

    premium: ModelTier = field(default_factory=default_premium)
    standard: ModelTier = field(default_factory=default_standard)

    # Guardrail.
    guardrail_block_threshold: float = field(default_factory=lambda: _env_float("GUARDRAIL_BLOCK_THRESHOLD", 0.75))
    max_prompt_chars: int = 16_000

    # Router.
    complexity_threshold: float = field(default_factory=lambda: _env_float("ROUTER_COMPLEXITY_THRESHOLD", 0.50))
    router_min_confidence: float = field(default_factory=lambda: _env_float("ROUTER_MIN_CONFIDENCE", 0.0))

    # RAG and evaluation.
    rag_top_k: int = 3
    # Retrieval scores below `rag_min_similarity` count as "nothing relevant found". The mock
    # model answers faithfully only when the top score reaches `rag_strong_similarity`.
    rag_min_similarity: float = field(default_factory=lambda: _env_float("RAG_MIN_SIMILARITY", 0.15))
    rag_strong_similarity: float = field(default_factory=lambda: _env_float("RAG_STRONG_SIMILARITY", 0.30))
    faithfulness_threshold: float = field(default_factory=lambda: _env_float("FAITHFULNESS_THRESHOLD", 0.70))
    relevance_threshold: float = field(default_factory=lambda: _env_float("RELEVANCE_THRESHOLD", 0.50))
    eval_mode: str = field(default_factory=lambda: os.getenv("EVAL_MODE", "heuristic"))

    # Observability.
    otlp_endpoint: str | None = field(
        default_factory=lambda: os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    )
    trace_buffer_size: int = 500

    # Auth. Unset keys leave the matching routes open, which suits local use only.
    api_key: str | None = field(default_factory=lambda: os.getenv("GATEWAY_API_KEY") or None)
    admin_key: str | None = field(default_factory=lambda: os.getenv("GATEWAY_ADMIN_KEY") or None)

    def tier(self, name: str) -> ModelTier:
        if name == "premium":
            return self.premium
        if name == "standard":
            return self.standard
        raise KeyError(f"unknown tier: {name}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
