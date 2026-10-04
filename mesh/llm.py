"""Model access through LiteLLM: one call signature, fallbacks and cost tracking.

In mock mode the call still goes through `litellm.acompletion`, using its
`mock_response` argument, so no API key or network access is needed. Token
counts and costs are computed with LiteLLM's tokenizer and price registry.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # never fetch the price map over the network

import litellm  # noqa: E402

from .config import ModelTier, Settings, get_settings  # noqa: E402

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True
litellm.telemetry = False
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

_TOKENIZER_MODEL = "gpt-4o"


class LLMUnavailableError(RuntimeError):
    """Raised when the primary model and every fallback failed."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    tier: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    baseline_cost_usd: float
    latency_ms: float
    tokens_per_sec: float
    fallback_used: bool
    attempts: int

    @property
    def saved_usd(self) -> float:
        return max(0.0, self.baseline_cost_usd - self.cost_usd)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tier": self.tier,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "baseline_cost_usd": self.baseline_cost_usd,
            "saved_usd": self.saved_usd,
            "latency_ms": round(self.latency_ms, 2),
            "tokens_per_sec": round(self.tokens_per_sec, 1),
            "fallback_used": self.fallback_used,
            "attempts": self.attempts,
        }


def _register_prices(tier: ModelTier) -> None:
    """Register the tier price with LiteLLM so cost maths goes through one registry."""
    entry = {
        "input_cost_per_token": tier.input_cost_per_token,
        "output_cost_per_token": tier.output_cost_per_token,
        "cache_creation_input_token_cost": 0.0,
        "cache_read_input_token_cost": 0.0,
        "litellm_provider": "openai",
        "mode": "chat",
        "max_tokens": 8192,
    }
    litellm.register_model({tier.alias: entry, f"{tier.alias}-fallback": entry})


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        for tier in (self.settings.premium, self.settings.standard):
            _register_prices(tier)
        self.count_tokens(text="warm up")  # load the tokenizer now, not on the first request

    def candidates(self, tier: ModelTier) -> list[str]:
        if self.settings.mock_llm:
            return [tier.alias, f"{tier.alias}-fallback"]
        return [tier.provider_model, *tier.fallbacks]

    @staticmethod
    def count_tokens(messages: list[dict[str, str]] | None = None, text: str | None = None) -> int:
        try:
            if messages is not None:
                return int(litellm.token_counter(model=_TOKENIZER_MODEL, messages=messages))
            return int(litellm.token_counter(model=_TOKENIZER_MODEL, text=text or ""))
        except Exception:  # noqa: BLE001 - tokenizer problems must not fail a request
            source = text if text is not None else " ".join(m.get("content", "") for m in messages or [])
            return max(1, len(source) // 4)

    async def _call_mock(
        self, model: str, tier: ModelTier, messages: list[dict[str, str]], mock_text: str
    ) -> tuple[str, int, int, float]:
        failing = model in self.settings.mock_fail_models
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            mock_response=RuntimeError(f"simulated outage for {model}") if failing else mock_text,
        )
        text = response.choices[0].message.content or ""
        prompt_tokens = self.count_tokens(messages=messages)
        completion_tokens = self.count_tokens(text=text)
        # Simulate time to first token plus generation time for the tier.
        simulated_s = (tier.ttft_ms / 1000 + completion_tokens / tier.tokens_per_sec) * self.settings.mock_latency_scale
        if simulated_s > 0:
            await asyncio.sleep(simulated_s)
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        return text, prompt_tokens, completion_tokens, float(prompt_cost + completion_cost)

    async def _call_provider(
        self, model: str, tier: ModelTier, messages: list[dict[str, str]]
    ) -> tuple[str, int, int, float]:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=self.settings.max_output_tokens,
            timeout=self.settings.llm_timeout_s,
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or self.count_tokens(messages=messages))
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or self.count_tokens(text=text))
        try:
            cost = float(litellm.completion_cost(completion_response=response))
        except Exception:  # noqa: BLE001 - model missing from the price map
            cost = tier.cost(prompt_tokens, completion_tokens)
        return text, prompt_tokens, completion_tokens, cost

    async def complete(
        self, tier_name: str, messages: list[dict[str, str]], mock_text: str = "{}"
    ) -> LLMResult:
        """Run a completion on the tier, walking its fallback chain on failure."""
        tier = self.settings.tier(tier_name)
        candidates = self.candidates(tier)
        last_error: Exception | None = None

        for attempt, model in enumerate(candidates, start=1):
            started = time.perf_counter()
            try:
                call = (
                    self._call_mock(model, tier, messages, mock_text)
                    if self.settings.mock_llm
                    else self._call_provider(model, tier, messages)
                )
                text, prompt_tokens, completion_tokens, cost = await asyncio.wait_for(
                    call, timeout=self.settings.llm_timeout_s
                )
            except Exception as exc:  # noqa: BLE001 - any provider failure moves to the next candidate
                last_error = exc
                logger.warning("model %s failed on attempt %d/%d: %s", model, attempt, len(candidates), exc)
                continue

            elapsed_ms = (time.perf_counter() - started) * 1000
            tokens_per_sec = completion_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0
            result = LLMResult(
                text=text,
                model=model,
                tier=tier.name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
                baseline_cost_usd=self.settings.premium.cost(prompt_tokens, completion_tokens),
                latency_ms=elapsed_ms,
                tokens_per_sec=tokens_per_sec,
                fallback_used=attempt > 1,
                attempts=attempt,
            )
            logger.info(
                "llm call model=%s tier=%s tokens=%d/%d cost=$%.6f latency=%.0fms",
                model, tier.name, prompt_tokens, completion_tokens, cost, elapsed_ms,
            )
            return result

        raise LLMUnavailableError(f"all models failed for tier '{tier.name}': {last_error}") from last_error
