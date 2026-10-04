"""Cost and performance simulation for the semantic router.

Generates synthetic traffic with known workload categories, runs every prompt
through the same `SemanticCostRouter` the gateway uses, and compares two
scenarios:

* without gateway: every request goes to the premium model
* with gateway: each request goes to the tier the router picks

Routing accuracy is therefore measured, not assumed. Token sizes and model
speeds are modelling assumptions and are exposed as parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import ModelTier, Settings, get_settings
from .router import CATEGORY_LABELS, CATEGORY_TIER, Category, SemanticCostRouter

DEFAULT_MIX: dict[Category, float] = {
    Category.SIMPLE_FORMATTING: 0.35,
    Category.FAQ: 0.30,
    Category.ADVANCED_LOGIC: 0.15,
    Category.CODING: 0.20,
}


@dataclass(frozen=True)
class TokenProfile:
    """Log-normal token sizes for one category: (median, sigma) for input and output."""

    input_median: float
    input_sigma: float
    output_median: float
    output_sigma: float


# Input sizes include system prompt and any retrieved context, which is why FAQ inputs are large.
TOKEN_PROFILES: dict[Category, TokenProfile] = {
    Category.SIMPLE_FORMATTING: TokenProfile(180, 0.50, 120, 0.50),
    Category.FAQ: TokenProfile(900, 0.35, 180, 0.40),
    Category.ADVANCED_LOGIC: TokenProfile(1400, 0.50, 700, 0.45),
    Category.CODING: TokenProfile(2200, 0.60, 900, 0.50),
}

_TOPICS = ("billing", "onboarding", "the mobile release", "the quarterly report", "vendor contracts", "the support queue",
           "our data pipeline", "the hiring plan", "customer churn", "the pricing page")
_TEXTS = ("the meeting is moved to thursday at 3pm", "pls send the invoice asap thx", "apples, pears, kiwis, plums",
          "our team shipped the feature late but customers liked it", "john smith; maria garcia; wei chen",
          "revenue grew in q2 and costs stayed flat", "2026-03-14, 2026-01-02, 2026-02-20")
_LANGS = ("Python", "TypeScript", "Go", "Java", "SQL", "Rust", "Bash")
_THINGS = ("a retry decorator with exponential backoff", "an LRU cache", "a CSV to JSON converter", "a webhook handler",
           "a pagination helper", "a rate limiter", "a binary search", "a JWT validation middleware", "a Kafka consumer",
           "a function that deduplicates records")
_PRODUCT = ("the Pro plan", "the Enterprise plan", "the free tier", "the API", "my account", "the mobile app", "SSO")

TEMPLATES: dict[Category, tuple[str, ...]] = {
    Category.SIMPLE_FORMATTING: (
        "Convert to uppercase: {text}",
        "Fix the spelling and grammar: {text}",
        "Turn this into bullet points: {text}",
        "Rewrite in a more formal tone: {text}",
        "Translate to French: {text}",
        "Sort these alphabetically: {text}",
        "Make this a single line: {text}",
        "Capitalize each word: {text}",
        "Reformat as a markdown table: {text}",
        "Shorten this message about {topic}: {text}",
        "Clean up the punctuation in this note: {text}",
        "Put these dates in ISO format: {text}",
    ),
    Category.FAQ: (
        "What is the refund policy for {product}?",
        "How do I reset the password for {product}?",
        "What are the support hours for {product}?",
        "How long is the trial for {product}?",
        "Where can I download the invoice for {product}?",
        "Is there a rate limit on {product}?",
        "How do I cancel {product}?",
        "Which regions is {product} available in?",
        "Who do I contact about {topic}?",
        "Does {product} include audit logs?",
        "What is the price of {product}?",
        "How do I upgrade to {product}?",
    ),
    Category.ADVANCED_LOGIC: (
        "Analyze the trade-offs of centralising {topic} versus keeping it with each team and recommend one.",
        "Given a 12 percent budget cut, work out step by step how to reprioritise {topic} and justify each choice.",
        "Evaluate three scenarios for {topic} over the next four quarters and state the assumptions behind each.",
        "Diagnose the most likely root cause of the decline in {topic} and rank the hypotheses by evidence.",
        "Compare a build versus buy strategy for {topic} and estimate the second-order implications.",
        "Design an experiment to test whether changing {topic} reduces churn, including sample size reasoning.",
        "Critique this plan for {topic} and identify the weakest assumptions and mitigations.",
        "Assuming demand doubles, derive the staffing model for {topic} and show the intermediate steps.",
        "What would happen if we delayed {topic} by two quarters? Reason through the constraints and risks.",
        "Prioritise these five initiatives around {topic} using expected value and explain the reasoning.",
    ),
    Category.CODING: (
        "Write {thing} in {lang} with unit tests.",
        "Implement {thing} in {lang} and explain the complexity.",
        "Refactor this {lang} module that contains {thing} to be async.",
        "Debug this {lang} stack trace from {thing}: TypeError at line 42.",
        "Write a {lang} script for {thing} that reads from stdin.",
        "Add type hints and error handling to {thing} written in {lang}.",
        "Create a REST endpoint in {lang} that exposes {thing}.",
        "Why does this {lang} code for {thing} deadlock? Fix the bug.",
        "Optimise the {lang} query behind {thing}; it does a full table scan.",
        "Write a Dockerfile and CI config for a {lang} service providing {thing}.",
    ),
}

# Prompts whose wording points away from their true category. They keep the measured accuracy honest.
HARD_PROMPTS: dict[Category, tuple[str, ...]] = {
    Category.SIMPLE_FORMATTING: (
        "Can you make this sound better? {text}",
        "What would this look like as a table? {text}",
        "Tidy this: {text}",
        "Same thing but shorter please. {text}",
        "How should I word this for a client? {text}",
    ),
    Category.FAQ: (
        "I was charged twice last month, is that expected?",
        "Tell me about data residency.",
        "Explain how {product} handles backups.",
        "Need the steps for adding a teammate.",
        "Is my information safe with you?",
    ),
    Category.ADVANCED_LOGIC: (
        "Why did {topic} get worse after the reorg?",
        "What should we do about {topic}?",
        "Is it worth moving {topic} in-house?",
        "How do I decide between hiring two juniors or one senior for {topic}?",
        "Which matters more for {topic}: speed or accuracy?",
    ),
    Category.CODING: (
        "My pod keeps restarting with exit status 137, what is wrong?",
        "How do I make {thing} faster?",
        "The build is red after merging, where do I look?",
        "What is the cleanest way to structure {thing}?",
        "Explain why the cron job for {topic} runs twice.",
    ),
}


def _fill(template: str, rng: np.random.Generator) -> str:
    return template.format(
        text=_TEXTS[rng.integers(len(_TEXTS))],
        topic=_TOPICS[rng.integers(len(_TOPICS))],
        product=_PRODUCT[rng.integers(len(_PRODUCT))],
        lang=_LANGS[rng.integers(len(_LANGS))],
        thing=_THINGS[rng.integers(len(_THINGS))],
    )


def generate_traffic(
    n_requests: int = 10_000,
    mix: dict[Category, float] | None = None,
    ambiguity: float = 0.12,
    seed: int = 7,
) -> pd.DataFrame:
    """Synthetic request log with a known category for every prompt."""
    rng = np.random.default_rng(seed)
    mix = mix or DEFAULT_MIX
    categories = list(mix)
    weights = np.array([mix[c] for c in categories], dtype=float)
    weights = weights / weights.sum()
    picks = rng.choice(len(categories), size=n_requests, p=weights)
    hard = rng.random(n_requests) < ambiguity

    prompts: list[str] = []
    labels: list[str] = []
    for idx, is_hard in zip(picks, hard):
        category = categories[idx]
        pool = HARD_PROMPTS[category] if is_hard else TEMPLATES[category]
        prompts.append(_fill(pool[rng.integers(len(pool))], rng))
        labels.append(category.value)

    frame = pd.DataFrame({"prompt": prompts, "true_category": labels, "hard": hard})
    input_tokens = np.zeros(n_requests)
    output_tokens = np.zeros(n_requests)
    for category in categories:
        mask = (frame["true_category"] == category.value).to_numpy()
        profile = TOKEN_PROFILES[category]
        count = int(mask.sum())
        input_tokens[mask] = rng.lognormal(np.log(profile.input_median), profile.input_sigma, count)
        output_tokens[mask] = rng.lognormal(np.log(profile.output_median), profile.output_sigma, count)
    frame["input_tokens"] = np.maximum(8, input_tokens).round().astype(int)
    frame["output_tokens"] = np.maximum(4, output_tokens).round().astype(int)
    return frame


@dataclass
class SimulationResult:
    requests: pd.DataFrame
    summary: dict[str, Any]
    by_category: pd.DataFrame = field(default_factory=pd.DataFrame)


def _latency_ms(tier: ModelTier, output_tokens: np.ndarray) -> np.ndarray:
    return tier.ttft_ms + output_tokens / tier.tokens_per_sec * 1000.0


def run_simulation(
    traffic: pd.DataFrame,
    router: SemanticCostRouter | None = None,
    settings: Settings | None = None,
    premium: ModelTier | None = None,
    standard: ModelTier | None = None,
    gateway_overhead_ms: float = 12.0,
    escalate_under_routed: bool = True,
) -> SimulationResult:
    """Price the traffic with and without routing.

    `escalate_under_routed`: when a request that needed the premium model is sent
    to the standard model, assume the evaluation step catches it and the request
    is retried on premium. That request then pays for both calls, in cost and in
    latency, which is the realistic penalty for a routing mistake.
    """
    settings = settings or get_settings()
    router = router or SemanticCostRouter(settings)
    premium = premium or settings.premium
    standard = standard or settings.standard

    decisions = {prompt: router.route(prompt) for prompt in traffic["prompt"].unique()}
    frame = traffic.copy()
    frame["predicted_category"] = [decisions[p].category.value for p in frame["prompt"]]
    frame["routed_tier"] = [decisions[p].tier for p in frame["prompt"]]
    frame["required_tier"] = frame["true_category"].map(lambda c: CATEGORY_TIER[Category(c)])
    frame["tier_correct"] = frame["routed_tier"] == frame["required_tier"]
    frame["category_correct"] = frame["predicted_category"] == frame["true_category"]

    tokens_in = frame["input_tokens"].to_numpy(dtype=float)
    tokens_out = frame["output_tokens"].to_numpy(dtype=float)
    to_standard = (frame["routed_tier"] == "standard").to_numpy()
    under_routed = to_standard & (frame["required_tier"] == "premium").to_numpy()
    over_routed = ~to_standard & (frame["required_tier"] == "standard").to_numpy()

    premium_cost = tokens_in * premium.input_cost_per_token + tokens_out * premium.output_cost_per_token
    standard_cost = tokens_in * standard.input_cost_per_token + tokens_out * standard.output_cost_per_token
    premium_latency = _latency_ms(premium, tokens_out)
    standard_latency = _latency_ms(standard, tokens_out)

    retried = under_routed if escalate_under_routed else np.zeros_like(under_routed)
    gateway_standard_cost = np.where(to_standard, standard_cost, 0.0)
    gateway_premium_cost = np.where(~to_standard | retried, premium_cost, 0.0)
    gateway_latency = (
        gateway_overhead_ms
        + np.where(to_standard, standard_latency, premium_latency)
        + np.where(retried, premium_latency, 0.0)
    )

    frame["cost_without_gateway"] = premium_cost
    frame["cost_with_gateway"] = gateway_standard_cost + gateway_premium_cost
    frame["gateway_standard_cost"] = gateway_standard_cost
    frame["gateway_premium_cost"] = gateway_premium_cost
    frame["latency_without_gateway_ms"] = premium_latency
    frame["latency_with_gateway_ms"] = gateway_latency
    frame["under_routed"] = under_routed
    frame["over_routed"] = over_routed

    total_without = float(premium_cost.sum())
    total_with = float(frame["cost_with_gateway"].sum())
    mean_without = float(premium_latency.mean())
    mean_with = float(gateway_latency.mean())
    summary = {
        "requests": int(len(frame)),
        "spend_without_gateway": total_without,
        "spend_with_gateway": total_with,
        "spend_with_gateway_standard": float(gateway_standard_cost.sum()),
        "spend_with_gateway_premium": float(gateway_premium_cost.sum()),
        "dollars_saved": total_without - total_with,
        "savings_pct": (1 - total_with / total_without) * 100 if total_without else 0.0,
        "avg_latency_without_ms": mean_without,
        "avg_latency_with_ms": mean_with,
        "latency_reduction_pct": (1 - mean_with / mean_without) * 100 if mean_without else 0.0,
        "p95_latency_without_ms": float(np.percentile(premium_latency, 95)),
        "p95_latency_with_ms": float(np.percentile(gateway_latency, 95)),
        "routing_accuracy_pct": float(frame["tier_correct"].mean() * 100),
        "category_accuracy_pct": float(frame["category_correct"].mean() * 100),
        "share_to_standard_pct": float(to_standard.mean() * 100),
        "under_routed": int(under_routed.sum()),
        "over_routed": int(over_routed.sum()),
    }

    grouped = frame.groupby("true_category", sort=False)
    by_category = grouped.agg(
        requests=("prompt", "size"),
        spend_without_gateway=("cost_without_gateway", "sum"),
        spend_with_gateway=("cost_with_gateway", "sum"),
        routing_accuracy_pct=("tier_correct", lambda s: s.mean() * 100),
        to_standard_pct=("routed_tier", lambda s: (s == "standard").mean() * 100),
        avg_latency_without_ms=("latency_without_gateway_ms", "mean"),
        avg_latency_with_ms=("latency_with_gateway_ms", "mean"),
    ).reset_index()
    by_category["saved"] = by_category["spend_without_gateway"] - by_category["spend_with_gateway"]
    by_category["label"] = by_category["true_category"].map(lambda c: CATEGORY_LABELS[Category(c)])
    order = {c.value: i for i, c in enumerate(Category)}
    by_category = by_category.sort_values("true_category", key=lambda s: s.map(order)).reset_index(drop=True)

    return SimulationResult(requests=frame, summary=summary, by_category=by_category)
