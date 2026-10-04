"""Semantic and cost router.

Classifies a prompt into a workload category, derives a complexity score and
picks the cheapest tier expected to answer it well.

The classifier blends two signals:
* semantic: embedding similarity to prototype prompts for each category
* structural: regex features such as code markers, reasoning verbs and length

Both are deterministic and run in well under a millisecond, which matters
because routing sits on the hot path of every request.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import numpy as np

from .config import ModelTier, Settings, get_settings
from .embeddings import Embedder, HashingEmbedder, tokenize

logger = logging.getLogger(__name__)


class Category(str, Enum):
    SIMPLE_FORMATTING = "simple_formatting"
    FAQ = "faq"
    ADVANCED_LOGIC = "advanced_logic"
    CODING = "coding"


CATEGORY_LABELS: dict[Category, str] = {
    Category.SIMPLE_FORMATTING: "Simple formatting",
    Category.FAQ: "FAQ",
    Category.ADVANCED_LOGIC: "Advanced logic",
    Category.CODING: "Coding",
}

# Which tier each category needs. Formatting and FAQ work is well within a small model.
CATEGORY_TIER: dict[Category, str] = {
    Category.SIMPLE_FORMATTING: "standard",
    Category.FAQ: "standard",
    Category.ADVANCED_LOGIC: "premium",
    Category.CODING: "premium",
}

BASE_COMPLEXITY: dict[Category, float] = {
    Category.SIMPLE_FORMATTING: 0.10,
    Category.FAQ: 0.25,
    Category.ADVANCED_LOGIC: 0.80,
    Category.CODING: 0.85,
}

PROTOTYPES: dict[Category, tuple[str, ...]] = {
    Category.SIMPLE_FORMATTING: (
        "convert this text to uppercase",
        "fix the grammar and spelling in this sentence",
        "reformat this list as bullet points",
        "turn these comma separated values into a markdown table",
        "rewrite this paragraph in a more formal tone",
        "translate this sentence to spanish",
        "capitalize each word in the title",
        "shorten this message to one line",
        "format this date as iso 8601",
        "remove duplicate lines and sort alphabetically",
        "summarize this note in one sentence",
        "clean up the punctuation in this email",
    ),
    Category.FAQ: (
        "what is your refund policy",
        "how do i reset my password",
        "what are your support hours",
        "where can i find my invoice",
        "how long is the free trial",
        "what is the api rate limit on the pro plan",
        "do you support single sign on",
        "which regions is the service available in",
        "how do i cancel my subscription",
        "what is the uptime sla",
        "how is my data encrypted",
        "who do i contact about billing",
    ),
    Category.ADVANCED_LOGIC: (
        "analyze the trade offs between two architectures and recommend one with reasoning",
        "prove that the algorithm terminates and derive its complexity",
        "given these constraints work out the optimal schedule step by step",
        "evaluate the risks of this strategy and propose mitigations",
        "compare three pricing models and estimate the impact on revenue and churn",
        "reason through this logic puzzle and explain each deduction",
        "design an experiment to test the hypothesis and justify the sample size",
        "build a financial projection under several scenarios and explain the assumptions",
        "diagnose the root cause from these symptoms and rank the hypotheses",
        "critique this argument and identify the flawed premises",
        "plan a multi quarter migration with dependencies and contingencies",
        "derive the formula and show the intermediate steps",
    ),
    Category.CODING: (
        "write a python function that parses a csv file and returns a dictionary",
        "debug this stack trace and fix the null pointer exception",
        "implement a rate limiter class in go with unit tests",
        "refactor this javascript code to use async await",
        "write a sql query joining orders and customers grouped by month",
        "create a rest api endpoint in fastapi with pydantic validation",
        "explain why this recursive function overflows and rewrite it iteratively",
        "write a regex to validate email addresses",
        "optimize this pandas dataframe loop",
        "implement binary search tree insertion and deletion in java",
        "write a dockerfile and kubernetes deployment yaml for this service",
        "add type hints and unit tests to this module",
    ),
}

_F = re.IGNORECASE
FEATURES: dict[Category, tuple[tuple[re.Pattern[str], float], ...]] = {
    Category.CODING: (
        (re.compile(r"```|\bdef \w+\(|\bclass \w+[:(]|=>|\{\s*\n|;\s*\n|\bimport \w+|#include|</\w+>", _F), 0.9),
        (
            re.compile(
                r"\b(python|javascript|typescript|java|golang|rust|c\+\+|sql|bash|kotlin|swift|react|fastapi|django|"
                r"pandas|numpy|kubernetes|dockerfile|terraform|regex|graphql)\b|\b(in|written in|using) go\b",
                _F,
            ),
            0.6,
        ),
        (
            re.compile(
                r"\b(function|method|script|code|snippet|query|endpoint|unit tests?|stack ?trace|exception|traceback|"
                r"compile|refactor|debug|bug|api client|algorithm|class|module|library|migration script)\b",
                _F,
            ),
            0.5,
        ),
        (re.compile(r"\b(implement|write|build|create|fix)\b.{0,40}\b(function|class|script|query|service|parser|test)", _F), 0.5),
    ),
    Category.ADVANCED_LOGIC: (
        (
            re.compile(
                r"\b(analy[sz]e|evaluate|assess|derive|prove|justify|critique|diagnose|forecast|model|optimi[sz]e|"
                r"recommend|prioriti[sz]e|reason(ing)?|deduce|estimate|simulate|experiment)\b",
                _F,
            ),
            0.6,
        ),
        (
            re.compile(
                r"\b(trade.?offs?|step by step|root cause|pros and cons|scenarios?|assumptions?|constraints?|"
                r"implications?|strategy|hypothes[ie]s|second.order|sensitivity|mitigations?)\b",
                _F,
            ),
            0.6,
        ),
        (re.compile(r"\b(given that|assuming|suppose|if .{3,60} then|under what conditions|what would happen if)\b", _F), 0.4),
        (re.compile(r"\b(compare|versus|vs\.?)\b.{0,80}\b(and|with|against)\b.{0,80}\b(recommend|which|better|best|should)\b", _F), 0.5),
    ),
    Category.SIMPLE_FORMATTING: (
        (
            re.compile(
                r"\b(convert|reformat|format|capitali[sz]e|uppercase|lowercase|title case|rephrase|reword|proofread|"
                r"translate|shorten|trim|sort|alphabeti[sz]e|bullet points?|punctuation|spelling|grammar)\b",
                _F,
            ),
            0.7,
        ),
        (re.compile(r"\b(rewrite|clean up|tidy|polish|fix)\b.{0,30}\b(text|sentence|paragraph|email|message|note|list|title)\b", _F), 0.6),
        (re.compile(r"\b(into|as|to) (a |an )?(table|list|json|csv|markdown|bullets?|one line|single line)\b", _F), 0.5),
        (re.compile(r":\s*[\"']?.{3,}", _F), 0.15),
    ),
    Category.FAQ: (
        (re.compile(r"^\s*(what|when|where|who|which|how (do|can|long|much|many)|do you|does|is there|can i|are there)\b", _F), 0.5),
        (
            re.compile(
                r"\b(policy|refund|pricing|price|plan|subscription|billing|invoice|password|account|support|sla|"
                r"trial|warranty|shipping|hours|contact|limit|available|cancel|upgrade)\b",
                _F,
            ),
            0.6,
        ),
        (re.compile(r"\?\s*$"), 0.2),
    ),
}

SEMANTIC_WEIGHT = 0.55
FEATURE_WEIGHT = 0.45
SOFTMAX_TEMPERATURE = 0.12
NO_SIGNAL_FLOOR = 0.08


@dataclass(frozen=True)
class RoutingDecision:
    category: Category
    tier: str
    model: str
    complexity: float
    confidence: float
    rationale: str
    scores: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "category": self.category.value,
            "tier": self.tier,
            "model": self.model,
            "complexity": round(self.complexity, 3),
            "confidence": round(self.confidence, 3),
            "rationale": self.rationale,
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
        }


class SemanticCostRouter:
    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None) -> None:
        self.settings = settings or get_settings()
        self._embedder = embedder or HashingEmbedder()
        self._categories = list(Category)
        self._prototype_matrices = {cat: self._embedder.embed(PROTOTYPES[cat]) for cat in self._categories}

    def _semantic_scores(self, prompt: str) -> dict[Category, float]:
        vec = self._embedder.embed([prompt])[0]
        scores: dict[Category, float] = {}
        for cat, matrix in self._prototype_matrices.items():
            sims = np.sort(matrix @ vec)[::-1]
            # Best match dominates, the next two add a little evidence.
            scores[cat] = float(0.7 * sims[0] + 0.3 * sims[1:3].mean())
        return scores

    @staticmethod
    def _feature_scores(prompt: str) -> dict[Category, float]:
        scores: dict[Category, float] = {}
        for cat, features in FEATURES.items():
            miss = 1.0
            for pattern, weight in features:
                if pattern.search(prompt):
                    miss *= 1.0 - weight
            scores[cat] = 1.0 - miss
        return scores

    @lru_cache(maxsize=50_000)
    def classify(self, prompt: str) -> tuple[Category, float, dict[str, float]]:
        """Return (category, confidence, per-category probabilities)."""
        semantic = self._semantic_scores(prompt)
        feature = self._feature_scores(prompt)
        raw = {cat: SEMANTIC_WEIGHT * max(semantic[cat], 0.0) + FEATURE_WEIGHT * feature[cat] for cat in self._categories}
        peak = max(raw.values())
        if peak < NO_SIGNAL_FLOOR:
            # Nothing matched. Short prompts (greetings, one-liners) are cheap to serve;
            # long unclassifiable prompts go to the stronger model.
            fallback = Category.FAQ if len(tokenize(prompt)) <= 25 else Category.ADVANCED_LOGIC
            probs = {cat.value: (0.4 if cat is fallback else 0.2) for cat in self._categories}
            return fallback, 0.4, probs
        exps = {cat: math.exp((val - peak) / SOFTMAX_TEMPERATURE) for cat, val in raw.items()}
        total = sum(exps.values())
        probs = {cat: val / total for cat, val in exps.items()}
        best = max(probs, key=lambda c: probs[c])
        return best, probs[best], {cat.value: probs[cat] for cat in self._categories}

    def complexity(self, prompt: str, probs: dict[str, float]) -> float:
        """Expected category complexity plus a small bump for long prompts."""
        expected = sum(BASE_COMPLEXITY[Category(name)] * p for name, p in probs.items())
        n_tokens = len(tokenize(prompt))
        length_bump = 0.15 * min(1.0, max(0.0, (n_tokens - 120) / 600))
        return float(min(1.0, expected + length_bump))

    def route(self, prompt: str) -> RoutingDecision:
        category, confidence, probs = self.classify(prompt)
        complexity = self.complexity(prompt, probs)
        tier_name = CATEGORY_TIER[category]
        reasons = [f"classified as {CATEGORY_LABELS[category]} ({confidence:.0%} confidence)"]

        if tier_name == "standard" and complexity >= self.settings.complexity_threshold:
            tier_name = "premium"
            reasons.append(f"complexity {complexity:.2f} at or above threshold {self.settings.complexity_threshold:.2f}")
        elif tier_name == "standard" and confidence < self.settings.router_min_confidence:
            tier_name = "premium"
            reasons.append(f"confidence below {self.settings.router_min_confidence:.0%}, escalated to be safe")
        else:
            reasons.append(f"complexity {complexity:.2f}")

        tier: ModelTier = self.settings.tier(tier_name)
        model = tier.alias if self.settings.mock_llm else tier.provider_model
        decision = RoutingDecision(
            category=category,
            tier=tier_name,
            model=model,
            complexity=complexity,
            confidence=confidence,
            rationale="; ".join(reasons),
            scores=probs,
        )
        logger.debug("routed prompt category=%s tier=%s complexity=%.2f", category.value, tier_name, complexity)
        return decision
