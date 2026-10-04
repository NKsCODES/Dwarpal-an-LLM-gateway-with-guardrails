"""Real-time RAG evaluation: faithfulness and answer relevance.

`HeuristicEvaluator` runs offline with no model calls:

* faithfulness: share of answer sentences whose content words and numbers are
  supported by a retrieved context sentence
* relevance: how well the answer covers the content words of the question,
  blended with embedding similarity between question and answer

`LLMJudgeEvaluator` asks a model to grade the same two metrics and is meant for
use with real providers. Both satisfy the `Evaluator` protocol.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .embeddings import Embedder, HashingEmbedder, content_tokens

if TYPE_CHECKING:
    from .llm import LLMClient

logger = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_NUMBER_RE = re.compile(r"\d+(?:[.,:]\d+)*")


@dataclass(frozen=True)
class EvalResult:
    faithfulness: float
    relevance: float
    evaluator: str
    unsupported_claims: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "faithfulness": round(self.faithfulness, 3),
            "relevance": round(self.relevance, 3),
            "evaluator": self.evaluator,
            "unsupported_claims": list(self.unsupported_claims),
        }


class Evaluator(Protocol):
    async def evaluate(self, question: str, answer: str, contexts: list[str]) -> EvalResult: ...


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if len(s.strip()) > 3]


class HeuristicEvaluator:
    name = "heuristic"

    def __init__(self, support_threshold: float = 0.6, embedder: Embedder | None = None) -> None:
        self.support_threshold = support_threshold
        self._embedder = embedder or HashingEmbedder()

    def _claim_support(self, claim: str, context_sentences: list[set[str]], context_numbers: set[str]) -> float:
        tokens = set(content_tokens(claim))
        if not tokens:
            return 1.0
        best = max((len(tokens & ctx) / len(tokens) for ctx in context_sentences), default=0.0)
        numbers = set(_NUMBER_RE.findall(claim))
        if numbers and not numbers <= context_numbers:
            # A figure that appears nowhere in the context is the clearest sign of fabrication.
            best *= 0.5
        return best

    def faithfulness(self, answer: str, contexts: list[str]) -> tuple[float, list[str]]:
        claims = split_sentences(answer)
        if not claims:
            return 0.0, []
        context_sentences = [set(content_tokens(s)) for ctx in contexts for s in split_sentences(ctx)]
        context_numbers = {n for ctx in contexts for n in _NUMBER_RE.findall(ctx)}
        supported = 0
        unsupported: list[str] = []
        for claim in claims:
            if self._claim_support(claim, context_sentences, context_numbers) >= self.support_threshold:
                supported += 1
            else:
                unsupported.append(claim)
        return supported / len(claims), unsupported

    def relevance(self, question: str, answer: str) -> float:
        q_tokens = set(content_tokens(question))
        if not q_tokens:
            return 1.0
        a_tokens = set(content_tokens(answer))
        coverage = len(q_tokens & a_tokens) / len(q_tokens)
        vectors = self._embedder.embed([question, answer])
        similarity = max(0.0, float(vectors[0] @ vectors[1]))
        # Similarity between a short question and a longer answer is low in absolute terms, so rescale it.
        return float(min(1.0, 0.7 * coverage + 0.3 * min(1.0, similarity * 3.0)))

    async def evaluate(self, question: str, answer: str, contexts: list[str]) -> EvalResult:
        faithfulness, unsupported = self.faithfulness(answer, contexts)
        relevance = self.relevance(question, answer)
        return EvalResult(faithfulness, relevance, self.name, tuple(unsupported))


JUDGE_PROMPT = """You are grading a retrieval-augmented answer.

Question:
{question}

Retrieved context:
{context}

Answer:
{answer}

Score two metrics from 0.0 to 1.0.
faithfulness: share of claims in the answer that are directly supported by the context.
relevance: how completely the answer addresses the question.

Reply with JSON only: {{"faithfulness": <float>, "relevance": <float>}}"""


class LLMJudgeEvaluator:
    """Model-graded evaluation. Falls back to the heuristic scorer if the judge reply cannot be parsed."""

    name = "llm_judge"

    def __init__(self, llm: "LLMClient", judge_tier: str = "standard") -> None:
        self._llm = llm
        self._judge_tier = judge_tier
        self._fallback = HeuristicEvaluator()

    async def evaluate(self, question: str, answer: str, contexts: list[str]) -> EvalResult:
        prompt = JUDGE_PROMPT.format(question=question, context="\n\n".join(contexts), answer=answer)
        try:
            result = await self._llm.complete(self._judge_tier, [{"role": "user", "content": prompt}])
            payload = json.loads(re.search(r"\{.*\}", result.text, re.DOTALL).group(0))  # type: ignore[union-attr]
            faithfulness = min(1.0, max(0.0, float(payload["faithfulness"])))
            relevance = min(1.0, max(0.0, float(payload["relevance"])))
            return EvalResult(faithfulness, relevance, self.name)
        except Exception:  # noqa: BLE001 - a broken judge must never fail the request
            logger.exception("LLM judge failed, falling back to heuristic evaluation")
            return await self._fallback.evaluate(question, answer, contexts)
