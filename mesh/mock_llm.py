"""Deterministic stand-in for model output, used when GATEWAY_MOCK_LLM is on.

The text returned here is passed to LiteLLM as `mock_response`, so the request
still travels through LiteLLM's completion path without any API key.

For knowledge queries the mock behaves like a model with three failure modes,
chosen by how well retrieval matched the question:

* strong match: answer copied from the retrieved context (faithful)
* weak match: one grounded sentence plus one invented detail (partly faithful)
* no match: a confident answer with invented specifics (unfaithful)

The invented text exists to exercise the evaluation gate and the human review
queue. It is clearly synthetic and is never produced when real providers are on.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass

from .embeddings import content_tokens
from .evals import split_sentences
from .rag import RetrievedChunk


@dataclass(frozen=True)
class MockRequest:
    prompt: str
    category: str
    contexts: tuple[RetrievedChunk, ...]
    is_knowledge_query: bool
    high_risk_actions: tuple[str, ...]
    min_similarity: float
    strong_similarity: float


def _pick(options: tuple[str, ...], seed_text: str) -> str:
    return options[zlib.crc32(seed_text.encode("utf-8")) % len(options)]


def _subject(prompt: str) -> str:
    """Best-effort subject of a question, used to keep invented answers on topic."""
    cleaned = re.sub(r"[?.!]+\s*$", "", prompt.strip())
    cleaned = re.sub(
        r"^\s*(what|which|when|where|who|how|do|does|is|are|can|could|will|would)\b\s*"
        r"(is|are|was|were|do|does|did|can|long|much|many|you|i|we|your|the|a|an|there)?\s*"
        r"(is|are|the|a|an|your|you|i|we)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip() or "that item"


def _best_sentences(prompt: str, chunk: RetrievedChunk, limit: int) -> list[str]:
    query = set(content_tokens(prompt))
    sentences = split_sentences(chunk.text)
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (-len(query & set(content_tokens(item[1]))), item[0]),
    )
    chosen = sorted(ranked[:limit], key=lambda item: item[0])
    return [sentence for _, sentence in chosen]


_INVENTED_DETAILS: tuple[str, ...] = (
    "It is covered for 24 months from the delivery date and claims are settled within 10 working days.",
    "It is included at no charge for the first 18 months and renews automatically at 49 USD per year.",
    "It applies worldwide and is handled by the regional service desk within 3 working days.",
)

_INVENTED_ADDONS: tuple[str, ...] = (
    "An express option is also available that completes within 2 hours for a 15 USD fee.",
    "Customers on legacy contracts receive an extra 45 day grace period.",
    "A dedicated hotline at extension 4417 handles these requests on weekends.",
)


def _knowledge_answer(req: MockRequest) -> str:
    top = req.contexts[0] if req.contexts else None
    if top is not None and top.score >= req.strong_similarity:
        return f"{top.title}: " + " ".join(_best_sentences(req.prompt, top, 2))
    if top is not None and top.score >= req.min_similarity:
        grounded = _best_sentences(req.prompt, top, 1)[0]
        return f"{top.title}: {grounded} {_pick(_INVENTED_ADDONS, req.prompt)}"
    subject = _subject(req.prompt)
    return f"Regarding {subject}: {_pick(_INVENTED_DETAILS, req.prompt)} {_pick(_INVENTED_ADDONS, req.prompt)}"


def _format_answer(prompt: str) -> str:
    head, sep, body = prompt.partition(":")
    text = body.strip() if sep and body.strip() else prompt.strip()
    instruction = head.lower() if sep else prompt.lower()
    if "upper" in instruction:
        return text.upper()
    if "lower" in instruction:
        return text.lower()
    if "title case" in instruction or "capitali" in instruction:
        return text.title()
    if "bullet" in instruction or "list" in instruction:
        items = [item.strip() for item in re.split(r"[,;\n]", text) if item.strip()]
        return "\n".join(f"- {item}" for item in items)
    if "sort" in instruction or "alphabet" in instruction:
        items = sorted((item.strip() for item in re.split(r"[,;\n]", text) if item.strip()), key=str.lower)
        return ", ".join(items)
    cleaned = re.sub(r"\s+", " ", text)
    return cleaned[:1].upper() + cleaned[1:] + ("" if cleaned.endswith((".", "!", "?")) else ".")


def _coding_answer(prompt: str) -> str:
    name = "_".join(content_tokens(prompt)[:3]) or "solution"
    return (
        "Here is an implementation with type hints and a docstring.\n\n"
        "```python\n"
        f"def {name}(items: list[int]) -> list[int]:\n"
        '    """Mock implementation returned by the offline model."""\n'
        "    result: list[int] = []\n"
        "    for item in items:\n"
        "        if item not in result:\n"
        "            result.append(item)\n"
        "    return sorted(result)\n"
        "```\n\n"
        "It runs in O(n log n) time. Add unit tests for empty input and duplicates before shipping."
    )


def _logic_answer(prompt: str) -> str:
    subject = " ".join(prompt.split()[:14])
    return (
        f"Analysis of: {subject}...\n\n"
        "1. Frame the decision: list the options, the constraints and the metric that decides success.\n"
        "2. Compare the options on cost, operational risk and time to value, noting which assumptions drive each estimate.\n"
        "3. Stress test the leading option against the two most likely failure scenarios.\n"
        "4. Recommendation: take the option that meets the constraints at the lowest operational risk, "
        "run it as a time-boxed pilot and define the rollback trigger in advance.\n\n"
        "This is a mock reasoning trace from the offline model."
    )


def _action_answer(req: MockRequest) -> str:
    actions = ", ".join(req.high_risk_actions)
    return (
        f"Proposed action plan for: {req.prompt.strip()}\n\n"
        "1. Confirm the target scope and take a snapshot or backup.\n"
        "2. Run the change in dry-run mode and record the affected resources.\n"
        "3. Execute the change inside the approved window and verify the result.\n\n"
        f"This request was classified as {actions}. Nothing is executed until an authorised reviewer approves it."
    )


def mock_answer(req: MockRequest) -> str:
    if req.high_risk_actions:
        return _action_answer(req)
    if req.is_knowledge_query:
        return _knowledge_answer(req)
    if req.category == "coding":
        return _coding_answer(req.prompt)
    if req.category == "advanced_logic":
        return _logic_answer(req.prompt)
    if req.category == "simple_formatting":
        return _format_answer(req.prompt)
    return "Hello. This is the gateway's offline mock model. Ask a product question, or send a formatting, reasoning or coding task."
