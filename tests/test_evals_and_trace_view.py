from __future__ import annotations

import asyncio

from mesh.evals import HeuristicEvaluator
from mesh.trace_view import SAMPLE_TRACE, build_waterfall

CONTEXT = [
    "Northwind Cloud offers a full refund within 30 days of purchase on annual plans. "
    "Refund requests are processed within 5 business days to the original payment method."
]
evaluator = HeuristicEvaluator()


def evaluate(question: str, answer: str):
    return asyncio.run(evaluator.evaluate(question, answer, CONTEXT))


def test_grounded_answer_scores_high() -> None:
    result = evaluate("What is the refund policy?", "Refund policy: a full refund is offered within 30 days of purchase on annual plans.")
    assert result.faithfulness == 1.0
    assert result.relevance >= 0.7


def test_invented_figures_are_caught() -> None:
    result = evaluate("What is the refund policy?", "Refund policy: a full refund is offered within 90 days of purchase on annual plans.")
    assert result.faithfulness == 0.0
    assert result.unsupported_claims


def test_partly_grounded_answer_scores_in_between() -> None:
    answer = (
        "Refund requests are processed within 5 business days to the original payment method. "
        "A dedicated hotline at extension 4417 handles these requests on weekends."
    )
    assert evaluate("How long do refunds take?", answer).faithfulness == 0.5


def test_off_topic_answer_has_low_relevance() -> None:
    assert evaluate("What is the refund policy?", "Our offices are closed on public holidays.").relevance < 0.5


def test_waterfall_orders_spans_and_nests_children() -> None:
    rows = build_waterfall(SAMPLE_TRACE["spans"])
    assert rows[0].name == "gateway_ingress" and rows[0].depth == 0
    assert [r.depth for r in rows[1:]] == [1] * (len(rows) - 1)
    assert [r.offset_ms for r in rows] == sorted(r.offset_ms for r in rows)
    assert "tok/s" in next(r for r in rows if r.name == "llm_generation_time").detail


def test_waterfall_collapses_reviewer_wait() -> None:
    def span(name: str, sid: str, parent: str | None, start_ms: float, dur_ms: float) -> dict:
        start = int(start_ms * 1e6)
        return {"span_id": sid, "parent_span_id": parent, "name": name, "start_ns": start,
                "end_ns": start + int(dur_ms * 1e6), "duration_ms": dur_ms, "status": "UNSET", "attributes": {}}

    spans = [
        span("gateway_ingress", "a", None, 0, 100),
        span("human_review_wait", "b", "a", 100, 60_000),
        span("gateway_resume", "c", "a", 60_100, 5),
    ]
    collapsed = {r.name: r for r in build_waterfall(spans, collapse_wait=True)}
    assert collapsed["human_review_wait"].collapsed and collapsed["human_review_wait"].duration_ms == 40.0
    assert collapsed["gateway_resume"].offset_ms == 140.0
    full = {r.name: r for r in build_waterfall(spans, collapse_wait=False)}
    assert full["gateway_resume"].offset_ms == 60_100.0
