"""Pure helpers that turn a list of spans into rows for a waterfall chart."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

WAIT_SPAN = "human_review_wait"
COLLAPSED_WAIT_MS = 40.0


@dataclass(frozen=True)
class WaterfallRow:
    span_id: str
    name: str
    depth: int
    offset_ms: float
    duration_ms: float
    detail: str
    status: str
    collapsed: bool
    attributes: dict[str, Any]


def _fmt_usd(value: float) -> str:
    return f"${value:.6f}" if value < 0.01 else f"${value:.4f}"


def span_detail(span: dict[str, Any]) -> str:
    """Short annotation for a span, matching what an operator looks for in each phase."""
    a = span.get("attributes", {})
    name = span["name"]
    if name in {"gateway_ingress", "gateway_resume"}:
        status = a.get("gateway.final_status") or a.get("gateway.status")
        return f"status {status}" if status else "total time"
    if name == "prompt_injection_check":
        verdict = "allowed" if a.get("guardrail.allowed", True) else "blocked"
        return f"{verdict}, risk {a.get('guardrail.risk_score', 0):.2f}"
    if name == "semantic_routing_decision":
        return f"{a.get('routing.model', '?')}, est. saved {_fmt_usd(a.get('routing.estimated_saved_usd', 0.0))}"
    if name == "rag_retrieval":
        return f"top score {a.get('retrieval.top_score', 0):.2f}"
    if name == "llm_generation_time":
        return f"{a.get('llm.tokens_per_sec', 0):.0f} tok/s, {a.get('llm.token_count.total', 0)} tokens, {_fmt_usd(a.get('llm.cost_usd', 0.0))}"
    if name == "rag_evaluation_latency":
        return f"faithfulness {a.get('eval.faithfulness', 0):.2f}, relevance {a.get('eval.relevance', 0):.2f}"
    if name == "human_approval_gate":
        action = a.get("hitl.action")
        return f"resumed: {action}" if action else "suspended for review"
    if name == WAIT_SPAN:
        return f"waited {a.get('hitl.wait_s', 0):.1f} s for reviewer"
    if name == "output":
        return str(a.get("gateway.final_status", ""))
    return ""


def build_waterfall(spans: list[dict[str, Any]], collapse_wait: bool = True) -> list[WaterfallRow]:
    """Order spans depth-first by start time and compute display offsets.

    With `collapse_wait`, time spent waiting on a human reviewer is squeezed to
    a fixed sliver so the machine phases stay readable.
    """
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: s["start_ns"])
    origin = ordered[0]["start_ns"]
    ids = {s["span_id"] for s in ordered}
    children: dict[str | None, list[dict[str, Any]]] = {}
    for span in ordered:
        parent = span["parent_span_id"] if span["parent_span_id"] in ids else None
        children.setdefault(parent, []).append(span)

    waits = [(s["start_ns"], s["end_ns"]) for s in ordered if s["name"] == WAIT_SPAN] if collapse_wait else []

    def shift_ms(at_ns: int) -> float:
        """Total wait time to remove from anything that starts at or after a wait began."""
        removed = 0.0
        for start, end in waits:
            if at_ns >= end:
                removed += (end - start) / 1e6 - COLLAPSED_WAIT_MS
        return removed

    rows: list[WaterfallRow] = []

    def visit(parent: str | None, depth: int) -> None:
        for span in children.get(parent, []):
            is_wait = collapse_wait and span["name"] == WAIT_SPAN
            offset = (span["start_ns"] - origin) / 1e6 - shift_ms(span["start_ns"])
            rows.append(
                WaterfallRow(
                    span_id=span["span_id"],
                    name=span["name"],
                    depth=depth,
                    offset_ms=offset,
                    duration_ms=COLLAPSED_WAIT_MS if is_wait else span["duration_ms"],
                    detail=span_detail(span),
                    status=span.get("status", "UNSET"),
                    collapsed=is_wait,
                    attributes=span.get("attributes", {}),
                )
            )
            visit(span["span_id"], depth + 1)

    visit(None, 0)
    return rows


def _sample_span(name: str, span_id: str, parent: str | None, start_ms: float, dur_ms: float, **attrs: Any) -> dict[str, Any]:
    start_ns = int(start_ms * 1e6)
    return {
        "trace_id": "sample",
        "span_id": span_id,
        "parent_span_id": parent,
        "name": name,
        "start_ns": start_ns,
        "end_ns": start_ns + int(dur_ms * 1e6),
        "duration_ms": dur_ms,
        "offset_ms": start_ms,
        "status": "UNSET",
        "attributes": attrs,
    }


# Shown by the dashboard when the API cannot be reached, so the panel is never empty.
SAMPLE_TRACE: dict[str, Any] = {
    "trace_id": "sample",
    "root_name": "gateway_ingress",
    "duration_ms": 1423.0,
    "spans": [
        _sample_span("gateway_ingress", "s0", None, 0, 1423.0, **{"gateway.status": "completed"}),
        _sample_span("prompt_injection_check", "s1", "s0", 1.2, 3.8, **{"guardrail.allowed": True, "guardrail.risk_score": 0.04}),
        _sample_span(
            "semantic_routing_decision", "s2", "s0", 5.6, 2.1,
            **{"routing.model": "mesh-standard", "routing.estimated_saved_usd": 0.004153, "routing.tier": "standard"},
        ),
        _sample_span("rag_retrieval", "s3", "s0", 8.4, 21.0, **{"retrieval.top_score": 0.71}),
        _sample_span(
            "llm_generation_time", "s4", "s0", 30.2, 1310.0,
            **{"llm.tokens_per_sec": 172.0, "llm.token_count.total": 468, "llm.cost_usd": 0.000165, "llm.model_name": "mesh-standard"},
        ),
        _sample_span("rag_evaluation_latency", "s5", "s0", 1341.0, 74.0, **{"eval.faithfulness": 0.92, "eval.relevance": 0.88}),
        _sample_span("output", "s6", "s0", 1416.0, 1.5, **{"gateway.final_status": "completed"}),
    ],
}
