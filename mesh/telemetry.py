"""Observability: OpenTelemetry tracing, an in-process trace store and counters.

Every gateway phase is recorded as an OpenTelemetry span. Spans always go to
`TraceStore`, a bounded in-memory exporter that backs the dashboard waterfall.
When an OTLP endpoint is configured (for example Arize Phoenix at
http://localhost:6006/v1/traces) the same spans are also exported there.

Span attributes follow the OpenInference naming that Phoenix understands
(`openinference.span.kind`, `llm.model_name`, `llm.token_count.*`).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Sequence

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

SPAN_KIND = "openinference.span.kind"


def _hex_trace(trace_id: int) -> str:
    return format(trace_id, "032x")


def _hex_span(span_id: int) -> str:
    return format(span_id, "016x")


class TraceStore(SpanExporter):
    """Keeps the most recent traces in memory, grouped by trace id."""

    def __init__(self, max_traces: int = 500) -> None:
        self._max_traces = max_traces
        self._traces: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            for span in spans:
                trace_id = _hex_trace(span.context.trace_id)
                start_ns = span.start_time or 0
                end_ns = span.end_time or start_ns
                record = {
                    "trace_id": trace_id,
                    "span_id": _hex_span(span.context.span_id),
                    "parent_span_id": _hex_span(span.parent.span_id) if span.parent else None,
                    "name": span.name,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "duration_ms": (end_ns - start_ns) / 1e6,
                    "status": span.status.status_code.name,
                    "attributes": {k: (list(v) if isinstance(v, tuple) else v) for k, v in (span.attributes or {}).items()},
                }
                self._traces.setdefault(trace_id, []).append(record)
                self._traces.move_to_end(trace_id)
            while len(self._traces) > self._max_traces:
                self._traces.popitem(last=False)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover - nothing to release
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:  # pragma: no cover
        return True

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        with self._lock:
            spans = [dict(s) for s in self._traces.get(trace_id, [])]
        if not spans:
            return None
        spans.sort(key=lambda s: s["start_ns"])
        origin = spans[0]["start_ns"]
        for span in spans:
            span["offset_ms"] = (span["start_ns"] - origin) / 1e6
        return {"trace_id": trace_id, "spans": spans, **self._summarise(spans)}

    @staticmethod
    def _summarise(spans: list[dict[str, Any]]) -> dict[str, Any]:
        root = next((s for s in spans if s["parent_span_id"] is None), spans[0])
        merged: dict[str, Any] = {}
        for span in spans:
            merged.update(span["attributes"])
        return {
            "root_name": root["name"],
            "started_at": root["start_ns"] / 1e9,
            "duration_ms": root["duration_ms"],
            "span_count": len(spans),
            "status": merged.get("gateway.final_status") or merged.get("gateway.status"),
            "model": merged.get("llm.model_name"),
            "tier": merged.get("routing.tier"),
            "category": merged.get("routing.category"),
            "cost_usd": merged.get("llm.cost_usd"),
            "prompt_preview": merged.get("gateway.prompt_preview"),
        }

    def list_traces(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            items = [(tid, [dict(s) for s in spans]) for tid, spans in self._traces.items()]
        summaries = [{"trace_id": tid, **self._summarise(sorted(spans, key=lambda s: s["start_ns"]))} for tid, spans in items]
        summaries.sort(key=lambda s: s["started_at"], reverse=True)
        return summaries[:limit]


class Telemetry:
    """Owns the tracer provider for one gateway instance."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = TraceStore(self.settings.trace_buffer_size)
        resource = Resource.create(
            {"service.name": self.settings.service_name, "deployment.environment": self.settings.environment}
        )
        self.provider = TracerProvider(resource=resource)
        self.provider.add_span_processor(SimpleSpanProcessor(self.store))
        if self.settings.otlp_endpoint:
            self._attach_otlp(self.settings.otlp_endpoint)
        self.tracer = self.provider.get_tracer("ai_gateway_mesh")

    def _attach_otlp(self, endpoint: str) -> None:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            self.provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
            logger.info("exporting spans over OTLP to %s", endpoint)
        except Exception:  # noqa: BLE001 - tracing must never stop the gateway from starting
            logger.exception("could not attach OTLP exporter for %s", endpoint)

    @staticmethod
    def remote_parent(trace_id_hex: str, span_id_hex: str) -> Context:
        """Context that parents new spans under a span from an earlier request."""
        parent = SpanContext(
            trace_id=int(trace_id_hex, 16),
            span_id=int(span_id_hex, 16),
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        return trace.set_span_in_context(NonRecordingSpan(parent))

    def record_interval(
        self, name: str, parent: Context, start_s: float, end_s: float, attributes: dict[str, Any]
    ) -> None:
        """Emit a span for an interval that already happened, such as time spent waiting on a reviewer."""
        span = self.tracer.start_span(name, context=parent, start_time=int(start_s * 1e9), attributes=attributes)
        span.end(end_time=int(end_s * 1e9))

    def shutdown(self) -> None:
        self.provider.shutdown()


class MetricsStore:
    """Running totals for the overview panel. Process-local by design."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = time.time()
        self._requests = 0
        self._by_status: dict[str, int] = {}
        self._by_tier: dict[str, int] = {}
        self._by_category: dict[str, int] = {}
        self._reviews: dict[str, int] = {}
        self._cost = 0.0
        self._baseline = 0.0
        self._latency_ms = 0.0
        self._generations = 0

    def record_request(
        self,
        status: str,
        tier: str | None,
        category: str | None,
        cost_usd: float,
        baseline_cost_usd: float,
        latency_ms: float,
    ) -> None:
        with self._lock:
            self._requests += 1
            self._by_status[status] = self._by_status.get(status, 0) + 1
            if tier:
                self._by_tier[tier] = self._by_tier.get(tier, 0) + 1
                self._generations += 1
                self._latency_ms += latency_ms
            if category:
                self._by_category[category] = self._by_category.get(category, 0) + 1
            self._cost += cost_usd
            self._baseline += baseline_cost_usd

    def record_review(self, action: str, final_status: str) -> None:
        """Count a reviewer decision and move the request out of the pending bucket."""
        with self._lock:
            self._reviews[action] = self._reviews.get(action, 0) + 1
            pending = self._by_status.get("pending_approval", 0)
            if pending > 0:
                self._by_status["pending_approval"] = pending - 1
            self._by_status[final_status] = self._by_status.get(final_status, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "uptime_s": round(time.time() - self._started, 1),
                "requests": self._requests,
                "by_status": dict(self._by_status),
                "by_tier": dict(self._by_tier),
                "by_category": dict(self._by_category),
                "reviews": dict(self._reviews),
                "cost_usd": self._cost,
                "baseline_cost_usd": self._baseline,
                "saved_usd": max(0.0, self._baseline - self._cost),
                "avg_llm_latency_ms": (self._latency_ms / self._generations) if self._generations else 0.0,
            }
