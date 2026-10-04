"""Gateway orchestration: the LangGraph state machine every request runs through.

    guardrail_node -> router_node -> rag_execution_node -> eval_node
                   -> conditional_human_gate -> output_node

A blocked prompt skips straight from the guardrail to the output node. When the
human gate fires, the graph raises a LangGraph interrupt: the full state is
saved by the checkpointer under the request's thread id, the caller receives a
pending status, and `GatewayMesh.resume` continues the same graph once a
reviewer decides.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from opentelemetry import trace

from mesh.config import Settings, get_settings
from mesh.evals import Evaluator, HeuristicEvaluator, LLMJudgeEvaluator
from mesh.guardrails import Guardrail, RegexSemanticGuardrail, assess_action_risk
from mesh.hitl import ApprovalRecord, ApprovalStore
from mesh.llm import LLMClient
from mesh.mock_llm import MockRequest, mock_answer
from mesh.rag import InMemoryRetriever, RetrievedChunk, Retriever
from mesh.router import Category, SemanticCostRouter
from mesh.telemetry import SPAN_KIND, MetricsStore, Telemetry

logger = logging.getLogger(__name__)

BLOCKED_MESSAGE = "This request was blocked by the gateway security policy."
REJECTED_MESSAGE = "This response was withheld after human review."

SYSTEM_PROMPT = (
    "You are an enterprise assistant behind a governed AI gateway. Be accurate and concise. "
    "When context is provided, answer only from that context and say so if it does not contain the answer."
)

_QUESTION_RE = re.compile(
    r"\?\s*$|^\s*(what|when|where|who|whom|which|why|how|do|does|did|is|are|can|could|will|would|should|may)\b",
    re.IGNORECASE,
)

# Rough completion sizes used only to estimate savings at routing time, before generation.
EXPECTED_OUTPUT_TOKENS: dict[str, int] = {
    Category.SIMPLE_FORMATTING.value: 120,
    Category.FAQ.value: 200,
    Category.ADVANCED_LOGIC.value: 700,
    Category.CODING.value: 900,
}


class GatewayState(TypedDict, total=False):
    request_id: str
    thread_id: str
    user_id: str
    prompt: str
    trace_id: str
    root_span_id: str
    guardrail: dict[str, Any]
    risk: dict[str, Any]
    routing: dict[str, Any]
    is_knowledge_query: bool
    contexts: list[dict[str, Any]]
    draft_answer: str
    generation: dict[str, Any]
    evaluation: dict[str, Any] | None
    intercept_reasons: list[str]
    review: dict[str, Any] | None
    final_answer: str | None
    status: str


class ThreadNotFoundError(KeyError):
    pass


class GatewayMesh:
    """Wires the gateway components together and exposes handle/resume."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        guardrail: Guardrail | None = None,
        router: SemanticCostRouter | None = None,
        retriever: Retriever | None = None,
        llm: LLMClient | None = None,
        evaluator: Evaluator | None = None,
        approvals: ApprovalStore | None = None,
        telemetry: Telemetry | None = None,
        metrics: MetricsStore | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.guardrail = guardrail or RegexSemanticGuardrail(self.settings.guardrail_block_threshold)
        self.router = router or SemanticCostRouter(self.settings)
        self.retriever = retriever or InMemoryRetriever()
        self.llm = llm or LLMClient(self.settings)
        self.evaluator = evaluator or self._default_evaluator()
        self.approvals = approvals or ApprovalStore()
        self.telemetry = telemetry or Telemetry(self.settings)
        self.metrics = metrics or MetricsStore()
        self.tracer = self.telemetry.tracer
        # InMemorySaver keeps suspended state for the life of the process. Use a
        # Postgres or SQLite checkpointer when state has to survive restarts.
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = self.build_graph()

    def _default_evaluator(self) -> Evaluator:
        if self.settings.eval_mode == "llm_judge" and not self.settings.mock_llm:
            return LLMJudgeEvaluator(self.llm)
        return HeuristicEvaluator()

    # ------------------------------------------------------------------ graph

    def build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(GatewayState)
        graph.add_node("guardrail_node", self.guardrail_node)
        graph.add_node("router_node", self.router_node)
        graph.add_node("rag_execution_node", self.rag_execution_node)
        graph.add_node("eval_node", self.eval_node)
        graph.add_node("conditional_human_gate", self.conditional_human_gate)
        graph.add_node("output_node", self.output_node)

        graph.add_edge(START, "guardrail_node")
        graph.add_conditional_edges(
            "guardrail_node",
            self._after_guardrail,
            {"blocked": "output_node", "allowed": "router_node"},
        )
        graph.add_edge("router_node", "rag_execution_node")
        graph.add_edge("rag_execution_node", "eval_node")
        graph.add_edge("eval_node", "conditional_human_gate")
        graph.add_edge("conditional_human_gate", "output_node")
        graph.add_edge("output_node", END)
        return graph.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _after_guardrail(state: GatewayState) -> str:
        return "allowed" if state["guardrail"]["allowed"] else "blocked"

    # ------------------------------------------------------------------ nodes

    async def guardrail_node(self, state: GatewayState) -> dict[str, Any]:
        with self.tracer.start_as_current_span("prompt_injection_check", attributes={SPAN_KIND: "GUARDRAIL"}) as span:
            verdict = self.guardrail.inspect(state["prompt"])
            risk = assess_action_risk(state["prompt"]) if verdict.allowed else None
            span.set_attributes(
                {
                    "guardrail.allowed": verdict.allowed,
                    "guardrail.risk_score": round(verdict.risk_score, 3),
                    "guardrail.categories": list(verdict.categories),
                    "guardrail.high_risk_action": bool(risk and risk.high_risk),
                }
            )
        update: dict[str, Any] = {
            "guardrail": verdict.as_dict(),
            "risk": risk.as_dict() if risk else {"high_risk": False, "actions": []},
        }
        if not verdict.allowed:
            update["status"] = "blocked"
        return update

    async def router_node(self, state: GatewayState) -> dict[str, Any]:
        with self.tracer.start_as_current_span("semantic_routing_decision", attributes={SPAN_KIND: "CHAIN"}) as span:
            decision = self.router.route(state["prompt"])
            routing = decision.as_dict()
            if state["risk"]["high_risk"] and routing["tier"] != "premium":
                premium = self.settings.premium
                routing["tier"] = "premium"
                routing["model"] = premium.alias if self.settings.mock_llm else premium.provider_model
                routing["rationale"] += "; high-risk action escalated to premium"

            prompt_tokens = self.llm.count_tokens(text=state["prompt"])
            output_tokens = EXPECTED_OUTPUT_TOKENS[routing["category"]]
            chosen = self.settings.tier(routing["tier"])
            estimated_saved = max(
                0.0,
                self.settings.premium.cost(prompt_tokens, output_tokens) - chosen.cost(prompt_tokens, output_tokens),
            )
            routing["estimated_saved_usd"] = estimated_saved
            span.set_attributes(
                {
                    "routing.category": routing["category"],
                    "routing.tier": routing["tier"],
                    "routing.model": routing["model"],
                    "routing.complexity": routing["complexity"],
                    "routing.confidence": routing["confidence"],
                    "routing.estimated_saved_usd": estimated_saved,
                    "routing.rationale": routing["rationale"],
                }
            )
        return {"routing": routing}

    async def rag_execution_node(self, state: GatewayState) -> dict[str, Any]:
        routing = state["routing"]
        risk_actions = tuple(state["risk"]["actions"])
        wants_retrieval = routing["category"] == Category.FAQ.value and not risk_actions

        chunks: list[RetrievedChunk] = []
        if wants_retrieval:
            with self.tracer.start_as_current_span("rag_retrieval", attributes={SPAN_KIND: "RETRIEVER"}) as span:
                chunks = await self.retriever.retrieve(state["prompt"], self.settings.rag_top_k)
                span.set_attributes(
                    {
                        "retrieval.top_k": len(chunks),
                        "retrieval.top_score": round(chunks[0].score, 3) if chunks else 0.0,
                        "retrieval.documents": [c.doc_id for c in chunks],
                    }
                )

        # A knowledge query is a question, or anything the knowledge base clearly covers.
        # Greetings and small talk fall through as plain chat and skip evaluation.
        top_score = chunks[0].score if chunks else 0.0
        is_knowledge = wants_retrieval and (
            bool(_QUESTION_RE.search(state["prompt"])) or top_score >= self.settings.rag_min_similarity
        )
        if not is_knowledge:
            chunks = []

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if chunks:
            context_block = "\n\n".join(f"[{c.doc_id}] {c.title}\n{c.text}" for c in chunks)
            messages.append({"role": "system", "content": f"Context:\n{context_block}"})
        messages.append({"role": "user", "content": state["prompt"]})

        mock_text = mock_answer(
            MockRequest(
                prompt=state["prompt"],
                category=routing["category"],
                contexts=tuple(chunks),
                is_knowledge_query=is_knowledge,
                high_risk_actions=risk_actions,
                min_similarity=self.settings.rag_min_similarity,
                strong_similarity=self.settings.rag_strong_similarity,
            )
        )

        with self.tracer.start_as_current_span("llm_generation_time", attributes={SPAN_KIND: "LLM"}) as span:
            result = await self.llm.complete(routing["tier"], messages, mock_text=mock_text)
            span.set_attributes(
                {
                    "llm.model_name": result.model,
                    "llm.tier": result.tier,
                    "llm.token_count.prompt": result.prompt_tokens,
                    "llm.token_count.completion": result.completion_tokens,
                    "llm.token_count.total": result.prompt_tokens + result.completion_tokens,
                    "llm.tokens_per_sec": round(result.tokens_per_sec, 1),
                    "llm.cost_usd": result.cost_usd,
                    "llm.baseline_cost_usd": result.baseline_cost_usd,
                    "llm.saved_usd": result.saved_usd,
                    "llm.fallback_used": result.fallback_used,
                    "llm.attempts": result.attempts,
                }
            )

        return {
            "is_knowledge_query": is_knowledge,
            "contexts": [c.as_dict() for c in chunks],
            "draft_answer": result.text,
            "generation": result.as_dict(),
        }

    async def eval_node(self, state: GatewayState) -> dict[str, Any]:
        reasons = [f"HIGH_RISK_ACTION = {action}" for action in state["risk"]["actions"]]
        evaluation: dict[str, Any] | None = None

        if state["is_knowledge_query"]:
            with self.tracer.start_as_current_span("rag_evaluation_latency", attributes={SPAN_KIND: "EVALUATOR"}) as span:
                result = await self.evaluator.evaluate(
                    state["prompt"], state["draft_answer"], [c["text"] for c in state["contexts"]]
                )
                faithful = result.faithfulness >= self.settings.faithfulness_threshold
                relevant = result.relevance >= self.settings.relevance_threshold
                evaluation = {**result.as_dict(), "passed": faithful and relevant}
                span.set_attributes(
                    {
                        "eval.faithfulness": round(result.faithfulness, 3),
                        "eval.relevance": round(result.relevance, 3),
                        "eval.passed": faithful and relevant,
                        "eval.evaluator": result.evaluator,
                        "eval.faithfulness_threshold": self.settings.faithfulness_threshold,
                    }
                )
            if not faithful:
                reasons.append(f"RAG_FAITHFULNESS_SCORE = {result.faithfulness:.2f}")
            if not relevant:
                reasons.append(f"RAG_RELEVANCE_SCORE = {result.relevance:.2f}")

        return {"evaluation": evaluation, "intercept_reasons": reasons}

    async def conditional_human_gate(self, state: GatewayState) -> dict[str, Any]:
        """Pause for a reviewer when evaluation or policy requires it.

        LangGraph re-runs this node from the top on resume, so everything before
        `interrupt` is side-effect free.
        """
        reasons = state.get("intercept_reasons") or []
        if not reasons:
            return {"review": None}

        with self.tracer.start_as_current_span(
            "human_approval_gate",
            attributes={SPAN_KIND: "CHAIN", "hitl.reasons": reasons, "hitl.state": "suspended"},
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            decision: dict[str, Any] = interrupt(
                {
                    "thread_id": state["thread_id"],
                    "request_id": state["request_id"],
                    "prompt": state["prompt"],
                    "draft_answer": state["draft_answer"],
                    "reasons": reasons,
                }
            )
            action = decision["action"]
            span.set_attributes(
                {"hitl.state": "resumed", "hitl.action": action, "hitl.reviewer": decision.get("reviewer", "")}
            )

        review = {
            "action": action,
            "reviewer": decision.get("reviewer"),
            "note": decision.get("note"),
            "decided_at": time.time(),
        }
        if action == "reject":
            return {"review": review, "status": "rejected"}
        if action == "edit":
            return {"review": review, "final_answer": decision["edited_response"]}
        return {"review": review, "final_answer": state["draft_answer"]}

    async def output_node(self, state: GatewayState) -> dict[str, Any]:
        with self.tracer.start_as_current_span("output", attributes={SPAN_KIND: "CHAIN"}) as span:
            status = state.get("status")
            if status == "blocked":
                answer: str | None = BLOCKED_MESSAGE
            elif status == "rejected":
                answer = REJECTED_MESSAGE
            else:
                status = "completed"
                answer = state.get("final_answer") or state.get("draft_answer")
            span.set_attribute("gateway.final_status", status)
        return {"status": status, "final_answer": answer}

    # ------------------------------------------------------------- public API

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def _response(self, values: dict[str, Any], status: str, approval_id: str | None = None) -> dict[str, Any]:
        generation = values.get("generation")
        usage = None
        if generation:
            usage = {
                key: generation[key]
                for key in (
                    "model", "prompt_tokens", "completion_tokens", "cost_usd", "baseline_cost_usd",
                    "saved_usd", "latency_ms", "tokens_per_sec", "fallback_used",
                )
            }
        guardrail = values.get("guardrail")
        routing = values.get("routing")
        evaluation = values.get("evaluation")
        review = values.get("review") or {}
        return {
            "request_id": values["request_id"],
            "thread_id": values["thread_id"],
            "status": status,
            "answer": values.get("final_answer") if status != "pending_approval" else None,
            "guardrail": {
                "allowed": guardrail["allowed"],
                "risk_score": guardrail["risk_score"],
                "categories": guardrail["categories"],
                "reason": guardrail["reason"],
            }
            if guardrail
            else None,
            "routing": {k: routing[k] for k in ("category", "tier", "model", "complexity", "confidence", "rationale")}
            if routing
            else None,
            "usage": usage,
            "evaluation": {k: evaluation[k] for k in ("faithfulness", "relevance", "passed", "evaluator")}
            if evaluation
            else None,
            "approval_id": approval_id,
            "intercept_reasons": values.get("intercept_reasons") or [],
            "reviewed_by": review.get("reviewer"),
            "trace_id": values.get("trace_id"),
        }

    async def handle(self, prompt: str, user_id: str = "anonymous") -> dict[str, Any]:
        """Run one request through the graph. Returns a ChatResponse-shaped dict."""
        request_id = uuid.uuid4().hex
        thread_id = uuid.uuid4().hex

        with self.tracer.start_as_current_span(
            "gateway_ingress",
            attributes={
                SPAN_KIND: "CHAIN",
                "gateway.request_id": request_id,
                "gateway.thread_id": thread_id,
                "gateway.user_id": user_id,
                "gateway.prompt_preview": prompt[:120],
                "gateway.prompt_chars": len(prompt),
            },
        ) as span:
            ctx = span.get_span_context()
            initial: GatewayState = {
                "request_id": request_id,
                "thread_id": thread_id,
                "user_id": user_id,
                "prompt": prompt,
                "trace_id": format(ctx.trace_id, "032x"),
                "root_span_id": format(ctx.span_id, "016x"),
            }
            try:
                values = await self.graph.ainvoke(initial, self._config(thread_id))
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                logger.exception("request %s failed", request_id)
                raise

            interrupts = values.get("__interrupt__")
            if interrupts:
                status = "pending_approval"
                evaluation = values.get("evaluation") or {}
                routing = values["routing"]
                await self.approvals.create(
                    ApprovalRecord(
                        approval_id=thread_id,
                        request_id=request_id,
                        thread_id=thread_id,
                        user_id=user_id,
                        prompt=prompt,
                        draft_answer=values["draft_answer"],
                        reasons=list(values["intercept_reasons"]),
                        faithfulness=evaluation.get("faithfulness"),
                        relevance=evaluation.get("relevance"),
                        model=values["generation"]["model"],
                        tier=routing["tier"],
                        category=routing["category"],
                        contexts=values.get("contexts") or [],
                        trace_id=initial["trace_id"],
                        root_span_id=initial["root_span_id"],
                    )
                )
                span.set_attribute("hitl.reasons", list(values["intercept_reasons"]))
                response = self._response(values, status, approval_id=thread_id)
            else:
                status = values["status"]
                response = self._response(values, status)
            span.set_attribute("gateway.status", status)

        generation = values.get("generation") or {}
        routing = values.get("routing") or {}
        self.metrics.record_request(
            status=status,
            tier=routing.get("tier"),
            category=routing.get("category"),
            cost_usd=generation.get("cost_usd", 0.0),
            baseline_cost_usd=generation.get("baseline_cost_usd", 0.0),
            latency_ms=generation.get("latency_ms", 0.0),
        )
        logger.info(
            "request %s status=%s tier=%s category=%s", request_id, status, routing.get("tier"), routing.get("category")
        )
        return response

    async def resume(
        self,
        approval_id: str,
        action: str,
        edited_response: str | None = None,
        reviewer: str = "admin",
        note: str | None = None,
    ) -> dict[str, Any]:
        """Apply a reviewer decision and continue the suspended graph from its checkpoint."""
        if action == "edit" and not (edited_response and edited_response.strip()):
            raise ValueError("edited_response is required when action is 'edit'")

        record = await self.approvals.claim(approval_id)
        try:
            parent = self.telemetry.remote_parent(record.trace_id, record.root_span_id)  # type: ignore[arg-type]
            now = time.time()
            self.telemetry.record_interval(
                "human_review_wait",
                parent,
                record.created_at,
                now,
                {SPAN_KIND: "CHAIN", "hitl.wait_s": round(now - record.created_at, 3), "hitl.reviewer": reviewer},
            )
            with self.tracer.start_as_current_span(
                "gateway_resume",
                context=parent,
                attributes={SPAN_KIND: "CHAIN", "hitl.action": action, "hitl.reviewer": reviewer},
            ) as span:
                values = await self.graph.ainvoke(
                    Command(resume={"action": action, "edited_response": edited_response, "reviewer": reviewer, "note": note}),
                    self._config(record.thread_id),
                )
                status = values["status"]
                span.set_attribute("gateway.final_status", status)
            await self.approvals.resolve(
                approval_id, action, values.get("final_answer") if action != "reject" else None, reviewer, note
            )
        except Exception:
            await self.approvals.release(approval_id)
            logger.exception("resume failed for approval %s", approval_id)
            raise

        self.metrics.record_review(action, status)
        return self._response(values, status, approval_id=approval_id)

    async def get_thread(self, thread_id: str) -> dict[str, Any]:
        """Current view of a request, for clients polling after a pending response."""
        snapshot = await self.graph.aget_state(self._config(thread_id))
        if not snapshot.values:
            raise ThreadNotFoundError(thread_id)
        if snapshot.next:
            return self._response(snapshot.values, "pending_approval", approval_id=thread_id)
        return self._response(snapshot.values, snapshot.values["status"])
