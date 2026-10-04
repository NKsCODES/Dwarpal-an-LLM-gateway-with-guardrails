"""FastAPI entry point for the Enterprise AI Gateway.

Run with:  uvicorn app:app --port 8000

Routes
    POST /api/v1/chat                         run a prompt through the gateway
    GET  /api/v1/chat/{thread_id}             poll a request (after a pending response)
    GET  /api/v1/approvals                    human review queue
    GET  /api/v1/approvals/{approval_id}
    POST /api/v1/approvals/{approval_id}/decision   approve, edit or reject and resume the graph
    GET  /api/v1/traces, /api/v1/traces/{trace_id}  recent OpenTelemetry traces
    GET  /api/v1/metrics                      running totals
    GET  /healthz
"""

from __future__ import annotations

import hmac
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from gateway import GatewayMesh, ThreadNotFoundError
from mesh import __version__
from mesh.config import Settings, get_settings
from mesh.hitl import ApprovalConflictError, ApprovalNotFoundError
from mesh.llm import LLMUnavailableError
from mesh.schemas import ApprovalDecision, ApprovalView, ChatRequest, ChatResponse

logger = logging.getLogger("gateway.api")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

STATUS_CODES: dict[str, int] = {
    "completed": status.HTTP_200_OK,
    "pending_approval": status.HTTP_202_ACCEPTED,
    "blocked": status.HTTP_403_FORBIDDEN,
    "rejected": status.HTTP_200_OK,
}


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )


def _check_key(expected: str | None, provided: str | None) -> None:
    """Constant-time key check. A route whose key is not configured stays open."""
    if expected is None:
        return
    if provided is None or not hmac.compare_digest(expected.encode(), provided.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing API key")


def get_mesh(request: Request) -> GatewayMesh:
    return request.app.state.mesh


def require_client(request: Request, key: str | None = Depends(_api_key_header)) -> None:
    _check_key(request.app.state.settings.api_key, key)


def require_admin(request: Request, key: str | None = Depends(_api_key_header)) -> None:
    settings: Settings = request.app.state.settings
    _check_key(settings.admin_key or settings.api_key, key)


def create_app(settings: Settings | None = None, mesh: GatewayMesh | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        app.state.mesh = mesh or GatewayMesh(settings)
        logger.info(
            "gateway started env=%s mock_llm=%s premium=%s standard=%s",
            settings.environment, settings.mock_llm, settings.premium.provider_model, settings.standard.provider_model,
        )
        if settings.api_key is None:
            logger.warning("GATEWAY_API_KEY is not set: all routes are open. Set it before exposing this service.")
        yield
        app.state.mesh.telemetry.shutdown()

    app = FastAPI(
        title="Enterprise AI Gateway & Orchestration Mesh",
        version=__version__,
        description="Guardrails, cost routing, RAG evaluation, human approval and tracing in front of LLMs.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def add_request_context(request: Request, call_next: Any) -> Response:
        http_request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        started = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = http_request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        logger.info("%s %s -> %d in %.1fms", request.method, request.url.path, response.status_code, elapsed_ms)
        return response

    @app.exception_handler(LLMUnavailableError)
    async def llm_unavailable(_: Request, exc: LLMUnavailableError) -> JSONResponse:
        logger.error("no model available: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "No model is currently available. Try again shortly."},
            headers={"Retry-After": "5"},
        )

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Internal gateway error."})

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "mock_llm": settings.mock_llm, "environment": settings.environment}

    @app.post(
        "/api/v1/chat",
        response_model=ChatResponse,
        tags=["gateway"],
        dependencies=[Depends(require_client)],
        responses={
            202: {"model": ChatResponse, "description": "Paused for human approval."},
            403: {"model": ChatResponse, "description": "Blocked by the security guardrail."},
        },
    )
    async def chat(body: ChatRequest, response: Response, mesh: GatewayMesh = Depends(get_mesh)) -> ChatResponse:
        result = await mesh.handle(body.prompt, body.user_id)
        response.status_code = STATUS_CODES[result["status"]]
        if result["status"] == "pending_approval":
            response.headers["Location"] = f"/api/v1/chat/{result['thread_id']}"
        return ChatResponse(**result)

    @app.get("/api/v1/chat/{thread_id}", response_model=ChatResponse, tags=["gateway"], dependencies=[Depends(require_client)])
    async def chat_status(thread_id: str, mesh: GatewayMesh = Depends(get_mesh)) -> ChatResponse:
        try:
            return ChatResponse(**await mesh.get_thread(thread_id))
        except ThreadNotFoundError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown thread id") from None

    @app.get("/api/v1/approvals", response_model=list[ApprovalView], tags=["hitl"], dependencies=[Depends(require_admin)])
    async def list_approvals(
        approval_status: Literal["pending", "approved", "edited", "rejected"] | None = Query(default=None, alias="status"),
        mesh: GatewayMesh = Depends(get_mesh),
    ) -> list[ApprovalView]:
        return [ApprovalView(**record.as_dict()) for record in await mesh.approvals.list(approval_status)]

    @app.get("/api/v1/approvals/{approval_id}", response_model=ApprovalView, tags=["hitl"], dependencies=[Depends(require_admin)])
    async def get_approval(approval_id: str, mesh: GatewayMesh = Depends(get_mesh)) -> ApprovalView:
        try:
            return ApprovalView(**(await mesh.approvals.get(approval_id)).as_dict())
        except ApprovalNotFoundError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown approval id") from None

    @app.post(
        "/api/v1/approvals/{approval_id}/decision",
        response_model=ChatResponse,
        tags=["hitl"],
        dependencies=[Depends(require_admin)],
    )
    async def decide(approval_id: str, body: ApprovalDecision, mesh: GatewayMesh = Depends(get_mesh)) -> ChatResponse:
        if body.action == "edit" and not body.edited_response:
            raise HTTPException(422, "edited_response is required when action is 'edit'")
        try:
            result = await mesh.resume(approval_id, body.action, body.edited_response, body.reviewer, body.note)
        except ApprovalNotFoundError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown approval id") from None
        except ApprovalConflictError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None
        return ChatResponse(**result)

    @app.get("/api/v1/traces", tags=["observability"], dependencies=[Depends(require_admin)])
    async def list_traces(limit: int = Query(default=50, ge=1, le=500), mesh: GatewayMesh = Depends(get_mesh)) -> list[dict[str, Any]]:
        return mesh.telemetry.store.list_traces(limit)

    @app.get("/api/v1/traces/{trace_id}", tags=["observability"], dependencies=[Depends(require_admin)])
    async def get_trace(trace_id: str, mesh: GatewayMesh = Depends(get_mesh)) -> dict[str, Any]:
        found = mesh.telemetry.store.get_trace(trace_id)
        if found is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown trace id")
        return found

    @app.get("/api/v1/metrics", tags=["observability"], dependencies=[Depends(require_admin)])
    async def metrics(mesh: GatewayMesh = Depends(get_mesh)) -> dict[str, Any]:
        snapshot = mesh.metrics.snapshot()
        snapshot["pending_approvals"] = len(await mesh.approvals.list("pending"))
        return snapshot

    return app


configure_logging()
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=os.getenv("GATEWAY_HOST", "127.0.0.1"), port=int(os.getenv("GATEWAY_PORT", "8000")))
