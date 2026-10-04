"""Human-in-the-loop approval queue.

Holds the review record for every request the graph suspended. The graph state
itself lives in the LangGraph checkpointer; this store is the index the admin
console reads and the audit trail of who decided what.

The default implementation is in-memory. Back it with a database by keeping the
same method signatures.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

PENDING = "pending"
RESOLVED_STATUS = {"approve": "approved", "edit": "edited", "reject": "rejected"}


class ApprovalNotFoundError(KeyError):
    pass


class ApprovalConflictError(RuntimeError):
    """The approval was already decided or is being decided by someone else."""


@dataclass
class ApprovalRecord:
    approval_id: str
    request_id: str
    thread_id: str
    user_id: str
    prompt: str
    draft_answer: str
    reasons: list[str]
    faithfulness: float | None = None
    relevance: float | None = None
    model: str | None = None
    tier: str | None = None
    category: str | None = None
    contexts: list[dict[str, Any]] = field(default_factory=list)
    trace_id: str | None = None
    root_span_id: str | None = None
    status: str = PENDING
    created_at: float = field(default_factory=time.time)
    decided_at: float | None = None
    final_answer: str | None = None
    reviewer: str | None = None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("root_span_id", None)
        return data


class ApprovalStore:
    def __init__(self) -> None:
        self._records: dict[str, ApprovalRecord] = {}
        self._in_progress: set[str] = set()
        self._lock = asyncio.Lock()

    async def create(self, record: ApprovalRecord) -> ApprovalRecord:
        async with self._lock:
            existing = self._records.get(record.approval_id)
            if existing is not None:
                return existing
            self._records[record.approval_id] = record
        logger.info("approval queued id=%s reasons=%s", record.approval_id, record.reasons)
        return record

    async def get(self, approval_id: str) -> ApprovalRecord:
        async with self._lock:
            record = self._records.get(approval_id)
        if record is None:
            raise ApprovalNotFoundError(approval_id)
        return record

    async def list(self, status: str | None = None) -> list[ApprovalRecord]:
        async with self._lock:
            records = list(self._records.values())
        if status:
            records = [r for r in records if r.status == status]
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    async def claim(self, approval_id: str) -> ApprovalRecord:
        """Reserve a pending approval so two reviewers cannot resume the same graph."""
        async with self._lock:
            record = self._records.get(approval_id)
            if record is None:
                raise ApprovalNotFoundError(approval_id)
            if record.status != PENDING or approval_id in self._in_progress:
                raise ApprovalConflictError(f"approval {approval_id} is already {record.status}")
            self._in_progress.add(approval_id)
            return record

    async def release(self, approval_id: str) -> None:
        async with self._lock:
            self._in_progress.discard(approval_id)

    async def resolve(
        self, approval_id: str, action: str, final_answer: str | None, reviewer: str, note: str | None
    ) -> ApprovalRecord:
        async with self._lock:
            record = self._records.get(approval_id)
            if record is None:
                raise ApprovalNotFoundError(approval_id)
            record.status = RESOLVED_STATUS[action]
            record.decided_at = time.time()
            record.final_answer = final_answer
            record.reviewer = reviewer
            record.note = note
            self._in_progress.discard(approval_id)
        logger.info("approval resolved id=%s action=%s reviewer=%s", approval_id, action, reviewer)
        return record
