"""Retrieval layer.

A small in-memory knowledge base for a fictional SaaS vendor, searched by
embedding similarity. Implement `Retriever` to point the gateway at a real
vector store (FAISS, Chroma, Pinecone) without touching the graph.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .embeddings import Embedder, HashingEmbedder, content_tokens

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    title: str
    text: str
    score: float

    def as_dict(self) -> dict[str, object]:
        return {"doc_id": self.doc_id, "title": self.title, "text": self.text, "score": round(self.score, 3)}


class Retriever(Protocol):
    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]: ...


KNOWLEDGE_BASE: tuple[Document, ...] = (
    Document(
        "kb-refunds",
        "Refund policy",
        "Northwind Cloud offers a full refund within 30 days of purchase on annual plans. "
        "Monthly plans are not refundable but can be cancelled at any time and stay active until the end of the billing period. "
        "Refund requests are submitted from the Billing page and are processed within 5 business days to the original payment method.",
    ),
    Document(
        "kb-sla",
        "Uptime SLA",
        "The uptime SLA is 99.9 percent per calendar month on the Pro plan and 99.99 percent on the Enterprise plan. "
        "If uptime falls below the SLA, customers receive a service credit of 10 percent of the monthly fee for each 0.1 percent missed. "
        "Scheduled maintenance announced 72 hours in advance is excluded from the SLA calculation.",
    ),
    Document(
        "kb-rate-limits",
        "API rate limits",
        "The API rate limit is 60 requests per minute on the Free plan, 600 requests per minute on the Pro plan and 6000 requests per minute on the Enterprise plan. "
        "Requests over the limit receive HTTP 429 with a Retry-After header. "
        "Rate limit increases can be requested through support on the Enterprise plan.",
    ),
    Document(
        "kb-retention",
        "Data retention",
        "Customer data is retained for 90 days after an account is cancelled and is then permanently deleted. "
        "Audit logs are retained for 12 months on the Enterprise plan and 30 days on the Pro plan. "
        "Customers can request earlier deletion of their data by contacting privacy@northwind.example.",
    ),
    Document(
        "kb-sso",
        "Single sign-on",
        "Single sign-on with SAML 2.0 and OpenID Connect is available on the Enterprise plan. "
        "Supported identity providers include Okta, Microsoft Entra ID and Google Workspace. "
        "SSO is configured by an organization owner under Settings, Security, Single sign-on.",
    ),
    Document(
        "kb-support",
        "Support hours",
        "Standard support is available Monday to Friday from 9:00 to 18:00 IST by email and chat. "
        "Enterprise customers have 24x7 support with a 1 hour response target for critical incidents. "
        "The support team can be reached at support@northwind.example.",
    ),
    Document(
        "kb-encryption",
        "Encryption and security",
        "All data is encrypted in transit with TLS 1.3 and at rest with AES-256. "
        "Encryption keys are managed in a dedicated key management service and rotated every 90 days. "
        "Enterprise customers can bring their own encryption keys.",
    ),
    Document(
        "kb-regions",
        "Hosting regions",
        "Northwind Cloud is hosted in three regions: Mumbai, Frankfurt and Virginia. "
        "The data region is chosen when the organization is created and cannot be changed later without a migration request. "
        "Data does not leave the selected region except for encrypted backups stored in the paired region.",
    ),
    Document(
        "kb-pricing",
        "Plans and pricing",
        "The Free plan includes 3 users and 1 project. "
        "The Pro plan costs 29 USD per user per month billed monthly or 24 USD per user per month billed annually. "
        "The Enterprise plan has custom pricing and includes SSO, audit logs and a dedicated account manager.",
    ),
    Document(
        "kb-trial",
        "Free trial",
        "The Pro plan has a free trial of 14 days with no credit card required. "
        "At the end of the trial the organization moves to the Free plan unless a payment method is added. "
        "Trial extensions of 7 days can be requested once per organization.",
    ),
    Document(
        "kb-password",
        "Password reset",
        "To reset a password, select Forgot password on the sign-in page and enter the account email address. "
        "A reset link is emailed and is valid for 30 minutes. "
        "Accounts that use single sign-on must reset the password with their identity provider.",
    ),
    Document(
        "kb-incidents",
        "Incident response",
        "Critical incidents are acknowledged within 15 minutes and status updates are posted every 30 minutes on status.northwind.example. "
        "A written post-incident review is published within 5 business days of resolution. "
        "Customers are notified of security incidents affecting their data within 72 hours.",
    ),
)


class InMemoryRetriever:
    """Hybrid retrieval: embedding cosine blended with IDF-weighted term overlap."""

    def __init__(
        self,
        documents: tuple[Document, ...] = KNOWLEDGE_BASE,
        embedder: Embedder | None = None,
        lexical_weight: float = 0.5,
    ) -> None:
        self._documents = documents
        self._embedder = embedder or HashingEmbedder()
        self._lexical_weight = lexical_weight
        indexed = [f"{doc.title}. {doc.text}" for doc in documents]
        self._matrix = self._embedder.embed(indexed)
        self._doc_tokens = [set(content_tokens(text)) for text in indexed]
        n_docs = len(documents)
        doc_freq: dict[str, int] = {}
        for tokens in self._doc_tokens:
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        self._idf = {token: math.log(1 + n_docs / freq) for token, freq in doc_freq.items()}
        self._max_idf = math.log(1 + n_docs)

    def _lexical_scores(self, query: str) -> np.ndarray:
        q_tokens = set(content_tokens(query))
        total = sum(self._idf.get(tok, self._max_idf) for tok in q_tokens)
        if total == 0:
            return np.zeros(len(self._documents), dtype=np.float32)
        return np.array(
            [sum(self._idf[tok] for tok in q_tokens & doc) / total for doc in self._doc_tokens],
            dtype=np.float32,
        )

    async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        query_vec = self._embedder.embed([query])[0]
        dense = np.clip(self._matrix @ query_vec, 0.0, 1.0)
        scores = (1 - self._lexical_weight) * dense + self._lexical_weight * self._lexical_scores(query)
        order = scores.argsort()[::-1][:top_k]
        chunks = [
            RetrievedChunk(self._documents[i].doc_id, self._documents[i].title, self._documents[i].text, float(scores[i]))
            for i in order
        ]
        logger.debug("retrieved %d chunks, top score %.3f", len(chunks), chunks[0].score if chunks else 0.0)
        return chunks
