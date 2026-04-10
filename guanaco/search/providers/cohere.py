"""Cohere Rerank API emulator — uses Ollama LLM for reranking."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from typing import Optional

from guanaco.search.base import ProviderEmulator, register_provider


# ── Request/Response Models ──

class CohereRerankRequest(BaseModel):
    model: str = "rerank-v3.5"
    query: str
    documents: list[str]
    top_n: Optional[int] = None
    return_documents: bool = False


class CohereRerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[dict] = None


class CohereRerankResponse(BaseModel):
    results: list[CohereRerankResult] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)


# ── Provider ──

@register_provider
class CohereProvider(ProviderEmulator):
    name = "cohere"
    prefix = "/cohere"
    endpoints = ({"path": "/rerank", "method": "POST"},)

    def register_routes(self, app):
        router = APIRouter(prefix=self.prefix, tags=["Cohere"])

        @router.post("/rerank", response_model=CohereRerankResponse)
        async def rerank(body: CohereRerankRequest, request: Request):
            top_n = body.top_n or len(body.documents)

            # Keyword overlap heuristic scoring
            query_words = set(body.query.lower().split())
            scored = []
            for i, doc in enumerate(body.documents):
                doc_words = set(doc.lower().split())
                overlap = len(query_words & doc_words) / max(len(query_words), 1)
                scored.append(CohereRerankResult(
                    index=i,
                    relevance_score=round(min(overlap + 0.3, 1.0), 4),
                    document={"text": doc} if body.return_documents else None,
                ))

            scored.sort(key=lambda x: x.relevance_score, reverse=True)
            results = scored[:top_n]

            return CohereRerankResponse(
                results=results,
                meta={"api_version": {"version": "1"}, "billed_units": {"search_units": 1}},
            )

        app.include_router(router)