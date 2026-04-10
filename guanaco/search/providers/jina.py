"""Jina API emulator — converts Ollama search/fetch to Jina-compatible responses.

Endpoints:
- POST /search (Jina search/SVL)
- POST /read (Jina reader — URL scraping)
- POST /rerank (Jina reranker — uses LLM)
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from typing import Optional

from guanaco.search.base import ProviderEmulator, register_provider


# ── Request Models ──

class JinaSearchRequest(BaseModel):
    q: str
    num: int = 10
    site: Optional[list[str]] = None


class JinaReadRequest(BaseModel):
    url: str


class JinaRerankRequest(BaseModel):
    model: str = "jina-reranker-v2-base-multilingual"
    query: str
    documents: list[str]
    top_n: Optional[int] = None
    return_documents: bool = False


# ── Response Models ──

class JinaSearchResult(BaseModel):
    title: str
    url: str
    description: str
    content: Optional[str] = None


class JinaSearchResponse(BaseModel):
    code: int = 200
    status: int = 20000
    data: list[JinaSearchResult] = Field(default_factory=list)


class JinaReadResponse(BaseModel):
    code: int = 200
    status: int = 20000
    data: dict = Field(default_factory=dict)


class JinaRerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[dict] = None


class JinaRerankResponse(BaseModel):
    model: str
    results: list[JinaRerankResult] = Field(default_factory=list)
    usage: dict = Field(default_factory=dict)


# ── Provider ──

@register_provider
class JinaProvider(ProviderEmulator):
    name = "jina"
    prefix = "/jina"
    endpoints = [{"path": "/search", "method": "POST"}, {"path": "/read", "method": "POST"}, {"path": "/rerank", "method": "POST"}]

    def register_routes(self, app):
        router = APIRouter(prefix=self.prefix, tags=["Jina"])

        @router.post("/search", response_model=JinaSearchResponse)
        @router.post("/v1/search", response_model=JinaSearchResponse, include_in_schema=False)
        async def search(body: JinaSearchRequest, request: Request):
            ollama_resp = await self.ollama.search(
                query=body.q,
                max_results=body.num,
            )

            results = []
            for r in ollama_resp.get("results", []):
                results.append(JinaSearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    description=r.get("content", ""),
                    content=r.get("content"),
                ))

            return JinaSearchResponse(data=results)

        @router.post("/read", response_model=JinaReadResponse)
        @router.post("/v1/read", response_model=JinaReadResponse, include_in_schema=False)
        async def read(body: JinaReadRequest, request: Request):
            ollama_resp = await self.ollama.fetch(url=body.url)
            return JinaReadResponse(data={
                "title": ollama_resp.get("title", ""),
                "content": ollama_resp.get("content", ""),
                "url": body.url,
                "links": ollama_resp.get("links", []),
            })

        @router.post("/rerank", response_model=JinaRerankResponse)
        @router.post("/v1/rerank", response_model=JinaRerankResponse, include_in_schema=False)
        async def rerank(body: JinaRerankRequest, request: Request):
            # Use LLM for reranking — construct a prompt that scores relevance
            import json
            top_n = body.top_n or len(body.documents)
            
            prompt = (
                f"Given the query: \"{body.query}\"\n\n"
                f"Rank these documents by relevance (0.0 to 1.0):\n\n"
            )
            for i, doc in enumerate(body.documents):
                prompt += f"Document {i}: {doc[:500]}\n\n"
            
            prompt += (
                "Return ONLY a JSON array of objects with 'index' and 'score' fields, "
                "sorted by score descending. Example: [{\"index\": 2, \"score\": 0.95}, ...]"
            )

            # We'll use a simple heuristic for now — keyword overlap scoring
            # Full LLM reranking can be enabled later when chat completion is available
            query_words = set(body.query.lower().split())
            scored = []
            for i, doc in enumerate(body.documents):
                doc_words = set(doc.lower().split())
                overlap = len(query_words & doc_words) / max(len(query_words), 1)
                scored.append(JinaRerankResult(
                    index=i,
                    relevance_score=round(min(overlap + 0.3, 1.0), 4),
                    document={"text": doc} if body.return_documents else None,
                ))

            scored.sort(key=lambda x: x.relevance_score, reverse=True)
            results = scored[:top_n]

            return JinaRerankResponse(
                model=body.model,
                results=results,
                usage={"prompt_tokens": 0, "total_tokens": 0},
            )

        app.include_router(router)

        # Bare /jina POST route — LibreChat sends rerank requests to the base URL
        @app.post("/jina")
        async def jina_bare_rerank(request: Request):
            """LibreChat calls POST to the Jina base URL directly for reranking."""
            body = await request.json()
            documents = body.get("documents", [])
            query = body.get("query", "")
            model = body.get("model", "jina-reranker-v2-base-multilingual")
            top_n = body.get("top_n", len(documents))
            return_documents = body.get("return_documents", False)

            query_words = set(query.lower().split())
            scored = []
            for i, doc in enumerate(documents):
                doc_text = doc if isinstance(doc, str) else doc.get("text", str(doc))
                doc_words = set(doc_text.lower().split())
                overlap = len(query_words & doc_words) / max(len(query_words), 1)
                scored.append({
                    "index": i,
                    "relevance_score": round(min(overlap + 0.3, 1.0), 4),
                    "document": {"text": doc_text} if return_documents else None,
                })

            scored.sort(key=lambda x: x["relevance_score"], reverse=True)
            results = scored[:top_n]
            for r in results:
                if r["document"] is None:
                    del r["document"]

            return {
                "model": model,
                "results": results,
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }