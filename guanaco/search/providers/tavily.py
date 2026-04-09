"""Tavily API emulator — converts Ollama search to Tavily-compatible responses."""

from __future__ import annotations

from fastapi import APIRouter, Header, Query, Request
from pydantic import BaseModel, Field
from typing import Optional

from guanaco.search.base import ProviderEmulator, register_provider


# ── Request/Response Models ──

class TavilySearchRequest(BaseModel):
    query: str
    search_depth: str = "basic"  # basic | advanced
    max_results: int = 5
    topic: str = "general"  # general | news
    include_answer: bool = False
    include_raw_content: bool = False
    include_images: bool = False
    include_image_descriptions: bool = False
    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)


class TavilySearchResult(BaseModel):
    title: str
    url: str
    content: str
    score: float = 0.0
    raw_content: Optional[str] = None


class TavilySearchResponse(BaseModel):
    query: str
    answer: Optional[str] = None
    results: list[TavilySearchResult] = Field(default_factory=list)
    response_time: float = 0.0


# ── Provider ──

@register_provider
class TavilyProvider(ProviderEmulator):
    name = "tavily"
    prefix = "/tavily"

    def register_routes(self, app):
        router = APIRouter(prefix=self.prefix, tags=["Tavily"])

        @router.post("/search", response_model=TavilySearchResponse)
        async def tavily_search(
            body: TavilySearchRequest,
            request: Request,
        ):
            import time
            start = time.time()

            # Use Ollama search
            ollama_resp = await self.ollama.search(
                query=body.query,
                max_results=body.max_results,
            )

            results = []
            for r in ollama_resp.get("results", []):
                results.append(TavilySearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                    score=0.5,  # Ollama doesn't return scores
                    raw_content=r.get("content") if body.include_raw_content else None,
                ))

            return TavilySearchResponse(
                query=body.query,
                answer=None,
                results=results,
                response_time=round(time.time() - start, 3),
            )

        app.include_router(router)