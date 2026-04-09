"""SearXNG API emulator — converts Ollama search to SearXNG-compatible responses."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from typing import Optional

from guanaco.search.base import ProviderEmulator, register_provider


# ── Response Models ──

class SearXNGResult(BaseModel):
    title: str
    url: str
    content: str
    engine: str = "ollama"
    engines: list[str] = Field(default_factory=lambda: ["ollama"])
    score: float = 0.0
    category: str = "general"
    parsed_url: Optional[list[str]] = None
    template: str = "default.html"


class SearXNGSearchResponse(BaseModel):
    query: str
    number_of_results: int = 0
    results: list[SearXNGResult] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    infoboxes: list[dict] = Field(default_factory=list)


# ── Provider ──

@register_provider
class SearXNGProvider(ProviderEmulator):
    name = "searxng"
    prefix = "/searxng"

    def register_routes(self, app):
        router = APIRouter(prefix=self.prefix, tags=["SearXNG"])

        @router.get("/search", response_model=SearXNGSearchResponse)
        async def searxng_search_get(
            q: str,
            format: str = "json",
            pageno: int = 1,
            categories: Optional[str] = None,
            request: Request = None,
        ):
            ollama_resp = await self.ollama.search(query=q, max_results=10)
            return _format_searxng(q, ollama_resp)

        @router.post("/search", response_model=SearXNGSearchResponse)
        async def searxng_search_post(
            q: str = "",
            format: str = "json",
            pageno: int = 1,
            categories: Optional[str] = None,
            request: Request = None,
        ):
            ollama_resp = await self.ollama.search(query=q, max_results=10)
            return _format_searxng(q, ollama_resp)

        # SearXNG also accepts requests at root /
        @router.get("/", response_model=SearXNGSearchResponse, include_in_schema=False)
        async def searxng_root_get(q: str, format: str = "json"):
            ollama_resp = await self.ollama.search(query=q, max_results=10)
            return _format_searxng(q, ollama_resp)

        app.include_router(router)


def _format_searxng(query: str, ollama_resp: dict) -> SearXNGSearchResponse:
    results = []
    for r in ollama_resp.get("results", []):
        url = r.get("url", "")
        parsed = url.replace("://", "/").split("/") if url else []
        results.append(SearXNGResult(
            title=r.get("title", ""),
            url=url,
            content=r.get("content", ""),
            parsed_url=parsed,
        ))
    return SearXNGSearchResponse(
        query=query,
        number_of_results=len(results),
        results=results,
    )