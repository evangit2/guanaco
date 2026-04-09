"""Brave Search API emulator — converts Ollama search to Brave Search-compatible responses."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from guanaco.search.base import ProviderEmulator, register_provider


# ── Response Models ──

class BraveWebResult(BaseModel):
    title: str
    url: str
    description: str


class BraveSearchResponse(BaseModel):
    type: str = "search"
    web: dict = Field(default_factory=dict)


# ── Provider ──

@register_provider
class BraveProvider(ProviderEmulator):
    name = "brave"
    prefix = "/brave"

    def register_routes(self, app):
        router = APIRouter(prefix=self.prefix, tags=["Brave"])

        @router.get("/search", response_model=BraveSearchResponse)
        async def search_get(
            q: str,
            count: int = 10,
            offset: int = 0,
            request: Request = None,
        ):
            ollama_resp = await self.ollama.search(query=q, max_results=count)
            return _format_brave(q, ollama_resp)

        @router.post("/search", response_model=BraveSearchResponse)
        async def search_post(body: dict, request: Request):
            q = body.get("q", "")
            count = body.get("count", 10)
            ollama_resp = await self.ollama.search(query=q, max_results=count)
            return _format_brave(q, ollama_resp)

        app.include_router(router)


def _format_brave(query: str, ollama_resp: dict) -> BraveSearchResponse:
    results = []
    for r in ollama_resp.get("results", []):
        results.append(BraveWebResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            description=r.get("content", ""),
        ))
    return BraveSearchResponse(
        type="search",
        web={"results": results},
    )