"""Serper API emulator — converts Ollama search/fetch to Serper-compatible responses."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from typing import Optional

from guanaco.search.base import ProviderEmulator, register_provider


# ── Request Models ──

class SerperSearchRequest(BaseModel):
    q: str
    gl: str = "us"
    hl: str = "en"
    num: int = 10
    page: int = 1
    type: Optional[str] = None  # news, images, videos, places


class SerperScrapeRequest(BaseModel):
    url: str


# ── Response Models ──

class SerperOrganicResult(BaseModel):
    title: str
    link: str
    snippet: str
    position: int = 0


class SerperKnowledgePanel(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None


class SerperSearchResponse(BaseModel):
    search_parameters: dict = Field(default_factory=dict)
    organic: list[SerperOrganicResult] = Field(default_factory=list)
    knowledge_graph: Optional[SerperKnowledgePanel] = None
    search_information: dict = Field(default_factory=dict)


class SerperScrapeResponse(BaseModel):
    url: str
    title: str
    content: str
    links: list[str] = Field(default_factory=list)


# ── Provider ──

@register_provider
class SerperProvider(ProviderEmulator):
    name = "serper"
    prefix = "/serper"
    endpoints = ({"path": "/search", "method": "POST"}, {"path": "/scrape", "method": "POST"})

    def register_routes(self, app):
        router = APIRouter(prefix=self.prefix, tags=["Serper"])

        @router.post("/search", response_model=SerperSearchResponse)
        async def search(body: SerperSearchRequest, request: Request):
            ollama_resp = await self.ollama.search(
                query=body.q,
                max_results=body.num,
            )

            organic = []
            for i, r in enumerate(ollama_resp.get("results", [])):
                organic.append(SerperOrganicResult(
                    title=r.get("title", ""),
                    link=r.get("url", ""),
                    snippet=r.get("content", ""),
                    position=i + 1,
                ))

            return SerperSearchResponse(
                search_parameters={"q": body.q, "gl": body.gl, "hl": body.hl},
                organic=organic,
                search_information={"total_results": len(organic)},
            )

        @router.post("/scrape", response_model=SerperScrapeResponse)
        async def scrape(body: SerperScrapeRequest, request: Request):
            ollama_resp = await self.ollama.fetch(url=body.url)
            return SerperScrapeResponse(
                url=body.url,
                title=ollama_resp.get("title", ""),
                content=ollama_resp.get("content", ""),
                links=ollama_resp.get("links", []),
            )

        app.include_router(router)