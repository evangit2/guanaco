"""Exa API emulator — converts Ollama search to Exa-compatible responses."""

from __future__ import annotations

import uuid
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from typing import Optional

from guanaco.search.base import ProviderEmulator, register_provider


# ── Request/Response Models ──

class ExaContentOptions(BaseModel):
    text: bool = False
    highlights: Optional[dict] = None
    summary: Optional[dict] = None


class ExaSearchRequest(BaseModel):
    query: str
    type: str = "auto"
    num_results: int = Field(default=10, alias="numResults")
    start_published_date: Optional[str] = Field(default=None, alias="startPublishedDate")
    end_published_date: Optional[str] = Field(default=None, alias="endPublishedDate")
    include_domains: list[str] = Field(default_factory=list, alias="includeDomains")
    exclude_domains: list[str] = Field(default_factory=list, alias="excludeDomains")
    contents: Optional[ExaContentOptions] = None


class ExaFindSimilarRequest(BaseModel):
    url: str
    num_results: int = Field(default=10, alias="numResults")
    include_domains: list[str] = Field(default_factory=list, alias="includeDomains")
    exclude_domains: list[str] = Field(default_factory=list, alias="excludeDomains")
    contents: Optional[ExaContentOptions] = None


class ExaResult(BaseModel):
    id: str
    title: str
    url: str
    published_date: Optional[str] = Field(default=None, alias="publishedDate")
    author: Optional[str] = None
    text: Optional[str] = None
    highlights: Optional[list[str]] = None
    summary: Optional[str] = None


class ExaSearchResponse(BaseModel):
    request_id: str = Field(default="", alias="requestId")
    results: list[ExaResult] = Field(default_factory=list)


class ExaFindSimilarResponse(BaseModel):
    request_id: str = Field(default="", alias="requestId")
    results: list[ExaResult] = Field(default_factory=list)


# ── Provider ──

@register_provider
class ExaProvider(ProviderEmulator):
    name = "exa"
    prefix = "/exa"
    endpoints = ({"path": "/search", "method": "POST"}, {"path": "/findSimilar", "method": "POST"})

    def register_routes(self, app):
        router = APIRouter(prefix=self.prefix, tags=["Exa"])

        @router.post("/search", response_model=ExaSearchResponse)
        async def exa_search(body: ExaSearchRequest, request: Request):
            ollama_resp = await self.ollama.search(
                query=body.query,
                max_results=body.num_results,
            )

            results = []
            for r in ollama_resp.get("results", []):
                result = ExaResult(
                    id=str(uuid.uuid4()),
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                )
                if body.contents:
                    if body.contents.text:
                        result.text = r.get("content", "")
                    if body.contents.highlights:
                        result.highlights = [r.get("content", "")[:200]]
                results.append(result)

            return ExaSearchResponse(
                request_id=str(uuid.uuid4()),
                results=results,
            )

        @router.post("/findSimilar", response_model=ExaFindSimilarResponse)
        async def exa_find_similar(body: ExaFindSimilarRequest, request: Request):
            # Use Ollama fetch to get the URL content, then search for similar
            fetch_resp = await self.ollama.fetch(url=body.url)
            title = fetch_resp.get("title", "")
            # Use the title as a search query for similar results
            ollama_resp = await self.ollama.search(
                query=title or body.url,
                max_results=body.num_results,
            )

            results = []
            for r in ollama_resp.get("results", []):
                result = ExaResult(
                    id=str(uuid.uuid4()),
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                )
                results.append(result)

            return ExaFindSimilarResponse(
                request_id=str(uuid.uuid4()),
                results=results,
            )

        app.include_router(router)