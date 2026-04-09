"""Firecrawl API emulator — converts Ollama search/fetch to Firecrawl-compatible responses.

Firecrawl is HIGH PRIORITY. We emulate:
- POST /scrape  (single URL scraping)
- POST /search  (search the web)
- POST /crawl   (multi-page crawl — uses fetch for each URL)
- POST /extract (extract structured data from URLs)
"""

from __future__ import annotations

import uuid
import time
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from guanaco.search.base import ProviderEmulator, register_provider


# ── Request Models ──

class ScrapeRequest(BaseModel):
    url: str
    formats: list[str] = Field(default_factory=lambda: ["markdown"])
    only_main_content: bool = True
    include_tags: list[str] = Field(default_factory=list)
    exclude_tags: list[str] = Field(default_factory=list)
    timeout: int = 30000
    actions: Optional[list[dict]] = None


class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    scrape_options: Optional[dict] = None
    lang: str = "en"


class CrawlRequest(BaseModel):
    url: str
    limit: int = 10
    scrape_options: Optional[dict] = None
    max_depth: int = 2


class ExtractRequest(BaseModel):
    urls: list[str] = Field(default_factory=list)
    prompt: Optional[str] = None
    schema_: Optional[dict] = Field(default=None, alias="schema")


# ── Response Models ──

class FirecrawlMetadata(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    source_url: Optional[str] = None
    status_code: int = 200


class ScrapeResponse(BaseModel):
    success: bool = True
    data: Optional[dict] = None
    metadata: Optional[FirecrawlMetadata] = None


class SearchResult(BaseModel):
    title: str
    url: str
    content: str
    description: Optional[str] = None


class SearchResponse(BaseModel):
    success: bool = True
    data: list[SearchResult] = Field(default_factory=list)


class CrawlResult(BaseModel):
    title: str
    url: str
    content: str
    markdown: str
    metadata: Optional[FirecrawlMetadata] = None


class CrawlResponse(BaseModel):
    success: bool = True
    status: str = "completed"
    completed: int = 0
    total: int = 0
    credits_used: int = 0
    data: list[CrawlResult] = Field(default_factory=list)


class ExtractResponse(BaseModel):
    success: bool = True
    data: dict = Field(default_factory=dict)


# ── Provider ──

@register_provider
class FirecrawlProvider(ProviderEmulator):
    name = "firecrawl"
    prefix = "/firecrawl"

    def register_routes(self, app):
        router = APIRouter(prefix=self.prefix, tags=["Firecrawl"])

        async def _scrape(body: ScrapeRequest):
            ollama_resp = await self.ollama.fetch(url=body.url)
            title = ollama_resp.get("title", "")
            content = ollama_resp.get("content", "")
            links = ollama_resp.get("links", [])

            data = {}
            if "markdown" in body.formats or not body.formats:
                data["markdown"] = content
            if "html" in body.formats:
                data["html"] = content  # Ollama returns text, approx
            if "rawHtml" in body.formats:
                data["rawHtml"] = content
            if "links" in body.formats:
                data["links"] = links

            return ScrapeResponse(
                success=True,
                data=data,
                metadata=FirecrawlMetadata(
                    title=title,
                    source_url=body.url,
                    status_code=200,
                ),
            )

        async def _search(body: SearchRequest):
            ollama_resp = await self.ollama.search(
                query=body.query,
                max_results=body.limit,
            )

            results = []
            for r in ollama_resp.get("results", []):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                    description=r.get("content", "")[:200],
                ))

            return SearchResponse(success=True, data=results)

        @router.post("/scrape", response_model=ScrapeResponse)
        async def scrape(body: ScrapeRequest, request: Request):
            return await _scrape(body)

        @router.post("/v2/scrape", response_model=ScrapeResponse, include_in_schema=False)
        async def scrape_v2(body: ScrapeRequest, request: Request):
            return await _scrape(body)

        @router.post("/search", response_model=SearchResponse)
        async def search(body: SearchRequest, request: Request):
            return await _search(body)

        @router.post("/v2/search", response_model=SearchResponse, include_in_schema=False)
        async def search_v2(body: SearchRequest, request: Request):
            return await _search(body)

        @router.post("/crawl", response_model=CrawlResponse)
        async def crawl(body: CrawlRequest, request: Request):
            # Crawl = scrape the seed URL + follow links
            ollama_resp = await self.ollama.fetch(url=body.url)
            title = ollama_resp.get("title", "")
            content = ollama_resp.get("content", "")
            links = ollama_resp.get("links", [])

            results = [CrawlResult(
                title=title,
                url=body.url,
                content=content,
                markdown=content,
                metadata=FirecrawlMetadata(title=title, source_url=body.url),
            )]

            # Follow up to limit-1 additional links
            for link in links[:body.limit - 1]:
                try:
                    link_resp = await self.ollama.fetch(url=link)
                    lt = link_resp.get("title", "")
                    lc = link_resp.get("content", "")
                    results.append(CrawlResult(
                        title=lt,
                        url=link,
                        content=lc,
                        markdown=lc,
                        metadata=FirecrawlMetadata(title=lt, source_url=link),
                    ))
                except Exception:
                    continue

            return CrawlResponse(
                success=True,
                status="completed",
                completed=len(results),
                total=len(results),
                data=results,
            )

        @router.post("/extract", response_model=ExtractResponse)
        async def extract(body: ExtractRequest, request: Request):
            # Extract uses fetch + LLM to pull structured data
            all_content = {}
            for url in body.urls[:5]:  # limit to 5 URLs
                try:
                    resp = await self.ollama.fetch(url=url)
                    all_content[url] = resp.get("content", "")
                except Exception:
                    all_content[url] = ""

            # If prompt provided, we could route through LLM later
            # For now, return raw content mapped by URL
            return ExtractResponse(success=True, data=all_content)

        app.include_router(router)