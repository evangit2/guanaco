"""Main FastAPI application — ties together LLM router, search providers, dashboard, and status."""

from __future__ import annotations

import os
import time
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from guanaco.config import load_config, get_config, AppConfig, get_base_url, get_tailscale_ip
from guanaco.client import OllamaClient
__version__ = "0.4.1"
from guanaco.router.router import create_router as create_llm_router
from guanaco.search.providers import ALL_PROVIDERS
from guanaco.dashboard import create_dashboard_router
from guanaco.utils.api_keys import ApiKeyManager
from guanaco.analytics import AnalyticsLogger


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create the combined FastAPI application with all routes on a single port."""
    if config is None:
        config = load_config()

    resolved_key = os.getenv("OLLAMA_API_KEY", "") or config.ollama_api_key or ""
    if not resolved_key:
        print("Warning: OLLAMA_API_KEY not set. Set it with 'guanaco setup' or export OLLAMA_API_KEY.")

    client = OllamaClient(api_key=resolved_key, session_cookie=config.usage.session_cookie)

    from guanaco.config import get_default_config_dir
    key_manager = ApiKeyManager(get_default_config_dir())
    analytics = AnalyticsLogger()
    providers_config = config.providers.model_dump()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        base_url = get_base_url(config)
        print(f"Guanaco running on http://{config.router.host}:{config.router.port}")
        print(f"   LLM Router:    {base_url}:{config.router.port}/v1/chat/completions")
        print(f"   Anthropic:     {base_url}:{config.router.port}/v1/messages")
        print(f"   Search APIs:    {base_url}:{config.router.port}/<provider>/...")
        print(f"   Dashboard:     {base_url}:{config.router.port}/dashboard")
        print(f"   Analytics DB:  {analytics.db_path}")
        analytics.log_status("info", "system", "Guanaco started", {
            "host": config.router.host, "port": config.router.port,
            "cache_beta": config.cache.beta_mode,
        })
        if config.cache.beta_mode:
            print(f"   Cache (BETA):  ENABLED — exact_ttl={config.cache.exact_cache_ttl}s, prefix_ttl={config.cache.session_prefix_ttl}s, dedup={config.cache.dedup_enabled}")
        else:
            print(f"   Cache (BETA):  DISABLED (enable with /v1/config/cache)")
        yield
        await client.close()

    app = FastAPI(
        title="Guanaco",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Search request analytics middleware ──
    @app.middleware("http")
    async def search_analytics_middleware(request: Request, call_next):
        path = request.url.path.strip("/")
        is_search = any(path.startswith(p) for p in [
            "tavily", "exa", "searxng", "firecrawl", "serper", "jina", "cohere", "brave",
            "v2/scrape", "v2/search", "v2/crawl", "v2/extract",
        ])

        if not is_search:
            return await call_next(request)

        # Map v2/ paths to firecrawl provider for analytics
        if path.startswith("v2/"):
            provider = "firecrawl"
        else:
            provider = path.split("/")[0] if path else ""
        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start

        analytics.log_search(
            provider=provider,
            endpoint=path,
            duration_seconds=round(elapsed, 3),
            error=None if response.status_code < 400 else f"HTTP {response.status_code}",
        )

        return response

    # ── API key auth middleware ──
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        path = request.url.path.strip("/")
        if not path or path.startswith("v1/") or path.startswith("v2/") or path.startswith("dashboard") or path.startswith("api/") or path == "health":
            return await call_next(request)

        provider_name = path.split("/")[0]
        prov_config = providers_config.get(provider_name, {})
        requires_key = prov_config.get("require_api_key", False)

        if requires_key:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
            else:
                token = request.headers.get("X-API-Key", "") or request.query_params.get("api_key", "")
            if not token or not key_manager.verify_key(token, provider=provider_name):
                raise HTTPException(status_code=401, detail=f"Invalid API key for {provider_name}")

        return await call_next(request)

    # ── Health check ──
    @app.get("/health")
    async def health():
        return {"status": "ok", "version": __version__}

    # ── LLM Router ──
    app.include_router(create_llm_router(client, analytics=analytics, config=config))

    # ── Search Providers ──
    for provider_cls in ALL_PROVIDERS:
        prov_name = provider_cls.name
        prov_cfg = providers_config.get(prov_name, {})
        if prov_cfg.get("enabled", True):
            provider = provider_cls(client, analytics=analytics)
            provider.register_routes(app)
            print(f"   [OK] {prov_name}")
        else:
            print(f"   [DISABLED] {prov_name}")

    # ── Firecrawl SDK v2 compatibility routes ──
    # The official Firecrawl Python SDK calls /v2/scrape, /v2/search etc.
    # Guanaco exposes these under /firecrawl/v2/... but the SDK sends to /v2/...
    # These top-level aliases let the SDK work without the /firecrawl prefix.
    try:
        firecrawl_prov = next(p for p in ALL_PROVIDERS if p.name == "firecrawl")
        fc_instance = firecrawl_prov(client, analytics=analytics)
        fc_compat = APIRouter(tags=["Firecrawl SDK Compat"])

        # Re-use the same handler logic by delegating to the provider's methods
        @fc_compat.post("/v2/scrape")
        async def fc_v2_scrape(request: Request):
            """Proxy /v2/scrape to the Firecrawl provider."""
            from guanaco.search.providers.firecrawl import ScrapeRequest
            body = await request.json()
            body_obj = ScrapeRequest(**body)
            # Access the provider's registered router handlers
            # Simpler: just call the ollama fetch directly
            ollama_resp = await client.fetch(url=body_obj.url)
            title = ollama_resp.get("title", "")
            content = ollama_resp.get("content", "")
            links = ollama_resp.get("links", [])
            # v2 SDK expects data to be a Document-like object with metadata nested inside
            data = {}
            if "markdown" in body_obj.formats or not body_obj.formats:
                data["markdown"] = content
            if "html" in body_obj.formats:
                data["html"] = content
            if "rawHtml" in body_obj.formats:
                data["rawHtml"] = content
            if "links" in body_obj.formats:
                data["links"] = links
            data["metadata"] = {
                "title": title,
                "sourceURL": body_obj.url,
                "statusCode": 200,
            }
            return {
                "success": True,
                "data": data,
            }

        @fc_compat.post("/v2/search")
        async def fc_v2_search(request: Request):
            """Proxy /v2/search to the Firecrawl provider."""
            from guanaco.search.providers.firecrawl import SearchRequest
            body = await request.json()
            body_obj = SearchRequest(**body)
            ollama_resp = await client.search(query=body_obj.query, max_results=body_obj.limit)
            results = []
            for r in ollama_resp.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "description": r.get("content", "")[:200],
                })
            return {"success": True, "data": {"web": results}}

        @fc_compat.post("/v2/crawl")
        async def fc_v2_crawl(request: Request):
            """Proxy /v2/crawl to the Firecrawl provider."""
            from guanaco.search.providers.firecrawl import CrawlRequest
            body = await request.json()
            body_obj = CrawlRequest(**body)
            ollama_resp = await client.fetch(url=body_obj.url)
            title = ollama_resp.get("title", "")
            content = ollama_resp.get("content", "")
            links = ollama_resp.get("links", [])
            results = [{
                "title": title,
                "url": body_obj.url,
                "content": content,
                "markdown": content,
                "metadata": {"title": title, "sourceURL": body_obj.url},
            }]
            for link in links[:body_obj.limit - 1]:
                try:
                    link_resp = await client.fetch(url=link)
                    lt = link_resp.get("title", "")
                    lc = link_resp.get("content", "")
                    results.append({
                        "title": lt,
                        "url": link,
                        "content": lc,
                        "markdown": lc,
                        "metadata": {"title": lt, "sourceURL": link},
                    })
                except Exception:
                    continue
            return {
                "success": True,
                "status": "completed",
                "completed": len(results),
                "total": len(results),
                "data": results,
            }

        @fc_compat.post("/v2/extract")
        async def fc_v2_extract(request: Request):
            """Proxy /v2/extract to the Firecrawl provider."""
            from guanaco.search.providers.firecrawl import ExtractRequest
            body = await request.json()
            body_obj = ExtractRequest(**body)
            all_content = {}
            for url in body_obj.urls[:5]:
                try:
                    resp = await client.fetch(url=url)
                    all_content[url] = resp.get("content", "")
                except Exception:
                    all_content[url] = ""
            return {"success": True, "data": all_content}

        app.include_router(fc_compat)
    except Exception as e:
        print(f"   [WARN] Firecrawl SDK compat routes not loaded: {e}")

    # ── Dashboard ──
    app.include_router(create_dashboard_router(key_manager, analytics, client), prefix="/dashboard")

    # ── Ollama status & models (top-level API) ──

    @app.get("/api/ollama/status")
    async def ollama_status():
        """Check Ollama Cloud API connectivity and list available models."""
        start = time.time()
        cfg = get_config()
        try:
            health = await client.health_check()
            latency_ms = health.get("latency_ms", round((time.time() - start) * 1000))

            if health["status"] == "connected":
                try:
                    models = await client.list_models()
                    model_count = len(models)
                except Exception:
                    model_count = health.get("model_count", 0)

                analytics.log_status("info", "ollama", "Health check OK", {"latency_ms": latency_ms})
                return {
                    "status": "connected",
                    "model_count": model_count,
                    "latency_ms": latency_ms,
                    "details": health,
                }
            else:
                latency_ms = round((time.time() - start) * 1000)
                analytics.log_status("error" if health["status"] in ("error", "auth_error") else "warning",
                                     "ollama", f"Health check failed: {health.get('message', health['status'])}",
                                     health)
                return {
                    "status": health["status"],
                    "error": health.get("message", str(health["status"])),
                    "model_count": 0,
                    "latency_ms": latency_ms,
                }
        except Exception as e:
            latency_ms = round((time.time() - start) * 1000)
            analytics.log_status("error", "ollama", f"Connection error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "model_count": 0,
                "latency_ms": latency_ms,
            }

    @app.get("/api/ollama/models")
    async def ollama_models():
        """List all available Ollama Cloud models with metadata."""
        try:
            models = await client.get_cloud_models()
            return {"models": models, "count": len(models)}
        except Exception as e:
            analytics.log_status("error", "ollama", f"Failed to list models: {str(e)}")
            raise HTTPException(status_code=502, detail=f"Cannot reach Ollama Cloud: {str(e)}")

    @app.get("/v1/usage")
    async def get_usage():
        """Get Ollama Cloud account usage/quota information."""
        try:
            usage_data = await client.get_usage()
            if usage_data.get("source") != "unavailable":
                session_pct = None
                weekly_pct = None
                plan = usage_data.get("plan", "")
                if isinstance(usage_data.get("session_usage"), dict):
                    session_pct = usage_data["session_usage"].get("used_percentage")
                elif usage_data.get("session_pct") is not None:
                    session_pct = usage_data["session_pct"]
                if isinstance(usage_data.get("weekly_usage"), dict):
                    weekly_pct = usage_data["weekly_usage"].get("used_percentage")
                elif usage_data.get("weekly_pct") is not None:
                    weekly_pct = usage_data["weekly_pct"]
                analytics.log_usage_snapshot(
                    session_pct=session_pct, weekly_pct=weekly_pct,
                    plan=plan, source=usage_data.get("source", "api"),
                )
            return usage_data
        except Exception as e:
            analytics.log_status("error", "ollama", f"Usage check failed: {str(e)}")
            return {"source": "error", "error": str(e)}

    # ── Status event endpoints ──

    @app.post("/api/status/log")
    async def log_status(request: Request):
        """Log a status event."""
        body = await request.json()
        entry_id = analytics.log_status(
            level=body.get("level", "info"),
            source=body.get("source", "api"),
            message=body.get("message", ""),
            details=body.get("details"),
        )
        return {"id": entry_id, "status": "logged"}

    @app.get("/api/status/events")
    async def get_status_events(limit: int = 50, level: Optional[str] = None, source: Optional[str] = None):
        """Get recent status events."""
        return analytics.get_status_events(limit=limit, level=level, source=source)

    return app


def main():
    """CLI entry point."""
    import uvicorn
    config = load_config()
    app = create_app(config)
    uvicorn.run(app, host=config.router.host, port=config.router.port)


if __name__ == "__main__":
    main()