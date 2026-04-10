"""Web dashboard for Guanaco management."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
import httpx

from guanaco.config import get_config, get_base_url, get_tailscale_ip, save_config, load_config
from guanaco.utils.api_keys import ApiKeyManager
from guanaco.analytics import AnalyticsLogger
from guanaco.client import OllamaClient


TEMPLATES_DIR = Path(__file__).parent / "templates"
LOGO_PATH = TEMPLATES_DIR / "logo.png"

def get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _generate_systemd_service() -> str:
    """Generate systemd unit file content for Guanaco."""
    import shutil
    import sys

    venv_python = shutil.which("python") or sys.executable
    working_dir = str(Path(__file__).resolve().parent.parent.parent)
    config_dir = str(Path.home() / ".guanaco")

    return f"""[Unit]
Description=Guanaco - LLM Proxy & Dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=PATH={Path(venv_python).parent}:/usr/bin:/usr/local/bin
WorkingDirectory={working_dir}
ExecStart={venv_python} -m uvicorn guanaco.app:create_app --factory --host 0.0.0.0 --port 8080
Restart=on-failure
RestartSec=5
Environment=OCT_CONFIG_DIR={config_dir}

[Install]
WantedBy=multi-user.target
"""


def create_dashboard_router(key_manager: ApiKeyManager, analytics: AnalyticsLogger, client=None) -> APIRouter:
    router = APIRouter(tags=["Dashboard"])

    @router.get("/logo.png")
    async def logo():
        return FileResponse(LOGO_PATH, media_type="image/png")

    @router.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        config = get_config()
        base_url = get_base_url(config)
        port = config.router.port

        html_path = TEMPLATES_DIR / "dashboard.html"
        html = html_path.read_text()

        # Inject config
        config_json = json.dumps({
            "base_url": base_url,
            "port": port,
            "router_port": port,
            "tailscale": config.router.use_tailscale,
            "tailscale_ip": get_tailscale_ip(),
            "local_ip": get_local_ip(),
            "tailscale_installed": get_tailscale_ip() is not None,
            "llm": config.llm.model_dump(),
            "available_models": config.llm.available_models,
        })
        html = html.replace("__CONFIG__", config_json)
        html = html.replace("__USAGE__", json.dumps(analytics.get_summary()))
        html = html.replace("__KEYS__", json.dumps(key_manager.list_keys()))
        html = html.replace("__FALLBACK__", json.dumps(config.fallback.model_dump()))
        providers_data = config.providers.model_dump()
        # Include endpoint metadata from provider classes
        from guanaco.search.providers import ALL_PROVIDERS
        provider_endpoints = {cls.name: [dict(ep) for ep in cls.endpoints] for cls in ALL_PROVIDERS}
        
        # Merge: config data + all known providers (in case config is missing some)
        all_provider_names = set(providers_data.keys()) | set(provider_endpoints.keys())
        html = html.replace("__PROVIDERS__", json.dumps({
            k: {
                "enabled": providers_data.get(k, {}).get("enabled", True),
                "require_api_key": providers_data.get(k, {}).get("require_api_key", False),
                "endpoints": provider_endpoints.get(k, []),
                "prefix": next((cls.prefix for cls in ALL_PROVIDERS if cls.name == k), f"/{k}"),
            }
            for k in all_provider_names
        }))

        return HTMLResponse(content=html)

    # ── API Keys ──

    @router.get("/api/keys")
    async def list_keys(request: Request):
        return key_manager.list_keys()

    @router.post("/api/keys/generate")
    async def generate_key(request: Request):
        body = await request.json()
        provider = body.get("provider", "general")
        name = body.get("name", "")
        key = key_manager.generate_key(provider=provider, name=name)
        return {"key": key, "provider": provider}

    @router.post("/api/keys/revoke")
    async def revoke_key(request: Request):
        body = await request.json()
        prefix = body.get("prefix", "")
        success = key_manager.revoke_by_prefix(prefix)
        return {"success": success}

    # ── Analytics ──

    @router.get("/api/analytics/summary")
    async def analytics_summary(request: Request):
        return analytics.get_summary()

    @router.get("/api/analytics/logs")
    async def analytics_logs(
        request: Request,
        limit: int = 100,
        offset: int = 0,
        type: Optional[str] = None,
        model: Optional[str] = None,
    ):
        return analytics.get_logs(limit=limit, offset=offset, type_filter=type, model_filter=model)

    @router.get("/api/analytics/timeseries")
    async def analytics_timeseries(request: Request, hours: int = 24):
        return analytics.get_timeseries(hours=hours)

    @router.post("/api/analytics/clear")
    async def analytics_clear(request: Request):
        analytics.clear()
        return {"status": "ok"}

    # ── Status Events ──

    @router.get("/api/status/events")
    async def status_events(
        request: Request,
        limit: int = 50,
        level: Optional[str] = None,
        source: Optional[str] = None,
    ):
        return analytics.get_status_events(limit=limit, level=level, source=source)

    @router.post("/api/status/log")
    async def log_status_event(request: Request):
        """Log a status event from the dashboard or external source."""
        body = await request.json()
        level = body.get("level", "info")
        source = body.get("source", "dashboard")
        message = body.get("message", "")
        details = body.get("details")
        entry_id = analytics.log_status(level=level, source=source, message=message, details=details)
        return {"id": entry_id, "status": "logged"}

    # ── Config Management ──

    @router.post("/api/fallback/test")
    async def test_fallback_connection(request: Request):
        """Test the fallback provider connection by sending a minimal chat request."""
        config = get_config()
        fb = config.fallback

        if not fb.enabled:
            return {"ok": False, "error": "Fallback is not enabled"}
        if not fb.base_url:
            return {"ok": False, "error": "Base URL is not configured"}
        if not fb.default_model:
            return {"ok": False, "error": "Default model is not configured"}

        # Normalize base_url — strip /chat/completions if user pasted the full path
        base_url = fb.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")]
        url = f"{base_url}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if fb.api_key:
            headers["Authorization"] = f"Bearer {fb.api_key}"

        payload = {
            "model": fb.default_model,
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 10,
            "stream": False,
        }

        timeout = fb.timeout or 30.0

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                start = time.time()
                resp = await client.post(url, json=payload, headers=headers)
                elapsed = round((time.time() - start) * 1000)

                if resp.status_code == 200:
                    data = resp.json()
                    model_used = ""
                    content_preview = ""
                    if data.get("choices"):
                        msg = data["choices"][0].get("message", {})
                        model_used = data.get("model", fb.default_model)
                        content_preview = (msg.get("content") or "")[:60]
                    return {
                        "ok": True,
                        "message": f"Connected ({elapsed}ms) — {model_used}: \"{content_preview}\"",
                    }
                else:
                    try:
                        err_body = resp.json()
                        err_msg = err_body.get("error", {})
                        if isinstance(err_msg, dict):
                            err_msg = err_msg.get("message", str(err_body))
                        elif not err_msg:
                            err_msg = str(err_body)
                    except Exception:
                        err_msg = resp.text[:200]
                    return {
                        "ok": False,
                        "error": f"HTTP {resp.status_code} ({elapsed}ms): {err_msg}",
                    }
        except httpx.ConnectError as e:
            return {"ok": False, "error": f"Connection failed: {str(e)}"}
        except httpx.TimeoutException:
            return {"ok": False, "error": f"Timeout after {timeout}s"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @router.post("/api/test-search")
    async def test_search(request: Request):
        """Test a search provider by forwarding the query to Ollama and formatting results."""
        from guanaco.search.providers import ALL_PROVIDERS

        body = await request.json()
        provider_name = body.get("provider", "")
        query = body.get("query", "")

        if not query:
            return {"error": "Query is required"}

        # Find the provider class
        provider_cls = next((cls for cls in ALL_PROVIDERS if cls.name == provider_name), None)
        if not provider_cls:
            return {"error": f"Unknown provider: {provider_name}"}

        config = get_config()
        ollama_client = OllamaClient(api_key=config.ollama_api_key or "")

        try:
            ollama_resp = await ollama_client.search(query=query, max_results=5)
        except Exception as e:
            return {"error": f"Ollama search failed: {str(e)}"}

        # Format results per provider's response model
        results = []
        for r in ollama_resp.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            })

        return {"query": query, "results": results}

    @router.post("/api/summarize")
    async def summarize_search(request: Request):
        """Search the web and summarize results using the configured summary model.

        BETA — combines Ollama web_search with LLM summarization.
        """
        body = await request.json()
        query = body.get("query", "")
        max_results = min(max(body.get("max_results", 5), 1), 10)

        if not query:
            return {"error": "Query is required"}

        config = get_config()
        ollama_client = OllamaClient(api_key=config.ollama_api_key or "")

        # Step 1: Search
        try:
            ollama_resp = await ollama_client.search(query=query, max_results=max_results)
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}

        results = []
        for r in ollama_resp.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            })

        if not results:
            return {"query": query, "results": [], "summary": "No results found.", "model": None}

        # Step 2: Summarize using the configured summary_model
        summary_model = getattr(config.llm, "summary_model", "") or ""
        summary = None
        model_used = summary_model

        if summary_model:
            try:
                # Build context from search results
                context_parts = []
                for i, r in enumerate(results, 1):
                    context_parts.append(f"[{i}] {r['title']}\n{r['content'][:500]}")
                context = "\n\n".join(context_parts)

                prompt = (
                    f"Summarize the following search results for the query: \"{query}\"\n\n"
                    f"Provide a concise, informative summary that directly answers the query. "
                    f"Include key facts and cite sources by number (e.g., [1], [2]).\n\n"
                    f"Search results:\n{context}"
                )

                payload = {
                    "model": summary_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4096,
                    "stream": False,
                }

                # Call through the LLM client directly
                llm_resp = await ollama_client.chat_completion(payload)
                choices = llm_resp.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    summary = msg.get("content", "") or msg.get("reasoning", "")
            except Exception as e:
                summary = f"Summarization failed: {str(e)}"

        return {
            "query": query,
            "results": results,
            "summary": summary,
            "model": model_used,
        }

    @router.get("/api/config")
    async def get_config_api(request: Request):
        """Get full config as JSON (llm settings + fallback settings)."""
        config = get_config()
        return {
            "llm": config.llm.model_dump(),
            "fallback": config.fallback.model_dump(),
            "search": config.search.model_dump(),
        }

    @router.post("/api/config")
    async def update_config_api(request: Request):
        """Update config (llm and/or fallback settings)."""
        body = await request.json()
        config = get_config()

        # Update LLM settings
        if "llm" in body:
            llm_updates = body["llm"]
            for key, value in llm_updates.items():
                if hasattr(config.llm, key):
                    setattr(config.llm, key, value)

        # Update search settings
        if "search" in body:
            s_updates = body["search"]
            s = config.search
            if "summarize_enabled" in s_updates:
                s.summarize_enabled = bool(s_updates["summarize_enabled"])
            if "summarize_all" in s_updates:
                s.summarize_all = bool(s_updates["summarize_all"])
            if "summary_model" in s_updates:
                s.summary_model = str(s_updates["summary_model"])

        # Update fallback settings
        if "fallback" in body:
            fb_updates = body["fallback"]
            fb = config.fallback
            if "enabled" in fb_updates:
                fb.enabled = fb_updates["enabled"]
            if "name" in fb_updates:
                fb.name = fb_updates["name"]
            if "base_url" in fb_updates:
                fb.base_url = fb_updates["base_url"]
            if "api_key" in fb_updates:
                fb.api_key = fb_updates["api_key"]
            if "model_map" in fb_updates:
                fb.model_map = fb_updates["model_map"]
            if "default_model" in fb_updates:
                fb.default_model = fb_updates["default_model"]
            if "timeout" in fb_updates:
                fb.timeout = float(fb_updates["timeout"])
            if "primary_timeout" in fb_updates:
                fb.primary_timeout = float(fb_updates["primary_timeout"])
            if "stream_chunk_timeout" in fb_updates:
                fb.stream_chunk_timeout = float(fb_updates["stream_chunk_timeout"])
            if "max_tokens" in fb_updates:
                fb.max_tokens = int(fb_updates["max_tokens"])
            if "stream_fallback" in fb_updates:
                fb.stream_fallback = fb_updates["stream_fallback"]
            if "supports_vision" in fb_updates:
                fb.supports_vision = fb_updates["supports_vision"]

        save_config(config)
        return {"status": "ok", "config": {"llm": config.llm.model_dump(), "fallback": config.fallback.model_dump(), "search": config.search.model_dump()}}

    # ── Emulation Config ──

    @router.post("/api/config/emulation")
    async def save_emulation_config(request: Request):
        """Save emulation toggle config (OpenAI/Anthropic endpoint modes)."""
        body = await request.json()
        config = get_config()
        if "emulate_openai" in body:
            config.llm.emulate_openai = bool(body["emulate_openai"])
        if "emulate_anthropic" in body:
            config.llm.emulate_anthropic = bool(body["emulate_anthropic"])
        save_config(config)
        return {"status": "ok", "emulate_openai": config.llm.emulate_openai, "emulate_anthropic": config.llm.emulate_anthropic}

    @router.get("/api/config/emulation")
    async def get_emulation_config(request: Request):
        """Get current emulation config."""
        config = get_config()
        return {"emulate_openai": config.llm.emulate_openai, "emulate_anthropic": config.llm.emulate_anthropic}

    # ── Model History ──

    @router.get("/api/analytics/model/{model_name}")
    async def model_history(request: Request, model_name: str, limit: int = 50):
        """Get detailed history for a specific model."""
        return analytics.get_model_history(model_name, limit=limit)

    # ── Autostart / Systemd ──

    @router.get("/api/autostart")
    async def get_autostart(request: Request):
        """Check if Guanaco is currently set to autostart via systemd."""
        import subprocess
        service_name = "guanaco.service"
        try:
            result = subprocess.run(
                ["systemctl", "is-enabled", service_name],
                capture_output=True, text=True, timeout=5
            )
            enabled = result.stdout.strip() == "enabled"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            enabled = False

        # Check if service exists
        try:
            result = subprocess.run(
                ["systemctl", "status", service_name],
                capture_output=True, text=True, timeout=5
            )
            installed = result.returncode != 4  # code 4 = unit not found
        except (FileNotFoundError, subprocess.TimeoutExpired):
            installed = False

        # Get runtime status
        try:
            result = subprocess.run(
                ["systemctl", "is-active", service_name],
                capture_output=True, text=True, timeout=5
            )
            active = result.stdout.strip() == "active"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            active = False

        config = get_config()
        return {
            "enabled": enabled or config.router.autostart,
            "installed": installed,
            "active": active,
        }

    @router.post("/api/autostart/enable")
    async def enable_autostart(request: Request):
        """Install and enable Guanaco systemd service for autostart."""
        import subprocess
        from pathlib import Path

        service_content = _generate_systemd_service()
        service_path = Path("/etc/systemd/system/guanaco.service")

        try:
            service_path.write_text(service_content)
        except PermissionError:
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Need sudo to write systemd service file. Run: sudo guanaco autostart enable")

        # Reload and enable
        subprocess.run(["systemctl", "daemon-reload"], check=True, capture_output=True, timeout=10)
        subprocess.run(["systemctl", "enable", "guanaco.service"], check=True, capture_output=True, timeout=10)

        # Start it now if not already running
        subprocess.run(["systemctl", "start", "guanaco.service"], capture_output=True, timeout=10)

        config = get_config()
        config.router.autostart = True
        save_config(config)

        return {"status": "ok", "enabled": True, "message": "Autostart enabled. Guanaco will start on boot."}

    @router.post("/api/autostart/disable")
    async def disable_autostart(request: Request):
        """Disable and remove Guanaco systemd service."""
        import subprocess

        try:
            subprocess.run(["systemctl", "stop", "guanaco.service"], capture_output=True, timeout=10)
            subprocess.run(["systemctl", "disable", "guanaco.service"], capture_output=True, timeout=10)
        except Exception:
            pass

        config = get_config()
        config.router.autostart = False
        save_config(config)

        return {"status": "ok", "enabled": False, "message": "Autostart disabled."}

    # ── Model Sync ──

    @router.post("/api/models/sync")
    async def sync_models_api(request: Request):
        """Trigger model sync from Ollama Cloud into config."""
        from guanaco.client import OllamaClient
        from guanaco.config import get_config as _get_config
        _cfg = _get_config()
        client = OllamaClient(api_key=_cfg.ollama_api_key or "")
        try:
            models = await client.list_models(force_refresh=True)
            config = get_config()
            model_names = []
            for m in models:
                name = m.get("name", m.get("model", ""))
                name = name.replace("-cloud", "") if name.endswith("-cloud") else name
                if name and name not in model_names:
                    model_names.append(name)

            existing = set(config.llm.available_models)
            for mn in model_names:
                existing.add(mn)

            config.llm.available_models = sorted(existing)
            save_config(config)

            return {"status": "ok", "synced": len(model_names), "total": len(config.llm.available_models), "models": config.llm.available_models}
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=502, detail=f"Cannot sync models: {str(e)}")

    # ── Usage / Session Cookie ──

    @router.get("/api/usage/config")
    async def get_usage_config(request: Request):
        config = get_config()
        uc = config.usage
        return {
            "session_cookie_set": bool(uc.session_cookie),
            "session_cookie_preview": uc.session_cookie[:8] + "..." if uc.session_cookie else "",
            "check_interval": uc.check_interval,
            "redirect_on_full": uc.redirect_on_full,
            "last_session_pct": uc.last_session_pct,
            "last_weekly_pct": uc.last_weekly_pct,
            "last_plan": uc.last_plan,
            "last_session_reset": uc.last_session_reset,
            "last_weekly_reset": uc.last_weekly_reset,
            "last_checked": uc.last_checked,
        }

    @router.post("/api/usage/session-cookie")
    async def set_session_cookie(request: Request):
        body = await request.json()
        config = get_config()
        # Update session cookie if provided
        if "session_cookie" in body:
            cookie = body.get("session_cookie", "").strip()
            config.usage.session_cookie = cookie
            if client:
                client._session_cookie = cookie
        # Update check interval if provided
        if "check_interval" in body:
            config.usage.check_interval = int(body["check_interval"])
        # Update redirect_on_full if provided
        if "redirect_on_full" in body:
            config.usage.redirect_on_full = bool(body["redirect_on_full"])
        save_config(config)
        return {
            "status": "ok",
            "cookie_set": bool(config.usage.session_cookie),
            "preview": config.usage.session_cookie[:8] + "..." if config.usage.session_cookie else "",
            "check_interval": config.usage.check_interval,
            "redirect_on_full": config.usage.redirect_on_full,
        }

    @router.post("/api/usage/check")
    async def check_usage_now(request: Request):
        config = get_config()
        cookie = config.usage.session_cookie
        if not cookie:
            return {"source": "unavailable", "error": "No session cookie configured. Paste your __Secure-session cookie in the Status tab."}
        try:
            usage_data = await client.get_usage(session_cookie=cookie)
            if usage_data.get("source") != "unavailable":
                config.usage.last_session_pct = usage_data.get("session_pct")
                config.usage.last_weekly_pct = usage_data.get("weekly_pct")
                config.usage.last_plan = usage_data.get("plan")
                config.usage.last_session_reset = usage_data.get("session_reset")
                config.usage.last_weekly_reset = usage_data.get("weekly_reset")
                config.usage.last_checked = time.time()
                save_config(config)
            return usage_data
        except Exception as e:
            return {"source": "error", "error": str(e)}

    # ── Update ──

    @router.get("/api/update/check")
    async def check_for_update(request: Request):
        """Check GitHub for the latest release and compare with current version."""
        from guanaco.app import __version__
        current_version = __version__

        result = {"current_version": current_version, "latest_version": None, "update_available": False, "error": None}

        try:
            # Get release info from GitHub API
            # Default: only check stable releases (/releases/latest)
            # If allow_prerelease is set in config, also check prereleases
            config = get_config()
            allow_prerelease = getattr(config.router, "allow_prerelease", False)
            import httpx
            async with httpx.AsyncClient(timeout=10) as hc:
                release_data = None
                # Always try stable release first
                resp = await hc.get(
                    "https://api.github.com/repos/evangit2/guanaco/releases/latest",
                    headers={"Accept": "application/vnd.github+json"}
                )
                if resp.status_code == 200:
                    release_data = resp.json()
                elif allow_prerelease:
                    # No stable release found — check all releases including prereleases
                    resp = await hc.get(
                        "https://api.github.com/repos/evangit2/guanaco/releases",
                        headers={"Accept": "application/vnd.github+json"}
                    )
                    if resp.status_code == 200 and resp.json():
                        release_data = resp.json()[0]  # GitHub sorts newest-first
                if release_data:
                    tag = release_data.get("tag_name", "")
                    # Strip leading 'v' if present
                    result["latest_version"] = tag.lstrip("v")
                    result["release_notes"] = release_data.get("body", "")[:500]
                    result["release_url"] = release_data.get("html_url", "")
                    result["prerelease"] = release_data.get("prerelease", False)
                else:
                    result["error"] = f"GitHub API returned {resp.status_code}"
        except Exception as e:
            result["error"] = str(e)

        if result["latest_version"]:
            # Compare versions (simple semver comparison)
            try:
                current_parts = [int(x) for x in current_version.split(".")]
                latest_parts = [int(x) for x in result["latest_version"].split(".")]
                # Pad to same length
                while len(current_parts) < len(latest_parts):
                    current_parts.append(0)
                while len(latest_parts) < len(current_parts):
                    latest_parts.append(0)
                result["update_available"] = latest_parts > current_parts
            except (ValueError, TypeError):
                # Fall back to string comparison
                result["update_available"] = result["latest_version"] != current_version

        # Include auto_update and allow_prerelease settings
        result["auto_update"] = config.router.auto_update
        result["allow_prerelease"] = getattr(config.router, "allow_prerelease", False)

        return result

    @router.post("/api/update/apply")
    async def apply_update(request: Request, background_tasks: BackgroundTasks):
        """Pull latest from git, reinstall, and restart the service.

        The restart happens in a BackgroundTask so the HTTP response is sent
        BEFORE the service kills itself — otherwise the client never sees the
        success message.
        """
        import subprocess
        from guanaco.app import __version__
        old_version = __version__

        project_dir = Path(__file__).resolve().parent.parent.parent

        try:
            # Step 1: Determine current branch and pull from it
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=project_dir, capture_output=True, text=True, timeout=10
            )
            current_branch = branch_result.stdout.strip() or "main"

            # Step 1b: Git fetch + pull
            fetch_result = subprocess.run(
                ["git", "fetch", "origin", current_branch],
                cwd=project_dir, capture_output=True, text=True, timeout=30
            )
            if fetch_result.returncode != 0:
                return {"status": "error", "step": "fetch", "message": fetch_result.stderr[:200]}

            pull_result = subprocess.run(
                ["git", "pull", "origin", current_branch],
                cwd=project_dir, capture_output=True, text=True, timeout=30
            )
            if pull_result.returncode != 0:
                return {"status": "error", "step": "pull", "message": pull_result.stderr[:200]}

            # Step 2: Reinstall
            # Check common venv locations: ~/.guanaco/venv (install.sh default), then repo-local
            install_dir = Path.home() / ".guanaco"
            venv_python = install_dir / "venv" / "bin" / "python"
            if not venv_python.exists():
                venv_python = project_dir / "venv" / "bin" / "python"
            install_result = subprocess.run(
                [str(venv_python), "-m", "pip", "install", "-e", ".", "--quiet"],
                cwd=project_dir, capture_output=True, text=True, timeout=60
            )
            if install_result.returncode != 0:
                return {"status": "error", "step": "install", "message": install_result.stderr[:200]}

            # Step 3: Validate the update can actually start before restarting
            validate_result = subprocess.run(
                [str(venv_python), "-c",
                 "from guanaco.app import create_app; app = create_app(); "
                 "from guanaco import __version__; print(__version__)"],
                cwd=project_dir, capture_output=True, text=True, timeout=15
            )
            if validate_result.returncode != 0:
                return {
                    "status": "error",
                    "step": "validate",
                    "message": f"Update installed but app failed to start: {validate_result.stderr[:200]}"
                }
            new_version = validate_result.stdout.strip()

            # Step 4: Schedule restart as BackgroundTask so the response is sent first
            async def _restart_service():
                import asyncio
                await asyncio.sleep(1)  # give the HTTP response time to be sent
                # Stop then start (more reliable than restart if process is stuck)
                subprocess.run(["sudo", "systemctl", "stop", "guanaco.service"],
                              capture_output=True, timeout=15)
                await asyncio.sleep(1)  # let the process fully exit
                subprocess.run(["sudo", "systemctl", "start", "guanaco.service"],
                              capture_output=True, timeout=15)

            background_tasks.add_task(_restart_service)

            return {
                "status": "ok",
                "old_version": old_version,
                "new_version": new_version,
                "message": f"Updated from {old_version} to {new_version}. Service restarting."
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "step": "timeout", "message": "Operation timed out"}
        except Exception as e:
            return {"status": "error", "step": "unknown", "message": str(e)[:200]}

    @router.post("/api/update/auto-toggle")
    async def toggle_auto_update(request: Request):
        """Enable or disable automatic updates."""
        body = await request.json()
        config = get_config()
        config.router.auto_update = bool(body.get("enabled", False))
        save_config(config)
        return {"status": "ok", "auto_update": config.router.auto_update}

    return router