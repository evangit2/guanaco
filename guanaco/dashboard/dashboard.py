"""Web dashboard for Guanaco management."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
import httpx

from guanaco.config import get_config, get_base_url, get_tailscale_ip, save_config, load_config
from guanaco.utils.api_keys import ApiKeyManager
from guanaco.analytics import AnalyticsLogger
from guanaco.client import OllamaClient


TEMPLATES_DIR = Path(__file__).parent / "templates"


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
            "llm": config.llm.model_dump(),
            "available_models": config.llm.available_models,
        })
        html = html.replace("__CONFIG__", config_json)
        html = html.replace("__USAGE__", json.dumps(analytics.get_summary()))
        html = html.replace("__KEYS__", json.dumps(key_manager.list_keys()))
        html = html.replace("__FALLBACK__", json.dumps(config.fallback.model_dump()))
        providers_data = config.providers.model_dump()
        html = html.replace("__PROVIDERS__", json.dumps({
            k: {"enabled": v.get("enabled", True), "require_api_key": v.get("require_api_key", False)}
            for k, v in providers_data.items()
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

    @router.get("/api/config")
    async def get_config_api(request: Request):
        """Get full config as JSON (llm settings + fallback settings)."""
        config = get_config()
        return {
            "llm": config.llm.model_dump(),
            "fallback": config.fallback.model_dump(),
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

        save_config(config)
        return {"status": "ok", "config": {"llm": config.llm.model_dump(), "fallback": config.fallback.model_dump()}}

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

    return router