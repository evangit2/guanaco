"""Ollama Cloud API client — handles search, fetch, chat, models, and usage."""

from __future__ import annotations

import json
import time
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE = "https://ollama.com"
OLLAMA_V1_URL = f"{OLLAMA_BASE}/v1"
OLLAMA_CHAT_URL = f"{OLLAMA_V1_URL}/chat/completions"
OLLAMA_MODELS_URL = f"{OLLAMA_V1_URL}/models"
OLLAMA_SEARCH_URL = f"{OLLAMA_BASE}/api/web_search"
OLLAMA_FETCH_URL = f"{OLLAMA_BASE}/api/web_fetch"
OLLAMA_USAGE_URL = f"{OLLAMA_BASE}/api/account/usage"
OLLAMA_SETTINGS_URL = f"{OLLAMA_BASE}/api/account/settings"

# Known cloud models (fallback + display info)
# Names must match /v1/models response (e.g. "gemma4:31b", "qwen3.5:397b")
KNOWN_CLOUD_MODELS = {
    "gemma4": {"sizes": ["31b"], "family": "gemma", "capabilities": ["vision", "tools", "thinking", "cloud"]},
    "gemma3": {"sizes": ["4b", "12b", "27b"], "family": "gemma", "capabilities": ["vision", "tools", "thinking", "cloud"]},
    "qwen3.5": {"sizes": ["397b"], "family": "qwen", "capabilities": ["vision", "tools", "thinking", "cloud"]},
    "qwen3-vl": {"sizes": ["235b", "235b-instruct"], "family": "qwen", "capabilities": ["vision", "tools", "thinking", "cloud"]},
    "qwen3-coder": {"sizes": ["480b"], "family": "qwen", "capabilities": ["tools", "cloud"]},
    "qwen3-coder-next": {"sizes": [], "family": "qwen", "capabilities": ["tools", "cloud"]},
    "qwen3-next": {"sizes": ["80b"], "family": "qwen", "capabilities": ["tools", "thinking", "cloud"]},
    "minimax-m2": {"sizes": [], "family": "minimax", "capabilities": ["tools", "thinking", "cloud"]},
    "minimax-m2.7": {"sizes": [], "family": "minimax", "capabilities": ["tools", "thinking", "cloud"]},
    "minimax-m2.5": {"sizes": [], "family": "minimax", "capabilities": ["tools", "thinking", "cloud"]},
    "minimax-m2.1": {"sizes": [], "family": "minimax", "capabilities": ["tools", "thinking", "cloud"]},
    "glm-5.1": {"sizes": [], "family": "glm", "capabilities": ["tools", "thinking", "cloud"]},
    "glm-5": {"sizes": [], "family": "glm", "capabilities": ["tools", "thinking", "cloud"]},
    "glm-4.7": {"sizes": [], "family": "glm", "capabilities": ["tools", "thinking", "cloud"]},
    "glm-4.6": {"sizes": [], "family": "glm", "capabilities": ["tools", "thinking", "cloud"]},
    "gpt-oss": {"sizes": ["20b", "120b"], "family": "gpt-oss", "capabilities": ["tools", "thinking", "cloud"]},
    "deepseek-v3.1": {"sizes": ["671b"], "family": "deepseek", "capabilities": ["thinking", "cloud"]},
    "deepseek-v3.2": {"sizes": [], "family": "deepseek", "capabilities": ["thinking", "cloud"]},
    "devstral-small-2": {"sizes": ["24b"], "family": "devstral", "capabilities": ["tools", "cloud"]},
    "devstral-2": {"sizes": ["123b"], "family": "devstral", "capabilities": ["tools", "cloud"]},
    "nemotron-3-super": {"sizes": [], "family": "nemotron", "capabilities": ["tools", "thinking", "cloud"]},
    "nemotron-3-nano": {"sizes": ["30b"], "family": "nemotron", "capabilities": ["tools", "thinking", "cloud"]},
    "mistral-large-3": {"sizes": ["675b"], "family": "mistral", "capabilities": ["tools", "thinking", "cloud"]},
    "ministral-3": {"sizes": ["3b", "8b", "14b"], "family": "mistral", "capabilities": ["tools", "cloud"]},
    "kimi-k2.5": {"sizes": [], "family": "kimi", "capabilities": ["vision", "tools", "thinking", "cloud"]},
    "kimi-k2-thinking": {"sizes": [], "family": "kimi", "capabilities": ["thinking", "cloud"]},
    "kimi-k2": {"sizes": ["1t"], "family": "kimi", "capabilities": ["tools", "thinking", "cloud"]},
    "cogito-2.1": {"sizes": ["671b"], "family": "cogito", "capabilities": ["thinking", "cloud"]},
    "gemini-3-flash-preview": {"sizes": [], "family": "gemini", "capabilities": ["vision", "tools", "thinking", "cloud"]},
    "rnj-1": {"sizes": ["8b"], "family": "rnj", "capabilities": ["tools", "cloud"]},
}


class OllamaClient:
    """Async client for Ollama Cloud API."""

    def __init__(self, api_key: str, timeout: float = 120.0, session_cookie: str = ""):
        self.api_key = api_key
        self.timeout = timeout
        self._session_cookie = session_cookie
        self._client: Optional[httpx.AsyncClient] = None
        self._models_cache: Optional[list[dict]] = None
        self._models_cache_time: float = 0
        self._models_cache_ttl: float = 300.0  # 5 minutes

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {"Content-Type": "application/json"}
            # Only send Authorization if we have a real API key (not empty, placeholder, or masked)
            if self.api_key and self.api_key not in ("***", "REPLACE_ME", "your_api_key_here"):
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=headers,
            )
        return self._client

    # ── Search & Fetch ──

    async def search(self, query: str, max_results: int = 10) -> dict:
        """Search the web using Ollama's web_search API."""
        client = await self._get_client()
        payload = {"query": query, "max_results": max(min(max_results, 10), 1)}
        resp = await client.post(OLLAMA_SEARCH_URL, json=payload)
        resp.raise_for_status()
        return resp.json()

    async def fetch(self, url: str) -> dict:
        """Fetch/scrape a URL using Ollama's web_fetch API."""
        client = await self._get_client()
        payload = {"url": url}
        resp = await client.post(OLLAMA_FETCH_URL, json=payload)
        resp.raise_for_status()
        return resp.json()

    # ── Models ──

    async def list_models(self, force_refresh: bool = False) -> list[dict]:
        """List available Ollama Cloud models, with caching.

        Uses the OpenAI-compatible /v1/models endpoint which returns
        model IDs in standard format (e.g. 'gemma4:31b', 'qwen3.5:397b').
        """
        now = time.time()
        if not force_refresh and self._models_cache and (now - self._models_cache_time) < self._models_cache_ttl:
            return self._models_cache

        client = await self._get_client()
        try:
            resp = await client.get(OLLAMA_MODELS_URL)
            if resp.status_code == 401:
                logger.error("Ollama API key is invalid or expired")
                raise httpx.HTTPStatusError("Invalid API key", request=resp.request, response=resp)
            resp.raise_for_status()
            data = resp.json()
            # OpenAI format: {"data": [{"id": "gemma4:31b", "object": "model", ...}]}
            raw_models = data.get("data", data.get("models", []))
            models = []
            for m in raw_models:
                if isinstance(m, dict):
                    model_id = m.get("id", m.get("name", m.get("model", "")))
                    models.append({
                        "name": model_id,
                        "model": model_id,
                        "id": model_id,
                        "modified_at": m.get("created", m.get("modified_at", "")),
                        "size": m.get("size", 0),
                        "digest": m.get("digest", ""),
                    })
                elif isinstance(m, str):
                    models.append({"name": m, "model": m, "id": m})
            self._models_cache = models
            self._models_cache_time = now
            return models
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to fetch models: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            raise

    async def check_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available on Ollama Cloud."""
        models = await self.list_models()
        available_names = {m.get("name", m.get("model", "")) for m in models}
        # Check with and without -cloud suffix
        return model_name in available_names or f"{model_name}-cloud" in available_names

    async def get_cloud_models(self) -> list[dict]:
        """Get list of cloud-capable models with metadata."""
        models = await self.list_models()
        cloud_models = []
        for m in models:
            name = m.get("name", m.get("model", ""))
            details = m.get("details", {})
            # Check if model has cloud capability (or is available via cloud API)
            is_cloud = True  # All models from /api/tags with auth are cloud-available
            size_info = details.get("parameter_size", "")
            family = details.get("family", "")
            quant = details.get("quantization_level", "")

            cloud_models.append({
                "name": name,
                "display_name": name.replace("-cloud", ""),
                "size_bytes": m.get("size", 0),
                "parameter_size": size_info,
                "family": family,
                "quantization": quant,
                "capabilities": self._get_model_capabilities(name),
                "modified_at": m.get("modified_at", ""),
                "digest": m.get("digest", "")[:12] if m.get("digest") else "",
            })
        return cloud_models

    def _get_model_capabilities(self, model_name: str) -> list[str]:
        """Get known capabilities for a model name."""
        base_name = model_name.split(":")[0].replace("-cloud", "")
        if base_name in KNOWN_CLOUD_MODELS:
            return KNOWN_CLOUD_MODELS[base_name].get("capabilities", ["cloud"])
        # Default capabilities for unknown models
        return ["cloud"]

    # ── Usage / Quota ──

    async def get_usage(self, session_cookie: str = "") -> dict:
        """Get account usage/quota information from Ollama Cloud.

        Uses the session cookie to scrape usage from /settings HTML page.
        Ollama doesn't have a public usage API, so we parse the rendered page.
        """
        cookie = session_cookie or self._session_cookie
        if not cookie:
            return {"source": "unavailable", "error": "No session cookie configured. Paste your __Secure-session cookie in the Status tab to enable usage tracking."}

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    "https://ollama.com/settings",
                    follow_redirects=True,
                    cookies={"__Secure-session": cookie},
                    headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
                )
                if resp.status_code == 200:
                    usage = self._parse_settings_html(resp.text)
                    if usage:
                        return {"source": "settings_html", **usage}
                    return {"source": "settings_html", "error": "Could not parse usage data from settings page. Cookie may be expired."}
                elif resp.status_code == 401 or resp.status_code == 302:
                    return {"source": "unavailable", "error": "Session cookie is expired or invalid. Please update it in the Status tab."}
                else:
                    return {"source": "unavailable", "error": f"Unexpected status {resp.status_code} from ollama.com/settings"}
        except Exception as e:
            logger.warning(f"Failed to check usage with session cookie: {e}")
            return {"source": "unavailable", "error": f"Failed to fetch usage: {str(e)}"}

    def _parse_settings_html(self, html: str) -> Optional[dict]:
        """Extract usage data from the Ollama settings page HTML.

        The page is server-rendered with patterns like:
          <span class="text-sm">Session usage</span>
          <span class="text-sm">4.6% used</span>
          ... Resets in 4 hours
          <span class="text-sm">Weekly usage</span>
          <span class="text-sm">30.9% used</span>
          ... Resets in 3 days
        """
        import re
        result = {}

        # Extract percentages: "N.N% used" near "Session" and "Weekly" contexts
        # Find all "X.X% used" occurrences in order
        pct_matches = re.findall(r'(\d+(?:\.\d+)?)%\s*used', html)
        reset_matches = re.findall(r'Resets in ([^<\n]+)', html)

        # Find session/weekly labels to determine which percentage is which
        session_idx = None
        weekly_idx = None

        # Look for "Session usage" label and find the nearest percentage
        session_label = re.search(r'Session usage.*?(\d+(?:\.\d+)?)%\s*used', html, re.DOTALL)
        if session_label:
            result["session_pct"] = float(session_label.group(1))
        elif len(pct_matches) >= 1:
            result["session_pct"] = float(pct_matches[0])

        weekly_label = re.search(r'Weekly usage.*?(\d+(?:\.\d+)?)%\s*used', html, re.DOTALL)
        if weekly_label:
            result["weekly_pct"] = float(weekly_label.group(1))
        elif len(pct_matches) >= 2:
            result["weekly_pct"] = float(pct_matches[1])

        # Reset timers
        if reset_matches:
            if len(reset_matches) >= 1:
                result["session_reset"] = reset_matches[0].strip()
            if len(reset_matches) >= 2:
                result["weekly_reset"] = reset_matches[1].strip()

        # Plan detection — find the badge right after "Cloud Usage"
        # Pattern: <span ...>Cloud Usage</span> ... <span ...>pro</span>
        plan_match = re.search(r'Cloud Usage\s*</span>\s*<span[^>]*>\s*(pro|max|free|team|starter)\s*</span', html, re.IGNORECASE)
        if not plan_match:
            # Fallback: look for a lowercase plan badge in a capitalize span
            plan_match = re.search(r'class=\"[^"]*capitalize[^"]*\">\s*(pro|max|free|team|starter)\s*</span', html, re.IGNORECASE)
        if plan_match:
            result["plan"] = plan_match.group(1).strip().lower()

        return result if result else None

    # ── Health Check ──

    async def health_check(self) -> dict:
        """Check Ollama Cloud API connectivity and key validity."""
        client = await self._get_client()
        start = time.time()
        try:
            resp = await client.get(OLLAMA_MODELS_URL)
            elapsed = time.time() - start
            if resp.status_code == 401:
                return {
                    "status": "auth_error",
                    "message": "Invalid or expired API key",
                    "latency_ms": round(elapsed * 1000),
                }
            if resp.status_code == 429:
                return {
                    "status": "rate_limited",
                    "message": "Rate limited by Ollama Cloud",
                    "latency_ms": round(elapsed * 1000),
                }
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", data.get("models", []))
            return {
                "status": "connected",
                "model_count": len(models),
                "latency_ms": round(elapsed * 1000),
            }
        except httpx.ConnectError:
            return {
                "status": "unreachable",
                "message": "Cannot connect to ollama.com",
                "latency_ms": round((time.time() - start) * 1000),
            }
        except httpx.TimeoutException:
            return {
                "status": "timeout",
                "message": "Connection to ollama.com timed out",
                "latency_ms": round((time.time() - start) * 1000),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "latency_ms": round((time.time() - start) * 1000),
            }

    # ── Chat Completions ──

    async def chat_completion(self, payload: dict) -> dict:
        """Send a chat completion to Ollama Cloud (OpenAI-compatible format)."""
        client = await self._get_client()
        start = time.time()
        resp = await client.post(OLLAMA_CHAT_URL, json=payload)
        elapsed = time.time() - start
        resp.raise_for_status()
        data = resp.json()

        # Extract metrics — Ollama Cloud returns standard OpenAI format but may
        # also include Ollama-native fields (eval_count, eval_duration, etc.)
        usage = data.get("usage", {})
        metrics = {
            "total_duration_ns": data.get("total_duration"),
            "load_duration_ns": data.get("load_duration"),
            "prompt_eval_count": data.get("prompt_eval_count") or usage.get("prompt_tokens"),
            "prompt_eval_duration_ns": data.get("prompt_eval_duration"),
            "eval_count": data.get("eval_count") or usage.get("completion_tokens"),
            "eval_duration_ns": data.get("eval_duration"),
            "elapsed_seconds": elapsed,
        }

        # Calculate derived metrics — prefer Ollama-native fields when available
        eval_duration_ns = metrics.get("eval_duration_ns")
        eval_count = metrics.get("eval_count") or 0
        if eval_duration_ns and eval_count and eval_duration_ns > 0:
            metrics["tps"] = round(eval_count / (eval_duration_ns / 1e9), 2)
        elif eval_count and elapsed > 0:
            # Fallback: TPS = completion_tokens / total_elapsed
            metrics["tps"] = round(eval_count / elapsed, 2)

        prompt_eval_duration_ns = metrics.get("prompt_eval_duration_ns")
        prompt_eval_count = metrics.get("prompt_eval_count")
        if prompt_eval_duration_ns and prompt_eval_count and prompt_eval_duration_ns > 0:
            metrics["prompt_tps"] = round(prompt_eval_count / (prompt_eval_duration_ns / 1e9), 2)
        elif prompt_eval_count and elapsed > 0:
            metrics["prompt_tps"] = round(prompt_eval_count / elapsed, 2)

        load_duration_ns = metrics.get("load_duration_ns")
        if load_duration_ns and prompt_eval_duration_ns:
            # TTFT = load_duration + prompt_eval_duration (Ollama-native)
            prompt_dur = prompt_eval_duration_ns or 0
            metrics["ttft_seconds"] = round((load_duration_ns + prompt_dur) / 1e9, 3)
        # Note: For non-streaming OpenAI-format responses, we can't measure true TTFT
        # (time to first token). Only streaming responses will have accurate TTFT.

        data["_oct_metrics"] = metrics
        return data

    async def chat_completion_stream(self, payload: dict):
        """Stream chat completion responses from Ollama Cloud, yielding metrics via _oct_stream_metrics."""
        client = await self._get_client()
        payload_copy = dict(payload)
        payload_copy["stream"] = True

        first_token_time = None
        total_tokens = 0
        start = time.time()

        async with client.stream("POST", OLLAMA_CHAT_URL, json=payload_copy) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        # Yield final chunk with metrics
                        elapsed = time.time() - start
                        metrics = {
                            "eval_count": total_tokens,
                            "elapsed_seconds": elapsed,
                        }
                        if total_tokens and elapsed > 0:
                            metrics["tps"] = round(total_tokens / elapsed, 2)
                        if first_token_time:
                            metrics["ttft_seconds"] = round(first_token_time - start, 3)
                        yield f"data: [DONE]\n\n"
                        # Store metrics on the response for analytics
                        yield f"__oct_metrics__:{json.dumps(metrics)}\n\n"
                        break
                    try:
                        chunk_data = json.loads(data)
                        # Count tokens from streaming chunks
                        for choice in chunk_data.get("choices", []):
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                total_tokens += 1
                    except (json.JSONDecodeError, KeyError):
                        pass
                    yield f"data: {data}\n\n"
                elif line.strip():
                    yield f"data: {line}\n\n"
            yield "data: [DONE]\n\n"

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None