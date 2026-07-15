"""Command Code Go API client — OpenAI-compatible chat completions via local proxy.

Command Code (commandcode.ai) offers a $1/mo Go plan with CLI access to 20+ models.
The Go plan does NOT include the official API endpoint ($15/mo Provider plan required),
but a local proxy (cmd_proxy.py, systemd service on port 5999) translates OpenAI-compatible
requests to the CLI's internal /alpha/generate endpoint, bypassing the restriction.

This client talks to the local proxy at http://localhost:5999/v1, which handles:
  - CLI header mimicry (x-session-id, x-command-code-version, etc.)
  - SSE streaming from /alpha/generate
  - Zero Data Retention (ZDR) mode
  - Model name resolution (short names → full Command Code IDs)

API: http://localhost:5999/v1/chat/completions (OpenAI-compatible)
Auth: Bearer <CMD_API_KEY> (user_... prefix, read from ~/.commandcode/auth.json)
Models: 20+ open-weight models, zero per-token cost ($1/mo flat rate)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncGenerator, Optional

import httpx

from guanaco.providers.base import BaseProvider, ProviderMetrics

logger = logging.getLogger(__name__)

CMDCODE_DEFAULT_BASE = "http://localhost:5999/v1"
CMDCODE_DEFAULT_CHAT_URL = f"{CMDCODE_DEFAULT_BASE}/chat/completions"
CMDCODE_DEFAULT_MODELS_URL = f"{CMDCODE_DEFAULT_BASE}/models"

# Static model list — Command Code Go plan offers 20+ models with ZDR support.
# The /v1/models endpoint on the proxy returns them dynamically, but we keep a
# static fallback for capability hints and offline use.
# Only models that worked in the July 15 2026 benchmark with ZDR enabled are listed.
CMDCODE_MODELS: dict[str, dict[str, Any]] = {
    "deepseek-v4-pro": {
        "family": "deepseek", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "deepseek-v4-flash": {
        "family": "deepseek", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "kimi-k2.7-code": {
        "family": "kimi", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "kimi-k2.7-code-highspeed": {
        "family": "kimi", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "kimi-k2.6": {
        "family": "kimi", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "kimi-k2.5": {
        "family": "kimi", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "glm-5.2": {
        "family": "glm", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "glm-5.2-fast": {
        "family": "glm", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "glm-5.1": {
        "family": "glm", "supports_vision": False, "supports_tools": True,
        "supports_thinking": False, "usage_multiplier": 0.0,
    },
    "glm-5": {
        "family": "glm", "supports_vision": False, "supports_tools": True,
        "supports_thinking": False, "usage_multiplier": 0.0,
    },
    "minimax-m3": {
        "family": "minimax", "supports_vision": False, "supports_tools": True,
        "supports_thinking": False, "usage_multiplier": 0.0,
    },
    "minimax-m2.7": {
        "family": "minimax", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "minimax-m2.5": {
        "family": "minimax", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "mimo-v2.5-pro": {
        "family": "mimo", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "mimo-v2.5": {
        "family": "mimo", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "qwen3.7-plus": {
        "family": "qwen", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "qwen3.6-plus": {
        "family": "qwen", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "tencent-hy3": {
        "family": "tencent", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "nemotron-3-ultra": {
        "family": "nvidia", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "step-3.5-flash": {
        "family": "stepfun", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
}


def _strip_cmdcode_prefix(model: str) -> str:
    """Return the model id without the cmdcode/ prefix."""
    model = model.strip()
    lower = model.lower()
    if lower.startswith("cmdcode/"):
        model = model[len("cmdcode/"):]
    return model


class CmdCodeClient(BaseProvider):
    """Async client for Command Code Go plan via local proxy.

    The proxy (cmd_proxy.py, systemd service cmd-proxy.service) runs on
    localhost:5999 and translates OpenAI-compatible requests to Command Code's
    internal /alpha/generate endpoint. This client simply talks to the proxy
    as a standard OpenAI-compatible API.

    Key difference from ClinePassClient: the proxy handles all the CLI header
    mimicry and SSE format translation, so this client is simpler.
    """

    provider_name = "cmdcode"
    supports_streaming = True
    supports_vision = False
    supports_thinking = True

    def __init__(self, api_key: str = "", timeout: float = 300.0, base_url: str = ""):
        super().__init__(api_key=api_key, timeout=timeout, base_url=base_url or CMDCODE_DEFAULT_BASE)
        self.chat_url = f"{self.base_url}/chat/completions"
        self.models_url = f"{self.base_url}/models"

    # ── Model listing ──

    async def list_models(self, force_refresh: bool = False, api_key: Optional[str] = None) -> list[dict]:
        """List available Command Code models from the proxy's /v1/models endpoint.

        Falls back to static model list if the proxy is unreachable.
        """
        now = time.time()
        if not force_refresh and not api_key and self._models_cache and (now - self._models_cache_time) < self._models_cache_ttl:
            return self._models_cache

        client = await self._get_client(api_key_override=api_key)
        is_temp = api_key is not None and api_key != self.api_key
        try:
            resp = await client.get(self.models_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                models = []
                model_list = data.get("data", []) if isinstance(data, dict) else data
                if isinstance(model_list, list):
                    for item in model_list:
                        if isinstance(item, dict):
                            model_id = item.get("id", item.get("name", ""))
                            # Only include short names (no slash) to avoid duplicates
                            if model_id and "/" not in model_id:
                                models.append({
                                    "id": model_id,
                                    "name": model_id,
                                    "model": model_id,
                                    "display_name": model_id,
                                    "details": item,
                                })
                if not models:
                    models = self._static_models()
            else:
                logger.warning("CmdCode proxy /models returned HTTP %s, using static list", resp.status_code)
                models = self._static_models()
        except Exception as e:
            logger.warning("CmdCode proxy /models fetch failed: %s, using static list", e)
            models = self._static_models()
        finally:
            if is_temp and not client.is_closed:
                await client.aclose()

        self._models_cache = models
        self._models_cache_time = now
        return models

    def _static_models(self) -> list[dict]:
        """Return static model list as fallback."""
        return [
            {"id": mid, "name": mid, "model": mid, "display_name": mid, "details": {}}
            for mid in CMDCODE_MODELS
        ]

    async def test_key(self, api_key: Optional[str] = None) -> dict:
        """Test connectivity by listing models from the proxy."""
        client = await self._get_client(api_key_override=api_key)
        is_temp = api_key is not None and api_key != self.api_key
        try:
            resp = await client.get(self.models_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                count = len(data.get("data", [])) if isinstance(data, dict) else len(data) if isinstance(data, list) else 0
                return {"ok": True, "error": None, "model_count": count}
            if resp.status_code == 401:
                return {"ok": False, "error": "Invalid Command Code API key or proxy not running"}
            return {"ok": False, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            logger.warning("CmdCode key test failed: %s", e)
            return {"ok": False, "error": f"Proxy unreachable: {str(e)[:150]}"}
        finally:
            if is_temp and not client.is_closed:
                await client.aclose()

    # ── Capabilities ──

    def _get_model_capabilities(self, model: str) -> dict:
        """Return capability dict for a Command Code model."""
        canonical = _strip_cmdcode_prefix(model)
        caps = CMDCODE_MODELS.get(canonical, {})
        return {
            "supports_vision": bool(caps.get("supports_vision", False)),
            "supports_tools": bool(caps.get("supports_tools", True)),
            "supports_thinking": bool(caps.get("supports_thinking", False)),
            "family": caps.get("family", canonical.split("-")[0] if "-" in canonical else "unknown"),
            "usage_multiplier": 0.0,  # $1/mo flat rate — zero per-token cost
            "provider": "cmdcode",
        }

    # ── Payload normalization ──

    def _prepare_payload(self, payload: dict) -> dict:
        """Strip cmdcode/ prefix and normalize payload."""
        payload = dict(payload)
        model = payload.get("model", "")
        payload["model"] = _strip_cmdcode_prefix(model)
        # Strip reasoning_content from assistant messages (same as Cline/UMANS)
        msgs = payload.get("messages")
        if isinstance(msgs, list):
            for m in msgs:
                if m.get("role") == "assistant":
                    m.pop("reasoning_content", None)
                    m.pop("reasoningContent", None)
        return payload

    # ── Chat completions ──

    async def chat_completion(self, payload: dict, api_key: Optional[str] = None) -> dict:
        """Non-streaming chat completion via the proxy."""
        payload = self._prepare_payload(payload)
        client = await self._get_client(api_key_override=api_key)
        is_temp = api_key is not None and api_key != self.api_key
        start = time.time()
        try:
            resp = await client.post(self.chat_url, json=payload)
        finally:
            if is_temp and not client.is_closed:
                await client.aclose()
        elapsed = time.time() - start
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        metrics = {
            "elapsed_seconds": elapsed,
            "prompt_eval_count": usage.get("prompt_tokens") or usage.get("input_tokens", 0),
            "eval_count": usage.get("completion_tokens") or usage.get("output_tokens", 0),
        }
        if metrics["eval_count"] and elapsed > 0:
            metrics["tps"] = round(min(metrics["eval_count"] / elapsed, 1000.0), 2)
        if elapsed > 0:
            metrics["ttft_seconds"] = round(elapsed, 3)
        data["_oct_metrics"] = metrics
        return data

    async def chat_completion_stream(self, payload: dict, api_key: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Streaming chat completion via the proxy.

        The proxy returns standard OpenAI SSE format with reasoning_content
        deltas for thinking models.
        """
        payload = self._prepare_payload(payload)
        client = await self._get_client(api_key_override=api_key)
        is_temp = api_key is not None and api_key != self.api_key
        payload_copy = dict(payload)
        payload_copy["stream"] = True

        first_token_time: Optional[float] = None
        content_chars = 0
        reasoning_chars = 0
        prompt_tokens = 0
        completion_tokens = 0
        start = time.time()

        try:
            async with client.stream("POST", self.chat_url, json=payload_copy) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            estimated_content_tokens = max(1, content_chars // 4) if content_chars else 0
                            estimated_reasoning_tokens = max(1, reasoning_chars // 4) if reasoning_chars else 0
                            final_tokens = completion_tokens or (estimated_content_tokens + estimated_reasoning_tokens)
                            elapsed = time.time() - start
                            ttft = (first_token_time - start) if first_token_time else None
                            _MIN_GENERATION_TIME = 0.05
                            if ttft is not None and (elapsed - ttft) > _MIN_GENERATION_TIME:
                                generation_time = elapsed - ttft
                            else:
                                generation_time = elapsed

                            metrics = {
                                "eval_count": final_tokens,
                                "prompt_eval_count": prompt_tokens,
                                "reasoning_tokens": estimated_reasoning_tokens,
                                "elapsed_seconds": round(elapsed, 3),
                                "ttft_seconds": round(ttft, 3) if ttft else None,
                            }
                            if final_tokens and generation_time > 0:
                                raw_tps = final_tokens / generation_time
                                metrics["tps"] = round(min(raw_tps, 1000.0), 2)
                            yield self._build_usage_chunk(
                                payload_copy.get("model", ""),
                                metrics.get("prompt_eval_count", 0),
                                metrics.get("eval_count", 0),
                                metrics.get("reasoning_tokens", 0),
                            )
                            yield "data: [DONE]\n\n"
                            yield f"__oct_metrics__:{json.dumps(metrics)}\n\n"
                            return
                        try:
                            chunk_data = json.loads(data)
                            for choice in chunk_data.get("choices", []):
                                delta = choice.get("delta", {})
                                content = delta.get("content", "")
                                reasoning = delta.get("reasoning", "") or delta.get("reasoning_content", "")
                                if content:
                                    content_chars += len(content)
                                if reasoning:
                                    reasoning_chars += len(reasoning)
                                if not first_token_time and (content or reasoning):
                                    first_token_time = time.time()
                            usage = chunk_data.get("usage", {})
                            if usage:
                                prompt_tokens = usage.get("prompt_tokens", prompt_tokens) or usage.get("input_tokens", prompt_tokens)
                                completion_tokens = usage.get("completion_tokens", completion_tokens) or usage.get("output_tokens", completion_tokens)
                        except json.JSONDecodeError:
                            logger.debug("Failed to decode CmdCode stream chunk: %s", data)
                        yield f"{line}\n\n"
                # Stream ended without [DONE]
                estimated_content_tokens = max(1, content_chars // 4) if content_chars else 0
                estimated_reasoning_tokens = max(1, reasoning_chars // 4) if reasoning_chars else 0
                final_tokens = completion_tokens or (estimated_content_tokens + estimated_reasoning_tokens)
                elapsed = time.time() - start
                ttft = (first_token_time - start) if first_token_time else None
                _MIN_GENERATION_TIME = 0.05
                if ttft is not None and (elapsed - ttft) > _MIN_GENERATION_TIME:
                    generation_time = elapsed - ttft
                else:
                    generation_time = elapsed

                metrics = {
                    "eval_count": final_tokens,
                    "prompt_eval_count": prompt_tokens,
                    "reasoning_tokens": estimated_reasoning_tokens,
                    "elapsed_seconds": round(elapsed, 3),
                    "ttft_seconds": round(ttft, 3) if ttft else None,
                }
                if final_tokens and generation_time > 0:
                    raw_tps = final_tokens / generation_time
                    metrics["tps"] = round(min(raw_tps, 1000.0), 2)
                yield self._build_usage_chunk(
                    payload_copy.get("model", ""),
                    metrics.get("prompt_eval_count", 0),
                    metrics.get("eval_count", 0),
                    metrics.get("reasoning_tokens", 0),
                )
                yield "data: [DONE]\n\n"
                yield f"__oct_metrics__:{json.dumps(metrics)}\n\n"
        finally:
            if is_temp and not client.is_closed:
                await client.aclose()