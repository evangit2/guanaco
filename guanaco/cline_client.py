"""Cline Pass API client — OpenAI-compatible chat completions for Cline Pass subscriptions.

Cline Pass is a flat-rate monthly subscription ($9.99/mo) providing access to 10
open-weight models through Cline's multi-provider gateway. The gateway routes
requests across multiple inference providers (Fireworks, Baseten, DeepInfra,
Moonshot, etc.) with automatic fallback.

API docs: https://api.cline.bot/api/v1
Auth: Bearer <API_KEY> (sk_... prefix)
Models: 10 open-weight models, zero per-token cost (subscription-based)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncGenerator, Optional

import httpx

from guanaco.providers.base import BaseProvider, ProviderMetrics

logger = logging.getLogger(__name__)

CLINE_BASE = "https://api.cline.bot/api/v1"
CLINE_CHAT_URL = f"{CLINE_BASE}/chat/completions"
CLINE_MODELS_URL = f"{CLINE_BASE}/models"

# Static model list — Cline Pass offers 10 models.
# The /models endpoint returns them dynamically, but we keep a static fallback
# for capability hints and offline use.
CLINE_MODELS: dict[str, dict[str, Any]] = {
    "glm-5.2": {
        "family": "glm", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "kimi-k3": {
        "family": "kimi", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "kimi-k2.7-code": {
        "family": "kimi", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "kimi-k2.6": {
        "family": "kimi", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "deepseek-v4-pro": {
        "family": "deepseek", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "deepseek-v4-flash": {
        "family": "deepseek", "supports_vision": False, "supports_tools": True,
        "supports_thinking": False, "usage_multiplier": 0.0,
    },
    "mimo-v2.5": {
        "family": "mimo", "supports_vision": False, "supports_tools": True,
        "supports_thinking": False, "usage_multiplier": 0.0,
    },
    "mimo-v2.5-pro": {
        "family": "mimo", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "minimax-m3": {
        "family": "minimax", "supports_vision": False, "supports_tools": True,
        "supports_thinking": False, "usage_multiplier": 0.0,
    },
    "qwen3.7-max": {
        "family": "qwen", "supports_vision": False, "supports_tools": True,
        "supports_thinking": False, "usage_multiplier": 0.0,
    },
    "qwen3.7-plus": {
        "family": "qwen", "supports_vision": False, "supports_tools": True,
        "supports_thinking": False, "usage_multiplier": 0.0,
    },
}


# Cline's gateway requires the format: modelType/model (e.g. "zai/glm-5.2").
# Each model belongs to a specific provider type. Discovered via API testing.
CLINE_MODEL_TYPES: dict[str, str] = {
    "glm-5.2": "zai",
    "kimi-k3": "moonshotai",
    "kimi-k2.7-code": "moonshotai",
    "kimi-k2.6": "moonshotai",
    "deepseek-v4-pro": "deepseek",
    "deepseek-v4-flash": "deepseek",
    "mimo-v2.5": "xiaomi",
    "mimo-v2.5-pro": "xiaomi",
    "minimax-m3": "minimax",
    "qwen3.7-max": "qwen",
    "qwen3.7-plus": "qwen",
}


def _strip_cline_prefix(model: str) -> str:
    """Return the model id in cline-pass/<model> format for subscription routing.

    Cline's API accepts two routing formats:
    - cline-pass/<model>  → routes through the Cline Pass subscription (free)
    - <type>/<model>      → routes through Cline Credits (pay-per-use)

    Using cline-pass/ ensures requests count against the subscription quota
    instead of draining the credit balance. This function strips any existing
    'cline/' prefix and prepends 'cline-pass/'.
    """
    model = model.strip()
    lower = model.lower()
    # Strip Guanaco's 'cline/' prefix if present
    if lower.startswith("cline/"):
        model = model[len("cline/"):]
    # If already in cline-pass/ format, use as-is
    if lower.startswith("cline-pass/"):
        return model
    # If the model has a type prefix (e.g. "zai/glm-5.2"), strip it — we want
    # the bare model name under cline-pass/
    if "/" in model and not model.lower().startswith("cline-pass/"):
        model = model.split("/", 1)[-1]
    return f"cline-pass/{model}"


class ClinePassClient(BaseProvider):
    """Async client for Cline Pass subscription API.

    Cline Pass is OpenAI-compatible, so this client follows the same pattern
    as UmansClient — streaming SSE, reasoning delta support, usage tracking.
    Key difference: zero per-token cost (flat-rate subscription).
    """

    provider_name = "cline"
    supports_streaming = True
    supports_vision = False
    supports_thinking = True

    def __init__(self, api_key: str = "", timeout: float = 300.0, base_url: str = ""):
        super().__init__(api_key=api_key, timeout=timeout, base_url=base_url or CLINE_BASE)
        self.chat_url = f"{self.base_url}/chat/completions"
        self.models_url = f"{self.base_url}/models"

    # ── Model listing ──

    async def list_models(self, force_refresh: bool = False, api_key: Optional[str] = None) -> list[dict]:
        """List available Cline Pass models from /models endpoint.

        Falls back to static model list if the API is unreachable.
        """
        now = time.time()
        if not force_refresh and not api_key and self._models_cache and (now - self._models_cache_time) < self._models_cache_ttl:
            return self._models_cache

        client = await self._get_client(api_key_override=api_key)
        is_temp = api_key is not None and api_key != self.api_key
        try:
            resp = await client.get(self.models_url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                models = []
                if isinstance(data, dict) and "data" in data:
                    for item in data["data"]:
                        if isinstance(item, dict):
                            model_id = item.get("id", item.get("name", ""))
                            if model_id:
                                models.append({
                                    "id": model_id,
                                    "name": model_id,
                                    "model": model_id,
                                    "display_name": item.get("id", model_id),
                                    "details": item,
                                })
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            model_id = item.get("id", item.get("name", ""))
                            if model_id:
                                models.append({
                                    "id": model_id,
                                    "name": model_id,
                                    "model": model_id,
                                    "display_name": model_id,
                                    "details": item,
                                })
                else:
                    # Unknown format — use static
                    models = self._static_models()
            else:
                logger.warning("Cline /models returned HTTP %s, using static list", resp.status_code)
                models = self._static_models()
        except Exception as e:
            logger.warning("Cline /models fetch failed: %s, using static list", e)
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
            for mid in CLINE_MODELS
        ]

    async def test_key(self, api_key: Optional[str] = None) -> dict:
        """Test an API key by listing models (fast, no cost)."""
        client = await self._get_client(api_key_override=api_key)
        is_temp = api_key is not None and api_key != self.api_key
        try:
            resp = await client.get(self.models_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                count = 0
                if isinstance(data, dict) and "data" in data:
                    count = len(data["data"])
                elif isinstance(data, list):
                    count = len(data)
                return {"ok": True, "error": None, "model_count": count}
            if resp.status_code == 401:
                return {"ok": False, "error": "Invalid or expired Cline Pass API key"}
            return {"ok": False, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            logger.warning("Cline key test failed: %s", e)
            return {"ok": False, "error": str(e)[:200]}
        finally:
            if is_temp and not client.is_closed:
                await client.aclose()

    # ── Capabilities ──

    def _get_model_capabilities(self, model: str) -> dict:
        """Return capability dict for a Cline Pass model."""
        canonical = _strip_cline_prefix(model)
        # canonical is "cline-pass/<model>" — extract bare name for CLINE_MODELS lookup
        bare = canonical.split("/", 1)[-1] if "/" in canonical else canonical
        caps = CLINE_MODELS.get(bare.lower(), {})
        return {
            "supports_vision": bool(caps.get("supports_vision", False)),
            "supports_tools": bool(caps.get("supports_tools", True)),
            "supports_thinking": bool(caps.get("supports_thinking", False)),
            "family": caps.get("family", canonical.split("-")[0] if "-" in canonical else "unknown"),
            "usage_multiplier": 0.0,  # Flat-rate subscription — zero per-token cost
            "provider": "cline",
        }

    # ── Payload normalization ──

    def _prepare_payload(self, payload: dict) -> dict:
        """Strip cline/ prefix and normalize payload."""
        payload = dict(payload)
        model = payload.get("model", "")
        payload["model"] = _strip_cline_prefix(model)
        # Strip reasoning_content from assistant messages (same as UMANS)
        msgs = payload.get("messages")
        if isinstance(msgs, list):
            for m in msgs:
                if m.get("role") == "assistant":
                    m.pop("reasoning_content", None)
                    m.pop("reasoningContent", None)
        return payload

    # ── Chat completions ──

    async def chat_completion(self, payload: dict, api_key: Optional[str] = None) -> dict:
        """Non-streaming chat completion."""
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

        # Cline wraps responses under a "data" key — unwrap it so the router
        # sees the standard OpenAI format with "choices" at top level.
        if isinstance(data, dict) and "data" in data and "choices" not in data:
            inner = data["data"]
            if isinstance(inner, dict):
                # Preserve any top-level fields like "success" but use inner for choices/usage
                inner["_oct_metrics"] = data.get("_oct_metrics", {})
                data = inner

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
        """Streaming chat completion from Cline Pass.

        Handles SSE with reasoning delta fields (extended thinking support).
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
                            logger.debug("Failed to decode Cline stream chunk: %s", data)
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
