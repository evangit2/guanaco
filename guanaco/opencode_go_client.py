"""OpenCode Go API client — OpenAI-compatible / Anthropic-compatible chat endpoints."""

from __future__ import annotations

import json
import time
import logging
from typing import Optional

import httpx


logger = logging.getLogger(__name__)

OPENCODE_GO_BASE = "https://opencode.ai/zen/go/v1"
OPENCODE_GO_CHAT_URL = f"{OPENCODE_GO_BASE}/chat/completions"
OPENCODE_GO_MESSAGES_URL = f"{OPENCODE_GO_BASE}/messages"
OPENCODE_GO_MODELS_URL = f"{OPENCODE_GO_BASE}/models"

# Models that should use the Anthropic /messages endpoint instead of OpenAI /chat/completions.
ANTHROPIC_ENDPOINT_MODELS = {
    "minimax-m3", "minimax-m2.7", "minimax-m2.5",
    "qwen3.7-max", "qwen3.7-plus", "qwen3.6-plus",
}


def _is_anthropic_model(model: str) -> bool:
    m = model.lower().strip()
    if m.startswith("opencode-go/"):
        m = m[len("opencode-go/"):]
    return any(name in m for name in ANTHROPIC_ENDPOINT_MODELS)


def _extract_cache_usage(data: dict) -> dict:
    """Extract cache-read / cache-write token info from OpenCode Go responses."""
    usage = data.get("usage", {})
    # OpenAI-style field names
    return {
        "cached_read_tokens": usage.get("cached_read_tokens") or usage.get("prompt_tokens_details", {}).get("cached_tokens"),
        "cached_write_tokens": usage.get("cached_write_tokens"),
        "uncached_prompt_tokens": usage.get("uncached_prompt_tokens"),
    }


class OpenCodeGoClient:
    """Minimal OpenCode Go client for chat completions and messages."""

    def __init__(self, api_key: str, timeout: float = 120.0, base_url: str = ""):
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = (base_url or OPENCODE_GO_BASE).rstrip("/")
        self._default_client: Optional[httpx.AsyncClient] = None

    async def _get_client(self, api_key_override: Optional[str] = None) -> httpx.AsyncClient:
        key = api_key_override or self.api_key
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return httpx.AsyncClient(timeout=self.timeout, headers=headers)

    async def close(self):
        if self._default_client and not self._default_client.is_closed:
            await self._default_client.aclose()

    async def list_models(self) -> list[dict]:
        """Fetch available models from the OpenCode Go API."""
        client = await self._get_client()
        try:
            resp = await client.get(f"{self.base_url}/models")
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", data.get("models", []))
        finally:
            await client.aclose()

    async def test_key(self, api_key: Optional[str] = None) -> dict:
        """Test an API key by listing models (fast, no cost)."""
        client = await self._get_client(api_key_override=api_key)
        try:
            resp = await client.get(f"{self.base_url}/models")
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", data.get("models", []))
                return {"ok": True, "model_count": len(models)}
            if resp.status_code in (401, 403):
                return {"ok": False, "error": "Invalid or revoked API key"}
            return {"ok": False, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            logger.warning(f"OpenCode Go key test failed: {e}")
            return {"ok": False, "error": str(e)[:200]}
        finally:
            await client.aclose()

    # Known Go model capabilities (static, since Go API doesn't return details).
    _GO_MODEL_CAPS: dict[str, dict] = {
        "deepseek-v4-flash": {"usage_multiplier": 0.10, "family": "deepseek"},
        "deepseek-v4-pro": {"usage_multiplier": 0.25, "family": "deepseek"},
        "glm-5": {"usage_multiplier": 0.20, "family": "glm"},
        "glm-5.1": {"usage_multiplier": 0.25, "family": "glm"},
        "kimi-k2.5": {"usage_multiplier": 0.20, "family": "kimi"},
        "kimi-k2.6": {"usage_multiplier": 0.25, "family": "kimi", "supports_vision": True},
        "kimi-k2.7-code": {"usage_multiplier": 0.30, "family": "kimi"},
        "minimax-m2.5": {"usage_multiplier": 0.15, "family": "minimax"},
        "minimax-m2.7": {"usage_multiplier": 0.20, "family": "minimax"},
        "minimax-m3": {"usage_multiplier": 0.25, "family": "minimax"},
        "qwen3.5-plus": {"usage_multiplier": 0.15, "family": "qwen"},
        "qwen3.6-plus": {"usage_multiplier": 0.18, "family": "qwen"},
        "qwen3.7-max": {"usage_multiplier": 0.25, "family": "qwen"},
        "mimo-v2.5": {"usage_multiplier": 0.10, "family": "mimo"},
        "mimo-v2.5-pro": {"usage_multiplier": 0.15, "family": "mimo"},
        "mimo-v2-omni": {"usage_multiplier": 0.18, "family": "mimo", "supports_vision": True},
        "hy3-preview": {"usage_multiplier": 0.12, "family": "hy"},
    }

    def _get_model_capabilities(self, model: str) -> dict:
        """Return capability dict for a Go model."""
        base = model.lower().replace("opencode-go/", "")
        caps = self._GO_MODEL_CAPS.get(base, {})
        return {
            "usage_multiplier": caps.get("usage_multiplier", 0.20),
            "supports_vision": caps.get("supports_vision", False),
            "provider": "opencode_go",
        }

    def _strip_provider_prefix(self, model: str) -> str:
        """Strip provider routing prefix before sending to the Go API."""
        m = model.strip()
        if m.lower().startswith("opencode-go/"):
            return m[len("opencode-go/"):]
        return m

    def chat_url(self, model: str) -> str:
        if _is_anthropic_model(model):
            return f"{self.base_url}/messages"
        return f"{self.base_url}/chat/completions"

    async def chat_completion(self, payload: dict, api_key: Optional[str] = None) -> dict:
        model = payload.get("model", "")
        payload["model"] = self._strip_provider_prefix(model)
        url = self.chat_url(payload["model"])
        client = await self._get_client(api_key_override=api_key)
        is_temp = api_key is not None and api_key != self.api_key
        start = time.time()
        try:
            resp = await client.post(url, json=payload)
        finally:
            if is_temp and not client.is_closed:
                await client.aclose()
        elapsed = time.time() - start
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        eval_count = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        prompt_count = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        metrics = {
            "elapsed_seconds": elapsed,
            "prompt_eval_count": prompt_count,
            "eval_count": eval_count,
            "prompt_cache_hit_tokens": (hit := usage.get("prompt_cache_hit_tokens") or usage.get("prompt_tokens_details", {}).get("cached_tokens") or 0),
            "prompt_cache_miss_tokens": usage.get("prompt_cache_miss_tokens") or max(0, prompt_count - hit),
            "cache_usage": _extract_cache_usage(data),
        }
        # TPS: completion_tokens / elapsed (no native timing fields from Go API)
        if eval_count and elapsed > 0:
            metrics["tps"] = round(eval_count / elapsed, 2)
        # TTFT for non-streaming ≈ elapsed (entire response arrives at once)
        if elapsed > 0:
            metrics["ttft_seconds"] = round(elapsed, 3)
        data["_oct_metrics"] = metrics
        return data

    async def chat_completion_stream(self, payload: dict, api_key: Optional[str] = None):
        model = payload.get("model", "")
        payload["model"] = self._strip_provider_prefix(model)
        url = self.chat_url(payload["model"])
        client = await self._get_client(api_key_override=api_key)
        is_temp = api_key is not None and api_key != self.api_key
        payload_copy = dict(payload)
        payload_copy["stream"] = True

        first_token_time = None
        content_chars = 0
        reasoning_chars = 0
        prompt_tokens = 0
        completion_tokens = 0
        start = time.time()

        try:
            async with client.stream("POST", url, json=payload_copy) as resp:
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
                            generation_time = (elapsed - ttft) if ttft and elapsed > ttft else elapsed

                            metrics = {
                                "eval_count": final_tokens,
                                "prompt_eval_count": prompt_tokens,
                                "reasoning_tokens": estimated_reasoning_tokens,
                                "elapsed_seconds": round(elapsed, 3),
                                "ttft_seconds": round(ttft, 3) if ttft else None,
                            }
                            if final_tokens and generation_time > 0:
                                metrics["tps"] = round(final_tokens / generation_time, 2)
                            usage_chunk = {
                                "id": "chatcmpl-final",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": payload_copy.get("model", ""),
                                "choices": [],
                                "usage": {
                                    "prompt_tokens": metrics.get("prompt_eval_count", 0),
                                    "completion_tokens": metrics.get("eval_count", 0),
                                    "total_tokens": metrics.get("prompt_eval_count", 0) + metrics.get("eval_count", 0),
                                },
                            }
                            if metrics.get("reasoning_tokens"):
                                usage_chunk["usage"]["completion_tokens_details"] = {"reasoning_tokens": metrics["reasoning_tokens"]}
                            yield f"data: {json.dumps(usage_chunk)}\n\n"
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
                            logger.debug("Failed to decode stream chunk: %s", data)
                            continue
                        yield f"{line}\n\n"
        finally:
            if is_temp and not client.is_closed:
                await client.aclose()
