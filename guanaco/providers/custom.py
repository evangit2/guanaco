"""Custom OpenAI/Anthropic-compatible provider.

Lets users add any OpenAI-compatible API endpoint (OpenAI, OpenRouter, Together,
Groq, LM Studio, vLLM, etc.) as a first-class provider in Guanaco — not just a
fallback. Supports multiple custom providers, each with its own base_url, API key,
model list, and optional concurrency limiting.

Config example (config.yaml):

    custom_providers:
      - name: openrouter
        base_url: https://openrouter.ai/api/v1
        api_key: sk-or-...
        models:
          - anthropic/claude-sonnet-4
          - google/gemini-2.5-flash
        max_concurrent_streams: 0   # 0 = unlimited (default)
      - name: lm-studio
        base_url: http://localhost:1234/v1
        api_key: ""
        models: []                   # auto-discover from /v1/models
        max_concurrent_streams: 4
"""

from __future__ import annotations

import json
import time
import logging
from typing import Any, AsyncGenerator, Optional

import httpx

from guanaco.providers.base import BaseProvider, ProviderMetrics

log = logging.getLogger(__name__)


class CustomProvider(BaseProvider):
    """Generic OpenAI-compatible provider.

    Works with any API that speaks the OpenAI /v1/chat/completions protocol.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str = "",
        models: Optional[list[str]] = None,
        timeout: float = 120.0,
        max_concurrent_streams: int = 0,
    ):
        super().__init__(api_key=api_key, timeout=timeout, base_url=base_url)
        self.provider_name = name
        self._configured_models = models or []
        self.max_concurrent_streams = max_concurrent_streams
        self._semaphore: Optional[Any] = None  # asyncio.Semaphore, created lazily

        # Endpoints
        self.chat_url = f"{self.base_url}/chat/completions"
        self.models_url = f"{self.base_url}/models"

    def _get_semaphore(self):
        """Lazily create the concurrency semaphore."""
        if self.max_concurrent_streams > 0 and self._semaphore is None:
            import asyncio
            self._semaphore = asyncio.Semaphore(self.max_concurrent_streams)
        return self._semaphore

    async def chat_completion(self, payload: dict, api_key: Optional[str] = None) -> dict:
        """Non-streaming chat completion."""
        payload = self._prepare_payload(payload)
        sem = self._get_semaphore()
        client = await self._get_client(api_key_override=api_key)
        is_temp = api_key is not None and api_key != self.api_key
        start = time.time()

        async def _call():
            return await client.post(self.chat_url, json=payload)

        try:
            if sem:
                async with sem:
                    resp = await _call()
            else:
                resp = await _call()
            elapsed = time.time() - start
            resp.raise_for_status()
            data = resp.json()

            usage = data.get("usage", {})
            metrics = ProviderMetrics.from_usage(usage, elapsed)
            data["_oct_metrics"] = metrics.to_dict()
            return data
        finally:
            if is_temp and not client.is_closed:
                await client.aclose()

    async def chat_completion_stream(self, payload: dict, api_key: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Streaming chat completion."""
        payload = self._prepare_payload(payload)
        payload["stream"] = True
        sem = self._get_semaphore()
        client = await self._get_client(api_key_override=api_key)
        is_temp = api_key is not None and api_key != self.api_key

        first_token_time: Optional[float] = None
        content_chars = 0
        reasoning_chars = 0
        prompt_tokens = 0
        completion_tokens = 0
        start = time.time()

        async def _stream():
            nonlocal first_token_time, content_chars, reasoning_chars, prompt_tokens, completion_tokens
            async with client.stream("POST", self.chat_url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            return
                        try:
                            chunk = json.loads(data)
                            for choice in chunk.get("choices", []):
                                delta = choice.get("delta", {})
                                content = delta.get("content", "")
                                reasoning = delta.get("reasoning", "") or delta.get("reasoning_content", "")
                                if content:
                                    content_chars += len(content)
                                if reasoning:
                                    reasoning_chars += len(reasoning)
                                if not first_token_time and (content or reasoning):
                                    first_token_time = time.time()
                            usage = chunk.get("usage", {})
                            if usage:
                                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                                completion_tokens = usage.get("completion_tokens", completion_tokens)
                        except json.JSONDecodeError:
                            pass
                    yield f"{line}\n\n"

        try:
            if sem:
                async with sem:
                    async for chunk in _stream():
                        yield chunk
            else:
                async for chunk in _stream():
                    yield chunk

            # Emit final usage chunk + metrics
            elapsed = time.time() - start
            est_content = max(1, content_chars // 4) if content_chars else 0
            est_reasoning = max(1, reasoning_chars // 4) if reasoning_chars else 0
            final_tokens = completion_tokens or (est_content + est_reasoning)
            metrics = ProviderMetrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=final_tokens,
                total_tokens=prompt_tokens + final_tokens,
                reasoning_tokens=est_reasoning,
                total_duration_seconds=round(elapsed, 3),
                ttft_seconds=round(first_token_time - start, 3) if first_token_time else None,
            )
            _MIN_GEN = 0.05
            gen_time = (elapsed - metrics.ttft_seconds) if metrics.ttft_seconds and (elapsed - metrics.ttft_seconds) > _MIN_GEN else elapsed
            if final_tokens and gen_time > 0:
                metrics.tps = round(min(final_tokens / gen_time, 1000.0), 2)

            yield self._build_usage_chunk(payload.get("model", ""), prompt_tokens, final_tokens, est_reasoning)
            yield "data: [DONE]\n\n"
            yield self._build_metrics_line(metrics)
        finally:
            if is_temp and not client.is_closed:
                await client.aclose()

    async def list_models(self, api_key: Optional[str] = None) -> list[dict]:
        """List models — uses configured list if set, otherwise fetches from /v1/models."""
        if self._configured_models:
            return [
                {"id": m, "name": m, "model": m, "provider": self.provider_name}
                for m in self._configured_models
            ]

        # Auto-discover from /v1/models
        now = time.time()
        if self._models_cache and (now - self._models_cache_time) < self._models_cache_ttl:
            return self._models_cache

        client = await self._get_client(api_key_override=api_key)
        is_temp = api_key is not None and api_key != self.api_key
        try:
            resp = await client.get(self.models_url, timeout=10)
            if resp.status_code != 200:
                log.warning("Custom provider %s: /v1/models returned %s", self.provider_name, resp.status_code)
                return []
            data = resp.json()
            models = []
            for m in data.get("data", data) if isinstance(data, dict) else data:
                if isinstance(m, dict):
                    mid = m.get("id", m.get("name", ""))
                    if mid:
                        models.append({"id": mid, "name": mid, "model": mid, "provider": self.provider_name})
            self._models_cache = models
            self._models_cache_time = now
            return models
        except Exception as e:
            log.warning("Custom provider %s: failed to list models: %s", self.provider_name, e)
            return []
        finally:
            if is_temp and not client.is_closed:
                await client.aclose()

    async def test_key(self, api_key: Optional[str] = None) -> dict:
        """Test the API key by listing models."""
        client = await self._get_client(api_key_override=api_key)
        is_temp = api_key is not None and api_key != self.api_key
        try:
            resp = await client.get(self.models_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                count = len(data.get("data", [])) if isinstance(data, dict) else len(data) if isinstance(data, list) else 0
                return {"ok": True, "error": None, "model_count": count}
            if resp.status_code == 401:
                return {"ok": False, "error": "Invalid or expired API key"}
            return {"ok": False, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}
        finally:
            if is_temp and not client.is_closed:
                await client.aclose()

    def _get_model_capabilities(self, model: str) -> dict:
        return {
            "supports_vision": self.supports_vision,
            "supports_tools": True,
            "supports_thinking": self.supports_thinking,
            "family": "custom",
            "usage_multiplier": 1.0,
            "provider": self.provider_name,
        }
