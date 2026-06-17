"""UMANS API client — OpenAI-compatible chat completions for UMANS subscriptions."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

import httpx


logger = logging.getLogger(__name__)

UMANS_BASE = "https://api.code.umans.ai/v1"
UMANS_CHAT_URL = f"{UMANS_BASE}/chat/completions"
UMANS_MODELS_URL = f"{UMANS_BASE}/models/info"
UMANS_USAGE_URL = f"{UMANS_BASE}/usage"

# Static capability hints for UMANS models that don't appear in /models/info map
# or whose info lacks capability metadata. Populated as we discover models.
UMANS_STATIC_CAPS: dict[str, dict[str, Any]] = {
    "umans-kimi-k2.7": {"family": "kimi", "supports_vision": True, "supports_tools": True, "supports_thinking": True},
    "umans-kimi-k2.6": {"family": "kimi", "supports_vision": True, "supports_tools": True, "supports_thinking": True},
    "umans-glm-5.1": {"family": "glm", "supports_vision": True, "supports_tools": True, "supports_thinking": True},
    "umans-glm-5": {"family": "glm", "supports_vision": True, "supports_tools": True, "supports_thinking": True},
}


def _strip_umans_prefix(model: str) -> str:
    """Return the canonical UMANS model id with umans- prefix."""
    model = model.strip()
    lower = model.lower()
    if lower.startswith("umans/"):
        model = model[len("umans/"):]
    if not lower.startswith("umans-"):
        model = f"umans-{model}"
    return model


def _normalize_model_id(model: str) -> str:
    """Return a display-ish id without the umans- prefix."""
    m = model.strip()
    for prefix in ("umans/", "umans-"):
        if m.lower().startswith(prefix):
            m = m[len(prefix):]
    return m


class UmansClient:
    """Async client for UMANS subscription API."""

    def __init__(self, api_key: str, timeout: float = 900.0, base_url: str = ""):
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = (base_url or UMANS_BASE).rstrip("/")
        self.chat_url = f"{self.base_url}/chat/completions"
        self.models_url = f"{self.base_url}/models/info"
        self.usage_url = f"{self.base_url}/usage"
        self._default_client: Optional[httpx.AsyncClient] = None
        self._models_cache: Optional[dict] = None
        self._models_cache_time: float = 0
        self._models_cache_ttl: float = 300.0  # 5 minutes
        # Runtime-configurable normalizers
        self.session_label_mode: str = "auto"  # yes | auto | no
        self.max_images_per_request: int = 0
        self._session_counter: int = 0

    async def _get_client(self, api_key_override: Optional[str] = None) -> httpx.AsyncClient:
        key = api_key_override or self.api_key
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if key and key not in ("***", "REPLACE_ME"):
            headers["Authorization"] = f"Bearer {key}"
        return httpx.AsyncClient(timeout=self.timeout, headers=headers)

    async def close(self):
        if self._default_client and not self._default_client.is_closed:
            await self._default_client.aclose()

    async def list_models(self, force_refresh: bool = False, api_key: Optional[str] = None) -> list[dict]:
        """List available UMANS models from /models/info.

        Returns OpenAI-ish model dicts with id/name and capabilities.
        """
        now = time.time()
        if not force_refresh and not api_key and self._models_cache and (now - self._models_cache_time) < self._models_cache_ttl:
            return self._models_cache["list"]

        client = await self._get_client(api_key_override=api_key)
        is_temp = api_key is not None and api_key != self.api_key
        try:
            resp = await client.get(self.models_url)
            resp.raise_for_status()
            data = resp.json()
        finally:
            if is_temp and not client.is_closed:
                await client.aclose()

        models = []
        if isinstance(data, dict):
            for model_id, info in data.items():
                if not isinstance(info, dict):
                    continue
                if not model_id or model_id.startswith("_"):
                    continue
                models.append({
                    "id": model_id,
                    "name": model_id,
                    "model": model_id,
                    "display_name": re.sub(r"^Umans\s+", "", info.get("display_name", ""), flags=re.I) if info.get("display_name") else _normalize_model_id(model_id),
                    "capabilities": info.get("capabilities", {}),
                    "details": info,
                })
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    model_id = item.get("id") or item.get("name") or item.get("model", "")
                    models.append({
                        "id": model_id,
                        "name": model_id,
                        "model": model_id,
                        "display_name": item.get("display_name", _normalize_model_id(model_id)),
                        "capabilities": item.get("capabilities", {}),
                        "details": item,
                    })

        self._models_cache = {"map": {m["id"]: m for m in models}, "list": models}
        self._models_cache_time = now
        return models

    async def test_key(self, api_key: Optional[str] = None) -> dict:
        """Test an API key by listing models (fast, no cost)."""
        client = await self._get_client(api_key_override=api_key)
        is_temp = api_key is not None and api_key != self.api_key
        try:
            resp = await client.get(self.models_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                models = []
                if isinstance(data, dict):
                    models = [k for k in data.keys() if isinstance(data[k], dict)]
                elif isinstance(data, list):
                    models = [m.get("id", m.get("name", "")) for m in data if isinstance(m, dict)]
                return {"ok": True, "error": None, "model_count": len(models)}
            if resp.status_code == 401:
                return {"ok": False, "error": "Invalid or expired API key"}
            return {"ok": False, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            logger.warning("UMANS key test failed: %s", e)
            return {"ok": False, "error": str(e)[:200]}
        finally:
            if is_temp and not client.is_closed:
                await client.aclose()

    def _get_model_capabilities(self, model: str) -> dict:
        """Return capability dict for a UMANS model."""
        canonical = _strip_umans_prefix(model)
        display = _normalize_model_id(canonical)
        caps = UMANS_STATIC_CAPS.get(canonical, {})
        # Prefer cached upstream metadata if available.
        if self._models_cache:
            m = self._models_cache["map"].get(canonical) or self._models_cache["map"].get(model)
            if m:
                caps = {**caps}
                upstream_caps = m.get("capabilities", {})
                if isinstance(upstream_caps, dict):
                    if "vision" in upstream_caps:
                        caps["supports_vision"] = bool(upstream_caps["vision"])
                    if "tools" in upstream_caps:
                        caps["supports_tools"] = bool(upstream_caps["tools"])
                    if "reasoning" in upstream_caps:
                        caps["supports_thinking"] = bool(upstream_caps["reasoning"])
                caps["upstream_capabilities"] = upstream_caps
        usage_multiplier = 0.20
        if any(x in display for x in ("kimi-k2.7", "glm-5.1")):
            usage_multiplier = 0.25
        return {
            "supports_vision": bool(caps.get("supports_vision", False)),
            "supports_tools": bool(caps.get("supports_tools", False)),
            "supports_thinking": bool(caps.get("supports_thinking", False)),
            "family": caps.get("family", display.split("-")[0] if "-" in display else "unknown"),
            "usage_multiplier": usage_multiplier,
            "provider": "umans",
            **{k: v for k, v in caps.items() if k not in ("supports_vision", "supports_tools", "supports_thinking", "family", "usage_multiplier", "provider")},
        }

    async def get_concurrency(self, api_key: Optional[str] = None) -> dict:
        """Fetch current concurrency usage/limit from UMANS /usage endpoint."""
        client = await self._get_client(api_key_override=api_key)
        is_temp = api_key is not None and api_key != self.api_key
        try:
            resp = await client.get(self.usage_url, timeout=10)
            if resp.status_code != 200:
                return {"concurrent": 0, "limit": None, "user_id": None}
            data = resp.json()
            concurrent = 0
            limit = None
            user_id = None
            if isinstance(data, dict):
                usage = data.get("usage", {})
                limits = data.get("limits", {})
                if isinstance(usage, dict):
                    concurrent = usage.get("concurrent_sessions", 0)
                    user_id = usage.get("user_id") or data.get("user_id")
                if isinstance(limits, dict) and isinstance(limits.get("concurrency"), dict):
                    limit = limits["concurrency"].get("limit")
                    if isinstance(limit, str):
                        try:
                            limit = int(limit)
                        except ValueError:
                            limit = None
            return {"concurrent": concurrent, "limit": limit, "user_id": user_id}
        except Exception as e:
            logger.warning("UMANS concurrency fetch failed: %s", e)
            return {"concurrent": 0, "limit": None, "user_id": None}
        finally:
            if is_temp and not client.is_closed:
                await client.aclose()

    # ── Payload normalizers inspired by umans-proxy ──

    def _session_label(self, mode: str, model: str) -> str:
        """Return a session label based on mode and model capabilities."""
        mode = mode or self.session_label_mode
        if mode == "yes":
            self._session_counter += 1
            return f"umans|sess{self._session_counter}"
        if mode == "auto":
            caps = self._get_model_capabilities(model)
            if caps.get("supports_thinking") or "thinking" in model.lower():
                self._session_counter += 1
                return f"umans|sess{self._session_counter}"
        return ""

    def _stamp_session_label(self, payload: dict, label: str) -> None:
        """Prepend a session label to the first user message content (text only)."""
        if not label:
            return
        msgs = payload.get("messages")
        if not isinstance(msgs, list):
            return
        for m in msgs:
            if m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str):
                    m["content"] = f"[{label}] {content}"
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            part["text"] = f"[{label}] {part.get('text', '')}"
                            break
                break

    @staticmethod
    def _strip_reasoning_content(payload: dict) -> None:
        """Remove reasoning_content / reasoningContent from assistant messages."""
        msgs = payload.get("messages")
        if not isinstance(msgs, list):
            return
        for m in msgs:
            if m.get("role") == "assistant":
                m.pop("reasoning_content", None)
                m.pop("reasoningContent", None)

    @staticmethod
    def _limit_images(payload: dict, max_images: int) -> None:
        """Keep only the newest max_images image parts across the conversation."""
        if max_images <= 0:
            return
        msgs = payload.get("messages")
        if not isinstance(msgs, list):
            return
        image_parts = []
        for mi, m in enumerate(msgs):
            if m.get("role") == "system":
                continue
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for pi, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    image_parts.append((content, pi, mi))
        if len(image_parts) <= max_images:
            return
        to_remove = len(image_parts) - max_images
        # Remove oldest first (smallest message index)
        for idx in range(to_remove):
            content, pi, _ = image_parts[idx]
            content.pop(pi)

    def _prepare_payload(
        self,
        payload: dict,
        session_label: str = "",
        max_images: int = -1,
    ) -> dict:
        """Make a shallow copy and apply UMANS-specific normalizations."""
        payload = dict(payload)
        model = payload.get("model", "")
        payload["model"] = _strip_umans_prefix(model)
        self._strip_reasoning_content(payload)
        if max_images < 0:
            max_images = self.max_images_per_request
        self._limit_images(payload, max_images)
        if session_label:
            self._stamp_session_label(payload, session_label)
        return payload

    async def chat_completion(
        self,
        payload: dict,
        api_key: Optional[str] = None,
        session_label: str = "",
        max_images: int = -1,
    ) -> dict:
        """Send a non-streaming chat completion to UMANS."""
        label = session_label or self._session_label(self.session_label_mode, payload.get("model", ""))
        payload = self._prepare_payload(payload, session_label=label, max_images=max_images)
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
            metrics["tps"] = round(metrics["eval_count"] / elapsed, 2)
        if elapsed > 0:
            metrics["ttft_seconds"] = round(elapsed, 3)
        data["_oct_metrics"] = metrics
        return data

    async def chat_completion_stream(
        self,
        payload: dict,
        api_key: Optional[str] = None,
        session_label: str = "",
        max_images: int = -1,
    ):
        """Stream chat completion responses from UMANS."""
        label = session_label or self._session_label(self.session_label_mode, payload.get("model", ""))
        payload = self._prepare_payload(payload, session_label=label, max_images=max_images)
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
                            logger.debug("Failed to decode UMANS stream chunk: %s", data)
                        yield f"{line}\n\n"
                # Stream ended without [DONE]
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
                yield "data: [DONE]\n\n"
                yield f"__oct_metrics__:{json.dumps(metrics)}\n\n"
        finally:
            if is_temp and not client.is_closed:
                await client.aclose()
