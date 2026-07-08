"""Base provider class — shared interface for all LLM providers.

Every provider (Ollama, OpenCode Go, UMANS, custom) implements this interface.
The router and MultiProviderChatClient call these methods without caring
about the underlying API differences.
"""

from __future__ import annotations

import json
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

import httpx

log = logging.getLogger(__name__)


@dataclass
class ProviderMetrics:
    """Metrics extracted from a provider response for analytics logging."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tps: Optional[float] = None
    prompt_tps: Optional[float] = None
    ttft_seconds: Optional[float] = None
    total_duration_seconds: Optional[float] = None
    reasoning_tokens: int = 0
    estimated_cost: float = 0.0

    @classmethod
    def from_usage(cls, usage: dict, elapsed: float, first_token_time: Optional[float] = None) -> "ProviderMetrics":
        """Build metrics from an OpenAI-style usage block."""
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)

        # Reasoning tokens from thinking models
        details = usage.get("completion_tokens_details", {})
        reasoning_tokens = details.get("reasoning_tokens", 0) if isinstance(details, dict) else 0

        ttft = (first_token_time - (time.time() - elapsed)) if first_token_time else None
        # Guard against providers that batch into one chunk
        _MIN_GEN_TIME = 0.05
        if ttft is not None and (elapsed - ttft) > _MIN_GEN_TIME:
            gen_time = elapsed - ttft
        else:
            gen_time = elapsed
            ttft = None

        tps = round(min(completion_tokens / gen_time, 1000.0), 2) if completion_tokens and gen_time > 0 else None

        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tps=tps,
            ttft_seconds=round(ttft, 3) if ttft else None,
            total_duration_seconds=round(elapsed, 3),
            reasoning_tokens=reasoning_tokens,
        )

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != 0}


class BaseProvider(ABC):
    """Abstract base class for LLM providers.

    Subclasses must implement:
        - chat_completion(payload, api_key) → dict
        - chat_completion_stream(payload, api_key) → AsyncGenerator[str, None]
        - list_models(api_key) → list[dict]
        - test_key(api_key) → dict

    Optional overrides:
        - _prepare_payload(payload) → dict  (provider-specific normalization)
        - _get_model_capabilities(model) → dict
        - close()
    """

    provider_name: str = "base"
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_thinking: bool = False

    def __init__(self, api_key: str = "", timeout: float = 120.0, base_url: str = ""):
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = base_url.rstrip("/") if base_url else ""
        self._default_client: Optional[httpx.AsyncClient] = None
        self._models_cache: Optional[list[dict]] = None
        self._models_cache_time: float = 0
        self._models_cache_ttl: float = 300.0  # 5 minutes

    # ── HTTP client management ──

    async def _get_client(self, api_key_override: Optional[str] = None) -> httpx.AsyncClient:
        """Get an httpx client with the right auth headers."""
        key = api_key_override or self.api_key
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if key and key not in ("***", "REPLACE_ME"):
            headers["Authorization"] = f"Bearer {key}"
        return httpx.AsyncClient(timeout=self.timeout, headers=headers)

    async def close(self):
        """Close the default HTTP client if one was created."""
        if self._default_client and not self._default_client.is_closed:
            await self._default_client.aclose()

    # ── Payload normalization ──

    def _prepare_payload(self, payload: dict) -> dict:
        """Normalize a chat payload before sending to the provider.

        Override in subclasses for provider-specific normalization
        (strip reasoning_content, prefix model names, etc.)
        """
        return dict(payload)

    # ── Reasoning effort ──

    def _apply_reasoning_effort(self, payload: dict, reasoning_effort: Optional[str]) -> dict:
        """Apply reasoning_effort to the payload in a provider-specific way.

        Override in subclasses for providers that use different parameter names
        (e.g. Anthropic uses 'thinking', UMANS uses 'reasoning').

        Default: pass through as 'reasoning_effort' (OpenAI-compatible).
        """
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        return payload

    def _prepare_payload_with_reasoning(self, payload: dict, reasoning_effort: Optional[str] = None) -> dict:
        """Prepare payload with reasoning effort applied.
        
        This is the main entry point — the router calls this instead of _prepare_payload
        when reasoning_effort is present in the request.
        """
        payload = self._prepare_payload(payload)
        if reasoning_effort:
            payload = self._apply_reasoning_effort(payload, reasoning_effort)
            # Remove the raw reasoning_effort if the provider mapped it to something else
            # (subclasses that rename it should pop the original)
        return payload

    # ── Streaming helpers ──

    @staticmethod
    def _build_usage_chunk(model: str, prompt_tokens: int, completion_tokens: int,
                           reasoning_tokens: int = 0) -> str:
        """Build a final SSE usage chunk for OpenAI streaming compliance."""
        chunk = {
            "id": "chatcmpl-final",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        if reasoning_tokens:
            chunk["usage"]["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
        return f"data: {json.dumps(chunk)}\n\n"

    @staticmethod
    def _build_metrics_line(metrics: ProviderMetrics) -> str:
        """Build the __oct_metrics__ sentinel line for the router to parse."""
        return f"__oct_metrics__:{json.dumps(metrics.to_dict())}\n\n"

    # ── Abstract interface ──

    @abstractmethod
    async def chat_completion(self, payload: dict, api_key: Optional[str] = None) -> dict:
        """Non-streaming chat completion. Returns the provider's JSON response."""
        ...

    @abstractmethod
    async def chat_completion_stream(self, payload: dict, api_key: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Streaming chat completion. Yields SSE-format lines."""
        ...
        yield ""  # type: ignore[unreachable]

    @abstractmethod
    async def list_models(self, api_key: Optional[str] = None) -> list[dict]:
        """List available models from the provider."""
        ...

    @abstractmethod
    async def test_key(self, api_key: Optional[str] = None) -> dict:
        """Test an API key. Returns {'ok': bool, 'error': Optional[str]}."""
        ...

    # ── Optional overrides ──

    def _get_model_capabilities(self, model: str) -> dict:
        """Return capability dict for a model. Override in subclasses."""
        return {
            "supports_vision": self.supports_vision,
            "supports_tools": False,
            "supports_thinking": self.supports_thinking,
            "family": "unknown",
            "usage_multiplier": 1.0,
            "provider": self.provider_name,
        }
