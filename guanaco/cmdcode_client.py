"""Command Code Go API client — direct integration (no external proxy needed).

Command Code (commandcode.ai) offers a $1/mo Go plan with CLI access to 20+ models.
The Go plan does NOT include the official OpenAI API endpoint ($15/mo Provider plan
required), but the CLI's internal /alpha/generate endpoint can be called directly
with the right headers and body structure.

This client talks DIRECTLY to https://api.commandcode.ai/alpha/generate, handling:
  - CLI header mimicry (x-session-id, x-command-code-version, x-cmd-zdr, etc.)
  - OpenAI → Command Code request body translation (memory, params, config fields)
  - SSE response translation (text-delta/reasoning-delta/finish → OpenAI chunks)
  - Zero Data Retention (ZDR) mode via x-cmd-zdr header

No external proxy process is needed — everything is self-contained in this client,
matching the architecture of ClinePassClient, UmansClient, and OpenCodeGoClient.

Auth: Bearer <API_KEY> (user_... prefix, from ~/.commandcode/auth.json or env var)
Models: 20+ open-weight models, zero per-token cost ($1/mo flat rate)
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import time
import uuid
from typing import Any, AsyncGenerator, Optional

import httpx

from guanaco.providers.base import BaseProvider, ProviderMetrics

logger = logging.getLogger(__name__)

CMDCODE_API_BASE = "https://api.commandcode.ai"
CMDCODE_GENERATE_URL = f"{CMDCODE_API_BASE}/alpha/generate"
CMDCODE_USAGE_URL = f"{CMDCODE_API_BASE}/alpha/usage/summary"
CMDCODE_CREDITS_URL = f"{CMDCODE_API_BASE}/alpha/billing/credits"
CMDCODE_SUBSCRIPTION_URL = f"{CMDCODE_API_BASE}/alpha/billing/subscriptions"
CMDCODE_CLI_VERSION = "0.44.1"

# Static model list — Command Code Go plan offers 20+ models with ZDR support.
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

# Model name mapping: OpenAI-style short names → Command Code full model IDs
MODEL_MAP: dict[str, str] = {
    "deepseek-v4-pro": "deepseek/deepseek-v4-pro",
    "deepseek-v4-flash": "deepseek/deepseek-v4-flash",
    "kimi-k2.7-code": "moonshotai/Kimi-K2.7-Code",
    "kimi-k2.7-code-highspeed": "moonshotai/Kimi-K2.7-Code-Highspeed",
    "kimi-k2.6": "moonshotai/Kimi-K2.6",
    "kimi-k2.5": "moonshotai/Kimi-K2.5",
    "glm-5.2": "zai-org/GLM-5.2",
    "glm-5.2-fast": "zai-org/GLM-5.2-Fast",
    "glm-5.1": "zai-org/GLM-5.1",
    "glm-5": "zai-org/GLM-5",
    "minimax-m3": "MiniMaxAI/MiniMax-M3",
    "minimax-m2.7": "MiniMaxAI/MiniMax-M2.7",
    "minimax-m2.5": "MiniMaxAI/MiniMax-M2.5",
    "mimo-v2.5-pro": "xiaomi/mimo-v2.5-pro",
    "mimo-v2.5": "xiaomi/mimo-v2.5",
    "qwen3.6-plus": "Qwen/Qwen3.6-Plus",
    "qwen3.7-plus": "Qwen/Qwen3.7-Plus",
    "tencent-hy3": "tencent/Hy3",
    "nemotron-3-ultra": "nvidia/Nemotron-3-Ultra",
    "step-3.5-flash": "stepfun/Step-3.5-Flash",
}


def _strip_cmdcode_prefix(model: str) -> str:
    """Return the model id without the cmdcode/ prefix."""
    model = model.strip()
    lower = model.lower()
    if lower.startswith("cmdcode/"):
        model = model[len("cmdcode/"):]
    return model


def _resolve_model(model_id: str) -> str:
    """Resolve short model name to full Command Code model ID."""
    if model_id in MODEL_MAP:
        return MODEL_MAP[model_id]
    if "/" in model_id:
        return model_id  # already a full ID
    for k, v in MODEL_MAP.items():
        if k.lower() == model_id.lower():
            return v
    return model_id  # pass through


class CmdCodeClient(BaseProvider):
    """Async client for Command Code Go plan — direct API integration.

    Calls https://api.commandcode.ai/alpha/generate directly, translating
    OpenAI chat completion requests to Command Code's internal format and
    translating the SSE response back to OpenAI-compatible chunks.

    No external proxy process is needed. This is self-contained, matching
    the architecture of ClinePassClient, UmansClient, and OpenCodeGoClient.
    """

    provider_name = "cmdcode"
    supports_streaming = True
    supports_vision = False
    supports_thinking = True

    def __init__(self, api_key: str = "", timeout: float = 300.0, base_url: str = ""):
        # base_url is accepted for config compatibility but ignored — we always
        # talk directly to api.commandcode.ai
        super().__init__(api_key=api_key, timeout=timeout, base_url=CMDCODE_API_BASE)

    # ── Header / body builders ──

    def _build_headers(self) -> dict[str, str]:
        """Build headers that mimic the Command Code CLI."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "x-command-code-version": CMDCODE_CLI_VERSION,
            "x-cli-environment": "cli",
            "x-session-id": str(uuid.uuid4()),
            "x-project-slug": "command-code",
            "x-taste-learning": "false",
            "x-taste-usage": "false",
            "x-cmd-zdr": "1",  # Zero Data Retention
            "User-Agent": "cli",
            "Accept": "application/json",
        }

    def _build_generate_body(self, openai_request: dict) -> dict:
        """Convert an OpenAI chat completion request to Command Code /alpha/generate format."""
        messages = openai_request.get("messages", [])
        model = _resolve_model(openai_request.get("model", "deepseek/deepseek-v4-flash"))
        max_tokens = openai_request.get("max_tokens", 64000)

        # Build system prompt from system messages
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        system_text = "\n".join(system_parts) if system_parts else "You are a helpful assistant."

        # Non-system messages
        conv_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages if m.get("role") != "system"
        ]
        if not conv_messages:
            conv_messages = [{"role": "user", "content": ""}]

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        return {
            "model": model,
            "messages": conv_messages,
            "max_tokens": max_tokens,
            "stream": True,  # ALWAYS stream from backend — collect for non-streaming clients
            "memory": "",
            "params": {
                "model": model,
                "messages": conv_messages,
                "tools": [],
                "system": system_text,
                "max_tokens": max_tokens,
                "stream": True,
            },
            "config": {
                "workingDir": os.getcwd(),
                "date": now,
                "environment": "cli",
                "structure": [],
                "isGitRepo": False,
                "currentBranch": "",
                "mainBranch": "",
                "gitStatus": "",
                "recentCommits": [],
            },
            "taste": None,
            "skills": None,
        }

    @staticmethod
    def _parse_cc_sse_line(line: str) -> dict | None:
        """Parse a single Command Code SSE line into an event dict.

        CC sends newline-delimited JSON objects (not SSE data: prefix):
        {"type":"text-delta","text":"..."}
        {"type":"reasoning-delta","text":"..."}
        {"type":"finish","finishReason":"stop","usage":{...}}
        {"type":"error","error":{"message":"..."}}
        """
        line = line.strip()
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _make_openai_chunk(
        model: str, content: str = "", reasoning: str = "", finish_reason: str | None = None,
    ) -> str:
        """Build an OpenAI-compatible streaming chunk (SSE format)."""
        delta: dict[str, Any] = {}
        if reasoning:
            delta["reasoning_content"] = reasoning
        if content:
            delta["content"] = content
        if finish_reason:
            delta = {}

        chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    @staticmethod
    def _make_openai_response(model: str, content: str, reasoning: str, usage: dict, finish_reason: str) -> dict:
        """Build an OpenAI-compatible non-streaming response."""
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": usage.get("promptTokens", 0),
                "completion_tokens": usage.get("completionTokens", 0),
                "total_tokens": usage.get("totalTokens", 0),
            } if usage else {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    # ── Model listing ──

    async def list_models(self, force_refresh: bool = False, api_key: Optional[str] = None) -> list[dict]:
        """List available Command Code models.

        Command Code doesn't have a /models endpoint on /alpha/generate, so we
        always return the static list. This matches the Cline pattern where
        the static list is the source of truth for capability hints.
        """
        now = time.time()
        if not force_refresh and not api_key and self._models_cache and (now - self._models_cache_time) < self._models_cache_ttl:
            return self._models_cache

        models = self._static_models()
        self._models_cache = models
        self._models_cache_time = now
        return models

    def _static_models(self) -> list[dict]:
        """Return static model list."""
        return [
            {"id": mid, "name": mid, "model": mid, "display_name": mid, "details": {}}
            for mid in CMDCODE_MODELS
        ]

    async def test_key(self, api_key: Optional[str] = None) -> dict:
        """Test the API key by making a minimal generate request."""
        key = api_key or self.api_key
        if not key:
            return {"ok": False, "error": "No Command Code API key configured"}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "x-command-code-version": CMDCODE_CLI_VERSION,
            "x-cli-environment": "cli",
            "x-session-id": str(uuid.uuid4()),
            "x-project-slug": "command-code",
            "x-taste-learning": "false",
            "x-taste-usage": "false",
            "x-cmd-zdr": "1",
            "User-Agent": "cli",
            "Accept": "application/json",
        }
        body = self._build_generate_body({
            "model": "deepseek/deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1,
        })
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(CMDCODE_GENERATE_URL, json=body, headers=headers)
                if resp.status_code == 200:
                    return {"ok": True, "error": None, "model_count": len(CMDCODE_MODELS)}
                if resp.status_code == 401:
                    return {"ok": False, "error": "Invalid or expired Command Code API key"}
                return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        except Exception as e:
            logger.warning("CmdCode key test failed: %s", e)
            return {"ok": False, "error": str(e)[:200]}

    # ── Usage / Billing ──

    async def fetch_usage(self) -> dict:
        """Fetch monthly usage summary from Command Code.

        Returns:
            {
                "total_requests": int,
                "completed": int,
                "failed": int,
                "success_rate": float,
                "tokens_in": int,
                "tokens_out": int,
                "total_tokens": int,
                "credits_used": float,
                "monthly_credits_used": float,
                "remaining_credits": float,
                "five_hour_used": float,
                "five_hour_cap": float,
                "weekly_used": float,
                "weekly_cap": float,
                "weekly_reset_at": str | None,
                "plan": str | None,
                "period_start": str | None,
                "period_end": str | None,
                "subscription_status": str | None,
            }
        """
        if not self.api_key:
            return {}
        headers = self._build_headers()
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                # Fetch usage summary, credits, and subscription in parallel
                usage_resp, credits_resp, sub_resp = await asyncio.gather(
                    client.get(CMDCODE_USAGE_URL, headers=headers),
                    client.get(CMDCODE_CREDITS_URL, headers=headers),
                    client.get(CMDCODE_SUBSCRIPTION_URL, headers=headers),
                    return_exceptions=True,
                )

            result: dict[str, Any] = {}

            if isinstance(usage_resp, httpx.Response) and usage_resp.status_code == 200:
                data = usage_resp.json()
                result.update({
                    "total_requests": data.get("totalCount", 0),
                    "completed": data.get("completedCount", 0),
                    "failed": data.get("failedCount", 0),
                    "success_rate": data.get("successRate", 0),
                    "tokens_in": data.get("totalTokensIn", 0),
                    "tokens_out": data.get("totalTokensOut", 0),
                    "total_tokens": data.get("totalTokens", 0),
                    "credits_used": data.get("totalCredits", 0),
                    "monthly_credits_used": data.get("totalMonthlyCredits", 0),
                })

            if isinstance(credits_resp, httpx.Response) and credits_resp.status_code == 200:
                data = credits_resp.json()
                credits = data.get("credits", {})
                window = data.get("windowLimits", {})
                five_hour = window.get("fiveHour", {})
                weekly = window.get("weekly", {})
                result.update({
                    "remaining_credits": credits.get("monthlyCredits", 0),
                    "five_hour_used": five_hour.get("used", 0),
                    "five_hour_cap": five_hour.get("cap", 0),
                    "weekly_used": weekly.get("used", 0),
                    "weekly_cap": weekly.get("cap", 0),
                    "weekly_reset_at": (
                        datetime.datetime.fromtimestamp(
                            weekly.get("resetAt", 0) / 1000,
                            tz=datetime.timezone.utc
                        ).isoformat() if weekly.get("resetAt") else None
                    ),
                })

            if isinstance(sub_resp, httpx.Response) and sub_resp.status_code == 200:
                data = sub_resp.json()
                sub = data.get("data", {})
                result.update({
                    "plan": sub.get("planId"),
                    "period_start": sub.get("currentPeriodStart"),
                    "period_end": sub.get("currentPeriodEnd"),
                    "subscription_status": sub.get("status"),
                })

            return result
        except Exception as e:
            logger.warning("CmdCode usage fetch failed: %s", e)
            return {}

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
        """Non-streaming chat completion.

        Always streams from the Command Code backend, collects the full response,
        then returns a single OpenAI-compatible JSON response.
        """
        payload = self._prepare_payload(payload)
        client_model = payload.get("model", "")
        cc_body = self._build_generate_body(payload)

        key = api_key or self.api_key
        headers = self._build_headers()
        if api_key and api_key != self.api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        start = time.time()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict = {}
        finish_reason = "stop"

        async with httpx.AsyncClient(timeout=self.timeout) as http_client:
            async with http_client.stream("POST", CMDCODE_GENERATE_URL, json=cc_body, headers=headers) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    event = self._parse_cc_sse_line(line)
                    if event is None:
                        continue
                    etype = event.get("type", "")
                    if etype == "text-delta":
                        content_parts.append(event.get("text", ""))
                    elif etype == "reasoning-delta":
                        reasoning_parts.append(event.get("text", ""))
                    elif etype == "finish":
                        finish_reason = event.get("finishReason", "stop")
                        u = event.get("usage", {})
                        if u:
                            usage = u
                    elif etype == "error":
                        msg = event.get("error", {}).get("message", "unknown error")
                        raise httpx.HTTPStatusError(
                            f"Command Code error: {msg}",
                            request=resp.request,
                            response=resp,
                        )

        elapsed = time.time() - start
        result = self._make_openai_response(
            model=client_model,
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            usage=usage,
            finish_reason=finish_reason,
        )

        u = result.get("usage", {})
        metrics = {
            "elapsed_seconds": elapsed,
            "prompt_eval_count": u.get("prompt_tokens", 0),
            "eval_count": u.get("completion_tokens", 0),
        }
        if metrics["eval_count"] and elapsed > 0:
            metrics["tps"] = round(min(metrics["eval_count"] / elapsed, 1000.0), 2)
        if elapsed > 0:
            metrics["ttft_seconds"] = round(elapsed, 3)
        result["_oct_metrics"] = metrics
        return result

    async def chat_completion_stream(self, payload: dict, api_key: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Streaming chat completion.

        Translates Command Code's SSE format (text-delta/reasoning-delta/finish
        JSON events) into OpenAI-compatible SSE chunks (data: {chat.completion.chunk}).
        """
        payload = self._prepare_payload(payload)
        client_model = payload.get("model", "")
        cc_body = self._build_generate_body(payload)

        key = api_key or self.api_key
        headers = self._build_headers()
        if api_key and api_key != self.api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        first_token_time: Optional[float] = None
        content_chars = 0
        reasoning_chars = 0
        prompt_tokens = 0
        completion_tokens = 0
        start = time.time()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as http_client:
                async with http_client.stream("POST", CMDCODE_GENERATE_URL, json=cc_body, headers=headers) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        event = self._parse_cc_sse_line(line)
                        if event is None:
                            continue

                        etype = event.get("type", "")

                        if etype == "text-delta":
                            text = event.get("text", "")
                            if text:
                                content_chars += len(text)
                                if not first_token_time:
                                    first_token_time = time.time()
                                yield self._make_openai_chunk(client_model, content=text)

                        elif etype == "reasoning-delta":
                            text = event.get("text", "")
                            if text:
                                reasoning_chars += len(text)
                                if not first_token_time:
                                    first_token_time = time.time()
                                yield self._make_openai_chunk(client_model, reasoning=text)

                        elif etype == "finish":
                            finish_reason = event.get("finishReason", "stop")
                            u = event.get("usage", {})
                            if u:
                                prompt_tokens = u.get("promptTokens", prompt_tokens)
                                completion_tokens = u.get("completionTokens", completion_tokens)
                            # Send final chunk with finish_reason
                            yield self._make_openai_chunk(client_model, finish_reason=finish_reason)

                        elif etype == "error":
                            msg = event.get("error", {}).get("message", "unknown error")
                            logger.error("Command Code stream error: %s", msg)
                            # Send error as content then stop
                            yield self._make_openai_chunk(client_model, content=f"[Error: {msg}]")
                            yield self._make_openai_chunk(client_model, finish_reason="error")

                    # Compute metrics
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
                        client_model,
                        metrics.get("prompt_eval_count", 0),
                        metrics.get("eval_count", 0),
                        metrics.get("reasoning_tokens", 0),
                    )
                    yield "data: [DONE]\n\n"
                    yield f"__oct_metrics__:{json.dumps(metrics)}\n\n"
        except httpx.HTTPStatusError as e:
            logger.error("Command Code HTTP error: %s", e)
            yield self._make_openai_chunk(client_model, content=f"[HTTP Error: {e.response.status_code}]")
            yield self._make_openai_chunk(client_model, finish_reason="error")
            yield "data: [DONE]\n\n"
            metrics = {
                "eval_count": 0,
                "prompt_eval_count": 0,
                "reasoning_tokens": 0,
                "elapsed_seconds": round(time.time() - start, 3),
                "ttft_seconds": None,
            }
            yield f"__oct_metrics__:{json.dumps(metrics)}\n\n"