"""Utility helpers for the Guanaco router.

Extracted from router.py for modularity. All functions are re-exported from
``guanaco.router.router`` for backward compatibility.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import time
from typing import Any, Optional

import httpx

from guanaco.router.models import ChatMessage

log = logging.getLogger("guanaco.router")


def _describe_error(exc: Exception) -> str:
    """Return a human-readable description for an exception, handling httpx
    timeout/connect errors whose str() is often empty or unhelpful."""
    if isinstance(exc, httpx.ReadTimeout):
        return "ReadTimeout: server did not respond within timeout"
    if isinstance(exc, httpx.ConnectTimeout):
        return "ConnectTimeout: could not establish connection within timeout"
    if isinstance(exc, httpx.WriteTimeout):
        return "WriteTimeout: could not send data within timeout"
    if isinstance(exc, httpx.PoolTimeout):
        return "PoolTimeout: connection pool exhausted"
    if isinstance(exc, httpx.ConnectError):
        return f"ConnectError: {exc}"
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            body = exc.response.text[:200]
        except Exception:
            body = "(response body not available)"
        return f"HTTP {exc.response.status_code}: {body}"
    msg = str(exc)
    if msg:
        return msg
    return f"{type(exc).__name__}: (no message)"


# ── Empty Response Retry ──

MAX_EMPTY_RETRIES = 1


def _is_empty_non_streaming_response(resp: dict) -> bool:
    """Check if a non-streaming chat completion response has no content."""
    choices = resp.get("choices", [])
    if not choices:
        return True
    for choice in choices:
        msg = choice.get("message", {})
        content = msg.get("content")
        if content and str(content).strip():
            return False
        reasoning = msg.get("reasoning_content")
        if reasoning and str(reasoning).strip():
            return False
        if msg.get("tool_calls"):
            return False
    return True


def _ensure_content_field(resp: dict) -> dict:
    """Copy reasoning_content into content for clients that expect it."""
    for choice in resp.get("choices", []):
        msg = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if content not in (None, ""):
            continue
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if reasoning:
            msg["content"] = reasoning
    return resp


# ── Vision helpers ──

def _has_vision_content(messages: list[ChatMessage]) -> bool:
    """Check if any message contains image/multimodal content."""
    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    return True
    return False


async def _convert_image_urls_to_base64(messages: list) -> list:
    """Download image URLs and convert to base64 data URIs for Ollama Cloud."""
    converted = []
    async with httpx.AsyncClient(timeout=15, follow_redirects=True, headers={"User-Agent": "Guanaco/0.3"}) as img_client:
        for msg in messages:
            if not isinstance(msg.content, list):
                converted.append(msg)
                continue
            new_parts = []
            changed = False
            for part in msg.content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url and url.startswith("http"):
                        try:
                            resp = await img_client.get(url)
                            if resp.status_code == 200:
                                content_type = resp.headers.get("content-type", "")
                                if not content_type or "image" not in content_type:
                                    ext = url.rsplit(".", 1)[-1].split("?")[0].lower()
                                    content_type = mimetypes.guess_type(f"img.{ext}")[0] or "image/png"
                                b64 = base64.b64encode(resp.content).decode("ascii")
                                data_uri = f"data:{content_type};base64,{b64}"
                                new_parts.append({"type": "image_url", "image_url": {"url": data_uri}})
                                changed = True
                            else:
                                log.warning("Failed to download image URL for base64 conversion: HTTP %d for %s", resp.status_code, url[:80])
                                new_parts.append(part)
                        except Exception as e:
                            log.warning("Error downloading image URL for base64 conversion: %s", _describe_error(e))
                            new_parts.append(part)
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)
            if changed:
                converted.append(msg.model_copy(update={"content": new_parts}))
            else:
                converted.append(msg)
    return converted


# ── Model resolution ──

def _resolve_model(model: str, config) -> str:
    """Resolve model name for Ollama Cloud API."""
    normalized = model
    if normalized.endswith(":cloud"):
        normalized = normalized[:-6]
    elif normalized.endswith(":local"):
        normalized = normalized[:-6]
    elif normalized.endswith("-cloud"):
        normalized = normalized[:-6]
    if normalized in config.llm.available_models:
        return normalized
    for available in config.llm.available_models:
        base = available.split(":")[0]
        if normalized == base:
            return available
    return normalized


def _map_model_to_fallback(model: str, fallback_config) -> str:
    """Map an Ollama model name to the corresponding fallback model."""
    if model in fallback_config.model_map:
        return fallback_config.model_map[model]
    base = model.split(":")[0]
    if base in fallback_config.model_map:
        return fallback_config.model_map[base]
    return fallback_config.default_model or model


def _is_quota_full(config) -> bool:
    """Check if Ollama Cloud usage quota is near or at limit (>= 99.5%)."""
    if not config or not config.usage.redirect_on_full:
        return False
    s = config.usage.last_session_pct
    w = config.usage.last_weekly_pct
    if s is not None and s >= 99.5:
        return True
    if w is not None and w >= 99.5:
        return True
    return False


async def _refresh_usage_background(client, config):
    """Background refresh of usage quota so we notice when it resets."""
    try:
        cookie = config.usage.session_cookie
        if not cookie:
            return
        usage = await client.get_usage(session_cookie=cookie)
        if usage.get("source") != "unavailable":
            config.usage.last_session_pct = usage.get("session_pct")
            config.usage.last_weekly_pct = usage.get("weekly_pct")
            config.usage.last_plan = usage.get("plan")
            config.usage.last_session_reset = usage.get("session_reset")
            config.usage.last_weekly_reset = usage.get("weekly_reset")
            config.usage.last_checked = time.time()
            from guanaco.config import save_config
            save_config(config)
            if not _is_quota_full(config):
                log.info("Quota recovered — session=%.1f%%, weekly=%.1f%%, routing back to Ollama",
                         config.usage.last_session_pct or 0, config.usage.last_weekly_pct or 0)
    except Exception as e:
        log.debug("Background usage refresh failed: %s", e)


# ── SSE / streaming helpers ──

def _is_empty_stream_buffer(chunks: list[str]) -> bool:
    """Check if buffered streaming chunks contain no actual content."""
    for chunk in chunks:
        if not chunk.startswith("data: ") or chunk.strip() == "data: [DONE]":
            continue
        try:
            data = json.loads(chunk[6:].strip())
            for choice in data.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content", "")
                reasoning = delta.get("reasoning", "") or delta.get("reasoning_content", "")
                if content and content.strip():
                    return False
                if reasoning and reasoning.strip():
                    return False
                if delta.get("tool_calls"):
                    return False
        except (json.JSONDecodeError, KeyError):
            continue
    return True


def _extract_sse_content(chunk: str) -> str:
    """Extract the content/reasoning text from an SSE data chunk."""
    try:
        if not chunk.startswith("data: ") or "__oct_metrics__" in chunk:
            return ""
        data_str = chunk[6:].strip()
        if data_str == "[DONE]":
            return ""
        data = json.loads(data_str)
        choices = data.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            parts = []
            if delta.get("content"):
                parts.append(delta["content"])
            reasoning = delta.get("reasoning", "") or delta.get("reasoning_content", "")
            if reasoning:
                parts.append(reasoning)
            return "".join(parts)
    except (json.JSONDecodeError, ValueError, KeyError, IndexError):
        pass
    return ""


def _accumulate_history_output(accumulated: list, chunk: str, history_kwargs: dict, config=None):
    """Extract text from an SSE chunk and append to the accumulator."""
    if not history_kwargs or not config or not config.history.enabled or not config.history.save_output:
        return
    text = _extract_sse_content(chunk)
    if text:
        accumulated.append(text)


async def _collect_stream_chunks(client, payload, api_key=None) -> tuple[list[str], dict]:
    """Collect all chunks from a stream into a buffer. Returns (chunks, metrics)."""
    chunks = []
    metrics = {}
    async for chunk in client.chat_completion_stream(payload, api_key=api_key):
        if chunk.startswith("__oct_metrics__:"):
            try:
                metrics = json.loads(chunk.split(":", 1)[1])
            except (json.JSONDecodeError, ValueError):
                pass
            continue
        chunks.append(chunk)
    return chunks, metrics


def _openai_to_anthropic_stop(reason: str) -> str:
    """Convert OpenAI finish_reason to Anthropic stop_reason."""
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "end_turn",
    }
    return mapping.get(reason, "end_turn")


def _build_history_kwargs(request, config, input_text: str = None, output_text: str = None) -> dict:
    """Build history kwargs from request context."""
    kwargs = {}
    if config and config.history.enabled:
        kwargs["source_ip"] = request.client.host if request.client else ""
        kwargs["user_agent"] = request.headers.get("user-agent", "")
        kwargs["input_text"] = input_text
        kwargs["output_text"] = output_text
    return kwargs
