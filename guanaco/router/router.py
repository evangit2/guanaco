"""OpenAI-compatible and Anthropic-compatible LLM router with usage tracking, analytics, and fallback."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Optional

import httpx
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field


def _describe_error(exc: Exception) -> str:
    """Return a human-readable description for an exception, handling httpx
    timeout/connect errors whose str() is often empty or unhelpful."""
    if isinstance(exc, httpx.ReadTimeout):
        return f"ReadTimeout: server did not respond within timeout"
    if isinstance(exc, httpx.ConnectTimeout):
        return f"ConnectTimeout: could not establish connection within timeout"
    if isinstance(exc, httpx.WriteTimeout):
        return f"WriteTimeout: could not send data within timeout"
    if isinstance(exc, httpx.PoolTimeout):
        return f"PoolTimeout: connection pool exhausted"
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
    # Fallback: use the exception class name if str() is empty
    return f"{type(exc).__name__}: (no message)"


async def _ollama_chat_with_primary_timeout(client, payload, fallback_config=None):
    """Call Ollama Cloud chat completion with a primary timeout.

    When fallback is configured, we use a shorter primary_timeout so that
    slow/unresponsive Ollama responses trigger fallback quickly instead of
    hanging for the full 120s client timeout.
    """
    if fallback_config and fallback_config.enabled and fallback_config.primary_timeout:
        try:
            return await asyncio.wait_for(
                client.chat_completion(payload),
                timeout=fallback_config.primary_timeout,
            )
        except asyncio.TimeoutError:
            raise httpx.ReadTimeout(
                f"Ollama did not respond within {fallback_config.primary_timeout}s primary timeout"
            )
    return await client.chat_completion(payload)

from guanaco.client import OllamaClient
from guanaco.cache import CacheEngine
from guanaco.analytics import _normalize_model_name

import logging

log = logging.getLogger("guanaco.router")

# ── Empty Response Retry ──
MAX_EMPTY_RETRIES = 1  # How many times to retry on empty responses


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
        # Some models (GLM) put output in reasoning_content while content is empty
        reasoning = msg.get("reasoning_content")
        if reasoning and str(reasoning).strip():
            return False
        # Check for tool_calls — those count as non-empty
        if msg.get("tool_calls"):
            return False
    return True


# ── Request/Response Models ──

class ChatMessage(BaseModel):
    role: str
    content: str | list | None = None
    name: Optional[str] = None
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None


def _has_vision_content(messages: list[ChatMessage]) -> bool:
    """Check if any message contains image/multimodal content that requires a vision-capable model."""
    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    return True
    return False


async def _convert_image_urls_to_base64(messages: list) -> list:
    """Download image URLs and convert to base64 data URIs for Ollama Cloud compatibility.
    
    Ollama Cloud doesn't support image URLs — it requires base64-encoded data URIs.
    This transforms {"type": "image_url", "image_url": {"url": "https://..."}} 
    into {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    """
    import base64
    import mimetypes
    
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
                        # Download and convert to base64
                        try:
                            resp = await img_client.get(url)
                            if resp.status_code == 200:
                                content_type = resp.headers.get("content-type", "")
                                if not content_type or "image" not in content_type:
                                    # Guess from URL extension
                                    ext = url.rsplit(".", 1)[-1].split("?")[0].lower()
                                    content_type = mimetypes.guess_type(f"img.{ext}")[0] or "image/png"
                                b64 = base64.b64encode(resp.content).decode("ascii")
                                data_uri = f"data:{content_type};base64,{b64}"
                                new_parts.append({
                                    "type": "image_url",
                                    "image_url": {"url": data_uri}
                                })
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
                new_msg = msg.model_copy(update={"content": new_parts})
                converted.append(new_msg)
            else:
                converted.append(msg)
    
    return converted


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[list[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[str | dict] = None
    response_format: Optional[dict] = None


# ── Anthropic Request Models ──

class AnthropicMessage(BaseModel):
    role: str
    content: str | list


class AnthropicRequest(BaseModel):
    model: str
    max_tokens: int = 4096
    messages: list[AnthropicMessage]
    system: Optional[str | list] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    stop_sequences: Optional[list[str]] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[dict] = None


def _resolve_model(model: str, config) -> str:
    """Resolve model name for Ollama Cloud API."""
    normalized = model
    # Strip routing suffixes used by Hermes/clients: :cloud, :local, -cloud
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


async def _call_fallback_provider(payload: dict, fallback_config, stream: bool = False):
    """Send a request to the fallback OpenAI-compatible provider."""
    base_url = fallback_config.base_url.rstrip("/")
    # Strip /chat/completions if user accidentally included the full path
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    if fallback_config.api_key:
        headers["Authorization"] = f"Bearer {fallback_config.api_key}"

    # Inject fallback max_tokens if not already set in the payload
    if fallback_config.max_tokens and "max_tokens" not in payload:
        payload = dict(payload)
        payload["max_tokens"] = fallback_config.max_tokens

    timeout = fallback_config.timeout or 60.0
    # For streaming, use a long connect timeout but generous read timeout for thinking models
    connect_timeout = min(timeout, 30.0)
    read_timeout = max(timeout, 120.0)

    if stream:
        # Streaming: use a long-lived client that stays open while the generator is consumed
        # Use generous read timeout for thinking models that can pause mid-stream
        client_timeout = httpx.Timeout(connect=connect_timeout, read=read_timeout, write=30.0, pool=30.0)
        client = httpx.AsyncClient(timeout=client_timeout)

        async def stream_from_fallback():
            try:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        yield line + "\n"
            finally:
                await client.aclose()

        return stream_from_fallback()
    else:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()


# ── Provider creation ──

def create_router(client: OllamaClient, analytics=None, config=None) -> APIRouter:
    router = APIRouter(tags=["LLM Router"])
    _analytics = analytics
    _config = config
    _cache = CacheEngine(config.cache) if config else None

    # ── OpenAI-compatible endpoints ──

    @router.get("/v1/models")
    async def list_models(request: Request):
        """List available models by querying Ollama Cloud dynamically."""
        try:
            models = await client.list_models()
            data = []
            for m in models:
                name = m.get("name", m.get("model", ""))
                display_name = name.replace("-cloud", "") if name.endswith("-cloud") else name
                details = m.get("details", {})
                data.append({
                    "id": display_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ollama",
                    "permission": [],
                    "root": display_name,
                    "parent": None,
                    "capabilities": client._get_model_capabilities(name),
                    "details": {
                        "parameter_size": details.get("parameter_size", ""),
                        "quantization": details.get("quantization_level", ""),
                        "family": details.get("family", ""),
                    },
                })
            # Add fallback provider models if configured
            if _config and _config.fallback.enabled and _config.fallback.default_model:
                fallback_models = set(_config.fallback.model_map.values())
                if _config.fallback.default_model:
                    fallback_models.add(_config.fallback.default_model)
                for fm in fallback_models:
                    if fm and not any(d["id"] == fm for d in data):
                        data.append({
                            "id": fm,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "fallback",
                            "permission": [],
                            "root": fm,
                            "parent": None,
                            "details": {"family": "fallback"},
                        })
            return {"object": "list", "data": data}
        except Exception as e:
            if _config:
                data = [
                    {"id": name, "object": "model", "created": int(time.time()), "owned_by": "ollama"}
                    for name in _config.llm.available_models
                ]
                return {"object": "list", "data": data}
            raise HTTPException(status_code=502, detail=f"Cannot reach Ollama Cloud: {str(e)}")

    @router.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        """OpenAI-compatible chat completions endpoint with fallback and smart caching (beta)."""
        start = time.time()
        resolved_model = _resolve_model(body.model, _config) if _config else body.model
        
        # Convert image URLs to base64 for Ollama Cloud compatibility
        if _has_vision_content(body.messages):
            body.messages = await _convert_image_urls_to_base64(body.messages)
            log.info("Converted image URLs to base64 for vision request on model %s", resolved_model)
        
        payload = body.model_dump(exclude_none=True)
        payload["model"] = resolved_model

        # ── Quota-full redirect: skip Ollama entirely, go straight to fallback ──
        # Exception: vision requests should skip fallback if the provider doesn't support them
        vision_request = _has_vision_content(body.messages)
        skip_fallback_for_vision = vision_request and _config.fallback.enabled and not _config.fallback.supports_vision
        if _is_quota_full(_config) and not skip_fallback_for_vision:
            if _config.fallback.enabled and _config.fallback.base_url:
                fallback_model = _map_model_to_fallback(resolved_model, _config.fallback)
                log.info("Quota full (session=%.1f%%, weekly=%.1f%%), redirecting %s to fallback %s (model: %s)",
                         _config.usage.last_session_pct or 0, _config.usage.last_weekly_pct or 0,
                         resolved_model, _config.fallback.name, fallback_model)
                # Refresh quota in background so we notice when it resets
                asyncio.ensure_future(_refresh_usage_background(client, _config))
                # Skip Ollama, go straight to fallback
                payload_fb = dict(payload)
                payload_fb["model"] = fallback_model
                if body.stream:
                    if _config.fallback.stream_fallback:
                        fallback_payload = dict(payload)
                        fallback_payload["model"] = fallback_model
                        return StreamingResponse(
                            await _call_fallback_provider(fallback_payload, _config.fallback, stream=True),
                            media_type="text/event-stream",
                        )
                    # Can't stream from fallback — do non-streaming
                    try:
                        fallback_resp = await _call_fallback_provider(payload_fb, _config.fallback)
                        return fallback_resp
                    except Exception as e:
                        raise HTTPException(status_code=502, detail=f"Quota full and fallback error: {_describe_error(e)}")
                else:
                    try:
                        fallback_resp = await _call_fallback_provider(payload_fb, _config.fallback)
                        # Normalize fallback response
                        if isinstance(fallback_resp, dict) and "choices" in fallback_resp:
                            for choice in fallback_resp.get("choices", []):
                                if isinstance(choice, dict) and "message" in choice:
                                    msg = choice["message"]
                                    if isinstance(msg, str):
                                        choice["message"] = {"role": "assistant", "content": msg}
                        fallback_resp["_oct_fallback"] = True
                        fallback_resp["_oct_fallback_provider"] = _config.fallback.name
                        fallback_resp["_oct_original_model"] = _normalize_model_name(resolved_model)
                        fallback_resp["_oct_quota_redirect"] = True
                        if _analytics:
                            _analytics.log_llm(
                                model=_normalize_model_name(fallback_model),
                                total_duration_seconds=time.time() - start,
                                provider=_config.fallback.name,
                                fallback_for=_normalize_model_name(resolved_model),
                            )
                        return fallback_resp
                    except Exception as e:
                        raise HTTPException(status_code=502, detail=f"Quota full and fallback error: {_describe_error(e)}")
            # No fallback configured — let it hit Ollama and probably get rate-limited

        # ── Smart Cache (beta) for non-streaming requests ──
        if _cache and _cache.is_enabled() and not body.stream:
            async def _fetch_from_upstream(p: dict) -> dict:
                """Fetch from Ollama Cloud with fallback, retrying on empty response."""
                try:
                    # ── Retry on empty response ──
                    for attempt in range(MAX_EMPTY_RETRIES + 1):
                        resp = await _ollama_chat_with_primary_timeout(client, p, _config.fallback if _config else None)
                        if not _is_empty_non_streaming_response(resp) or attempt == MAX_EMPTY_RETRIES:
                            break
                        log.warning("Empty cached-response from %s (attempt %d/%d), retrying...", resolved_model, attempt + 1, MAX_EMPTY_RETRIES + 1)

                    elapsed = time.time() - start
                    metrics = resp.pop("_oct_metrics", {})
                    usage = resp.get("usage", {})

                    if _analytics:
                        _analytics.log_llm(
                            model=resolved_model,
                            prompt_tokens=usage.get("prompt_tokens", metrics.get("prompt_eval_count", 0)),
                            completion_tokens=usage.get("completion_tokens", metrics.get("eval_count", 0)),
                            total_tokens=usage.get("total_tokens", 0),
                            tps=metrics.get("tps"),
                            prompt_tps=metrics.get("prompt_tps"),
                            ttft_seconds=metrics.get("ttft_seconds"),
                            total_duration_seconds=elapsed,
                            load_duration_seconds=metrics.get("load_duration_ns", 0) / 1e9 if metrics.get("load_duration_ns") else None,
                            provider="ollama",
                        )

                    return resp

                except Exception as ollama_error:
                    # Try fallback provider if configured
                    if _config and _config.fallback.enabled and _config.fallback.base_url:
                        fallback_model = _map_model_to_fallback(resolved_model, _config.fallback)
                        log.info("Ollama error for %s (cached path), trying fallback %s (model: %s)", resolved_model, _config.fallback.name, fallback_model)
                        fallback_payload = dict(p)
                        fallback_payload["model"] = fallback_model

                        try:
                            fallback_resp = await _call_fallback_provider(fallback_payload, _config.fallback)
                            elapsed = time.time() - start

                            # Normalize fallback response: ensure choices[].message is a dict
                            if isinstance(fallback_resp, dict) and "choices" in fallback_resp:
                                for choice in fallback_resp.get("choices", []):
                                    if isinstance(choice, dict) and "message" in choice:
                                        msg = choice["message"]
                                        if isinstance(msg, str):
                                            choice["message"] = {"role": "assistant", "content": msg}

                            if _analytics:
                                _analytics.log_llm(
                                    model=_normalize_model_name(fallback_model),
                                    total_duration_seconds=elapsed,
                                    provider=_config.fallback.name,
                                    fallback_for=_normalize_model_name(resolved_model),
                                )

                            fallback_resp["_oct_fallback"] = True
                            fallback_resp["_oct_fallback_provider"] = _config.fallback.name
                            fallback_resp["_oct_original_model"] = _normalize_model_name(resolved_model)
                            return fallback_resp

                        except Exception as fallback_err:
                            log.warning("Fallback to %s failed for model %s (cached path): %s", _config.fallback.name, resolved_model, _describe_error(fallback_err))
                            if _analytics:
                                _analytics.log_llm(model=resolved_model, error=f"ollama: {_describe_error(ollama_error)}; fallback: {_describe_error(fallback_err)}", total_duration_seconds=time.time() - start)
                            raise HTTPException(status_code=502, detail=f"Ollama Cloud error: {_describe_error(ollama_error)}; Fallback error: {_describe_error(fallback_err)}")

                    if _analytics:
                        _analytics.log_llm(model=resolved_model, error=str(ollama_error), total_duration_seconds=time.time() - start)
                    raise HTTPException(status_code=502, detail=f"Ollama Cloud error: {str(ollama_error)}")

            # Use cache for non-streaming
            response = await _cache.get_or_fetch(
                model=resolved_model,
                messages=[m.model_dump(exclude_none=True) for m in body.messages],
                params=payload,
                fetch_fn=_fetch_from_upstream,
                provider="ollama",
            )

            # Log cache metadata in analytics
            if response.get("_oct_cached"):
                elapsed = time.time() - start
                if _analytics:
                    _analytics.log_llm(
                        model=resolved_model,
                        total_duration_seconds=elapsed,
                        provider=f"cache:{response.get('_oct_cache_type', 'unknown')}",
                    )

            return response

        # ── Original path for streaming or cache disabled ──
        # Try Ollama Cloud first
        try:
            if body.stream:
                return await _stream_completion_openai(client, payload, resolved_model, _analytics, start, _config)

            # ── Non-streaming: retry on empty response ──
            for attempt in range(MAX_EMPTY_RETRIES + 1):
                resp = await _ollama_chat_with_primary_timeout(client, payload, _config.fallback if _config else None)
                if not _is_empty_non_streaming_response(resp) or attempt == MAX_EMPTY_RETRIES:
                    break
                log.warning("Empty response from %s (attempt %d/%d), retrying...", resolved_model, attempt + 1, MAX_EMPTY_RETRIES + 1)

            elapsed = time.time() - start
            metrics = resp.pop("_oct_metrics", {})
            usage = resp.get("usage", {})

            if _analytics:
                _analytics.log_llm(
                    model=resolved_model,
                    prompt_tokens=usage.get("prompt_tokens", metrics.get("prompt_eval_count", 0)),
                    completion_tokens=usage.get("completion_tokens", metrics.get("eval_count", 0)),
                    total_tokens=usage.get("total_tokens", 0),
                    tps=metrics.get("tps"),
                    prompt_tps=metrics.get("prompt_tps"),
                    ttft_seconds=metrics.get("ttft_seconds"),
                    total_duration_seconds=elapsed,
                    load_duration_seconds=metrics.get("load_duration_ns", 0) / 1e9 if metrics.get("load_duration_ns") else None,
                    provider="ollama",
                )

            return resp

        except Exception as ollama_error:
            # Try fallback provider if configured
            if _config and _config.fallback.enabled and _config.fallback.base_url:
                fallback_model = _map_model_to_fallback(resolved_model, _config.fallback)
                log.info("Ollama error for %s, trying fallback %s (model: %s)", resolved_model, _config.fallback.name, fallback_model)
                fallback_payload = dict(payload)
                fallback_payload["model"] = fallback_model

                try:
                    if body.stream and _config.fallback.stream_fallback:
                        return await _stream_fallback_openai(fallback_payload, _config, fallback_model, _analytics, start, "ollama_fallback")

                    fallback_resp = await _call_fallback_provider(fallback_payload, _config.fallback)
                    elapsed = time.time() - start

                    if _analytics:
                        _analytics.log_llm(
                            model=fallback_model,
                            total_duration_seconds=elapsed,
                            provider=_config.fallback.name, fallback_for=resolved_model,
                        )

                    # Tag response so dashboard can show it came from fallback
                    fallback_resp["_oct_fallback"] = True
                    fallback_resp["_oct_fallback_provider"] = _config.fallback.name
                    fallback_resp["_oct_original_model"] = resolved_model
                    return fallback_resp

                except Exception as fallback_err:
                    log.warning("Fallback to %s failed for model %s: %s", _config.fallback.name, resolved_model, _describe_error(fallback_err))
                    if _analytics:
                        _analytics.log_llm(model=resolved_model, error=f"ollama: {_describe_error(ollama_error)}; fallback: {_describe_error(fallback_err)}", total_duration_seconds=time.time() - start)
                    raise HTTPException(status_code=502, detail=f"Ollama Cloud error: {_describe_error(ollama_error)}; Fallback error: {_describe_error(fallback_err)}")

            if _analytics:
                _analytics.log_llm(model=resolved_model, error=_describe_error(ollama_error), total_duration_seconds=time.time() - start)
            raise HTTPException(status_code=502, detail=f"Ollama Cloud error: {_describe_error(ollama_error)}")

    @router.post("/v1/chat/completions/refresh_models")
    async def refresh_models(request: Request):
        """Force-refresh the model list cache."""
        try:
            models = await client.list_models(force_refresh=True)
            return {"status": "ok", "model_count": len(models)}
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Cannot refresh models: {str(e)}")

    @router.get("/v1/usage")
    async def get_usage(request: Request):
        """Get Ollama Cloud account usage/quota information."""
        try:
            config = get_config()
            session_cookie = config.usage.session_cookie
            usage_data = await client.get_usage(session_cookie=session_cookie)
            # Persist last-known values for dashboard display
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

    @router.get("/v1/analytics")
    async def get_analytics(request: Request):
        """Get local analytics summary."""
        if _analytics:
            return _analytics.get_summary()
        return {"total_requests": 0}

    @router.post("/v1/analytics/reset")
    async def reset_analytics(request: Request):
        """Reset local analytics data."""
        if _analytics:
            _analytics.clear()
        return {"status": "ok"}

    # ── Anthropic-compatible endpoints ──

    @router.post("/v1/messages")
    async def anthropic_messages(body: AnthropicRequest, request: Request):
        """Anthropic-compatible /v1/messages endpoint."""
        start = time.time()
        resolved_model = _resolve_model(body.model, _config) if _config else body.model

        # Convert Anthropic format to OpenAI format
        openai_messages = []

        if body.system:
            sys_content = body.system if isinstance(body.system, str) else json.dumps(body.system)
            openai_messages.append({"role": "system", "content": sys_content})

        for msg in body.messages:
            content = msg.content
            if isinstance(content, list):
                text_parts = []
                tool_use_blocks = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_result":
                            tool_content = block.get("content", "")
                            if isinstance(tool_content, list):
                                for c in tool_content:
                                    if isinstance(c, dict) and c.get("type") == "text":
                                        text_parts.append(c.get("text", ""))
                            else:
                                text_parts.append(str(tool_content))
                        elif block.get("type") == "tool_use":
                            tool_use_blocks.append(block)
                    else:
                        text_parts.append(str(block))
                content = "\n".join(text_parts) if text_parts else ""
                if tool_use_blocks:
                    tool_info = json.dumps(tool_use_blocks)
                    content = f"{content}\n[Tool calls: {tool_info}]" if content else f"[Tool calls: {tool_info}]"
            openai_messages.append({"role": msg.role, "content": content})

        openai_payload = {
            "model": resolved_model,
            "messages": openai_messages,
            "max_tokens": body.max_tokens,
            "stream": body.stream,
        }
        if body.temperature is not None:
            openai_payload["temperature"] = body.temperature
        if body.top_p is not None:
            openai_payload["top_p"] = body.top_p
        if body.stop_sequences:
            openai_payload["stop"] = body.stop_sequences

        if body.tools:
            openai_tools = []
            for tool in body.tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    }
                })
            openai_payload["tools"] = openai_tools
            if body.tool_choice:
                if isinstance(body.tool_choice, dict):
                    if tool_choice_type := body.tool_choice.get("type"):
                        if tool_choice_type == "auto":
                            openai_payload["tool_choice"] = "auto"
                        elif tool_choice_type == "any":
                            openai_payload["tool_choice"] = "required"
                        elif tool_choice_type == "tool":
                            openai_payload["tool_choice"] = {"type": "function", "function": {"name": body.tool_choice.get("name", "")}}
                elif isinstance(body.tool_choice, str):
                    openai_payload["tool_choice"] = body.tool_choice

        try:
            if body.stream:
                return await _stream_completion_anthropic(client, openai_payload, resolved_model, body.max_tokens, _analytics, start)

            resp = await client.chat_completion(openai_payload)
            elapsed = time.time() - start
            metrics = resp.pop("_oct_metrics", {})
            usage = resp.get("usage", {})

            if _analytics:
                _analytics.log_llm(
                    model=resolved_model,
                    prompt_tokens=usage.get("prompt_tokens", metrics.get("prompt_eval_count", 0)),
                    completion_tokens=usage.get("completion_tokens", metrics.get("eval_count", 0)),
                    total_tokens=usage.get("total_tokens", 0),
                    tps=metrics.get("tps"),
                    ttft_seconds=metrics.get("ttft_seconds"),
                    total_duration_seconds=elapsed,
                    provider="ollama",
                )

            # Convert OpenAI response to Anthropic format
            choices = resp.get("choices", [])
            content_text = ""
            finish_reason = "end_turn"
            tool_use_response = []
            if choices:
                msg = choices[0].get("message", {})
                content_text = msg.get("content", "")
                fr = choices[0].get("finish_reason", "stop")
                finish_reason = _openai_to_anthropic_stop(fr)

                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        tool_use_response.append({
                            "type": "tool_use",
                            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                            "name": func.get("name", ""),
                            "input": json.loads(func.get("arguments", "{}")) if isinstance(func.get("arguments"), str) else func.get("arguments", {}),
                        })

            content_blocks = []
            if content_text:
                content_blocks.append({"type": "text", "text": content_text})
            if tool_use_response:
                content_blocks.extend(tool_use_response)

            if not content_blocks:
                content_blocks = [{"type": "text", "text": ""}]

            anthropic_resp = {
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "role": "assistant",
                "model": body.model,
                "content": content_blocks,
                "stop_reason": "tool_use" if tool_use_response else finish_reason,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", metrics.get("prompt_eval_count", 0)),
                    "output_tokens": usage.get("completion_tokens", metrics.get("eval_count", 0)),
                },
            }
            return anthropic_resp

        except Exception as e:
            if _analytics:
                _analytics.log_llm(model=resolved_model, error=str(e), total_duration_seconds=time.time() - start)
            raise HTTPException(status_code=502, detail=f"Ollama Cloud error: {str(e)}")

    # ── Model selection endpoint ──

    @router.post("/v1/config/model")
    async def set_model(request: Request):
        """Update model selection for reranker/scraper/summary."""
        from guanaco.config import save_config, get_config
        body = await request.json()
        config = get_config()
        updated = False

        if "reranker_model" in body:
            config.llm.reranker_model = body["reranker_model"]
            updated = True
        if "scraper_model" in body:
            config.llm.scraper_model = body["scraper_model"]
            updated = True
        if "summary_model" in body:
            config.llm.summary_model = body["summary_model"]
            updated = True
        if "default_model" in body:
            config.llm.default_model = body["default_model"]
            updated = True
        if "fallback_model" in body:
            config.llm.fallback_model = body["fallback_model"]
            updated = True

        if updated:
            save_config(config)
            return {"status": "ok", "config": config.llm.model_dump()}
        return {"status": "no_changes", "config": config.llm.model_dump()}

    @router.get("/v1/config/model")
    async def get_model_config(request: Request):
        """Get current model selection config."""
        from guanaco.config import get_config
        config = get_config()
        return config.llm.model_dump()

    # ── Fallback provider config ──

    @router.get("/v1/config/fallback")
    async def get_fallback_config(request: Request):
        """Get current fallback provider config."""
        from guanaco.config import get_config
        config = get_config()
        fb = config.fallback
        return {
            "enabled": fb.enabled,
            "name": fb.name,
            "base_url": fb.base_url,
            "default_model": fb.default_model,
            "model_map": fb.model_map,
            "timeout": fb.timeout,
            "stream_fallback": fb.stream_fallback,
            "has_api_key": bool(fb.api_key or os.environ.get("FALLBACK_API_KEY", "")),
        }

    @router.post("/v1/config/fallback")
    async def set_fallback_config(request: Request):
        """Update fallback provider config."""
        from guanaco.config import save_config, get_config
        import os as _os
        body = await request.json()
        config = get_config()
        fb = config.fallback

        if "enabled" in body:
            fb.enabled = body["enabled"]
        if "name" in body:
            fb.name = body["name"]
        if "base_url" in body:
            fb.base_url = body["base_url"]
        if "api_key" in body:
            fb.api_key = body["api_key"]
        if "model_map" in body:
            fb.model_map = body["model_map"]
        if "default_model" in body:
            fb.default_model = body["default_model"]
        if "timeout" in body:
            fb.timeout = float(body["timeout"])
        if "stream_fallback" in body:
            fb.stream_fallback = body["stream_fallback"]

        save_config(config)

        return {
            "status": "ok",
            "fallback": {
                "enabled": fb.enabled,
                "name": fb.name,
                "base_url": fb.base_url,
                "default_model": fb.default_model,
                "model_map": fb.model_map,
                "timeout": fb.timeout,
                "stream_fallback": fb.stream_fallback,
                "has_api_key": bool(fb.api_key or _os.environ.get("FALLBACK_API_KEY", "")),
            },
        }

    # ── Model sync endpoint ──

    @router.post("/v1/config/sync_models")
    async def sync_models(request: Request):
        """Sync available_models from Ollama Cloud API into config."""
        from guanaco.config import save_config, get_config
        try:
            models = await client.list_models(force_refresh=True)
            config = get_config()
            model_names = []
            for m in models:
                name = m.get("name", m.get("model", ""))
                # Strip -cloud suffix
                name = name.replace("-cloud", "") if name.endswith("-cloud") else name
                if name and name not in model_names:
                    model_names.append(name)

            # Merge with existing config models
            existing = set(config.llm.available_models)
            for mn in model_names:
                existing.add(mn)

            config.llm.available_models = sorted(existing)
            save_config(config)

            return {"status": "ok", "synced": len(model_names), "total": len(config.llm.available_models), "models": config.llm.available_models}
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Cannot sync models: {str(e)}")

    # ── Cache management endpoints (beta) ──

    @router.get("/v1/cache/stats")
    async def cache_stats(request: Request):
        """Get cache statistics and configuration."""
        if not _cache:
            return {"beta_mode": False, "message": "Cache not initialized"}
        return _cache.get_stats()

    @router.post("/v1/cache/clear")
    async def cache_clear(request: Request):
        """Clear all cached responses."""
        if not _cache:
            return {"status": "not_initialized"}
        _cache.clear()
        return {"status": "ok", "message": "Cache cleared"}

    @router.post("/v1/config/cache")
    async def update_cache_config(request: Request):
        """Update cache configuration at runtime."""
        from guanaco.config import save_config, get_config as _get_config
        body = await request.json()
        config = _get_config()

        updated = False
        if "beta_mode" in body:
            config.cache.beta_mode = body["beta_mode"]
            updated = True
        if "exact_cache_ttl" in body:
            config.cache.exact_cache_ttl = int(body["exact_cache_ttl"])
            updated = True
        if "session_prefix_ttl" in body:
            config.cache.session_prefix_ttl = int(body["session_prefix_ttl"])
            updated = True
        if "max_entries" in body:
            config.cache.max_entries = int(body["max_entries"])
            updated = True
        if "exact_cache_enabled" in body:
            config.cache.exact_cache_enabled = body["exact_cache_enabled"]
            updated = True
        if "session_prefix_enabled" in body:
            config.cache.session_prefix_enabled = body["session_prefix_enabled"]
            updated = True
        if "dedup_enabled" in body:
            config.cache.dedup_enabled = body["dedup_enabled"]
            updated = True
        if "min_prompt_chars" in body:
            config.cache.min_prompt_chars = int(body["min_prompt_chars"])
            updated = True

        if updated:
            nonlocal _cache
            # Update the live cache config
            if _cache:
                _cache.config = config.cache
            save_config(config)
            # Re-init cache if beta_mode changed
            if config.cache.beta_mode and _cache is None:
                _cache = CacheEngine(config.cache)
            elif not config.cache.beta_mode and _cache:
                _cache.clear()
            return {"status": "ok", "cache": config.cache.model_dump()}
        return {"status": "no_changes", "cache": config.cache.model_dump()}

    @router.get("/v1/config/cache")
    async def get_cache_config(request: Request):
        """Get current cache configuration."""
        from guanaco.config import get_config as _get_config
        config = _get_config()
        return config.cache.model_dump()

    @router.post("/v1/cache/evict")
    async def cache_evict_expired(request: Request):
        """Force eviction of expired cache entries."""
        if not _cache:
            return {"status": "not_initialized"}
        _cache.evict_expired()
        stats = _cache.get_stats()
        return {"status": "ok", "remaining_entries": stats["exact_cache_entries"] + stats["prefix_cache_entries"]}

    return router


# ── Streaming helpers ──

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
                # tool_calls count as non-empty
                if delta.get("tool_calls"):
                    return False
        except (json.JSONDecodeError, KeyError):
            continue
    return True


async def _collect_stream_chunks(client, payload) -> tuple[list[str], dict]:
    """Collect all chunks from a stream into a buffer. Returns (chunks, metrics)."""
    chunks = []
    metrics = {}
    async for chunk in client.chat_completion_stream(payload):
        if chunk.startswith("__oct_metrics__:"):
            try:
                metrics = json.loads(chunk.split(":", 1)[1])
            except (json.JSONDecodeError, ValueError):
                pass
            continue
        chunks.append(chunk)
    return chunks, metrics


async def _iter_stream_with_timeouts(aiter, first_chunk_timeout, inter_chunk_timeout):
    """Wrap an async iterator with per-chunk timeouts.

    - first_chunk_timeout: max seconds to wait for the FIRST chunk
    - inter_chunk_timeout: max seconds to wait for each SUBSEQUENT chunk

    Raises asyncio.TimeoutError if any deadline is missed.
    """
    first = True
    async for item in aiter:
        first = False
        yield item
        # After yielding, set up timeout for next chunk
    # If we never got any item, the aiter ended normally (empty stream)


async def _stream_completion_openai(client, payload, model, analytics, start_time, config=None):
    """Stream OpenAI-format SSE responses, with fallback and timeout support.

    Key design: When fallback is configured with primary_timeout, we apply
    per-chunk timeouts to the Ollama stream. If the first chunk doesn't arrive
    within primary_timeout seconds, we fall back to the fallback provider
    BEFORE yielding any data to the client. This means Hermes never sees
    a timeout — it either gets Ollama chunks or fallback chunks.
    """
    from fastapi.responses import StreamingResponse

    fb = config.fallback if config else None
    use_timeouts = (fb and fb.enabled and fb.base_url and fb.primary_timeout
                    and fb.primary_timeout > 0)

    async def generate():
        stream_metrics = {}
        used_fallback = False
        fallback_model = None
        original_error = None
        try:
            if use_timeouts:
                chunk_timeout = fb.stream_chunk_timeout if fb.stream_chunk_timeout else 180.0
                # ── Timed streaming: fail fast on first chunk, tolerate gaps after ──
                ollama_stream = client.chat_completion_stream(payload)
                stream_closed = False
                # Wait for the first chunk with a strict timeout (triggers fallback fast)
                try:
                    first_chunk = await asyncio.wait_for(
                        ollama_stream.__anext__(), timeout=fb.primary_timeout
                    )
                except asyncio.TimeoutError:
                    # First chunk timeout — no data sent to client yet, can still fallback
                    try:
                        await ollama_stream.aclose()
                    except RuntimeError:
                        pass  # Generator already cleaned up by cancellation
                    stream_closed = True
                    raise httpx.ReadTimeout(
                        f"Ollama did not produce first stream chunk within {fb.primary_timeout}s"
                    )
                except StopAsyncIteration:
                    # Empty stream — treat as error so fallback can handle it
                    try:
                        await ollama_stream.aclose()
                    except RuntimeError:
                        pass
                    stream_closed = True
                    raise httpx.ReadTimeout(
                        f"Ollama stream ended before producing any chunks"
                    )

                # Got first chunk — process it (metrics chunks are internal, not yield)
                if first_chunk.startswith("__oct_metrics__:"):
                    try:
                        stream_metrics = json.loads(first_chunk.split(":", 1)[1])
                    except (json.JSONDecodeError, ValueError):
                        pass
                else:
                    yield first_chunk
                try:
                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                ollama_stream.__anext__(), timeout=chunk_timeout
                            )
                            if chunk.startswith("__oct_metrics__:"):
                                try:
                                    stream_metrics = json.loads(chunk.split(":", 1)[1])
                                except (json.JSONDecodeError, ValueError):
                                    pass
                                continue
                            yield chunk
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError:
                            # Inter-chunk timeout — we've already sent data, can't fallback
                            # Send an error SSE chunk and stop
                            log.warning("Ollama stream inter-chunk timeout for %s after %ss", model, chunk_timeout)
                            error_data = json.dumps({"error": {"message": f"Stream stalled: no data for {chunk_timeout}s", "type": "server_error"}})
                            yield f"data: {error_data}\n\n"
                            yield "data: [DONE]\n\n"
                            original_error = f"Stream stalled: no data for {chunk_timeout}s"
                            return
                finally:
                    if not stream_closed:
                        try:
                            await ollama_stream.aclose()
                        except RuntimeError:
                            pass  # Already closed
            else:
                # ── No timeout wrapping: original buffered behavior ──
                for attempt in range(MAX_EMPTY_RETRIES + 1):
                    chunks, stream_metrics = await _collect_stream_chunks(client, payload)
                    if not _is_empty_stream_buffer(chunks) or attempt == MAX_EMPTY_RETRIES:
                        break
                    log.warning("Empty streaming response from %s (attempt %d/%d), retrying...", model, attempt + 1, MAX_EMPTY_RETRIES + 1)

                for chunk in chunks:
                    yield chunk

        except Exception as e:
            original_error = _describe_error(e)
            # Try fallback if configured — we haven't sent any data yet, so we can cleanly switch
            if config and config.fallback.enabled and config.fallback.base_url and config.fallback.stream_fallback:
                fallback_model = _map_model_to_fallback(model, config.fallback)
                log.info("Ollama stream error for %s (%s), trying fallback %s (model: %s)", model, original_error, config.fallback.name, fallback_model)
                fallback_payload = dict(payload)
                fallback_payload["model"] = fallback_model
                try:
                    async for chunk in await _call_fallback_provider(fallback_payload, config.fallback, stream=True):
                        used_fallback = True
                        yield chunk
                except Exception as fallback_err:
                    log.warning("Stream fallback to %s failed for model %s: %s", config.fallback.name, model, _describe_error(fallback_err))
                    error_data = json.dumps({"error": {"message": f"Ollama: {original_error}; Fallback: {_describe_error(fallback_err)}", "type": "server_error"}})
                    yield f"data: {error_data}\n\n"
                    yield "data: [DONE]\n\n"
            else:
                error_data = json.dumps({"error": {"message": original_error, "type": "server_error"}})
                yield f"data: {error_data}\n\n"
                yield "data: [DONE]\n\n"
        finally:
            elapsed = time.time() - start_time
            if analytics:
                if used_fallback and fallback_model:
                    analytics.log_llm(
                        model=_normalize_model_name(fallback_model),
                        prompt_tokens=stream_metrics.get("prompt_eval_count", 0),
                        completion_tokens=stream_metrics.get("eval_count"),
                        tps=stream_metrics.get("tps"),
                        ttft_seconds=stream_metrics.get("ttft_seconds"),
                        total_duration_seconds=stream_metrics.get("elapsed_seconds", elapsed),
                        provider=config.fallback.name if config else "fallback",
                        fallback_for=_normalize_model_name(model),
                    )
                elif original_error:
                    analytics.log_llm(
                        model=_normalize_model_name(model),
                        error=original_error,
                        total_duration_seconds=elapsed,
                        provider="ollama",
                    )
                else:
                    analytics.log_llm(
                        model=_normalize_model_name(model),
                        prompt_tokens=stream_metrics.get("prompt_eval_count", 0),
                        completion_tokens=stream_metrics.get("eval_count"),
                        tps=stream_metrics.get("tps"),
                        ttft_seconds=stream_metrics.get("ttft_seconds"),
                        total_duration_seconds=stream_metrics.get("elapsed_seconds", elapsed),
                        provider="ollama",
                    )

    return StreamingResponse(generate(), media_type="text/event-stream")


async def _stream_fallback_openai(payload, config, fallback_model, analytics, start_time, provider_tag="fallback"):
    """Stream from fallback provider in OpenAI format."""
    from fastapi.responses import StreamingResponse

    async def generate():
        try:
            async for chunk in await _call_fallback_provider(payload, config.fallback, stream=True):
                yield chunk
        except Exception as e:
            error_data = json.dumps({"error": {"message": str(e), "type": "server_error"}})
            yield f"data: {error_data}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            elapsed = time.time() - start_time
            if analytics:
                analytics.log_llm(model=_normalize_model_name(fallback_model), total_duration_seconds=elapsed, provider=provider_tag)

    return StreamingResponse(generate(), media_type="text/event-stream")


async def _stream_completion_anthropic(client, payload, model, max_tokens, analytics, start_time):
    """Stream Anthropic-format SSE responses, translating from Ollama's OpenAI format."""
    from fastapi.responses import StreamingResponse
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    async def generate():
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

        total_tokens = 0
        first_token_time = None
        async for chunk in client.chat_completion_stream(payload):
            # Capture metrics from stream
            if chunk.startswith("__oct_metrics__:"):
                stream_metrics_raw = chunk.split(":", 1)[1]
                try:
                    stream_metrics = json.loads(stream_metrics_raw)
                    # Log with streaming metrics
                    if analytics:
                        analytics.log_llm(
                            model=model,
                            completion_tokens=stream_metrics.get("eval_count", total_tokens),
                            tps=stream_metrics.get("tps"),
                            ttft_seconds=stream_metrics.get("ttft_seconds") or (round(first_token_time - start_time, 3) if first_token_time else None),
                            total_duration_seconds=stream_metrics.get("elapsed_seconds", time.time() - start_time),
                        )
                except (json.JSONDecodeError, ValueError):
                    pass
                continue
            try:
                if "data: " in chunk:
                    data_str = chunk[6:].strip()
                    if data_str == "[DONE]":
                        continue
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    for choice in choices:
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if first_token_time is None:
                                first_token_time = time.time()
                            total_tokens += 1
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': content}})}\n\n"
            except (json.JSONDecodeError, KeyError):
                continue

        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': total_tokens}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")



def _openai_to_anthropic_stop(reason: str) -> str:
    """Convert OpenAI finish_reason to Anthropic stop_reason."""
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    return mapping.get(reason, "end_turn")