"""Fallback provider logic for the Guanaco router.

Extracted from router.py for modularity. Re-exported from
``guanaco.router.router`` for backward compatibility.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from guanaco.accounts import provider_for_model, strip_provider_prefix
from guanaco.concurrency import OllamaConcurrencyLimiter

log = logging.getLogger("guanaco.router")


async def _ollama_chat_with_primary_timeout(
    client, payload, fallback_config=None, limiter=None,
    api_key=None, account_name=None, account_pool=None, provider_priority=None
):
    """Call Ollama Cloud chat completion with a primary timeout and optional concurrency limit.

    When fallback is configured, we use a shorter primary_timeout so that
    slow/unresponsive Ollama responses trigger fallback quickly instead of
    hanging for the full 120s client timeout.
    """
    # Collect sub-clients for cross-provider failover
    _sub_clients: dict = {}
    if hasattr(client, "_clients") and isinstance(client._clients, dict):
        _sub_clients = client._clients
    _current_provider = provider_for_model(payload.get("model", "")) or "ollama"

    async def _do_call():
        """Execute the actual Ollama call with 429 retry logic."""
        nonlocal _current_provider
        current_key = api_key
        current_account = account_name
        current_call_client = client
        while True:
            try:
                return await current_call_client.chat_completion(payload, api_key=current_key)
            except Exception as e:
                should_failover = (
                    isinstance(e, httpx.HTTPStatusError)
                    and e.response.status_code == 429
                    and account_pool is not None
                    and len(account_pool.accounts) > 1
                )
                if should_failover:
                    if current_account and account_pool:
                        account_pool.mark_429(current_account)
                    next_acc = account_pool.next_account_for_failover(
                        current_account or "ollama",
                        provider=_current_provider,
                        model=payload.get("model"),
                        provider_priority=provider_priority,
                    ) if account_pool else None
                    if next_acc is None:
                        raise
                    current_key = next_acc.api_key
                    current_account = next_acc.name
                    # Cross-provider swap
                    if next_acc.provider != _current_provider:
                        _current_provider = next_acc.provider
                        sub = _sub_clients.get(_current_provider)
                        if sub is not None:
                            current_call_client = sub
                        payload["model"] = strip_provider_prefix(payload["model"])
                        log.info("Cross-provider failover: → %s (model=%s)", _current_provider, payload["model"])
                    log.info("429 failover: trying account '%s'", current_account)
                    continue
                if limiter and limiter.should_retry_429(e):
                    await limiter.backoff_and_retry(0)
                    continue
                raise

    if fallback_config and fallback_config.enabled and fallback_config.primary_timeout:
        try:
            return await asyncio.wait_for(_do_call(), timeout=fallback_config.primary_timeout)
        except asyncio.TimeoutError:
            raise httpx.ReadTimeout(
                f"Ollama did not respond within {fallback_config.primary_timeout}s primary timeout"
            )
    return await _do_call()


async def _call_fallback_provider(payload: dict, fallback_config, stream: bool = False):
    """Send a request to the fallback OpenAI-compatible provider."""
    base_url = fallback_config.base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if fallback_config.api_key:
        headers["Authorization"] = f"Bearer {fallback_config.api_key}"

    payload = dict(payload)
    payload["stream"] = stream

    if fallback_config.max_tokens and "max_tokens" not in payload:
        payload["max_tokens"] = fallback_config.max_tokens

    timeout = fallback_config.timeout or 60.0
    connect_timeout = min(timeout, 30.0)
    read_timeout = max(timeout, 120.0)

    if stream:
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
