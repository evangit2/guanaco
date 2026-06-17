"""Tests that the smart cache path does not consume account rotation on cache hits."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from guanaco.config import AppConfig, CacheConfig, RouterConfig
from guanaco.router.router import create_router, ChatCompletionRequest, ChatMessage


def _make_config(cache_beta: bool = True) -> AppConfig:
    return AppConfig(
        ollama_api_key="primary-key",
        router=RouterConfig(host="127.0.0.1", port=8081),
        cache=CacheConfig(
            beta_mode=cache_beta,
            exact_cache_ttl=3600,
            session_prefix_ttl=3600,
            dedup_enabled=False,
        ),
    )


@pytest.fixture
def router_deps():
    """Minimal mocks for create_router."""
    client = MagicMock()
    client.api_key = "primary-key"
    client.session_cookie = None

    config = _make_config()
    pool = MagicMock()
    # Real list so len() works inside _select_account; need 2+ accounts to exercise rotation
    second_account = config.primary_account.model_copy()
    second_account.name = "sub-2"
    second_account.api_key = "sub-2-key"
    pool.accounts = [config.primary_account, second_account]

    analytics = MagicMock()
    analytics.log_llm = MagicMock()

    return client, config, pool, analytics


def _fake_request():
    req = MagicMock()
    req.client.host = "127.0.0.1"
    req.client.port = 12345
    req.headers.get.return_value = "test-agent"
    return req


@patch("guanaco.router.router.CacheEngine")
@pytest.mark.asyncio
async def test_cache_hit_does_not_select_account(CacheEngineMock, router_deps):
    """A cached non-streaming response must never ask the account pool for a key."""
    client, config, pool, analytics = router_deps

    mock_cache = MagicMock()
    mock_cache.is_enabled.return_value = True
    mock_cache.get_or_fetch = AsyncMock(return_value={
        "id": "cached-1",
        "object": "chat.completion",
        "created": 123,
        "model": "gemma3:4b",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "_oct_cached": True,
        "_oct_cache_type": "exact",
    })
    CacheEngineMock.return_value = mock_cache

    router = create_router(client, analytics=analytics, config=config, account_pool=pool)
    chat_route = next(r for r in router.routes if getattr(r, "path", None) == "/v1/chat/completions")
    chat_handler = chat_route.endpoint

    body = ChatCompletionRequest(
        model="gemma3:4b",
        messages=[ChatMessage(role="user", content="hello")],
        stream=False,
    )

    resp = await chat_handler(body, _fake_request())

    assert resp["choices"][0]["message"]["content"] == "hi"
    pool.get_account.assert_not_called()
    analytics.log_llm.assert_called_once()
    logged = analytics.log_llm.call_args.kwargs
    assert logged["provider"] == "cache:exact"
    assert "account_name" not in logged


@patch("guanaco.router.router._ollama_chat_with_primary_timeout")
@patch("guanaco.router.router.CacheEngine")
@pytest.mark.asyncio
async def test_cache_miss_uses_selected_account(CacheEngineMock, mock_upstream, router_deps):
    """A cache miss must select an account and pass it upstream."""
    client, config, pool, analytics = router_deps

    account_mock = MagicMock()
    account_mock.api_key = "sub-2-key"
    account_mock.name = "sub-2"
    pool.get_account.return_value = account_mock

    upstream_response = {
        "id": "upstream-1",
        "object": "chat.completion",
        "created": 123,
        "model": "gemma3:4b",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "yo"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        "_oct_metrics": {"eval_count": 1},
    }
    mock_upstream.return_value = upstream_response

    mock_cache = MagicMock()
    mock_cache.is_enabled.return_value = True

    async def fake_cache_get_or_fetch(**kwargs):
        return await kwargs["fetch_fn"](kwargs.get("params", {}))

    mock_cache.get_or_fetch.side_effect = fake_cache_get_or_fetch
    CacheEngineMock.return_value = mock_cache

    router = create_router(client, analytics=analytics, config=config, account_pool=pool)
    chat_route = next(r for r in router.routes if getattr(r, "path", None) == "/v1/chat/completions")
    chat_handler = chat_route.endpoint

    body = ChatCompletionRequest(
        model="gemma3:4b",
        messages=[ChatMessage(role="user", content="hello")],
        stream=False,
    )

    resp = await chat_handler(body, _fake_request())

    assert resp["choices"][0]["message"]["content"] == "yo"
    pool.get_account.assert_called_once()
    analytics.log_llm.assert_called_once()
    logged = analytics.log_llm.call_args.kwargs
    assert logged["provider"] == "ollama:sub-2"
    assert logged["account_name"] == "sub-2"

    mock_upstream.assert_called_once()
    _, kwargs = mock_upstream.call_args
    assert kwargs["api_key"] == "sub-2-key"
    assert kwargs["account_name"] == "sub-2"
