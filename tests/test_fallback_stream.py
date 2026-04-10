"""Test that fallback payloads always include the correct 'stream' value in the JSON body.

Reproduction of the bug:
- Request with stream=true + max_tokens > 4096
- Ollama fails, falls back to Fireworks/custom provider
- Old code: 'stream' was only passed as a function arg, not in the JSON body
- Fireworks rejects: "Requests with max_tokens > 4096 must have stream=true"
- Fixed: _call_fallback_provider now always injects payload["stream"] = stream
"""
import pytest
import copy
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from guanaco.config import FallbackProviderConfig


def _make_config():
    return FallbackProviderConfig(
        enabled=True,
        name="fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
        api_key="test-key",
        default_model="accounts/fireworks/models/llama-v3p1-70b-instruct",
        max_tokens=8192,
        stream_fallback=True,
    )


@pytest.mark.asyncio
async def test_fallback_non_stream_payload_includes_stream_false():
    """Non-streaming fallback must have 'stream': false in the JSON body."""
    from guanaco.router.router import _call_fallback_provider

    config = _make_config()
    payload = {
        "model": "llama-v3p1-70b-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8192,
    }

    with patch("guanaco.router.router.httpx.AsyncClient") as mock_client_cls:
        mock_instance = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "test",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "Hi"}, "index": 0}],
        }
        mock_response.raise_for_status = MagicMock()
        mock_instance.post = AsyncMock(return_value=mock_response)

        await _call_fallback_provider(payload, config, stream=False)

        sent_json = mock_instance.post.call_args[1]["json"]
        assert sent_json["stream"] == False, f"Expected stream=False, got {sent_json.get('stream')}"
        assert sent_json["max_tokens"] == 8192


@pytest.mark.asyncio
async def test_fallback_stream_payload_includes_stream_true():
    """Streaming fallback must have 'stream': true in the JSON body.

    This is the critical fix for Fireworks' "max_tokens > 4096 requires stream=true" error.

    The function returns an async generator — we need to consume it to trigger
    client.stream() which is inside the generator body.
    """
    from guanaco.router.router import _call_fallback_provider

    config = _make_config()
    payload = {
        "model": "llama-v3p1-70b-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8192,
    }

    # Capture what gets passed to client.stream()
    captured_payload = {}

    class FakeStreamResponse:
        def raise_for_status(self):
            pass
        def aiter_lines(self):
            return AsyncIter([])

    class FakeStreamContext:
        async def __aenter__(self):
            return FakeStreamResponse()
        async def __aexit__(self, *args):
            pass

    class FakeAsyncClient:
        def __init__(self, timeout=None):
            pass

        def stream(self, method, url, json=None, headers=None):
            captured_payload.update(json or {})
            return FakeStreamContext()

        async def aclose(self):
            pass

    with patch("guanaco.router.router.httpx.AsyncClient", FakeAsyncClient):
        gen = await _call_fallback_provider(payload, config, stream=True)
        # Consume the generator to trigger the client.stream() call inside
        async for _ in gen:
            pass

    assert captured_payload.get("stream") == True, \
        f"CRITICAL: stream=true not in fallback JSON body! Got {captured_payload.get('stream')}. " \
        f"Fireworks will reject max_tokens={captured_payload.get('max_tokens')} > 4096 without stream=true."
    assert captured_payload.get("max_tokens") == 8192


@pytest.mark.asyncio
async def test_fallback_does_not_mutate_original_payload():
    """Ensure the original payload dict isn't mutated (should be safe to reuse)."""
    from guanaco.router.router import _call_fallback_provider

    config = _make_config()
    original = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5000,
    }
    original_copy = copy.deepcopy(original)

    with patch("guanaco.router.router.httpx.AsyncClient") as mock_client_cls:
        mock_instance = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "t", "object": "chat.completion", "choices": []}
        mock_response.raise_for_status = MagicMock()
        mock_instance.post = AsyncMock(return_value=mock_response)

        await _call_fallback_provider(original, config, stream=True)

        # Original should be unchanged — no "stream" key added
        assert original == original_copy, f"Original payload was mutated! {original} != {original_copy}"


class AsyncIter:
    def __init__(self, items):
        self._items = iter(items)
    def __aiter__(self):
        return self
    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_fallback_non_stream_payload_includes_stream_false())
    print("✅ Non-streaming fallback: stream=false in body")
    asyncio.run(test_fallback_stream_payload_includes_stream_true())
    print("✅ Streaming fallback: stream=true in body (Fireworks fix)")
    asyncio.run(test_fallback_does_not_mutate_original_payload())
    print("✅ Original payload not mutated")
    print("\n✅ All fallback stream tests passed!")