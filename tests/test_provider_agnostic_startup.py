"""Tests for provider-agnostic startup when Ollama Cloud is not configured."""

import os

import pytest

from guanaco.app import create_app
from guanaco.config import AppConfig, LLMConfig, RouterConfig
from guanaco.multi_provider_client import MultiProviderChatClient


class FakeOllamaClient:
    async def list_models(self):
        return [{"name": "llama3"}]

    def _get_model_capabilities(self, model: str):
        return {"vision": False}

    async def close(self):
        pass


class FakeGoClient:
    async def list_models(self):
        return [{"id": "deepseek-v4-flash"}]

    def _get_model_capabilities(self, model: str):
        return {"vision": False, "family": "deepseek"}

    async def close(self):
        pass


def _bare_config() -> AppConfig:
    return AppConfig(
        llm=LLMConfig(available_models=["unknown-model"]),
        router=RouterConfig(host="127.0.0.1", port=0),
    )


def test_ollama_only_client():
    clients = {"ollama": FakeOllamaClient()}
    mpc = MultiProviderChatClient(clients)
    assert mpc.provider_keys == ["ollama"]
    assert mpc._client_for("llama3") is clients["ollama"]


def test_go_only_client():
    clients = {"opencode_go": FakeGoClient()}
    mpc = MultiProviderChatClient(clients)
    assert mpc.provider_keys == ["opencode_go"]
    assert mpc._client_for("deepseek-v4-flash") is clients["opencode_go"]


def test_go_prefixed_model_without_ollama():
    clients = {"opencode_go": FakeGoClient()}
    mpc = MultiProviderChatClient(clients)
    assert mpc._client_for("opencode-go/glm-5") is clients["opencode_go"]


def test_no_provider_returns_none_for_client():
    mpc = MultiProviderChatClient({})
    assert mpc._client_for("anything") is None


def test_create_app_without_ollama_key_does_not_crash(monkeypatch):
    monkeypatch.setitem(os.environ, "OLLAMA_API_KEY", "")
    cfg = _bare_config()
    app = create_app(cfg)
    assert app is not None


def test_create_app_with_only_go_key_does_not_crash(monkeypatch):
    monkeypatch.setitem(os.environ, "OLLAMA_API_KEY", "")
    monkeypatch.setitem(os.environ, "OPENCODE_GO_API_KEY", "sk-go-test")
    cfg = _bare_config()
    app = create_app(cfg)
    assert app is not None


async def _collect(ait):
    return [item async for item in ait]


@pytest.mark.asyncio
async def test_multi_provider_list_models_aggregates_providers():
    clients = {"ollama": FakeOllamaClient(), "opencode_go": FakeGoClient()}
    mpc = MultiProviderChatClient(clients)
    models = await mpc.list_models()
    ids = {m["id"] for m in models}
    assert "llama3" in ids
    assert "opencode-go/deepseek-v4-flash" in ids


@pytest.mark.asyncio
async def test_multi_provider_list_models_without_ollama():
    clients = {"opencode_go": FakeGoClient()}
    mpc = MultiProviderChatClient(clients)
    models = await mpc.list_models()
    assert [m["id"] for m in models] == ["opencode-go/deepseek-v4-flash"]
