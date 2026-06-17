"""Tests for UMANS provider integration."""

import pytest

from guanaco.accounts import AccountPool, provider_for_model
from guanaco.config import AppConfig, RouterConfig, OllamaAccount, infer_provider_from_key
from guanaco.multi_provider_client import MultiProviderChatClient
from guanaco.umans_client import UmansClient, _strip_umans_prefix


def test_provider_for_model_umans():
    assert provider_for_model("umans-kimi-k2.7") == "umans"
    assert provider_for_model("umans/kimi-k2.7") == "umans"
    # Short "kimi-k2.7" is ambiguous and currently resolves to OpenCode Go because
    # that provider used the bare alias first. Use the umans- or umans/ prefix.
    assert provider_for_model("kimi-k2.7") == "opencode_go"


def test_infer_provider_from_key_umans():
    assert infer_provider_from_key("a" * 80) == "umans"
    assert infer_provider_from_key("sk-xxxx") == "opencode_go"
    assert infer_provider_from_key("short") == "ollama"


def test_strip_umans_prefix():
    assert _strip_umans_prefix("kimi-k2.7") == "umans-kimi-k2.7"
    assert _strip_umans_prefix("umans/kimi-k2.7") == "umans-kimi-k2.7"
    assert _strip_umans_prefix("umans-glm-5.1") == "umans-glm-5.1"


def test_umans_account_pool():
    acc = OllamaAccount(name="u1", provider="umans", api_key="a" * 80)
    pool = AccountPool([acc])
    selected = pool.get_account(provider="umans", model="umans-kimi-k2.7")
    assert selected is acc


class FakeUmansClient:
    async def list_models(self, force_refresh=False, api_key=None):
        return [{"id": "umans-kimi-k2.7"}]

    def _get_model_capabilities(self, model: str):
        return {"provider": "umans"}

    async def close(self):
        pass


def test_multi_provider_client_routes_umans():
    mpc = MultiProviderChatClient({"umans": FakeUmansClient()})
    assert mpc._client_for("umans-kimi-k2.7") is mpc._clients["umans"]
    assert mpc._client_for("umans/kimi-k2.7") is mpc._clients["umans"]
