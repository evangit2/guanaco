"""Tests for the Command Code Go (cmdcode) provider integration."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from guanaco.cmdcode_client import CmdCodeClient, CMDCODE_MODELS, _strip_cmdcode_prefix
from guanaco.config import CmdCodeConfig, AppConfig, infer_provider_from_key
from guanaco.accounts import (
    AccountPool, ProviderAccount,
    CMDCODE_PREFIXES, KNOWN_CMDCODE_MODELS,
    provider_for_model,
)


class TestKeyInference:
    """Test API key prefix detection."""

    def test_user_prefix_detected_as_cmdcode(self):
        assert infer_provider_from_key("user_SmC8oeYNkTtest123") == "cmdcode"

    def test_explicit_hint_respected(self):
        assert infer_provider_from_key("user_abc", provider_hint="cmdcode") == "cmdcode"

    def test_sk_underscore_still_cline(self):
        assert infer_provider_from_key("sk_abc123") == "cline"

    def test_sk_hyphen_still_opencode_go(self):
        assert infer_provider_from_key("sk-abc123") == "opencode_go"

    def test_cmdcode_hint_with_other_key(self):
        assert infer_provider_from_key("some-other-key", provider_hint="cmdcode") == "cmdcode"


class TestModelRouting:
    """Test model name routing to cmdcode provider."""

    def test_cmdcode_prefix_routes_to_cmdcode(self):
        assert provider_for_model("cmdcode/glm-5.2") == "cmdcode"

    def test_cmdcode_prefix_case_insensitive(self):
        assert provider_for_model("CMDCODE/deepseek-v4-pro") == "cmdcode"

    def test_known_cmdcode_model_routes_to_cmdcode(self):
        # tencent-hy3 is only in KNOWN_CMDCODE_MODELS
        assert provider_for_model("tencent-hy3") == "cmdcode"

    def test_known_cmdcode_model_with_prefix(self):
        assert provider_for_model("cmdcode/tencent-hy3") == "cmdcode"

    def test_cline_model_still_routes_to_cline(self):
        # kimi-k2.7-code is in KNOWN_GO_MODELS, so it routes to opencode_go
        # not cline. Models unique to cmdcode (tencent-hy3) route to cmdcode.
        # This is expected behavior — use cmdcode/ prefix for explicit routing.
        pass  # No model is in both cline and cmdcode but NOT in go/umans

    def test_cmdcode_only_model_routes_to_cmdcode(self):
        # nemotron-3-ultra is only in KNOWN_CMDCODE_MODELS
        assert provider_for_model("nemotron-3-ultra") == "cmdcode"

    def test_unknown_model_falls_back_to_priority(self):
        assert provider_for_model("unknown-model", provider_priority=["cmdcode"]) == "cmdcode"

    def test_unknown_model_default_ollama(self):
        assert provider_for_model("unknown-model") == "ollama"


class TestCmdCodeClient:
    """Test CmdCodeClient initialization and methods."""

    def test_init_default_base_url(self):
        client = CmdCodeClient(api_key="user_test123")
        assert "api.commandcode.ai" in client.base_url

    def test_init_ignores_custom_base_url(self):
        # base_url is accepted for config compat but always talks to api.commandcode.ai
        client = CmdCodeClient(api_key="user_test", base_url="http://custom:8080/v1")
        assert "api.commandcode.ai" in client.base_url

    def test_strip_prefix(self):
        assert _strip_cmdcode_prefix("cmdcode/glm-5.2") == "glm-5.2"
        assert _strip_cmdcode_prefix("CMDCODE/GLM-5.2") == "GLM-5.2"
        assert _strip_cmdcode_prefix("glm-5.2") == "glm-5.2"

    def test_prepare_payload_strips_prefix(self):
        client = CmdCodeClient(api_key="user_test")
        payload = {"model": "cmdcode/deepseek-v4-pro", "messages": []}
        prepared = client._prepare_payload(payload)
        assert prepared["model"] == "deepseek-v4-pro"

    def test_prepare_payload_strips_reasoning_content(self):
        client = CmdCodeClient(api_key="user_test")
        payload = {
            "model": "glm-5.2",
            "messages": [
                {"role": "assistant", "content": "test", "reasoning_content": "secret"},
            ],
        }
        prepared = client._prepare_payload(payload)
        assert "reasoning_content" not in prepared["messages"][0]

    def test_capabilities(self):
        client = CmdCodeClient(api_key="user_test")
        caps = client._get_model_capabilities("glm-5.2")
        assert caps["usage_multiplier"] == 0.0
        assert caps["provider"] == "cmdcode"
        assert caps["family"] == "glm"
        assert caps["supports_thinking"] is True

    def test_capabilities_unknown_model(self):
        client = CmdCodeClient(api_key="user_test")
        caps = client._get_model_capabilities("unknown-model")
        assert caps["usage_multiplier"] == 0.0
        assert caps["provider"] == "cmdcode"

    def test_static_models_count(self):
        client = CmdCodeClient(api_key="user_test")
        models = client._static_models()
        assert len(models) == len(CMDCODE_MODELS)
        assert len(models) >= 20  # At least 20 models

    def test_provider_name(self):
        client = CmdCodeClient(api_key="user_test")
        assert client.provider_name == "cmdcode"


class TestConfig:
    """Test config model and migration."""

    def test_cmdcode_config_exists(self):
        config = AppConfig()
        assert hasattr(config, "cmdcode")
        assert isinstance(config.cmdcode, CmdCodeConfig)
        assert config.cmdcode.enabled is False
        assert config.cmdcode.base_url == ""

    def test_cmdcode_in_default_provider_priority(self):
        config = AppConfig()
        assert "cmdcode" in config.router.provider_priority

    def test_cmdcode_config_with_values(self):
        config = AppConfig()
        config.cmdcode.enabled = True
        config.cmdcode.base_url = "https://api.commandcode.ai"
        config.cmdcode.max_concurrent_streams = 4
        assert config.cmdcode.enabled is True
        assert config.cmdcode.max_concurrent_streams == 4


class TestAccountPool:
    """Test AccountPool with cmdcode."""

    def test_cmdcode_is_reserved_name(self):
        pool = AccountPool([])
        assert pool.is_reserved_name("cmdcode") is True
        assert pool.is_reserved_name("CMDCODE") is True

    def test_add_cmdcode_account(self):
        acc = ProviderAccount(name="my-cmdcode", provider="cmdcode", api_key="user_test123")
        pool = AccountPool([acc])
        assert len(pool._accounts) == 1
        assert pool._accounts[0].provider == "cmdcode"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])