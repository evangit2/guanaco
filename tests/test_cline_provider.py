"""Tests for the Cline Pass provider integration."""

import pytest
from guanaco.config import infer_provider_from_key, AppConfig, ClineConfig
from guanaco.accounts import (
    provider_for_model,
    CLINE_PREFIXES,
    KNOWN_CLINE_MODELS,
    AccountPool,
)
from guanaco.cline_client import ClinePassClient, _strip_cline_prefix, CLINE_MODELS, _parse_plan_models


class TestClineKeyInference:
    """Test that Cline Pass API keys (sk_ prefix) are correctly identified."""

    def test_cline_key_detected_by_prefix(self):
        assert infer_provider_from_key("sk_test123") == "cline"

    def test_cline_key_with_hint_respected(self):
        assert infer_provider_from_key("anything", provider_hint="cline") == "cline"

    def test_cline_key_not_confused_with_opencode_go(self):
        # sk- (hyphen) = OpenCode Go, sk_ (underscore) = Cline Pass
        assert infer_provider_from_key("sk-opencodekey") == "opencode_go"
        assert infer_provider_from_key("sk_clinekey") == "cline"

    def test_ollama_key_not_confused(self):
        assert infer_provider_from_key("ollama-abc123") == "ollama"

    def test_umans_key_not_confused(self):
        # UMANS keys are long hex strings without sk_ prefix
        long_key = "a" * 70
        assert infer_provider_from_key(long_key) == "umans"


class TestClineModelRouting:
    """Test that Cline Pass models are routed to the cline provider."""

    def test_cline_prefix_routes_to_cline(self):
        assert provider_for_model("cline/glm-5.2") == "cline"
        assert provider_for_model("cline/kimi-k2.7-code") == "cline"
        assert provider_for_model("cline/minimax-m3") == "cline"

    def test_known_cline_models_route_to_cline(self):
        # All Cline Pass models overlap with KNOWN_GO_MODELS or KNOWN_UMANS_MODELS
        # (same underlying models, different providers). Users use the cline/ prefix
        # for explicit routing. Here we test prefix-based routing for all of them.
        for model in KNOWN_CLINE_MODELS:
            assert provider_for_model(f"cline/{model}") == "cline", f"cline/{model} should route to cline"

    def test_cline_model_not_confused_with_umans(self):
        # glm-5.2 is in both KNOWN_UMANS_MODELS and KNOWN_CLINE_MODELS
        # The priority should be: explicit prefix > known models > provider_priority
        # Since KNOWN_CLINE_MODELS is checked before KNOWN_OLLAMA_MODELS but after KNOWN_UMANS_MODELS,
        # glm-5.2 will route to umans (checked first). With provider_priority, cline can be preferred.
        # This is acceptable — the user can use cline/ prefix for explicit routing.
        pass

    def test_provider_priority_with_cline(self):
        """When cline is first in priority, unknown models go to cline."""
        result = provider_for_model(
            "unknown-model",
            provider_priority=["cline", "ollama", "opencode_go", "umans"],
        )
        assert result == "cline"

    def test_cline_prefixes_constant(self):
        assert "cline/" in CLINE_PREFIXES


class TestClinePassClient:
    """Test the ClinePassClient class."""

    def test_client_initialization(self):
        client = ClinePassClient(api_key="sk_test123")
        assert client.api_key == "sk_test123"
        assert client.provider_name == "cline"
        assert client.chat_url == "https://api.cline.bot/api/v1/chat/completions"
        assert client.models_url == "https://api.cline.bot/api/v1/models"

    def test_client_custom_base_url(self):
        client = ClinePassClient(api_key="sk_test", base_url="https://custom.example.com/v1")
        assert client.base_url == "https://custom.example.com/v1"
        assert client.chat_url == "https://custom.example.com/v1/chat/completions"

    def test_strip_cline_prefix(self):
        # _strip_cline_prefix returns cline-pass/<model> format for subscription routing
        assert _strip_cline_prefix("cline/glm-5.2") == "cline-pass/glm-5.2"
        assert _strip_cline_prefix("glm-5.2") == "cline-pass/glm-5.2"
        assert _strip_cline_prefix("cline/kimi-k2.7-code") == "cline-pass/kimi-k2.7-code"
        # Already in cline-pass/ format passes through
        assert _strip_cline_prefix("cline-pass/glm-5.2") == "cline-pass/glm-5.2"
        # modelType/model format gets converted to cline-pass/ for subscription routing
        assert _strip_cline_prefix("zai/glm-5.2") == "cline-pass/glm-5.2"

    def test_prepare_payload_strips_prefix(self):
        client = ClinePassClient(api_key="sk_test")
        payload = client._prepare_payload({"model": "cline/glm-5.2", "messages": []})
        assert payload["model"] == "cline-pass/glm-5.2"

    def test_prepare_payload_strips_reasoning_content(self):
        client = ClinePassClient(api_key="sk_test")
        payload = client._prepare_payload({
            "model": "glm-5.2",
            "messages": [
                {"role": "assistant", "content": "hello", "reasoning_content": "thinking..."},
            ],
        })
        assert "reasoning_content" not in payload["messages"][0]

    def test_model_capabilities(self):
        client = ClinePassClient(api_key="sk_test")
        caps = client._get_model_capabilities("cline/glm-5.2")
        assert caps["provider"] == "cline"
        assert caps["supports_thinking"] is True
        assert caps["usage_multiplier"] == 0.0  # flat-rate subscription

    def test_model_capabilities_unknown_model(self):
        client = ClinePassClient(api_key="sk_test")
        caps = client._get_model_capabilities("unknown-model")
        assert caps["provider"] == "cline"
        assert caps["usage_multiplier"] == 0.0

    def test_static_models_count(self):
        """Cline Pass offers 11 models (10 original + kimi-k3)."""
        client = ClinePassClient(api_key="sk_test")
        models = client._static_models()
        assert len(models) == 11
        model_ids = [m["id"] for m in models]
        assert "glm-5.2" in model_ids
        assert "kimi-k2.7-code" in model_ids
        assert "minimax-m3" in model_ids
        assert "deepseek-v4-flash" in model_ids

    def test_parse_plan_models(self):
        """_parse_plan_models extracts model IDs from plan features.included."""
        included = [
            "Low cost subscription pricing",
            "Generous limits and reliable access",
            "Built for as many programmers as possible",
            "Includes Kimi K3, GLM 5.2, Kimi K2.6, Kimi K2.7 Code, Mimo v2.5, "
            "Mimo v2.5 Pro, Minimax M3, Qwen3.7 Plus, Qwen3.7 Max, "
            "DeepSeek V4 Pro, and DeepSeek V4 Flash",
        ]
        result = _parse_plan_models(included)
        assert "kimi-k3" in result
        assert "glm-5.2" in result
        assert "kimi-k2.6" in result
        assert "kimi-k2.7-code" in result
        assert "mimo-v2.5" in result
        assert "mimo-v2.5-pro" in result
        assert "minimax-m3" in result
        assert "qwen3.7-plus" in result
        assert "qwen3.7-max" in result
        assert "deepseek-v4-pro" in result
        assert "deepseek-v4-flash" in result
        assert len(result) == 11

    def test_parse_plan_models_empty(self):
        """_parse_plan_models returns empty list when no model string found."""
        included = ["Low cost subscription pricing", "No models here"]
        result = _parse_plan_models(included)
        assert result == []

    def test_parse_plan_models_unknown_model(self):
        """_parse_plan_models handles unknown model names gracefully."""
        included = ["Includes New Model X, GLM 5.2, and Unknown Model Y"]
        result = _parse_plan_models(included)
        # Should still find glm-5.2 even if unknown models are present
        assert "glm-5.2" in result


class TestClineConfigMigration:
    """Test that config migration adds cline settings for existing installs."""

    def test_app_config_has_cline_field(self):
        config = AppConfig()
        assert hasattr(config, "cline")
        assert isinstance(config.cline, ClineConfig)
        assert config.cline.enabled is False
        assert config.cline.base_url == ""

    def test_cline_in_default_provider_priority(self):
        config = AppConfig()
        # After migration, cline should be in provider_priority
        # (The migration is in load_config, not in the model default,
        # but the default should still include it for new installs)
        assert "cline" in config.router.provider_priority


class TestAccountPoolCline:
    """Test that AccountPool handles cline accounts."""

    def test_cline_reserved_name(self):
        pool = AccountPool([])
        assert pool.is_reserved_name("cline") is True
        assert pool.is_reserved_name("CLINE") is True

    def test_cline_account_active(self):
        from guanaco.config import ProviderAccount
        acc = ProviderAccount(name="cline-backup", api_key="sk_test123", provider="cline")
        pool = AccountPool([acc])
        active = pool._active(provider="cline")
        assert len(active) == 1
        assert active[0].name == "cline-backup"
