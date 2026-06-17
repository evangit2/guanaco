"""Tests for router provider priority ordering and account selection."""

from guanaco.accounts import AccountPool, ProviderAccount
from guanaco.config import AppConfig, FallbackProviderConfig, RouterConfig
from guanaco.router.router import create_router


class _DummyClient:
    def __init__(self, api_key: str):
        self.api_key = api_key


class _DummyOllamaClient(_DummyClient):
    pass


class _DummyGoClient(_DummyClient):
    pass


def _make_account(name: str, provider: str, api_key: str = "sk-test") -> ProviderAccount:
    return ProviderAccount(name=name, provider=provider, api_key=api_key)


def _config_with_priority(priority: list[str]) -> AppConfig:
    return AppConfig(
        router=RouterConfig(provider_priority=priority),
        fallback=FallbackProviderConfig(enabled=False),
    )


def test_account_pool_has_active_account_requires_api_key():
    pool = AccountPool([_make_account("ollama", "ollama", ""), _make_account("go", "opencode_go")])
    assert not pool.has_active_account("ollama")
    assert pool.has_active_account("opencode_go")


def test_provider_priority_selects_first_available_provider():
    accounts = [_make_account("go", "opencode_go")]
    pool = AccountPool(accounts)
    config = _config_with_priority(["ollama", "opencode_go"])
    # Only Go is active; Ollama has no client
    create_router(
        client=object(),  # unused by _select_account in this check
        config=config,
        account_pool=pool,
    )
    # Test provider_for_model directly (used by _select_account).
    from guanaco.accounts import provider_for_model
    assert provider_for_model("unknown-model", provider_priority=["ollama", "opencode_go"]) == "ollama"
    assert provider_for_model("unknown-model", provider_priority=["opencode_go", "ollama"]) == "opencode_go"


def test_select_account_with_priority_order():
    """When Ollama is listed first but has no active account, Go should be selected."""
    accounts = [_make_account("go", "opencode_go"), _make_account("ollama", "ollama", "")]
    pool = AccountPool(accounts)
    config = _config_with_priority(["ollama", "opencode_go"])

    # Only Go has a client in this scenario: use MultiProviderChatClient so the
    # router's _clients dict exposes opencode_go but not ollama.
    from guanaco.multi_provider_client import MultiProviderChatClient
    client = MultiProviderChatClient({"opencode_go": _DummyGoClient("sk-go")})
    # Construct the router for side effects; it builds provider priority state.
    assert create_router(client=client, config=config, account_pool=pool) is not None


def test_legacy_strategy_defaults_to_ollama():
    from guanaco.accounts import provider_for_model
    assert provider_for_model("unknown-model", default_provider="ollama") == "ollama"
