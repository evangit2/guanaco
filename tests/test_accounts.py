"""Tests for multi-account Ollama key rotation (AccountPool)."""

from guanaco.accounts import AccountPool
from guanaco.config import OllamaAccount


def _acc(name: str, key: str = "sk-", usage: float | None = None, plan: str = "free", mode: str = "usage") -> OllamaAccount:
    return OllamaAccount(
        name=name,
        api_key=key,
        last_session_pct=usage,
        last_plan=plan,
        rotation_mode=mode,
    )


def test_empty_pool_returns_default():
    pool = AccountPool([])
    acc = pool.get_account()
    assert acc.name == "ollama"


def test_single_account_always_selected():
    a = _acc("alpha", "sk-alpha")
    pool = AccountPool([a])
    assert pool.get_account() is a
    assert pool.get_account(model="glm-5.1") is a


def test_usage_prefers_fresh_account():
    heavy = _acc("heavy", "sk-heavy", usage=80.0)
    fresh = _acc("fresh", "sk-fresh", usage=None)
    pool = AccountPool([heavy, fresh])
    assert pool.get_account().name == "fresh"


def test_usage_prefers_lowest_usage():
    high = _acc("high", "sk-high", usage=90.0)
    low = _acc("low", "sk-low", usage=10.0)
    pool = AccountPool([high, low])
    assert pool.get_account().name == "low"


def test_round_robin_advances_each_request():
    a = _acc("a", "sk-a", mode="round_robin")
    b = _acc("b", "sk-b", mode="round_robin")
    pool = AccountPool([a, b])
    assert pool.get_account().name == "a"
    assert pool.get_account().name == "b"
    assert pool.get_account().name == "a"
    assert pool.get_account().name == "b"


def test_mixed_rotation_mode_defaults_to_round_robin():
    a = _acc("a", "sk-a", mode="usage")
    b = _acc("b", "sk-b", mode="round_robin")
    pool = AccountPool([a, b])
    names = [pool.get_account().name for _ in range(4)]
    assert names == ["a", "b", "a", "b"]


def test_premium_model_filters_free_accounts():
    free = _acc("free", "sk-free", plan="free", mode="round_robin")
    pro = _acc("pro", "sk-pro", plan="pro", mode="round_robin")
    pool = AccountPool([free, pro])
    assert pool.get_account(model="kimi-k2.6").name == "pro"
    assert pool.get_account(model="glm-5.1").name == "pro"


def test_premium_model_falls_back_when_no_paid_accounts():
    free1 = _acc("free1", "sk-free1", plan="free", mode="round_robin")
    free2 = _acc("free2", "sk-free2", plan="free", mode="round_robin")
    pool = AccountPool([free1, free2])
    # No paid accounts, so it warns and tries all active accounts
    selected = pool.get_account(model="kimi-k2.6")
    assert selected.name in {"free1", "free2"}


def test_429_failover_skips_exhausted_account():
    a = _acc("a", "sk-a", mode="round_robin")
    b = _acc("b", "sk-b", mode="round_robin")
    c = _acc("c", "sk-c", mode="round_robin")
    pool = AccountPool([a, b, c])
    pool.mark_429("a")
    next_acc = pool.next_account_for_failover("a")
    assert next_acc is not None
    assert next_acc.name in {"b", "c"}


def test_429_failover_returns_none_when_all_exhausted():
    a = _acc("a", "sk-a")
    b = _acc("b", "sk-b")
    pool = AccountPool([a, b])
    pool.mark_429("a")
    pool.mark_429("b")
    assert pool.next_account_for_failover("a") is None


def test_429_updates_usage_estimate():
    a = _acc("a", "sk-a", usage=50.0)
    pool = AccountPool([a])
    pool.mark_429("a")
    assert a.last_session_pct == 75.0


def test_429_updates_usage_estimate_no_prior_data():
    a = _acc("a", "sk-a", usage=None)
    pool = AccountPool([a])
    pool.mark_429("a")
    assert a.last_session_pct == 75.0


def test_update_accounts_keeps_exhausted_for_existing_names():
    a = _acc("a", "sk-a")
    b = _acc("b", "sk-b")
    pool = AccountPool([a, b])
    pool.mark_429("a")
    # replace b with c
    pool.update_accounts([a, _acc("c", "sk-c")])
    assert "a" in pool._exhausted
    assert "b" not in pool._exhausted


def test_preferred_account_selected_when_active():
    a = _acc("a", "sk-a", mode="round_robin")
    b = _acc("b", "sk-b", mode="round_robin")
    pool = AccountPool([a, b])
    assert pool.get_account(preferred="b").name == "b"


def test_preferred_account_ignored_when_not_active():
    a = _acc("a", "sk-a", mode="round_robin")
    b = _acc("b", "sk-b", mode="round_robin")
    pool = AccountPool([a, b])
    assert pool.get_account(preferred="missing").name == "a"
