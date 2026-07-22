"""Multi-provider account key rotation with quota-aware and round-robin selection."""

from __future__ import annotations

import logging
import time
from typing import Optional

from guanaco.config import ProviderAccount

logger = logging.getLogger(__name__)

# How long a 429-exhausted account stays marked before being eligible again.
_EXHAUSTION_TTL_SECONDS = 60

# Models that require a paid Ollama plan (not available on free tier).
PREMIUM_MODELS = {"kimi-k2.6", "glm-5.1"}

# Provider hints from model names
OPENCODE_GO_PREFIXES = ("opencode-go/",)
OLLAMA_PREFIXES = ("ollama/",)
UMANS_PREFIXES = ("umans/", "umans-")
CLINE_PREFIXES = ("cline/",)
CMDCODE_PREFIXES = ("cmdcode/",)

# Known unprefixed OpenCode Go model names that should default to the Go provider.
KNOWN_GO_MODELS = {
    "glm-5.1", "glm-5", "glm-5-1", "glm5.1", "glm5",
    "kimi-k2.7-code", "kimi-k2.7", "kimi-k2.6", "kimi-k2.5",
    "kimi-k2-7-code", "kimi-k2-7", "kimi-k2-6", "kimi-k2-5",
    "deepseek-v4-pro", "deepseek-v4-flash",
    "deepseek-v4pro", "deepseek-v4flash",
    "mimo-v2.5", "mimo-v2.5-pro", "mimo-v2-5", "mimo-v2-5-pro",
    "minimax-m3", "minimax-m2.7", "minimax-m2.5",
    "minimax-m-3", "minimax-m-2-7", "minimax-m-2-5",
    "qwen3.7-max", "qwen3.7-plus", "qwen3.6-plus",
    "qwen3-7-max", "qwen3-7-plus", "qwen3-6-plus",
}

# Known Ollama Cloud models that should default to Ollama provider.
KNOWN_OLLAMA_MODELS = {
    "qwen3", "qwen3:480b", "qwen3:30b", "qwen3:235b",
    "gpt-oss", "gpt-oss:20b", "gpt-oss:120b",
    "deepseek-v3.1", "deepseek-v3.1:671b", "deepseek-r1", "deepseek-r1:671b",
    "llama4", "llama4:109b", "llama4-scout", "llama4-maverick",
    "glm-5", "glm-5.1", "glm-5:72b",
    "minimax-m2.7", "minimax-m3",
    "nemotron-3-nano", "nemotron-3-nano:30b", "nemotron-4", "nemotron-4:340b",
}

# Known UMANS model short names (without umans- prefix) that should default to UMANS provider.
KNOWN_UMANS_MODELS = {
    "kimi-k2.7", "kimi-k2.6", "kimi-k2.5",
    "kimi-k2-7", "kimi-k2-6", "kimi-k2-5",
    "glm-5.2", "glm-5.1", "glm-5", "glm-5-2", "glm-5-1", "glm5.2", "glm5.1", "glm5",
    "qwen3.7-max", "qwen3.7-plus", "qwen3.6-plus",
    "qwen3-7-max", "qwen3-7-plus", "qwen3-6-plus",
}

# Known Cline Pass model names that should default to the Cline provider.
KNOWN_CLINE_MODELS = {
    "glm-5.2", "kimi-k3", "kimi-k2.7-code", "kimi-k2.6",
    "deepseek-v4-pro", "deepseek-v4-flash",
    "mimo-v2.5", "mimo-v2.5-pro",
    "minimax-m3", "qwen3.7-max", "qwen3.7-plus",
}

# Known Command Code Go model names that should default to the cmdcode provider.
# Only models that work with ZDR enabled (verified July 15 2026 benchmark).
KNOWN_CMDCODE_MODELS = {
    "deepseek-v4-pro", "deepseek-v4-flash",
    "kimi-k2.7-code", "kimi-k2.7-code-highspeed", "kimi-k2.6", "kimi-k2.5",
    "glm-5.2", "glm-5.2-fast", "glm-5.1", "glm-5",
    "minimax-m3", "minimax-m2.7", "minimax-m2.5",
    "mimo-v2.5-pro", "mimo-v2.5",
    "qwen3.7-plus", "qwen3.6-plus",
    "tencent-hy3", "nemotron-3-ultra", "step-3.5-flash",
}


def _normalize_model_for_provider(model: str) -> str:
    """Return a canonical lowercased identifier for provider detection."""
    m = model.lower().strip()
    for prefix in OPENCODE_GO_PREFIXES + OLLAMA_PREFIXES + UMANS_PREFIXES + CLINE_PREFIXES + CMDCODE_PREFIXES:
        if m.startswith(prefix):
            m = m[len(prefix):]
    return m.split(":")[0].replace("_", "-")


def strip_provider_prefix(model: str) -> str:
    """Strip any known provider prefix from a model id, returning the bare name.

    Strips repeatedly — "umans/umans-kimi-k2.7" → "kimi-k2.7" (both the
    "umans/" provider prefix and the "umans-" model-name prefix are stripped).

    Examples:
        "umans/umans-kimi-k2.7" → "kimi-k2.7"
        "umans-kimi-k2.7"       → "kimi-k2.7"
        "cline/glm-5.2"         → "glm-5.2"
        "cmdcode/deepseek-v4-flash" → "deepseek-v4-flash"
        "kimi-k2.7"             → "kimi-k2.7"  (no prefix, unchanged)
    """
    m = model.strip()
    lower = m.lower()
    for prefix in OPENCODE_GO_PREFIXES + OLLAMA_PREFIXES + UMANS_PREFIXES + CLINE_PREFIXES + CMDCODE_PREFIXES:
        if lower.startswith(prefix):
            m = m[len(prefix):]
            lower = m.lower()
    return m


def provider_for_model(model: str, default_provider: str = "ollama", provider_priority: Optional[list[str]] = None) -> str:
    """Infer provider from model id.

    Explicit prefixes always win.  Otherwise we check known aliases.  If the
    model is still ambiguous we fall back to default_provider (set by router
    config or account availability).
    """
    m = model.lower().strip()
    for prefix in OPENCODE_GO_PREFIXES:
        if m.startswith(prefix):
            return "opencode_go"
    for prefix in OLLAMA_PREFIXES:
        if m.startswith(prefix):
            return "ollama"
    for prefix in UMANS_PREFIXES:
        if m.startswith(prefix):
            return "umans"
    for prefix in CLINE_PREFIXES:
        if m.startswith(prefix):
            return "cline"
    for prefix in CMDCODE_PREFIXES:
        if m.startswith(prefix):
            return "cmdcode"
    canon = _normalize_model_for_provider(model)
    # Build a set of all providers that claim this model as "known".
    claiming_providers = []
    if canon in KNOWN_GO_MODELS:
        claiming_providers.append("opencode_go")
    if canon in KNOWN_UMANS_MODELS:
        claiming_providers.append("umans")
    if canon in KNOWN_CLINE_MODELS:
        claiming_providers.append("cline")
    if canon in KNOWN_CMDCODE_MODELS:
        claiming_providers.append("cmdcode")
    if canon in KNOWN_OLLAMA_MODELS:
        claiming_providers.append("ollama")
    # If only one provider claims it, use that.
    if len(claiming_providers) == 1:
        return claiming_providers[0]
    # If multiple providers claim it, pick the first one in provider_priority.
    # Falls back to the hardcoded order above if no priority list.
    if len(claiming_providers) > 1:
        if provider_priority:
            for p in provider_priority:
                if p in claiming_providers:
                    return p
        # No priority list — use the old hardcoded order (Go > UMANS > Cline > CmdCode > Ollama)
        return claiming_providers[0]
    # Model is not in any known set — use provider_priority or default
    if provider_priority:
        for p in provider_priority:
            if p in ("ollama", "opencode_go", "umans", "cline", "cmdcode"):
                return p
    return default_provider


def model_requires_premium(model: str) -> bool:
    """Check if a model requires a paid Ollama plan."""
    model_lower = model.lower().strip()
    for pm in PREMIUM_MODELS:
        if pm in model_lower:
            return True
    return False





class NoEnabledAccounts(Exception):
    """Raised when no accounts are available for selection."""


class AccountPool:
    """Manages a pool of provider accounts and selects the best one per provider.

    Selection strategies:
    - "usage" (default): prefer accounts with the lowest known session usage.
    - "round_robin": classic round-robin across active accounts, advancing on
      every request. Useful for OpenCode Go subscriptions where each account is a
      full paid seat and you want to spread requests evenly.

    429 failover:
    - mark_429() records that an account is rate-limited.
    - next_account_for_failover() returns the next eligible account so callers
      can retry the same request with a fresh key.
    """

    def __init__(self, accounts: list[ProviderAccount]):
        self._accounts: list[ProviderAccount] = accounts
        self._rr_index: int = 0
        self._exhausted: set[str] = set()
        self._exhausted_ts: dict[str, float] = {}  # account name → timestamp of 429

    @property
    def accounts(self) -> list[ProviderAccount]:
        return self._accounts

    def update_accounts(self, accounts: list[ProviderAccount]) -> None:
        """Replace the account list (e.g., after config save)."""
        self._accounts = accounts
        valid_names = {a.name for a in accounts}
        self._exhausted = self._exhausted & valid_names

    def _active(self, provider: str = "ollama", model: Optional[str] = None) -> list[ProviderAccount]:
        """Return enabled accounts for a provider, optionally filtered for premium models."""
        active = [
            a for a in self._accounts
            if a.provider == provider and a.api_key and a.api_key not in ("***", "REPLACE_ME")
        ]
        if not active:
            return []

        if provider == "ollama":
            premium_needed = model and model_requires_premium(model)
            if premium_needed:
                eligible = [a for a in active if a.last_plan and a.last_plan.lower() != "free"]
                if eligible:
                    logger.info(f"Model '{model}' requires premium plan, filtered to {len(eligible)} eligible accounts")
                    return eligible
                logger.warning(f"Model '{model}' requires premium but no paid accounts available, trying all")
        elif provider == "umans":
            # UMANS subscription bypasses Ollama-style usage metrics; every key is treated as premium.
            pass

        return active

    def get_account(self, provider: str = "ollama", preferred: Optional[str] = None, model: Optional[str] = None) -> ProviderAccount:
        """Select the best account for the next request for a provider."""
        active = self._active(provider=provider, model=model)
        if not active:
            fallback = [a for a in self._accounts if a.provider == provider]
            return fallback[0] if fallback else ProviderAccount(name=provider, provider=provider)

        if len(active) == 1:
            return active[0]

        if preferred:
            for a in active:
                if a.name == preferred:
                    return a

        strategy = "round_robin" if any(a.rotation_mode == "round_robin" for a in active) else "usage"

        if strategy == "round_robin":
            return self._select_round_robin(active)

        return self._select_usage(active)

    def has_active_account(self, provider: str, model: Optional[str] = None) -> bool:
        """Return True if at least one active (keyed) account exists for this provider."""
        return bool(self._active(provider=provider, model=model))


    def _select_round_robin(self, active: list[ProviderAccount]) -> ProviderAccount:
        """Classic round-robin: advance index each request."""
        idx = self._rr_index % len(active)
        self._rr_index += 1
        account = active[idx]
        logger.debug(f"Selected account '{account.name}' (round-robin)")
        return account

    def _select_usage(self, active: list[ProviderAccount]) -> ProviderAccount:
        """Quota-aware selection: prefer accounts with no/fresh usage data, then lowest session_pct."""
        without_usage = [a for a in active if a.last_session_pct is None]
        with_usage = [a for a in active if a.last_session_pct is not None]

        if without_usage:
            idx = self._rr_index % len(without_usage)
            self._rr_index += 1
            account = without_usage[idx]
            logger.debug(f"Selected account '{account.name}' (no usage data, round-robin)")
            return account

        best = min(with_usage, key=lambda a: a.last_session_pct or 100)
        logger.debug(f"Selected account '{best.name}' (session: {best.last_session_pct}%)")
        return best

    def next_account_for_failover(
        self,
        current_name: str,
        provider: str = "ollama",
        model: Optional[str] = None,
        provider_priority: Optional[list[str]] = None,
    ) -> Optional[ProviderAccount]:
        """Pick the next account after a 429, skipping exhausted/current ones.

        If the current provider has no remaining accounts, fall through to
        the next provider in ``provider_priority`` and try its accounts.
        This enables cross-provider failover (e.g. UMANS 429 → Cline → Ollama)
        instead of immediately giving up when a single-provider pool is exhausted.
        """
        self._exhausted.add(current_name)
        self._exhausted_ts[current_name] = time.time()
        self._prune_expired()

        # Phase 1: try other accounts within the same provider
        active = self._active(provider=provider, model=model)
        for a in active:
            if a.name not in self._exhausted:
                logger.info(f"Failover from '{current_name}' to account '{a.name}'")
                return a

        # Phase 2: cross-provider failover — walk the priority list
        if provider_priority:
            for next_provider in provider_priority:
                if next_provider == provider:
                    continue
                active = self._active(provider=next_provider, model=model)
                for a in active:
                    if a.name not in self._exhausted:
                        logger.info(
                            f"Cross-provider failover: '{current_name}' (%s) → '%s' (%s)",
                            current_name, a.name, next_provider,
                        )
                        return a

        logger.warning(f"All active accounts exhausted (last 429 from '{current_name}')")
        return None

    def reset_exhausted(self) -> None:
        """Clear the in-process 429 exhaustion set."""
        self._exhausted.clear()
        self._exhausted_ts.clear()

    def _prune_expired(self) -> None:
        """Remove exhaustion entries that have exceeded their TTL (60s).

        Without this, a single 429 permanently marks an account as exhausted
        for the process lifetime, preventing failover even after the upstream
        rate limit has recovered.
        """
        now = time.time()
        expired = [
            name for name, ts in list(self._exhausted_ts.items())
            if now - ts > _EXHAUSTION_TTL_SECONDS
        ]
        for name in expired:
            self._exhausted.discard(name)
            del self._exhausted_ts[name]

    def update_usage(self, account_name: str, session_pct: Optional[float],
                     weekly_pct: Optional[float], plan: Optional[str] = None,
                     session_reset: Optional[str] = None, weekly_reset: Optional[str] = None) -> None:
        """Update usage data for a specific account."""
        for a in self._accounts:
            if a.name == account_name:
                a.last_session_pct = session_pct
                a.last_weekly_pct = weekly_pct
                a.last_plan = plan
                a.last_session_reset = session_reset
                a.last_weekly_reset = weekly_reset
                a.last_checked = time.time()
                break

    def mark_429(self, account_name: str) -> None:
        """Mark that an account hit a 429. Temporarily deprioritize it."""
        self._exhausted.add(account_name)
        self._exhausted_ts[account_name] = time.time()
        for a in self._accounts:
            if a.name == account_name:
                if a.last_session_pct is not None:
                    a.last_session_pct = min(a.last_session_pct + 25, 100)
                else:
                    a.last_session_pct = 75
                logger.info(f"Account '{account_name}' hit 429, deprioritizing (session est: {a.last_session_pct}%)")
                break

    def account_names(self) -> list[str]:
        """List all account names."""
        return [a.name for a in self._accounts]

    def get_by_name(self, name: str) -> Optional[ProviderAccount]:
        """Find account by name."""
        for a in self._accounts:
            if a.name == name:
                return a
        return None

    def name_taken(self, name: str) -> bool:
        """Check if an account name is already used."""
        return any(a.name == name for a in self._accounts)

    def is_reserved_name(self, name: str) -> bool:
        """Check if a name is reserved (case-insensitive)."""
        return name.lower() in ("ollama", "opencode_go", "umans", "cline", "cmdcode", "primary", "default")
