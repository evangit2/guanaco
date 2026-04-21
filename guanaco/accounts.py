"""Multi-account Ollama key rotation with quota-aware selection."""

import logging
import time
from typing import Optional

from guanaco.config import OllamaAccount

logger = logging.getLogger(__name__)

# Models that require a paid Ollama plan (not available on free tier).
PREMIUM_MODELS = {"kimi-k2.6", "glm-5.1"}


def model_requires_premium(model: str) -> bool:
    """Check if a model requires a paid Ollama plan (pro/max).

    Matches case-insensitively against model name substrings.
    E.g. 'kimi-k2.6-0915' matches 'k2.6'.
    """
    model_lower = model.lower().strip()
    for pm in PREMIUM_MODELS:
        if pm in model_lower:
            return True
    return False


class AccountPool:
    """Manages a pool of Ollama accounts and selects the best one for each request.

    Selection strategy:
    1. Among accounts with usage data, pick the one with the lowest session_pct
    2. Among accounts without usage data, round-robin
    3. If only one account, always use it
    """

    def __init__(self, accounts: list[OllamaAccount]):
        self._accounts: list[OllamaAccount] = accounts
        self._rr_index: int = 0

    @property
    def accounts(self) -> list[OllamaAccount]:
        return self._accounts

    def update_accounts(self, accounts: list[OllamaAccount]) -> None:
        """Replace the account list (e.g., after config save)."""
        self._accounts = accounts

    def get_account(self, preferred: Optional[str] = None, model: Optional[str] = None) -> OllamaAccount:
        """Select the best account for the next request.

        Strategy: prefer accounts with the most available quota.
        - If model is premium, skip free-tier accounts.
        - Accounts with no usage data are assumed fresh (0%) — preferred.
        - Among accounts with usage data, pick lowest session_pct.
        - Tie-break with round-robin for equal-priority accounts.

        Args:
            preferred: If set, try this account name first.
            model: If set, check if model requires premium and filter accordingly.

        Returns:
            The selected OllamaAccount.
        """
        active = [a for a in self._accounts if a.api_key and a.api_key not in ("***", "REPLACE_ME")]
        if not active:
            return self._accounts[0] if self._accounts else OllamaAccount(name="ollama")

        # If model requires premium, filter out free-tier accounts
        premium_needed = model and model_requires_premium(model)
        if premium_needed:
            eligible = [a for a in active if a.last_plan and a.last_plan.lower() != "free"]
            if eligible:
                active = eligible
                logger.info(f"Model '{model}' requires premium plan, filtered to {len(eligible)} eligible accounts")
            else:
                logger.warning(f"Model '{model}' requires premium but no paid accounts available, trying all")

        if len(active) == 1:
            return active[0]

        if preferred:
            for a in active:
                if a.name == preferred:
                    return a

        # Split by whether we have usage data
        without_usage = [a for a in active if a.last_session_pct is None]
        with_usage = [a for a in active if a.last_session_pct is not None]

        # Prefer accounts with no usage data (fresh/unknown quota) over known-heavy ones
        if without_usage:
            idx = self._rr_index % len(without_usage)
            self._rr_index += 1
            account = without_usage[idx]
            logger.debug(f"Selected account '{account.name}' (no usage data, round-robin)")
            return account

        # All have usage data — pick the one with lowest usage
        best = min(with_usage, key=lambda a: a.last_session_pct or 100)
        logger.debug(f"Selected account '{best.name}' (session: {best.last_session_pct}%)")
        return best

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

    def get_by_name(self, name: str) -> Optional[OllamaAccount]:
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
        return name.lower() in ("ollama", "primary", "default")