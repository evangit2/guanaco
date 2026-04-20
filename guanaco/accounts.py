"""Multi-account Ollama key rotation with quota-aware selection."""

import logging
import time
from typing import Optional

from guanaco.config import OllamaAccount

logger = logging.getLogger(__name__)


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

    def get_account(self, preferred: Optional[str] = None) -> OllamaAccount:
        """Select the best account for the next request.

        Args:
            preferred: If set, try this account name first.

        Returns:
            The selected OllamaAccount.
        """
        active = [a for a in self._accounts if a.api_key and a.api_key not in ("***", "REPLACE_ME")]
        if not active:
            return self._accounts[0] if self._accounts else OllamaAccount(name="ollama")

        if len(active) == 1:
            return active[0]

        if preferred:
            for a in active:
                if a.name == preferred:
                    return a

        with_usage = [a for a in active if a.last_session_pct is not None]
        without_usage = [a for a in active if a.last_session_pct is None]

        if with_usage:
            best = min(with_usage, key=lambda a: a.last_session_pct or 100)
            logger.debug(f"Selected account '{best.name}' (session: {best.last_session_pct}%)")
            return best

        if without_usage:
            idx = self._rr_index % len(without_usage)
            self._rr_index += 1
            return without_usage[idx]

        return active[0]

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