"""Provider depletion tracker — monitors credit/usage quotas and marks providers as depleted.

A background task polls each provider's usage API every N minutes (default 5).
When a provider's usage hits the configured threshold (default 99.9%), it is marked
as "depleted" and skipped for unprefixed model routing (requests that don't explicitly
specify a provider).

Providers with no usage API (UMANS, Cline, OpenCode Go) are never marked depleted.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ProviderDepletionTracker:
    """Tracks per-provider depletion state, refreshed by a background polling task."""

    def __init__(
        self,
        clients: dict[str, Any],
        *,
        check_interval: int = 300,
        depletion_threshold: float = 99.9,
        ollama_session_cookie: str = "",
    ):
        self._clients = clients
        self._check_interval = check_interval
        self._threshold = depletion_threshold
        self._ollama_session_cookie = ollama_session_cookie
        self._depleted: set[str] = set()
        self._last_check: dict[str, dict] = {}  # provider -> last usage snapshot
        self._last_check_time: float = 0.0
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # ── Public API ──

    def is_depleted(self, provider: str) -> bool:
        """Return True if the provider is currently marked as depleted."""
        return provider in self._depleted

    def status(self) -> dict:
        """Return a snapshot of all provider depletion states for the dashboard."""
        result = {}
        for name, client in self._clients.items():
            entry: dict[str, Any] = {
                "configured": True,
                "depleted": name in self._depleted,
                "last_check": self._last_check.get(name),
            }
            if name not in ("ollama", "cmdcode"):
                entry["note"] = "No usage API — depletion tracking not available"
            result[name] = entry
        result["_meta"] = {
            "check_interval_seconds": self._check_interval,
            "depletion_threshold_pct": self._threshold,
            "last_poll": self._last_check_time,
        }
        return result

    # ── Background task ──

    async def start(self):
        """Start the background polling loop."""
        if self._task and not self._task.done():
            return
        self._running = True
        # Do one immediate check at startup
        await self._poll_once()
        self._task = asyncio.create_task(self._loop())
        logger.info("Provider depletion tracker started (interval=%ss, threshold=%s%%)",
                    self._check_interval, self._threshold)

    async def stop(self):
        """Stop the background polling loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Provider depletion tracker stopped")

    async def _loop(self):
        """Main polling loop — runs every _check_interval seconds."""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval)
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Depletion tracker poll error: %s", e)
                # Wait a bit before retrying to avoid tight error loops
                await asyncio.sleep(60)

    async def _poll_once(self):
        """Check all providers with a usage API and update depletion state."""
        self._last_check_time = time.time()
        tasks = []

        # Ollama — session_pct and weekly_pct from HTML scrape
        if "ollama" in self._clients:
            tasks.append(self._check_ollama())

        # CmdCode — remaining credits and weekly window from API
        if "cmdcode" in self._clients:
            tasks.append(self._check_cmdcode())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_ollama(self):
        """Check Ollama Cloud usage via session cookie scraping."""
        client = self._clients.get("ollama")
        if not client:
            return
        try:
            usage = await client.get_usage(self._ollama_session_cookie)
            self._last_check["ollama"] = usage

            session_pct = usage.get("session_pct")
            weekly_pct = usage.get("weekly_pct")
            # Also check nested dict form
            if isinstance(usage.get("session_usage"), dict):
                session_pct = usage["session_usage"].get("used_percentage", session_pct)
            if isinstance(usage.get("weekly_usage"), dict):
                weekly_pct = usage["weekly_usage"].get("used_percentage", weekly_pct)

            depleted = False
            if session_pct is not None and session_pct >= self._threshold:
                depleted = True
                logger.info("Ollama depleted: session usage %.1f%% >= %.1f%%", session_pct, self._threshold)
            if weekly_pct is not None and weekly_pct >= self._threshold:
                depleted = True
                logger.info("Ollama depleted: weekly usage %.1f%% >= %.1f%%", weekly_pct, self._threshold)

            if depleted:
                self._depleted.add("ollama")
            else:
                self._depleted.discard("ollama")
        except Exception as e:
            logger.warning("Ollama depletion check failed: %s", e)
            self._last_check["ollama"] = {"error": str(e)}

    async def _check_cmdcode(self):
        """Check Command Code usage via the /alpha/billing/credits API."""
        client = self._clients.get("cmdcode")
        if not client or not hasattr(client, "fetch_usage"):
            return
        try:
            usage = await client.fetch_usage()
            self._last_check["cmdcode"] = usage

            remaining = usage.get("remaining_credits", 0)
            monthly_used = usage.get("monthly_credits_used", 0)
            total_credits = remaining + monthly_used

            weekly_used = usage.get("weekly_used", 0)
            weekly_cap = usage.get("weekly_cap", 0)

            depleted = False

            # Monthly credits: if remaining < 0.1% of total → depleted
            if total_credits > 0:
                remaining_pct = (remaining / total_credits) * 100
                used_pct = 100 - remaining_pct
                if used_pct >= self._threshold:
                    depleted = True
                    logger.info("CmdCode depleted: monthly usage %.1f%% (remaining $%.4f of $%.4f)",
                                used_pct, remaining, total_credits)

            # Weekly window: if used >= threshold% of cap → depleted
            if weekly_cap > 0:
                weekly_pct = (weekly_used / weekly_cap) * 100
                if weekly_pct >= self._threshold:
                    depleted = True
                    logger.info("CmdCode depleted: weekly window %.1f%% (%.4f / %.4f)",
                                weekly_pct, weekly_used, weekly_cap)

            if depleted:
                self._depleted.add("cmdcode")
            else:
                self._depleted.discard("cmdcode")
        except Exception as e:
            logger.warning("CmdCode depletion check failed: %s", e)
            self._last_check["cmdcode"] = {"error": str(e)}
