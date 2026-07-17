"""UMANS concurrency tracker — monitors concurrent sessions and gates routing.

A background task polls the UMANS /v1/usage endpoint every N seconds (default 15).
When concurrent_sessions >= the configured threshold (default 3), UMANS is marked
as "saturated" and skipped for unprefixed model routing — requests fall through
to the next provider in the priority list (e.g. ollama, opencode_go, cmdcode).

Explicit provider prefixes (e.g. umans/glm-5.2) always bypass the saturation
check, matching the same pattern used by the depletion tracker.

The tracker also records a rolling history of concurrent session counts for the
dashboard's live gauge and historical graph.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Max history entries to keep in memory (for dashboard sparkline)
_MAX_HISTORY = 240  # 240 * 15s = 1 hour of history


class UmansConcurrencyTracker:
    """Tracks UMANS concurrent session usage, refreshed by a background polling task.

    Usage:
        tracker = UmansConcurrencyTracker(clients, check_interval=15, saturation_threshold=3)
        await tracker.start()

        if tracker.is_saturated("umans"):
            # route to next provider
            ...

        tracker.status()  # for dashboard
    """

    def __init__(
        self,
        clients: dict[str, Any],
        *,
        check_interval: int = 15,
        saturation_threshold: int = 3,
        enabled: bool = True,
    ):
        self._clients = clients
        self._check_interval = check_interval
        self._threshold = saturation_threshold
        self._enabled = enabled
        self._saturated_providers: set[str] = set()
        self._last_check: dict[str, dict] = {}
        self._last_check_time: float = 0.0
        self._history: deque[dict] = deque(maxlen=_MAX_HISTORY)
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # ── Public API ──

    def is_saturated(self, provider: str) -> bool:
        """Return True if the provider is currently at/above the concurrency threshold."""
        if not self._enabled:
            return False
        return provider in self._saturated_providers

    def status(self) -> dict:
        """Return a snapshot of concurrency state for the dashboard."""
        result: dict[str, Any] = {}
        for name, client in self._clients.items():
            if name != "umans":
                continue
            entry: dict[str, Any] = {
                "configured": True,
                "saturated": name in self._saturated_providers,
                "last_check": self._last_check.get(name),
            }
            result[name] = entry

        result["_meta"] = {
            "enabled": self._enabled,
            "check_interval_seconds": self._check_interval,
            "saturation_threshold": self._threshold,
            "last_poll": self._last_check_time,
            "history": list(self._history),
        }
        return result

    def get_concurrent_count(self) -> int:
        """Return the last known concurrent session count for UMANS."""
        last = self._last_check.get("umans", {})
        return last.get("concurrent_sessions", 0)

    def get_limit(self) -> Optional[int]:
        """Return the concurrency limit from the last check."""
        last = self._last_check.get("umans", {})
        return last.get("limit")

    # ── Background task ──

    async def start(self):
        """Start the background polling loop."""
        if not self._enabled:
            logger.info("UMANS concurrency tracker disabled, skipping start")
            return
        if self._task and not self._task.done():
            return
        self._running = True
        # Do one immediate check at startup
        await self._poll_once()
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "UMANS concurrency tracker started (interval=%ss, threshold=%s)",
            self._check_interval,
            self._threshold,
        )

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
        logger.info("UMANS concurrency tracker stopped")

    async def _loop(self):
        """Main polling loop — runs every _check_interval seconds."""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval)
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("UMANS concurrency tracker poll error: %s", e)
                await asyncio.sleep(self._check_interval)

    async def _poll_once(self):
        """Check UMANS concurrency and update saturation state."""
        self._last_check_time = time.time()

        umans_client = self._clients.get("umans")
        if not umans_client or not hasattr(umans_client, "get_concurrency"):
            return

        try:
            data = await umans_client.get_concurrency()
            concurrent = data.get("concurrent", 0)
            limit = data.get("limit")
            user_id = data.get("user_id")

            snapshot = {
                "concurrent_sessions": concurrent,
                "limit": limit,
                "user_id": user_id,
                "ts": self._last_check_time,
            }
            self._last_check["umans"] = snapshot

            # Record to history
            self._history.append({
                "ts": self._last_check_time,
                "concurrent": concurrent,
                "limit": limit,
            })

            # Update saturation state
            if concurrent >= self._threshold:
                if "umans" not in self._saturated_providers:
                    logger.info(
                        "UMANS concurrency saturated: %d/%s sessions (threshold=%s) — "
                        "routing unprefixed models to fallback providers",
                        concurrent,
                        limit or "?",
                        self._threshold,
                    )
                self._saturated_providers.add("umans")
            else:
                if "umans" in self._saturated_providers:
                    logger.info(
                        "UMANS concurrency recovered: %d/%s sessions — resuming normal routing",
                        concurrent,
                        limit or "?",
                    )
                self._saturated_providers.discard("umans")

        except Exception as e:
            logger.warning("UMANS concurrency check failed: %s", e)
            self._last_check["umans"] = {"error": str(e), "ts": self._last_check_time}
