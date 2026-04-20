"""Concurrency limiter for Ollama Cloud requests.

Prevents 429 "too many concurrent requests" errors by:
1. Bounding concurrent in-flight requests via an asyncio.Semaphore
2. Auto-retrying 429s with exponential backoff
3. Tracking recent 429 rate to auto-suggest reducing max_concurrent
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Optional

import httpx

log = logging.getLogger("guanaco.concurrency")


class OllamaConcurrencyLimiter:
    """Limits concurrent Ollama Cloud requests and handles 429 backoff.

    Usage:
        limiter = OllamaConcurrencyLimiter(max_concurrent=8)
        async with limiter:
            result = await client.chat_completion(payload)
    """

    def __init__(self, max_concurrent: int = 0, max_429_retries: int = 2, base_backoff: float = 1.0):
        """
        Args:
            max_concurrent: Max simultaneous Ollama requests. 0 = unlimited (no semaphore).
            max_429_retries: How many times to retry on HTTP 429 before giving up.
            base_backoff: Base backoff in seconds for 429 retry (doubles each retry).
        """
        self.max_concurrent = max_concurrent
        self.max_429_retries = max_429_retries
        self.base_backoff = base_backoff
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._429_times: deque = deque(maxlen=50)  # Track last 50 429 timestamps
        self._active_count = 0
        self._lock = asyncio.Lock()

    def _ensure_semaphore(self):
        """Lazily create semaphore (can't do in __init__ if no event loop yet)."""
        if self.max_concurrent > 0 and self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def recent_429_rate(self) -> float:
        """429s per minute over the last 60 seconds. Used for dashboard display."""
        now = time.time()
        while self._429_times and now - self._429_times[0] > 60:
            self._429_times.popleft()
        return len(self._429_times)  # count in last 60s

    def _record_429(self):
        self._429_times.append(time.time())
        log.warning(
            "Ollama 429 rate limit hit (recent 429s: %d/min, active: %d/%s)",
            self.recent_429_rate, self._active_count,
            self.max_concurrent or "∞"
        )

    async def __aenter__(self):
        self._ensure_semaphore()
        if self._semaphore:
            await self._semaphore.acquire()
        async with self._lock:
            self._active_count += 1
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        async with self._lock:
            self._active_count -= 1
        if self._semaphore:
            self._semaphore.release()
        return False  # Don't suppress exceptions

    def should_retry_429(self, exc: Exception) -> bool:
        """Check if an exception is a 429 that we should retry."""
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
            self._record_429()
            return True
        return False

    async def backoff_and_retry(self, attempt: int) -> float:
        """Calculate and sleep for exponential backoff. Returns the backoff duration."""
        backoff = self.base_backoff * (2 ** attempt)
        backoff = min(backoff, 10.0)  # Cap at 10s per retry
        # Add jitter (±25%) to avoid thundering herd
        import random
        jitter = backoff * 0.25 * (random.random() * 2 - 1)
        wait = max(0.1, backoff + jitter)
        log.info("429 backoff: waiting %.1fs (attempt %d)", wait, attempt + 1)
        await asyncio.sleep(wait)
        return wait

    def get_stats(self) -> dict:
        """Return current concurrency stats for dashboard display."""
        return {
            "max_concurrent": self.max_concurrent,
            "active_count": self._active_count,
            "recent_429_rate": self.recent_429_rate,
        }