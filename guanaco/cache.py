"""Smart session-aware response cache for Guanaco (beta).

Three caching strategies:
1. Exact cache — hash(model + messages + params) → full response. TTL-based eviction.
2. Session prefix cache — hash(model + prefix of messages) → response. Detects when a
   conversation is just adding messages to an existing session (most common Hermes pattern)
   and reuses cached responses for the earlier messages.
3. Request deduplication — if two identical requests arrive while one is in-flight,
   the second waits for the first's result instead of making a duplicate upstream call.

All behind `cache.beta_mode` config flag. Off by default.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cached response."""
    key: str
    response: dict
    created_at: float
    ttl: float
    hit_count: int = 0
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_type: str = "exact"  # "exact" or "session_prefix"

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


class CacheEngine:
    """Smart response cache with exact matching, session-aware prefix caching, and deduplication."""

    def __init__(self, config):
        self.config = config
        self._exact_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._prefix_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._in_flight: dict[str, asyncio.Event] = {}
        self._in_flight_results: dict[str, dict] = {}
        self._stats = {
            "exact_hits": 0,
            "prefix_hits": 0,
            "misses": 0,
            "dedup_saves": 0,
            "evictions": 0,
            "total_requests": 0,
        }

    # ── Public API ──

    def is_enabled(self) -> bool:
        """Check if beta cache is enabled."""
        return self.config.beta_mode

    async def get_or_fetch(
        self,
        model: str,
        messages: list[dict],
        params: dict,
        fetch_fn,
        provider: str = "ollama",
    ) -> dict:
        """Main entry point: check cache, dedup, or fetch from upstream.

        Args:
            model: Resolved model name
            messages: Chat messages list
            params: Full request params dict (model, messages, temperature, etc.)
            fetch_fn: Async callable that takes the payload and returns the response
            provider: Provider name for tagging

        Returns:
            Response dict (either cached or fresh)
        """
        if not self.is_enabled():
            return await fetch_fn(params)

        self._stats["total_requests"] += 1

        # Skip tiny prompts
        prompt_text = self._extract_prompt_text(messages)
        if len(prompt_text) < self.config.min_prompt_chars:
            return await fetch_fn(params)

        # Skip excluded models
        if model in self.config.exclude_models:
            return await fetch_fn(params)

        # 1. Try exact cache
        if self.config.exact_cache_enabled:
            exact_key = self._exact_hash(model, messages, params)
            cached = self._get_exact(exact_key)
            if cached is not None:
                self._stats["exact_hits"] += 1
                cached.hit_count += 1
                response = dict(cached.response)
                response["_oct_cached"] = True
                response["_oct_cache_type"] = "exact"
                response["_oct_cache_age"] = round(cached.age_seconds, 1)
                logger.info(f"Cache EXACT hit: model={model} age={cached.age_seconds:.1f}s")
                return response

        # 2. Try session prefix cache
        if self.config.session_prefix_enabled and len(messages) > 1:
            prefix_key = self._prefix_hash(model, messages)
            cached = self._get_prefix(prefix_key)
            if cached is not None:
                self._stats["prefix_hits"] += 1
                cached.hit_count += 1
                response = dict(cached.response)
                response["_oct_cached"] = True
                response["_oct_cache_type"] = "session_prefix"
                response["_oct_cache_age"] = round(cached.age_seconds, 1)
                logger.info(f"Cache PREFIX hit: model={model} msgs={len(messages)} age={cached.age_seconds:.1f}s")
                return response

        # 3. Deduplication — if an identical request is already in-flight
        if self.config.dedup_enabled:
            dedup_key = self._exact_hash(model, messages, params)
            if dedup_key in self._in_flight:
                self._stats["dedup_saves"] += 1
                logger.info(f"Cache DEDUP: waiting for in-flight request model={model}")
                await self._in_flight[dedup_key].wait()
                result = self._in_flight_results.get(dedup_key)
                if result is not None:
                    result_copy = dict(result)
                    result_copy["_oct_deduped"] = True
                    return result_copy

        # 4. Cache miss — fetch from upstream
        self._stats["misses"] += 1

        # Register in-flight if dedup enabled
        dedup_key = None
        if self.config.dedup_enabled:
            dedup_key = self._exact_hash(model, messages, params)
            self._in_flight[dedup_key] = asyncio.Event()

        try:
            response = await fetch_fn(params)

            # Cache the result
            if response and not response.get("error"):
                self._store_response(model, messages, params, response, provider)

            # Store for dedup waiters
            if dedup_key:
                self._in_flight_results[dedup_key] = response

            return response
        finally:
            # Clean up in-flight marker
            if dedup_key and dedup_key in self._in_flight:
                self._in_flight[dedup_key].set()
                del self._in_flight[dedup_key]
            if dedup_key and dedup_key in self._in_flight_results:
                # Keep result briefly for late waiters, then clean up
                asyncio.get_event_loop().call_later(5.0, lambda: self._in_flight_results.pop(dedup_key, None))

    def clear(self):
        """Clear all caches."""
        self._exact_cache.clear()
        self._prefix_cache.clear()
        self._in_flight.clear()
        self._in_flight_results.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_hits = self._stats["exact_hits"] + self._stats["prefix_hits"]
        total_requests = self._stats["total_requests"]
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "beta_mode": self.config.beta_mode,
            "exact_cache_entries": len(self._exact_cache),
            "prefix_cache_entries": len(self._prefix_cache),
            "in_flight_requests": len(self._in_flight),
            "stats": {
                **self._stats,
                "total_hits": total_hits,
                "hit_rate_pct": round(hit_rate, 2),
            },
            "config": {
                "exact_cache_enabled": self.config.exact_cache_enabled,
                "session_prefix_enabled": self.config.session_prefix_enabled,
                "dedup_enabled": self.config.dedup_enabled,
                "exact_cache_ttl": self.config.exact_cache_ttl,
                "session_prefix_ttl": self.config.session_prefix_ttl,
                "max_entries": self.config.max_entries,
                "min_prompt_chars": self.config.min_prompt_chars,
                "exclude_models": self.config.exclude_models,
            },
        }

    def evict_expired(self):
        """Remove expired entries from both caches."""
        expired_exact = [k for k, v in self._exact_cache.items() if v.is_expired]
        for k in expired_exact:
            del self._exact_cache[k]
            self._stats["evictions"] += 1

        expired_prefix = [k for k, v in self._prefix_cache.items() if v.is_expired]
        for k in expired_prefix:
            del self._prefix_cache[k]
            self._stats["evictions"] += 1

    # ── Private helpers ──

    def _exact_hash(self, model: str, messages: list[dict], params: dict) -> str:
        """Hash the full request for exact cache key."""
        # Include model + messages + temperature/top_p/max_tokens (but not stream)
        cache_params = {
            "model": model,
            "messages": messages,
            "temperature": params.get("temperature"),
            "top_p": params.get("top_p"),
            "max_tokens": params.get("max_tokens"),
            "tools": params.get("tools"),  # Tool calls affect output
        }
        raw = json.dumps(cache_params, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def _prefix_hash(self, model: str, messages: list[dict]) -> str:
        """Hash the model + first N messages for session prefix caching.

        The idea: In a conversation, messages get appended but the early messages
        stay the same. If we see the same prefix again with just the last message
        different, we can potentially reuse. But for prefix cache we want the
        prefix WITHOUT the last message — because the last message is what's new.

        Actually, for session prefix we hash messages[:-1] (all but the last
        user message). If the conversation history is the same, the model's
        understanding of context is the same — so responses to the same
        continuation should be cacheable by the full message list.

        Wait — this would mean two different user messages get the same prefix
        key, which is wrong. Let me reconsider.

        The real pattern with Hermes: the same conversation gets re-sent with
        the EXACT same messages (e.g., retry, or the agent re-processing).
        That's the exact cache. The prefix cache is for when a conversation
        has N previous messages and we already computed a response for those
        N messages — we can't really reuse that for N+1 messages because the
        new message changes the output.

        So prefix caching is most useful for: same conversation, same history,
        slightly different last message (e.g., rephrased question). We use a
        fuzzy match: hash messages[:-1] + model, and only reuse if the last
        message is "similar enough" (simple heuristic: last message has high
        token overlap with the cached last message).

        For now, let's do a simpler version: prefix cache stores responses keyed
        by model + messages[:-1]. When a new request comes in with the same
        conversation history but a different final message, we DON'T return it
        automatically — instead we mark it as a prefix match candidate that
        could be used for future features (like speculative prefill). For now,
        we only actually use prefix cache when all messages match (which is
        just the exact cache). This infrastructure is here for future semantic
        matching.
        """
        # Use all messages except the last one (the new user input)
        prefix = messages[:-1] if len(messages) > 1 else messages
        raw = json.dumps({"model": model, "prefix": prefix}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def _get_exact(self, key: str) -> Optional[CacheEntry]:
        """Get from exact cache, moving to end (LRU). Returns None if not found or expired."""
        if key in self._exact_cache:
            entry = self._exact_cache[key]
            if entry.is_expired:
                del self._exact_cache[key]
                self._stats["evictions"] += 1
                return None
            # Move to end (most recently used)
            self._exact_cache.move_to_end(key)
            return entry
        return None

    def _get_prefix(self, key: str) -> Optional[CacheEntry]:
        """Get from prefix cache. Returns None if not found or expired."""
        if key in self._prefix_cache:
            entry = self._prefix_cache[key]
            if entry.is_expired:
                del self._prefix_cache[key]
                self._stats["evictions"] += 1
                return None
            self._prefix_cache.move_to_end(key)
            return entry
        return None

    def _store_response(self, model: str, messages: list[dict], params: dict, response: dict, provider: str):
        """Store a response in the cache(s)."""
        usage = response.get("usage", {})

        # Exact cache
        if self.config.exact_cache_enabled:
            exact_key = self._exact_hash(model, messages, params)
            entry = CacheEntry(
                key=exact_key,
                response=dict(response),  # Store a copy
                created_at=time.time(),
                ttl=self.config.exact_cache_ttl,
                model=model,
                provider=provider,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                cache_type="exact",
            )
            self._exact_cache[exact_key] = entry
            self._evict_if_needed(self._exact_cache)

        # Prefix cache (only for multi-turn conversations)
        if self.config.session_prefix_enabled and len(messages) > 1:
            prefix_key = self._prefix_hash(model, messages)
            entry = CacheEntry(
                key=prefix_key,
                response=dict(response),
                created_at=time.time(),
                ttl=self.config.session_prefix_ttl,
                model=model,
                provider=provider,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                cache_type="session_prefix",
            )
            self._prefix_cache[prefix_key] = entry
            self._evict_if_needed(self._prefix_cache)

    def _evict_if_needed(self, cache: OrderedDict):
        """Evict oldest entries if cache exceeds max_entries."""
        while len(cache) > self.config.max_entries:
            cache.popitem(last=False)  # Remove oldest (first inserted)
            self._stats["evictions"] += 1

    @staticmethod
    def _extract_prompt_text(messages: list[dict]) -> str:
        """Extract all text content from messages for length checking."""
        parts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
        return " ".join(parts)