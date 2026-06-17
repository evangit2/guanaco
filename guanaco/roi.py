"""
OpenRouter price-based subscription value calculator.

This module:
1. Fetches live model prices from OpenRouter's API
2. Maps Ollama Cloud model names to OpenRouter model IDs
3. Calculates "what would this usage have cost on OpenRouter?"
4. Compares against subscription price to show value multiplier

Prices are cached for 1 hour to avoid rate limits.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# ── OpenRouter API ──
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_CACHE_TTL = 3600  # 1 hour

# Model family mappings: ollama_name_fragment -> openrouter_id_fragment
# These are used when exact match fails
FAMILY_MAP = {
    "gemma": ("google/gemma", "google/gemma"),
    "gemma3": ("google/gemma", "google/gemma"),
    "gemma4": ("google/gemma", "google/gemma"),
    "qwen": ("qwen/qwen", "qwen/qwen"),
    "qwen3": ("qwen/qwen3", "qwen/qwen"),
    "qwen3.5": ("qwen/qwen3.5", "qwen/qwen"),
    "qwen3-vl": ("qwen/qwen3-vl", "qwen/qwen3-vl"),
    "qwen3-coder": ("qwen/qwen3-coder", "qwen/qwen"),
    "qwen3-next": ("qwen/qwen3-next", "qwen/qwen"),
    "deepseek": ("deepseek/deepseek", "deepseek/deepseek"),
    "deepseek-v3": ("deepseek/deepseek", "deepseek/deepseek-v3"),
    "deepseek-v4": ("deepseek/deepseek", "deepseek/deepseek-v4"),
    "gpt-oss": ("openai/gpt-oss", "openai/gpt"),
    "minimax": ("minimax/minimax", "minimax/minimax"),
    "glm": ("zhipu/glm", "zhipu/glm"),
    "glm-5": ("zhipu/glm-5", "zhipu/glm"),
    "kimi": ("moonshot/kimi", "moonshot/kimi"),
    "kimi-k2": ("moonshot/kimi", "moonshot/kimi"),
    "devstral": ("mistral/devstral", "mistral/devstral"),
    "mistral": ("mistral/mistral", "mistral/mistral"),
    "ministral": ("mistral/ministral", "mistral/ministral"),
    "nemotron": ("nvidia/nemotron", "nvidia/nemotron"),
    "cogito": ("cogito/cogito", "cogito/"),
    "gemini": ("google/gemini", "google/gemini"),
    "rnj": ("", ""),
}


def _normalized(name: str) -> str:
    """Strip provider prefix, ~leaderboard prefix, :cloud/:local suffixes, and lower-case."""
    base = name.split(":")[0].lower()
    # Strip ~ prefix (leaderboard indicator on OpenRouter)
    if base.startswith("~"):
        base = base[1:]
    # Strip provider/ prefix (e.g. moonshotai/kimi-k2.6 → kimi-k2.6)
    if "/" in base:
        base = base.split("/", 1)[1]
    if base.endswith("-cloud"):
        base = base[:-6]
    return base


def _model_size(name: str) -> int:
    """Extract parameter size in billions from model name, 0 if unknown."""
    import re
    m = re.search(r"(\d+)(b|t)", name, re.I)
    if not m:
        return 0
    n = int(m.group(1))
    unit = m.group(2).lower()
    return n * 1000 if unit == "t" else n


class PriceCache:
    """Holds cached OpenRouter prices in memory with TTL."""
    def __init__(self):
        self.prices: dict[str, dict] = {}
        self.fetched_at: float = 0

    def is_fresh(self) -> bool:
        return self.prices and (time.time() - self.fetched_at) < OPENROUTER_CACHE_TTL

    def fetch(self) -> dict[str, dict]:
        if self.is_fresh():
            logger.debug("Using cached OpenRouter prices")
            return self.prices

        prices = {}
        try:
            logger.info("Fetching OpenRouter model prices...")
            r = httpx.get(OPENROUTER_MODELS_URL, timeout=30)
            r.raise_for_status()
            data = r.json()
            for model in data.get("data", []):
                model_id = model.get("id", "")
                pricing = model.get("pricing", {})
                prompt = float(pricing.get("prompt", 0) or 0)
                completion = float(pricing.get("completion", 0) or 0)
                cache_read = float(pricing.get("input_cache_read", 0) or 0)
                if prompt > 0 or completion > 0:
                    # Convert from per-token ($/token) to per-million-tokens ($/Mt)
                    entry = {
                        "prompt": prompt * 1_000_000,
                        "completion": completion * 1_000_000,
                    }
                    if cache_read > 0:
                        entry["input_cache_read"] = cache_read * 1_000_000
                    prices[model_id] = entry
            self.prices = prices
            self.fetched_at = time.time()
            logger.info(f"Fetched {len(prices)} OpenRouter price entries")
        except Exception as e:
            logger.warning(f"Failed to fetch OpenRouter prices: {e}")
        return self.prices


# Singleton cache
_price_cache = PriceCache()


def _find_best_price(prices: dict, ollama_name: str) -> dict:
    """
    Given OpenRouter prices dict {model_id: {prompt, completion}} and an
    Ollama model name, return best matching price dict.
    """
    norm = _normalized(ollama_name)
    size = _model_size(ollama_name)

    # 1. Exact normalized match (handles provider-prefixed OR IDs like moonshotai/kimi-k2.6)
    for orouter_id, price_info in prices.items():
        if _normalized(orouter_id) == norm:
            return price_info

    # 2. Family prefix match — use raw orouter_id so provider/ prefix matches
    best_family_price = None
    best_family_score = -9999
    for orouter_id, price_info in prices.items():
        for frag, (family_exact, family_prefix) in FAMILY_MAP.items():
            if frag in norm and family_prefix and family_prefix in orouter_id:
                # Score by size closeness
                o_size = _model_size(orouter_id)
                score = -(abs(o_size - size))  # higher = closer size
                if score > best_family_score:
                    best_family_score = score
                    best_family_price = price_info
    if best_family_price:
        return best_family_price

    # 3. Same parameter size match
    if size > 0:
        for orouter_id, price_info in prices.items():
            if _model_size(orouter_id) == size:
                return price_info

    # 4. Size-window fallback
    candidates = []
    for orouter_id, price_info in prices.items():
        o_size = _model_size(orouter_id)
        if o_size == 0:
            continue
        window = max(20, size * 0.5)
        if abs(o_size - size) <= window:
            candidates.append(price_info)
    if candidates:
        candidates.sort(key=lambda p: p["completion"] + p["prompt"])
        return candidates[len(candidates) // 2]

    # 5. Global average
    all_prices = [p for p in prices.values() if p["prompt"] > 0 or p["completion"] > 0]
    if all_prices:
        avg_prompt = sum(p["prompt"] for p in all_prices) / len(all_prices)
        avg_comp = sum(p["completion"] for p in all_prices) / len(all_prices)
        return {"prompt": avg_prompt, "completion": avg_comp}

    return {"prompt": 0.0, "completion": 0.0}


def _map_usage_to_prices(usage_by_model: dict, prices: dict, cache_hit_pct: float = 0.0) -> dict:
    """
    Map usage to prices, optionally applying prompt-cache hit discount.

    For models with input_cache_read pricing (e.g. Claude Fable, Qwen, Minimax):
      - uncached_prompt = prompt_tokens * (1 - cache_hit_pct)
      - cached_prompt   = prompt_tokens * cache_hit_pct
      - prompt cost     = uncached_prompt * prompt_price + cached_prompt * cache_read_price
    """
    result = {}
    cache_rate = max(0.0, min(100.0, cache_hit_pct)) / 100.0
    for model, usage in usage_by_model.items():
        price = _find_best_price(prices, model)
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        
        # Apply cache discount if model supports it
        if "input_cache_read" in price and cache_rate > 0:
            uncached_pt = pt * (1 - cache_rate)
            cached_pt = pt * cache_rate
            prompt_cost = (uncached_pt / 1_000_000 * price["prompt"]) + (cached_pt / 1_000_000 * price["input_cache_read"])
            # Store effective prompt price for display
            effective_prompt = prompt_cost / (pt / 1_000_000) if pt > 0 else price["prompt"]
        else:
            prompt_cost = (pt / 1_000_000) * price["prompt"]
            effective_prompt = price["prompt"]
        
        comp_cost = (ct / 1_000_000) * price["completion"]
        cost = prompt_cost + comp_cost

        result[model] = {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "prompt_per_mt": effective_prompt,
            "completion_per_mt": price["completion"],
            "cost": cost,
            "matched_price_model": _find_best_price.__module__,
            "cache_applied": "input_cache_read" in price and cache_rate > 0,
            "cache_read_per_mt": price.get("input_cache_read"),
        }
    return result


def get_usage_from_analytics(db_path: Path | str, since: float = 0) -> tuple[dict, float]:
    usage_by_model = {}
    total_weighted = 0.0
    try:
        conn = sqlite3.connect(str(db_path))
        # Detect whether the request_log table has the usage_multiplier column
        # (older DBs created before v0.4.3 may not).
        has_multiplier = False
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(request_log)")}
            has_multiplier = "usage_multiplier" in cols
        except Exception:
            pass

        if has_multiplier:
            sql = """SELECT model,
                            IFNULL(SUM(prompt_tokens),0),
                            IFNULL(SUM(completion_tokens),0),
                            IFNULL(SUM(prompt_tokens * IFNULL(usage_multiplier,1.0)),0),
                            IFNULL(SUM(completion_tokens * IFNULL(usage_multiplier,1.0)),0),
                            COUNT(*)
                     FROM request_log WHERE type='llm' AND ts > ? GROUP BY model"""
        else:
            sql = """SELECT model,
                            IFNULL(SUM(prompt_tokens),0),
                            IFNULL(SUM(completion_tokens),0),
                            COUNT(*)
                     FROM request_log WHERE type='llm' AND ts > ? GROUP BY model"""

        rows = conn.execute(sql, (since,)).fetchall()
        for row in rows:
            if has_multiplier:
                model, pt, ct, w_pt, w_ct, req_count = row
            else:
                model, pt, ct, req_count = row
                w_pt, w_ct = float(pt), float(ct)
            usage_by_model[model] = {
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "weighted_prompt": w_pt,
                "weighted_completion": w_ct,
                "requests": req_count,
            }
            total_weighted += (w_pt + w_ct)
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to read analytics DB: {e}")
    return usage_by_model, total_weighted


def _get_ollama_week_start() -> float:
    """Return Unix timestamp of the most recent Sunday at 20:00 UTC.

    Ollama resets its weekly quota every Sunday at 8 PM UTC.
    """
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    days_since_sunday = now.weekday() + 1 if now.weekday() != 6 else 0
    sunday = now - timedelta(days=days_since_sunday)
    reset = sunday.replace(hour=20, minute=0, second=0, microsecond=0)
    if now < reset:
        reset -= timedelta(days=7)
    return reset.timestamp()


def calculate_roi(
    db_path: Path | str,
    subscription_monthly: float = 20.0,
    weekly_pct_used: float = 0.0,
    cache_hit_pct: float = 0.0,
) -> dict:
    """
    Calculate subscription value vs OpenRouter pay-as-you-go.

    Args:
        db_path: path to analytics SQLite DB
        subscription_monthly: monthly sub cost (20 Pro, 100 Max)
        weekly_pct_used: % of weekly quota consumed (from usage check)
        cache_hit_pct: estimated % of prompt tokens hitting cache (0-100).
                       Used for models with input_cache_read pricing on OpenRouter.

    Returns dict with:
        total_cost, total_prompt_tokens, total_completion_tokens,
        total_weighted_tokens, weekly_value, monthly_value,
        subscription_monthly, plan ("pro"|"max"), roi_multiplier,
        weekly_pct_used, cache_hit_pct, prices_stale,
        by_model[] with prompt_tokens, completion_tokens, prompt_per_mt, completion_per_mt, cost,
        unmatched_models[] names with no price match
    """
    # 1. Fetch prices
    prices = _price_cache.fetch()
    prices_stale = not prices or len(prices) < 10
    plan = "pro" if subscription_monthly <= 25 else "max"

    # 2. Get usage since Ollama's weekly reset (Sunday 20:00 UTC)
    since = _get_ollama_week_start()
    usage_by_model, total_weighted = get_usage_from_analytics(db_path, since)

    # 3. Map usage to prices (with cache hit estimation)
    priced = _map_usage_to_prices(usage_by_model, prices, cache_hit_pct)

    total_cost = sum(m["cost"] for m in priced.values())
    total_prompt = sum(m["prompt_tokens"] for m in priced.values())
    total_comp = sum(m["completion_tokens"] for m in priced.values())

    # 4. Extrapolate to 100% weekly
    if weekly_pct_used > 0:
        weekly_value = total_cost / (weekly_pct_used / 100.0)
    else:
        weekly_value = total_cost

    monthly_value = weekly_value * 4
    roi_multiplier = (monthly_value / subscription_monthly) if subscription_monthly > 0 else 0

    unmatched = [m for m in usage_by_model if priced.get(m, {}).get("cost", 0) == 0]

    # Per-model breakdown
    by_model = []
    for model, detail in priced.items():
        pt = detail["prompt_tokens"]
        ct = detail["completion_tokens"]
        by_model.append({
            "model": model,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "prompt_per_mt": round(detail["prompt_per_mt"], 6),
            "completion_per_mt": round(detail["completion_per_mt"], 6),
            "cost": round(detail["cost"], 4),
            "pct_of_total": round((detail["cost"] / total_cost * 100), 2) if total_cost > 0 else 0,
            "cache_applied": detail.get("cache_applied", False),
            "cache_read_per_mt": round(detail.get("cache_read_per_mt"), 6) if detail.get("cache_read_per_mt") else None,
        })
    by_model.sort(key=lambda x: x["cost"], reverse=True)

    return {
        "total_cost": round(total_cost, 2),
        "total_prompt_tokens": int(total_prompt),
        "total_completion_tokens": int(total_comp),
        "total_raw_tokens": int(total_prompt + total_comp),
        "total_weighted_tokens": int(total_weighted),
        "weekly_value": round(weekly_value, 2),
        "monthly_value": round(monthly_value, 2),
        "subscription_monthly": subscription_monthly,
        "plan": plan,
        "roi_multiplier": round(roi_multiplier, 2),
        "weekly_pct_used": weekly_pct_used,
        "cache_hit_pct": cache_hit_pct,
        "prices_stale": prices_stale,
        "by_model": by_model,
        "unmatched_models": unmatched,
        "price_models_available": len(prices),
    }


def calculate_model_value_comparison(
    db_path: Path | str,
    subscription_monthly: float = 20.0,
    weekly_pct_used: float = 0.0,
    session_pct_used: float = 0.0,
    period: str = "weekly",  # "weekly" or "session"
) -> dict:
    """
    Score each model: positive = gave more value than its fair share of sub.

    For each model actually used:
      - actual_value = what those tokens would cost on OpenRouter
      - fair_share   = (model's weighted tokens / total weighted tokens) * subscription_cost_for_period
      - score        = actual_value - fair_share
                     positive = model punches above its weight (good deal)
                     negative = model is expensive for its token share (bad deal)
    """
    prices = _price_cache.fetch()
    if not prices:
        return {"error": "No OpenRouter prices available", "models": []}

    # Determine time window and usage %
    now = time.time()
    if period == "session":
        since = now - (5 * 3600)  # 5-hour session window
        pct_used = session_pct_used
    else:
        since = now - (7 * 24 * 3600)  # 7-day weekly window
        pct_used = weekly_pct_used

    usage_by_model, total_weighted = get_usage_from_analytics(db_path, since)
    if not usage_by_model:
        return {"models": [], "summary": {}}

    # Period subscription cost
    weekly_sub_cost = subscription_monthly / 4.0
    if pct_used > 0:
        period_sub_cost = weekly_sub_cost * (pct_used / 100.0)
    else:
        period_sub_cost = weekly_sub_cost

    # Map to prices
    priced = _map_usage_to_prices(usage_by_model, prices)

    # Per-model scoring
    models = []
    total_actual_value = 0.0

    for model, usage in usage_by_model.items():
        detail = priced.get(model, {})
        actual_value = detail.get("cost", 0.0)
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        model_raw = pt + ct
        w_pt = usage.get("weighted_prompt", 0)
        w_ct = usage.get("weighted_completion", 0)
        model_weighted = w_pt + w_ct

        # Fair share of subscription based on WEIGHTED token proportion
        # (subscription quota is weighted; actual value uses raw tokens at OpenRouter prices)
        fair_share = (model_weighted / total_weighted * period_sub_cost) if total_weighted > 0 else 0
        score = actual_value - fair_share
        score_pct = (score / fair_share * 100) if fair_share > 0 else 0

        total_actual_value += actual_value

        models.append({
            "model": model,
            "requests": usage.get("requests", 0),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": int(model_raw),
            "weighted_tokens": int(model_weighted),
            "pct_of_total_tokens": round((model_weighted / total_weighted * 100), 2) if total_weighted > 0 else 0,
            "actual_value": round(actual_value, 2),
            "fair_share": round(fair_share, 2),
            "score": round(score, 2),
            "score_pct": round(score_pct, 1),
            "prompt_per_mt": round(detail.get("prompt_per_mt", 0), 6),
            "completion_per_mt": round(detail.get("completion_per_mt", 0), 6),
        })

    # Sort by score descending (best value first)
    models.sort(key=lambda x: x["score"], reverse=True)

    # Summary
    net_score = total_actual_value - period_sub_cost

    # Compute total raw tokens across all models for the summary
    total_raw = sum(m.get("prompt_tokens", 0) + m.get("completion_tokens", 0) for m in models)

    return {
        "period": period,
        "subscription_monthly": subscription_monthly,
        "period_sub_cost": round(period_sub_cost, 2),
        "total_actual_value": round(total_actual_value, 2),
        "total_raw_tokens": int(total_raw),
        "total_weighted_tokens": int(total_weighted),
        "net_score": round(net_score, 2),
        "models": models,
        "prices_stale": len(prices) < 10,
        "price_models_available": len(prices),
    }


def get_cached_roi() -> dict:
    """Return last calculated ROI, or minimal default."""
    return _price_cache.prices  # placeholder; real caching is in config
