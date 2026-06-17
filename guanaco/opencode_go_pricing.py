"""OpenCode Go per-model pricing.

Prices are per 1M tokens. Cached read/write prices are used to compute
OpenCode Go's prompt-cache-hit/miss token costs for local spend tracking.
Source: https://opencode.ai/docs/go (June 2026)
"""

from __future__ import annotations

from typing import Optional

# Provider hint: model_id -> endpoint override. Most models use the OpenAI-compatible
# /chat/completions endpoint; MiniMax and Qwen models use Anthropic's /messages.
GO_MESSAGES_MODELS = {
    "minimax-m3", "minimax-m2.7", "minimax-m2.5",
    "qwen3.7-max", "qwen3.7-plus", "qwen3.6-plus",
}

# Per-1M-token prices in USD.
MODEL_PRICING: dict[str, dict[str, Optional[float]]] = {
    "glm-5.1":        {"input": 1.40, "output": 4.40, "cached_read": 0.26,  "cached_write": None},
    "glm-5":          {"input": 1.00, "output": 3.20, "cached_read": 0.20,  "cached_write": None},
    "kimi-k2.7-code": {"input": 0.95, "output": 4.00, "cached_read": 0.19,  "cached_write": None},
    "kimi-k2.6":      {"input": 0.95, "output": 4.00, "cached_read": 0.16,  "cached_write": None},
    "kimi-k2.7":      {"input": 0.95, "output": 4.00, "cached_read": 0.19,  "cached_write": None},
    "kimi-k2.5":      {"input": 0.95, "output": 4.00, "cached_read": 0.19,  "cached_write": None},
    "mimo-v2.5":      {"input": 0.14, "output": 0.28, "cached_read": 0.0028,"cached_write": None},
    "mimo-v2.5-pro":  {"input": 1.74, "output": 3.48, "cached_read": 0.0145,"cached_write": None},
    "minimax-m3":     {"input": 0.30, "output": 1.20, "cached_read": 0.06,  "cached_write": 0.375},
    "minimax-m2.7":   {"input": 0.30, "output": 1.20, "cached_read": 0.06,  "cached_write": 0.375},
    "minimax-m2.5":   {"input": 0.30, "output": 1.20, "cached_read": 0.06,  "cached_write": 0.375},
    "qwen3.7-max":    {"input": 2.50, "output": 7.50, "cached_read": 0.50,  "cached_write": 3.125},
    "qwen3.7-plus":   {"input": 0.40, "output": 1.60, "cached_read": 0.04,  "cached_write": 0.50},
    "qwen3.6-plus":   {"input": 0.50, "output": 3.00, "cached_read": 0.05,  "cached_write": 0.625},
    "deepseek-v4-pro":{"input": 1.74, "output": 3.48, "cached_read": 0.0145,"cached_write": None},
    "deepseek-v4-flash":{"input": 0.14,"output": 0.28, "cached_read": 0.0028,"cached_write": None},
}


def normalize_go_model_id(model: str) -> str:
    """Strip provider prefix and suffix."""
    if model.startswith("opencode-go/"):
        model = model.split("/", 1)[1]
    return model.split(":")[0].lower()


def get_pricing(model_id: str) -> dict[str, Optional[float]]:
    """Return pricing for a model id, or zeros if unknown."""
    base = normalize_go_model_id(model_id)
    return MODEL_PRICING.get(base, {"input": 0.0, "output": 0.0, "cached_read": 0.0, "cached_write": None})


def estimate_cost(model_id: str, prompt_tokens: int, completion_tokens: int,
                  prompt_cache_hit_tokens: int = 0, prompt_cache_miss_tokens: int = 0) -> float:
    """Estimate dollar cost for an OpenCode Go request.

    Uses cached-read price for hit tokens, input price for miss tokens,
    and output price for completion tokens.
    """
    p = get_pricing(model_id)
    cached_read = p.get("cached_read") or p["input"]
    # Treat any prompt tokens not explicitly reported as hits as input-price tokens.
    if prompt_cache_hit_tokens or prompt_cache_miss_tokens:
        prompt_cost = (
            prompt_cache_hit_tokens * (cached_read / 1_000_000)
            + prompt_cache_miss_tokens * (p["input"] / 1_000_000)
        )
    else:
        prompt_cost = prompt_tokens * (p["input"] / 1_000_000)
    completion_cost = completion_tokens * (p["output"] / 1_000_000)
    return prompt_cost + completion_cost


def uses_messages_endpoint(model_id: str) -> bool:
    base = normalize_go_model_id(model_id)
    return base in GO_MESSAGES_MODELS
