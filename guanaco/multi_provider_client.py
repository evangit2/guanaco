"""Multi-provider chat client dispatcher.

The LLM router calls chat_completion/chat_completion_stream on a single client.
This wrapper routes those calls to Ollama Cloud or OpenCode Go based on the
model id in the payload, while still honoring per-account api_key overrides.
"""

from __future__ import annotations

from typing import Any, Optional

from guanaco.accounts import AccountPool, provider_for_model


class MultiProviderChatClient:
    """Looks like OllamaClient to the router but dispatches by model provider."""

    def __init__(self, clients: dict[str, Any], account_pool: Optional[AccountPool] = None):
        self._clients = clients
        self._account_pool = account_pool
        self.api_key = ""  # Router may read this; real keys come from account_pool overrides
        self.timeout = 120.0
        self._provider_priority: Optional[list[str]] = None
        # Saturated/depleted providers to skip (set by router at runtime)
        self._skip_providers: set[str] = set()

    def set_provider_priority(self, priority: Optional[list[str]]):
        """Called by the router to propagate the configured priority order."""
        self._provider_priority = priority

    def set_skip_providers(self, skip: set[str]):
        """Called by the router to mark providers that should be skipped
        (e.g. UMANS when saturated).  Set to empty set to clear."""
        self._skip_providers = skip

    def _client_for(self, model: str):
        provider = provider_for_model(model, provider_priority=self._provider_priority)
        # If the resolved provider is marked for skipping (saturated/depleted),
        # try the next provider in the priority list that also claims this model.
        # BUT: explicit provider prefixes (e.g. "umans/glm-5.2") always bypass
        # saturation — the user explicitly requested that provider.
        _model_lower = (model or "").lower().strip()
        _has_explicit_prefix = any(
            _model_lower.startswith(p)
            for p in ("umans/", "umans-", "ollama/", "opencode-go/",
                      "cline/", "cmdcode/")
        )
        if provider in self._skip_providers and self._provider_priority and not _has_explicit_prefix:
            from guanaco.accounts import (
                _normalize_model_for_provider,
                KNOWN_GO_MODELS, KNOWN_OLLAMA_MODELS,
                KNOWN_UMANS_MODELS, KNOWN_CLINE_MODELS, KNOWN_CMDCODE_MODELS,
            )
            canon = _normalize_model_for_provider(model)
            claiming = []
            if canon in KNOWN_GO_MODELS: claiming.append("opencode_go")
            if canon in KNOWN_UMANS_MODELS: claiming.append("umans")
            if canon in KNOWN_CLINE_MODELS: claiming.append("cline")
            if canon in KNOWN_CMDCODE_MODELS: claiming.append("cmdcode")
            if canon in KNOWN_OLLAMA_MODELS: claiming.append("ollama")
            # If no provider claims this model, fall through the priority list
            # to find any non-skipped provider.
            search_list = claiming if claiming else list(self._provider_priority)
            for p in self._provider_priority:
                if p in search_list and p not in self._skip_providers:
                    provider = p
                    break
        client = self._clients.get(provider)
        if client is None and provider == "opencode_go":
            # If user omitted the prefix but has Go accounts, see if the model is a known Go model.
            go_models = {
                "glm-5.1", "glm-5", "kimi-k2.7-code", "kimi-k2.7", "kimi-k2.6", "kimi-k2.5",
                "deepseek-v4-pro", "deepseek-v4-flash", "mimo-v2.5", "mimo-v2.5-pro",
                "minimax-m3", "minimax-m2.7", "minimax-m2.5",
                "qwen3.7-max", "qwen3.7-plus", "qwen3.6-plus",
            }
            if any(m in model.lower() for m in go_models):
                client = self._clients.get("opencode_go")
        if client is None and provider == "ollama":
            # Known UMANS models have a "umans-" prefix even when no provider is configured.
            if model.lower().startswith("umans-") or model.lower().startswith("umans/"):
                client = self._clients.get("umans")
        # Check custom providers by name prefix (e.g. "openrouter/anthropic/claude-...")
        if client is None and "/" in model:
            prefix = model.split("/")[0]
            client = self._clients.get(prefix)
        if client is None:
            # Provider requested but not configured — fall back to whatever is available.
            for fallback_provider in ("ollama", "opencode_go", "umans", "cline", "cmdcode"):
                client = self._clients.get(fallback_provider)
                if client is not None:
                    break
        return client

    @property
    def provider_keys(self) -> list:
        """Return configured provider names for routing introspection."""
        return list(self._clients.keys())

    async def chat_completion(self, payload: dict, api_key: Optional[str] = None):
        payload = dict(payload)
        model = payload.get("model", "")
        client = self._client_for(model)
        if not client:
            raise RuntimeError("No LLM provider configured")
        # Strip provider prefixes
        if model.startswith("opencode-go/") and "opencode_go" in self._clients and client is self._clients["opencode_go"]:
            payload["model"] = model[len("opencode-go/"):]
        if client is self._clients.get("umans") and (model.startswith("umans/") or model.lower().startswith("umans-")):
            payload["model"] = model
        if client is self._clients.get("cline") and model.lower().startswith("cline/"):
            payload["model"] = model[len("cline/"):].lower() if model.startswith("cline/") else model
        if client is self._clients.get("cmdcode") and model.lower().startswith("cmdcode/"):
            payload["model"] = model[len("cmdcode/"):].lower() if model.startswith("cmdcode/") else model
        # Strip custom provider prefix (e.g. "openrouter/anthropic/..." → "anthropic/...")
        if "/" in model:
            prefix = model.split("/")[0]
            if prefix in self._clients and prefix not in ("ollama", "opencode-go", "umans", "cline", "cmdcode"):
                # Only strip the first prefix segment
                payload["model"] = model[len(prefix)+1:]
        return await client.chat_completion(payload, api_key=api_key)

    async def chat_completion_stream(self, payload: dict, api_key: Optional[str] = None):
        """Stream a chat completion from the appropriate provider."""
        payload = dict(payload)
        model = payload.get("model", "")
        client = self._client_for(model)
        if not client:
            raise RuntimeError("No LLM provider configured")
        if model.startswith("opencode-go/") and "opencode_go" in self._clients and client is self._clients["opencode_go"]:
            payload["model"] = model[len("opencode-go/"):]
        if client is self._clients.get("umans") and (model.startswith("umans/") or model.lower().startswith("umans-")):
            payload["model"] = model
        if client is self._clients.get("cline") and model.lower().startswith("cline/"):
            payload["model"] = model[len("cline/"):].lower() if model.startswith("cline/") else model
        if client is self._clients.get("cmdcode") and model.lower().startswith("cmdcode/"):
            payload["model"] = model[len("cmdcode/"):].lower() if model.startswith("cmdcode/") else model
        if "/" in model:
            prefix = model.split("/")[0]
            if prefix in self._clients and prefix not in ("ollama", "opencode-go", "umans", "cline", "cmdcode"):
                payload["model"] = model[len(prefix)+1:]
        async for chunk in client.chat_completion_stream(payload, api_key=api_key):
            yield chunk

    async def close(self):
        for client in self._clients.values():
            if hasattr(client, "close"):
                await client.close()

    async def list_models(self) -> list[dict]:
        """Aggregate models from all configured providers."""
        results: list[dict] = []
        seen_ids: set[str] = set()
        for provider, client in self._clients.items():
            if not hasattr(client, "list_models"):
                continue
            try:
                models = await client.list_models()
            except Exception:
                continue
            for m in models:
                if not isinstance(m, dict):
                    continue
                if provider == "ollama":
                    raw_id = m.get("name", m.get("model", m.get("id", "")))
                    display_id = raw_id.replace("-cloud", "") if raw_id.endswith("-cloud") else raw_id
                    if not display_id or display_id in seen_ids:
                        continue
                    seen_ids.add(display_id)
                    details = m.get("details", {})
                    caps = {}
                    if hasattr(client, "_get_model_capabilities"):
                        try:
                            caps = client._get_model_capabilities(raw_id)
                        except Exception:
                            pass
                    results.append({
                        "id": display_id,
                        "provider": provider,
                        "capabilities": caps,
                        "details": {
                            "parameter_size": details.get("parameter_size", ""),
                            "quantization": details.get("quantization_level", ""),
                            "family": details.get("family", ""),
                        },
                    })
                    continue

                raw_id = m.get("id", m.get("name", m.get("model", "")))
                if not raw_id:
                    continue
                display_id = raw_id
                if provider == "opencode_go":
                    display_id = f"opencode-go/{raw_id}"
                elif provider == "umans":
                    display_id = f"umans/{raw_id}"
                elif provider == "cline":
                    display_id = f"cline/{raw_id}"
                elif provider == "cmdcode":
                    display_id = f"cmdcode/{raw_id}"
                elif provider not in ("ollama",):
                    # Custom providers get their name as prefix
                    display_id = f"{provider}/{raw_id}"
                if display_id in seen_ids:
                    continue
                seen_ids.add(display_id)
                caps = {}
                if hasattr(client, "_get_model_capabilities"):
                    try:
                        caps = client._get_model_capabilities(raw_id)
                    except Exception:
                        pass
                results.append({
                    "id": display_id,
                    "provider": provider,
                    "capabilities": caps,
                    "details": {"family": caps.get("family", "")},
                })
        return results

    def _get_model_capabilities(self, model: str) -> dict:
        """Delegate to the available client."""
        for client in self._clients.values():
            if hasattr(client, "_get_model_capabilities"):
                try:
                    return client._get_model_capabilities(model)
                except Exception:
                    pass
        return {}

    async def get_cloud_models(self) -> list:
        """Delegate cloud model metadata to Ollama client if configured."""
        ollama = self._clients.get("ollama")
        if ollama and hasattr(ollama, "get_cloud_models"):
            return await ollama.get_cloud_models()
        return []
