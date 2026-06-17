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

    def _client_for(self, model: str):
        provider = provider_for_model(model)
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
        if client is None:
            # Provider requested but not configured — fall back to whatever is available.
            for fallback_provider in ("ollama", "opencode_go"):
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
        if model.startswith("opencode-go/") and "opencode_go" in self._clients and client is self._clients["opencode_go"]:
            payload["model"] = model[len("opencode-go/"):]
        return await client.chat_completion(payload, api_key=api_key)

    async def chat_completion_stream(self, payload: dict, api_key: Optional[str] = None):
        payload = dict(payload)
        model = payload.get("model", "")
        client = self._client_for(model)
        if not client:
            raise RuntimeError("No LLM provider configured")
        if model.startswith("opencode-go/") and "opencode_go" in self._clients and client is self._clients["opencode_go"]:
            payload["model"] = model[len("opencode-go/"):]
        return client.chat_completion_stream(payload, api_key=api_key)

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
                display_id = f"opencode-go/{raw_id}" if provider == "opencode_go" else raw_id
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
