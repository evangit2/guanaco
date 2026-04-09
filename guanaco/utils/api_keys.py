"""API key generation and validation for search endpoints."""

from __future__ import annotations

import hashlib
import secrets
import time
from pathlib import Path
from typing import Optional

import yaml


KEYS_FILE = "api_keys.yaml"


class ApiKeyManager:
    """Manage per-provider API keys for the search emulator endpoints."""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self._keys_path = config_dir / KEYS_FILE
        self._keys: dict[str, dict] = {}  # key_hash -> {provider, name, created_at, key_prefix}
        self._load()

    def _load(self):
        if self._keys_path.exists():
            with open(self._keys_path) as f:
                data = yaml.safe_load(f) or {}
            self._keys = data

    def _save(self):
        self._keys_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._keys_path, "w") as f:
            yaml.dump(self._keys, f, default_flow_style=False)

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def generate_key(self, provider: str, name: str = "") -> str:
        """Generate a new API key for a provider. Returns the plaintext key (shown once)."""
        raw_key = f"guanca_{provider}_{secrets.token_urlsafe(24)}"
        key_hash = self._hash_key(raw_key)
        self._keys[key_hash] = {
            "provider": provider,
            "name": name or f"{provider}-key",
            "prefix": raw_key[:12] + "...",
            "created_at": time.time(),
        }
        self._save()
        return raw_key

    def verify_key(self, key: str, provider: Optional[str] = None) -> bool:
        """Verify a key is valid. Optionally check it's for a specific provider.
        
        Accepts guanca_ prefixed keys.
        """
        key_hash = self._hash_key(key)
        entry = self._keys.get(key_hash)
        if entry:
            if provider and entry["provider"] != provider:
                return False
            return True
            return True
        return False

    def list_keys(self) -> list[dict]:
        """List all keys (masked)."""
        return [
            {"prefix": v["prefix"], "provider": v["provider"], "name": v["name"], "created_at": v["created_at"]}
            for v in self._keys.values()
        ]

    def revoke_key(self, key: str) -> bool:
        """Revoke an API key."""
        key_hash = self._hash_key(key)
        if key_hash in self._keys:
            del self._keys[key_hash]
            self._save()
            return True
        return False

    def revoke_by_prefix(self, prefix: str) -> bool:
        """Revoke a key by its prefix."""
        for k, v in list(self._keys.items()):
            if v["prefix"] == prefix:
                del self._keys[k]
                self._save()
                return True
        return False