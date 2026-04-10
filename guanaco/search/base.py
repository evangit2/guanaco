"""Provider emulator base and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from guanaco.client import OllamaClient
    from guanaco.analytics import AnalyticsLogger


class ProviderEmulator(ABC):
    """Base class for search/scrape API emulators."""

    name: str = ""
    prefix: str = ""
    endpoints: list[dict] = []

    def __init__(self, ollama_client: "OllamaClient", analytics: Optional["AnalyticsLogger"] = None):
        self.ollama = ollama_client
        self.analytics = analytics

    @abstractmethod
    def register_routes(self, app):
        ...


_PROVIDERS: dict[str, type[ProviderEmulator]] = {}


def register_provider(cls: type[ProviderEmulator]) -> type[ProviderEmulator]:
    _PROVIDERS[cls.name] = cls
    return cls


def get_provider(name: str) -> Optional[type[ProviderEmulator]]:
    return _PROVIDERS.get(name)


def get_all_providers() -> dict[str, type[ProviderEmulator]]:
    return dict(_PROVIDERS)