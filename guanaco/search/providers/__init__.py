"""Search provider package — auto-discovers all providers."""

from guanaco.search.providers.tavily import TavilyProvider
from guanaco.search.providers.exa import ExaProvider
from guanaco.search.providers.searxng import SearXNGProvider
from guanaco.search.providers.firecrawl import FirecrawlProvider
from guanaco.search.providers.serper import SerperProvider
from guanaco.search.providers.jina import JinaProvider
from guanaco.search.providers.cohere import CohereProvider
from guanaco.search.providers.brave import BraveProvider

ALL_PROVIDERS = [
    TavilyProvider,
    ExaProvider,
    SearXNGProvider,
    FirecrawlProvider,
    SerperProvider,
    JinaProvider,
    CohereProvider,
    BraveProvider,
]

__all__ = [
    "TavilyProvider", "ExaProvider", "SearXNGProvider", "FirecrawlProvider",
    "SerperProvider", "JinaProvider", "CohereProvider", "BraveProvider",
    "ALL_PROVIDERS",
]