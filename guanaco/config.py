"""Configuration management for Guanaco."""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


def get_default_config_dir() -> Path:
    """Get the default config directory.
    
    Checks GUANACO_CONFIG_DIR env var first, then defaults to ~/.guanaco.
    """
    if "GUANACO_CONFIG_DIR" in os.environ:
        return Path(os.environ["GUANACO_CONFIG_DIR"])
    return Path.home() / ".guanaco"


def _config_dir_has_content(p: Path) -> bool:
    """Check if a config directory has existing config files."""
    if not p.exists():
        return False
    return (p / "config.yaml").exists() or list(p.glob("*.yaml")) or list(p.glob("*.json"))


def get_default_config_path() -> Path:
    return get_default_config_dir() / "config.yaml"


class RouterConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    use_tailscale: bool = False
    autostart: bool = False


class LLMConfig(BaseModel):
    """LLM model selection config."""
    reranker_model: str = "gpt-oss:120b"
    scraper_model: str = "gemma4:31b"
    summary_model: str = "qwen3.5:397b"
    default_model: str = "gemma4:31b"
    available_models: list[str] = Field(default_factory=lambda: [
        "qwen3.5:397b", "qwen3-coder:480b", "qwen3-vl:235b", "qwen3-next:80b",
        "gpt-oss:120b", "gpt-oss:20b", "deepseek-v3.1:671b", "deepseek-v3.2",
        "gemma4:31b", "gemma3:27b", "glm-5.1", "glm-5",
        "minimax-m2.7", "minimax-m2.5", "minimax-m2.1",
        "devstral-small-2:24b", "devstral-2:123b", "nemotron-3-super",
        "cogito-2.1:671b", "mistral-large-3:675b", "kimi-k2.5", "ministral-3:14b",
    ])
    emulate_anthropic: bool = True
    emulate_openai: bool = True
    # When a requested model isn't found on Ollama Cloud, fall back to this model
    fallback_model: str = "gemma4:31b"


class FallbackProviderConfig(BaseModel):
    """External OpenAI-compatible provider to use when Ollama Cloud fails or model not found."""
    enabled: bool = False
    name: str = "custom"                    # Display name
    base_url: str = ""                       # e.g. "https://api.openai.com/v1" or "http://localhost:1234/v1"
    api_key: str = ""                        # API key for the fallback provider
    # Model name mapping: ollama_model -> fallback_model
    # If a model isn't in the map, fallback_model_default is used
    model_map: dict[str, str] = Field(default_factory=dict)
    default_model: str = ""                  # Default model to use on the fallback provider
    timeout: float = 60.0                    # Request timeout in seconds (for fallback calls)
    primary_timeout: float = 30.0           # Max seconds to wait for Ollama first chunk/response before trying fallback
    stream_chunk_timeout: float = 180.0    # Max seconds between stream chunks (tolerates long reasoning pauses)
    max_tokens: int = 128000                 # Default max_tokens sent to fallback provider
    stream_fallback: bool = True              # Also fallback streaming requests


class ProviderConfig(BaseModel):
    """Per-provider enable/disable and API key settings."""
    enabled: bool = True
    require_api_key: bool = False
    api_keys: list[str] = Field(default_factory=list)


class AllProvidersConfig(BaseModel):
    tavily: ProviderConfig = Field(default_factory=ProviderConfig)
    exa: ProviderConfig = Field(default_factory=ProviderConfig)
    searxng: ProviderConfig = Field(default_factory=ProviderConfig)
    firecrawl: ProviderConfig = Field(default_factory=lambda: ProviderConfig(require_api_key=True))
    serper: ProviderConfig = Field(default_factory=ProviderConfig)
    jina: ProviderConfig = Field(default_factory=ProviderConfig)
    cohere: ProviderConfig = Field(default_factory=ProviderConfig)
    brave: ProviderConfig = Field(default_factory=ProviderConfig)


class CacheConfig(BaseModel):
    """Smart session-aware response cache (beta)."""
    beta_mode: bool = False                # Master switch — must be True for any caching
    exact_cache_ttl: int = 600             # Seconds to cache exact-match responses (default 10 min)
    session_prefix_ttl: int = 3600         # Seconds for session prefix cache (default 1 hr)
    max_entries: int = 500                 # Max cache entries before LRU eviction
    dedup_enabled: bool = True             # Merge identical concurrent requests into one upstream call
    session_prefix_enabled: bool = True    # Enable session-aware prefix caching
    exact_cache_enabled: bool = True       # Enable exact hash caching
    min_prompt_chars: int = 50             # Don't cache tiny prompts (not worth it)
    exclude_models: list[str] = Field(default_factory=list)  # Models to never cache

class UsageConfig(BaseModel):
    """Ollama Cloud usage quota scraping via session cookie."""
    session_cookie: str = ""                  # __Secure-1PSID or __Secure-session cookie value
    check_interval: int = 0                   # Auto-check interval in seconds (0 = disabled)
    last_session_pct: Optional[float] = None  # Last known session usage %
    last_weekly_pct: Optional[float] = None   # Last known weekly usage %
    last_plan: Optional[str] = None            # Last known plan name
    last_session_reset: Optional[str] = None   # e.g. "Resets in 7 minutes"
    last_weekly_reset: Optional[str] = None    # e.g. "Resets in 3 days"
    last_checked: Optional[float] = None       # Unix timestamp of last successful check
    redirect_on_full: bool = False             # Route all requests to fallback when quota is near limit


class AppConfig(BaseModel):
    ollama_api_key: str = ""
    router: RouterConfig = Field(default_factory=RouterConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    fallback: FallbackProviderConfig = Field(default_factory=FallbackProviderConfig)
    providers: AllProvidersConfig = Field(default_factory=AllProvidersConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    usage: UsageConfig = Field(default_factory=UsageConfig)

    @property
    def ollama_api_key_resolved(self) -> str:
        """Resolve API key from config or environment."""
        return self.ollama_api_key or os.environ.get("OLLAMA_API_KEY", "")


_config: Optional[AppConfig] = None


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load configuration from YAML file, falling back to defaults."""
    global _config
    path = path or get_default_config_path()

    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _config = AppConfig(**data)
    else:
        _config = AppConfig()

    return _config


def save_config(config: AppConfig, path: Optional[Path] = None) -> None:
    """Save configuration to YAML file."""
    path = path or get_default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Don't persist env-resolved API keys back to file
    dump = config.model_dump()
    if not config.ollama_api_key and "ollama_api_key" in dump:
        # Keep whatever was in the file, don't overwrite with empty
        pass

    with open(path, "w") as f:
        yaml.dump(dump, f, default_flow_style=False)


def get_config() -> AppConfig:
    """Get current config, loading if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def generate_api_key(prefix: str = "guanca") -> str:
    """Generate a random API key."""
    return f"{prefix}_{secrets.token_urlsafe(32)}"


def get_base_url(config: Optional[AppConfig] = None) -> str:
    """Get the base URL for the services, using Tailscale IP if configured."""
    config = config or get_config()
    if config.router.use_tailscale:
        ts_ip = get_tailscale_ip()
        if ts_ip:
            return f"http://{ts_ip}"
    return f"http://{get_local_ip()}"


def get_local_ip() -> str:
    """Get the LAN IP of this machine (not 127.0.0.1)."""
    import socket
    try:
        # Create a UDP socket to determine the default route interface IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("100.100.100.100", 1))  # Tailscale DERP; won't actually send
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # Fallback: check if Tailscale IP is available
        ts_ip = get_tailscale_ip()
        if ts_ip:
            return ts_ip
        return "127.0.0.1"


def get_tailscale_ip() -> Optional[str]:
    """Get the Tailscale IP address of this machine, or None if not installed."""
    import subprocess
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        return None  # Tailscale not installed
    except subprocess.TimeoutExpired:
        pass
    return None