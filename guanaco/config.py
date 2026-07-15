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
    auto_update: bool = False
    allow_prerelease: bool = False
    # Optional name for the systemd service (allows multiple isolated instances)
    service_name: str = "guanaco"
    # Visibility level for the dashboard/API listener:
    #   "localhost" -> 127.0.0.1 only (default, safest)
    #   "tailscale" -> 0.0.0.0, reachable on the Tailscale network
    #   "all"       -> 0.0.0.0, reachable on all interfaces
    visibility: str = "localhost"
    # When a model name has no provider prefix and both providers are available,
    # choose the default provider using this strategy: round_robin, usage, ollama, opencode_go.
    # DEPRECATED: kept for backward compatibility. Use provider_priority for ordered fallback.
    unprefixed_provider_strategy: str = "round_robin"
    # Ordered list of providers to try for unprefixed models. Earlier entries are preferred.
    # Built-in providers: "ollama", "opencode_go", "umans", "cline", "cmdcode". A configured fallback OpenAI-compatible
    # provider can be included as "fallback".
    provider_priority: list[str] = Field(default_factory=lambda: ["ollama", "opencode_go", "umans", "cline", "cmdcode"])


class SearchConfig(BaseModel):
    """Search provider settings — moved from Models tab to Search tab."""
    summarize_enabled: bool = False         # BETA — opt-in summarize for supported providers
    summarize_all: bool = False              # Secondary toggle — summarize ALL responses, not just native ones
    summary_model: str = "qwen3.5:397b"     # Model used for summarization




class HistoryConfig(BaseModel):
    """Full request/response history logging settings."""
    enabled: bool = False              # Opt-in — must be explicitly enabled
    save_input: bool = True            # Save input text (prompts)
    save_output: bool = True           # Save output text (responses)
    retention_days: int = 30           # Auto-delete after this many days (0 = keep forever)
    max_content_size: int = 100000     # Max chars to save per input/output (truncates if larger)
    log_to_files: bool = False        # Also write plaintext log files (opt-in, separate from DB)
    log_dir: str = ""                 # Directory for log files (default: <config_dir>/history_logs)

    def get_log_dir(self, config_dir: Optional[Path] = None) -> Path:
        """Resolve the log directory, creating it if needed."""
        if self.log_dir:
            p = Path(self.log_dir)
        else:
            p = (config_dir or get_default_config_dir()) / "history_logs"
        p.mkdir(parents=True, exist_ok=True)
        return p

class LLMConfig(BaseModel):
    """LLM model selection config."""
    reranker_model: str = "nemotron-3-nano:30b"
    scraper_model: str = "nemotron-3-nano:30b"
    summary_model: str = "nemotron-3-nano:30b"
    default_model: str = "nemotron-3-nano:30b"
    available_models: list[str] = Field(default_factory=lambda: [
        "qwen3.5:397b", "qwen3-coder:480b", "qwen3-vl:235b", "qwen3-next:80b",
        "gpt-oss:120b", "gpt-oss:20b", "deepseek-v3.1:671b", "deepseek-v3.2", "deepseek-v4-pro", "deepseek-v4-flash",
        "gemma4:31b", "gemma3:27b", "glm-5.1", "glm-5", "gemini-3-flash-preview",
        "minimax-m2.7", "minimax-m2.5", "minimax-m2.1",
        "devstral-small-2:24b", "devstral-2:123b", "nemotron-3-super",
        "nemotron-3-nano:30b",
        "cogito-2.1:671b", "mistral-large-3:675b", "kimi-k2.5", "kimi-k2.6", "ministral-3:14b",
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
    primary_timeout: float = 120.0          # Max seconds to wait for Ollama first chunk/response before trying fallback
    stream_chunk_timeout: float = 180.0    # Max seconds between stream chunks (tolerates long reasoning pauses)
    max_tokens: int = 128000                 # Default max_tokens sent to fallback provider
    stream_fallback: bool = True              # Also fallback streaming requests
    supports_vision: bool = False             # Whether the fallback provider handles image/vision requests
    max_concurrent_ollama: int = 8            # Max simultaneous Ollama requests (0 = unlimited)
    max_429_retries: int = 2                  # How many times to retry Ollama on HTTP 429 before falling back
    backoff_base: float = 1.0                 # Base backoff in seconds for 429 retry (doubles each attempt)


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
    # v0.4.3+ fields — added for multi-account migration
    last_plan: Optional[str] = None           # Last known plan (free/pro/max)
    last_session_reset: Optional[str] = None  # Human-readable time until session resets
    last_weekly_reset: Optional[str] = None  # Human-readable time until weekly resets
    last_checked: Optional[float] = None      # Unix timestamp of last successful check
    redirect_on_full: bool = False            # Route to fallback when quota near limit

class ROIConfig(BaseModel):
    """Experimental: subscription value comparison vs OpenRouter pay-as-you-go."""
    enabled: bool = False
    subscription_price: float = 0.0
    # OpenRouter prompt-cache hit estimate (0-100%). Affects cost calc for models with
    # input_cache_read pricing (e.g. Claude Fable, Qwen, Minimax).
    cache_hit_pct: float = 0.0

    last_price_cache: float = 0.0
    cached_prices: dict[str, dict] = Field(default_factory=dict)
    last_roi_calc: float = 0.0
    last_roi_detail: dict = Field(default_factory=dict)

class ProviderAccount(BaseModel):
    """A single provider account with its own API key and usage tracking."""
    name: str                                      # Display name (unique, "ollama" is reserved for primary)
    api_key: str = ""                              # API key for this account
    provider: str = "ollama"                       # Provider type: "ollama", "opencode_go", or "umans"
    base_url: str = ""                             # Optional endpoint override; defaults per provider if empty
    session_cookie: str = ""                       # __Secure-session cookie for usage scraping (Ollama | UMANS)
    # Usage tracking (updated by background check)
    last_session_pct: Optional[float] = None
    last_weekly_pct: Optional[float] = None
    last_plan: Optional[str] = None
    last_session_reset: Optional[str] = None
    last_weekly_reset: Optional[str] = None
    last_checked: Optional[float] = None
    # Multi-account rotation mode. "usage" = quota-aware (default). "round_robin" = round-robin.
    rotation_mode: str = "usage"


class CustomProviderConfig(BaseModel):
    """Configuration for a custom OpenAI-compatible provider.

    Lets users add any OpenAI-compatible API (OpenRouter, Together, Groq,
    LM Studio, vLLM, etc.) as a first-class provider.
    """
    name: str = ""                              # Provider name (used in model prefixes)
    base_url: str = ""                          # e.g. "https://openrouter.ai/api/v1"
    api_key: str = ""                           # API key (empty for local servers)
    models: list[str] = Field(default_factory=list)  # Empty = auto-discover from /v1/models
    timeout: float = 120.0                      # Request timeout
    max_concurrent_streams: int = 0             # 0 = unlimited (default), >0 = limit concurrent streams


class UmansConfig(BaseModel):
    """UMANS subscription provider settings."""
    enabled: bool = False
    # Optional UMANS app session cookie for usage scraping (from __Secure-authjs.session-token)
    session_cookie: str = ""
    # Optional UMANS credentials to fetch the session cookie automatically
    email: str = ""
    password: str = ""
    # Stamp a session label into the first user message:
    #   yes -> always stamp with "[umans|sessN]"
    #   auto -> stamp only for models whose name contains "thinking" or supports_thinking
    session_label_mode: str = "auto"
    # Max number of images to keep in a single request (0 = unlimited)
    max_images_per_request: int = 0
    # Override base URL for testing
    base_url: str = ""
    # Max concurrent streams to UMANS (0 = unlimited)
    max_concurrent_streams: int = 0


class ClineConfig(BaseModel):
    """Cline Pass subscription provider settings.

    Cline Pass is a flat-rate monthly subscription ($9.99/mo) providing
    OpenAI-compatible access to 10 open-weight models via Cline's gateway.
    Zero per-token cost — subscription-based.
    """
    enabled: bool = False
    # Override base URL for testing (default: https://api.cline.bot/api/v1)
    base_url: str = ""
    # Max concurrent streams (0 = unlimited)
    max_concurrent_streams: int = 0


class CmdCodeConfig(BaseModel):
    """Command Code Go plan provider settings.

    Command Code (commandcode.ai) offers a $1/mo Go plan with CLI access to 20+
    open-weight models. A local proxy (cmd_proxy.py) translates OpenAI-compatible
    requests to the CLI's internal /alpha/generate endpoint. Zero per-token cost.
    Zero Data Retention (ZDR) enabled by default.
    """
    enabled: bool = False
    # Override proxy URL (default: http://localhost:5999/v1)
    base_url: str = ""
    # Max concurrent streams (0 = unlimited)
    max_concurrent_streams: int = 0


# Backward-compatible alias
OllamaAccount = ProviderAccount


def infer_provider_from_key(api_key: str, provider_hint: Optional[str] = None) -> str:
    """Return the most likely provider for an account key.

    OpenCode Go keys start with ``sk-``; Ollama keys do not.
    UMANS keys are long hex-ish strings (no uniform prefix, but too long for Go/Ollama).
    Cline Pass keys start with ``sk_`` (underscore, not hyphen).
    If ``provider_hint`` is provided and valid, it is respected.
    """
    key = api_key.strip()
    if provider_hint in ("ollama", "opencode_go", "umans", "cline", "cmdcode"):
        return provider_hint
    if key.startswith("sk_"):
        return "cline"
    if key.startswith("user_"):
        return "cmdcode"
    if key.lower().startswith("sk-"):
        return "opencode_go"
    if len(key) >= 64 and not key.startswith("ollama"):
        # UMANS JWT-looking keys are very long hex-encoded strings
        return "umans"
    return "ollama"


class AppConfig(BaseModel):
    ollama_api_key: str = ""
    ollama_accounts: list[OllamaAccount] = Field(default_factory=list)
    router: RouterConfig = Field(default_factory=RouterConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    fallback: FallbackProviderConfig = Field(default_factory=FallbackProviderConfig)
    providers: AllProvidersConfig = Field(default_factory=AllProvidersConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    usage: UsageConfig = Field(default_factory=UsageConfig)
    roi: ROIConfig = Field(default_factory=ROIConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    umans: UmansConfig = Field(default_factory=UmansConfig)
    cline: ClineConfig = Field(default_factory=ClineConfig)
    cmdcode: CmdCodeConfig = Field(default_factory=CmdCodeConfig)
    custom_providers: list[CustomProviderConfig] = Field(default_factory=list)

    @property
    def ollama_api_key_resolved(self) -> str:
        """Resolve API key from config or environment."""
        return self.ollama_api_key or os.environ.get("OLLAMA_API_KEY", "")

    @property
    def primary_account(self) -> "OllamaAccount":
        """Get or create the primary 'ollama' account."""
        for acc in self.ollama_accounts:
            if acc.name == "ollama":
                return acc
        # Auto-create from legacy single-key config, merging usage cookie/data
        # Use ollama_api_key_resolved so env-var-only setups get a working key
        return OllamaAccount(
            name="ollama",
            api_key=self.ollama_api_key_resolved,
            session_cookie=self.usage.session_cookie if hasattr(self, 'usage') else "",
            last_session_pct=self.usage.last_session_pct if hasattr(self, 'usage') else None,
            last_weekly_pct=self.usage.last_weekly_pct if hasattr(self, 'usage') else None,
            last_plan=self.usage.last_plan if hasattr(self, 'usage') else None,
            last_session_reset=self.usage.last_session_reset if hasattr(self, 'usage') else None,
            last_weekly_reset=self.usage.last_weekly_reset if hasattr(self, 'usage') else None,
            last_checked=self.usage.last_checked if hasattr(self, 'usage') else None,
            rotation_mode="usage",
        )

    @property
    def active_accounts(self) -> list["OllamaAccount"]:
        """All accounts that have a non-empty API key."""
        return [a for a in self.ollama_accounts if a.api_key and a.api_key not in ("***", "REPLACE_ME")]


_config: Optional[AppConfig] = None


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load configuration from YAML file, falling back to defaults.

    Includes migration for backward compatibility:
    - v0.4.2 configs missing UsageConfig fields get auto-populated with defaults.
    """
    global _config
    path = path or get_default_config_path()

    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    # ── Config migration ──
    # v0.4.2 → v0.4.3+: UsageConfig gained last_plan, redirect_on_full, etc.
    usage = data.setdefault("usage", {})
    for field, default in (
        ("last_plan", None),
        ("last_session_reset", None),
        ("last_weekly_reset", None),
        ("last_checked", None),
        ("redirect_on_full", False),
    ):
        if field not in usage:
            usage[field] = default

    # v0.4.1 → v0.4.2+: RouterConfig gained auto_update, allow_prerelease
    router = data.setdefault("router", {})
    for field, default in (
        ("auto_update", False),
        ("allow_prerelease", False),
    ):
        if field not in router:
            router[field] = default

    # v0.4.4+: service_name added for multi-instance systemd support
    if "service_name" not in router:
        router["service_name"] = "guanaco"

    # v0.4.4+: provider_priority replaces unprefixed_provider_strategy
    if "provider_priority" not in router:
        old_strategy = str(router.get("unprefixed_provider_strategy", "round_robin")).lower()
        if old_strategy == "opencode_go":
            router["provider_priority"] = ["opencode_go", "ollama", "umans"]
        else:
            router["provider_priority"] = ["ollama", "opencode_go", "umans"]

    # v0.5.6+: ensure provider_priority includes "umans" if missing
    if "provider_priority" in router and isinstance(router["provider_priority"], list):
        if "umans" not in router["provider_priority"]:
            router["provider_priority"].append("umans")

    # v0.5.6+: ensure umans config exists for migration
    if "umans" not in data:
        data["umans"] = {}

    # v0.7.1+: ensure cline config exists for migration
    if "cline" not in data:
        data["cline"] = {}

    # v0.7.1+: ensure provider_priority includes "cline" if missing
    if "provider_priority" in router and isinstance(router["provider_priority"], list):
        if "cline" not in router["provider_priority"]:
            router["provider_priority"].append("cline")

    # v0.7.2+: ensure cmdcode config exists for migration
    if "cmdcode" not in data:
        data["cmdcode"] = {}

    # v0.7.2+: ensure provider_priority includes "cmdcode" if missing
    if "provider_priority" in router and isinstance(router["provider_priority"], list):
        if "cmdcode" not in router["provider_priority"]:
            router["provider_priority"].append("cmdcode")

    # v0.6.0+: visibility setting controls host binding
    if "visibility" not in router:
        router["visibility"] = "localhost"
    # Keep host aligned with visibility so existing installs behave consistently
    if router.get("visibility") in ("tailscale", "all"):
        router["host"] = "0.0.0.0"
    elif router.get("visibility") == "localhost":
        router["host"] = "127.0.0.1"

    _config = AppConfig(**data)

    # v0.5.3+: Auto-correct accounts whose provider field doesn't match their key,
    # but respect an explicitly configured provider hint.
    for acc in _config.ollama_accounts:
        if acc.name == "ollama":
            acc.provider = "ollama"
        else:
            inferred = infer_provider_from_key(acc.api_key, provider_hint=acc.provider)
            if inferred != acc.provider:
                acc.provider = inferred
    if not any(a.name == "ollama" for a in _config.ollama_accounts):
        # Create primary from the legacy single-key config + usage data
        # Use ollama_api_key_resolved so env-var-only setups get a working key
        _config.ollama_accounts.insert(0, OllamaAccount(
            name="ollama",
            api_key=_config.ollama_api_key_resolved,
            session_cookie=_config.usage.session_cookie if hasattr(_config, 'usage') else "",
            last_session_pct=_config.usage.last_session_pct if hasattr(_config, 'usage') else None,
            last_weekly_pct=_config.usage.last_weekly_pct if hasattr(_config, 'usage') else None,
            last_plan=_config.usage.last_plan if hasattr(_config, 'usage') else None,
            last_session_reset=_config.usage.last_session_reset if hasattr(_config, 'usage') else None,
            last_weekly_reset=_config.usage.last_weekly_reset if hasattr(_config, 'usage') else None,
            last_checked=_config.usage.last_checked if hasattr(_config, 'usage') else None,
            rotation_mode="usage",
        ))

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