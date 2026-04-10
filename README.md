<div align="center">
  <img src="docs/logo.png" width="200" alt="Guanaco logo"/>
  
# Guanaco 🦙

[![PyPI version](https://img.shields.io/pypi/v/guanaco?color=brightgreen)](https://pypi.org/project/guanaco/)
[![Python](https://img.shields.io/pypi/pyversions/guanaco)](https://pypi.org/project/guanaco/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

**Maximize your Ollama Cloud subscription.**

Guanaco is a self-hosted FastAPI proxy that sits between your applications and Ollama Cloud. It provides an OpenAI-compatible `/v1/chat/completions` endpoint, emulates 8 major search and scrape APIs, tracks token usage, supports transparent fallback to external providers, and ships with a real-time management dashboard — all on a single port.

</div>

```bash
curl -sSL https://raw.githubusercontent.com/evangit2/guanaco/main/install.sh | bash
```

---

## Features

- **LLM Router** — OpenAI-compatible `/v1/chat/completions` and Anthropic-compatible `/v1/messages` proxy with streaming, token tracking, and analytics
- **8 Search/Scrape Emulators** — Drop-in replacements for Tavily, Exa, SearXNG, Firecrawl, Serper, Jina, Cohere, and Brave Search
- **Fallback Provider** — Automatically route to a secondary OpenAI-compatible provider when Ollama Cloud is slow, rate-limited, or unavailable; also kicks in when your Ollama Cloud usage quota is exhausted
- **Usage Tracking** — Monitor Ollama Cloud session and weekly quota usage in real time
- **Smart Caching** — Optional exact-match and session-aware prefix caching (BETA) to reduce redundant API calls
- **Web Dashboard** — Real-time analytics, model configuration, API key management, and service status at `http://localhost:8080/dashboard`
- **Docker & systemd** — Production-ready deployment with included service unit files

---

## Quick Start

### 1. Install

```bash
curl -sSL https://raw.githubusercontent.com/evangit2/guanaco/main/install.sh | bash
```

The installer will check for prerequisites (git, Python 3.10+, venv) and auto-install them if missing, then prompt you for your Ollama API key and preferred port.

For platform-specific instructions, see [WSL Installation](#wsl-installation) and [macOS Installation](#macos-installation).

### 2. Reload your shell

The installer adds `guanaco` to your PATH, but you need to reload for it to take effect:

```bash
source ~/.bashrc   # or ~/.zshrc on macOS
```

After this, `guanaco` is available as a system command from anywhere.

The installer starts Guanaco automatically (as a systemd service or in the foreground).

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `guanaco start` | Start the proxy server (router + search APIs + dashboard) |
| `guanaco setup` | Interactive configuration wizard |
| `guanaco status` | Show service status and Ollama Cloud connectivity |
| `guanaco models` | List available Ollama Cloud models |
| `guanaco models --refresh` | Force-refresh model list from Ollama API |
| `guanaco models --capabilities` | Show model capabilities and sizes |
| `guanaco usage` | Check current Ollama Cloud session/weekly quota |
| `guanaco key generate` | Generate a new API key |
| `guanaco key list` | List all API keys |
| `guanaco key revoke` | Revoke an API key |
| `guanaco analytics` | View request analytics summary |
| `guanaco analytics --errors` | Show recent errors |
| `guanaco analytics --model <name>` | Show history for a specific model |
| `guanaco config --show` | Show current configuration |
| `guanaco config --set <key> <value>` | Update a config value |
| `guanaco version` | Show version |

---

## Dashboard

The built-in web dashboard is available at `http://localhost:8080/dashboard`.

![Guanaco Dashboard](https://i.ibb.co/KzzP6yNw/Screenshot-2026-04-09-223634.png)

Features: real-time request analytics, token usage graphs, model configuration, fallback provider setup, API key management, and Ollama Cloud quota monitoring.

---

## Configuration

Guanaco stores configuration in `~/.guanaco/config.yaml`. You can change the config directory:

```bash
export GUANACO_CONFIG_DIR=/path/to/config
```

### Full `config.yaml` Reference

```yaml
# ── Required ──
ollama_api_key: "sk-ollama-..."       # Or set via OLLAMA_API_KEY env var

# ── Server ──
router:
  host: "127.0.0.1"                   # Bind address
  port: 8080                           # Listen port
  use_tailscale: false                # Use Tailscale IP for endpoint URLs
  autostart: false

# ── LLM Model Selection ──
llm:
  default_model: "gemma4:31b"        # Model used when none specified
  reranker_model: "gpt-oss:120b"     # Used for search result reranking
  scraper_model: "gemma4:31b"         # Used for web page summarization
  summary_model: "qwen3.5:397b"      # Used for content summarization
  fallback_model: "gemma4:31b"        # Used when requested model unavailable
  emulate_openai: true                # Enable /v1/chat/completions endpoint
  emulate_anthropic: true             # Enable /v1/messages proxy endpoint
  # available_models: [...]

# ── Fallback Provider (when Ollama Cloud is unavailable) ──
fallback:
  enabled: false
  name: "openai"                      # Display name
  base_url: "https://api.openai.com/v1"
  api_key: ""
  default_model: "gpt-4o"
  timeout: 60.0                       # Request timeout in seconds
  primary_timeout: 30.0               # Max seconds to wait for Ollama first
                                       # chunk before trying fallback
  stream_chunk_timeout: 180.0         # Max seconds between stream chunks
  max_tokens: 128000
  stream_fallback: true
  model_map: {}                        # ollama_model -> fallback_model mapping

# ── Search/Scrape Provider API Keys ──
providers:
  tavily:     { enabled: true }
  exa:        { enabled: true }
  searxng:    { enabled: true }
  firecrawl:  { enabled: true, require_api_key: false }
  serper:     { enabled: true }
  jina:       { enabled: true }
  cohere:     { enabled: true }
  brave:      { enabled: true }

# ── Smart Cache (BETA) ──
cache:
  beta_mode: false                    # Master switch — must be true to enable
  exact_cache_ttl: 600                # Seconds for exact-match response cache
  session_prefix_ttl: 3600            # Seconds for session prefix cache
  max_entries: 500
  dedup_enabled: true                 # Merge identical concurrent upstream calls
  session_prefix_enabled: true
  exact_cache_enabled: true
  min_prompt_chars: 50                # Don't cache tiny prompts

# ── Ollama Cloud Usage Tracking ──
usage:
  session_cookie: ""                   # __Secure-session cookie from ollama.com
  check_interval: 0                   # Auto-check interval (0 = disabled)
  redirect_on_full: false             # Route to fallback when quota near limit
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OLLAMA_API_KEY` | Ollama Cloud API key (takes precedence over config file) |
| `GUANACO_CONFIG_DIR` | Path to config directory (default `~/.guanaco`) |

---

## Fallback Provider Setup

When Ollama Cloud is slow, rate-limited, or a requested model isn't available, Guanaco can automatically forward requests to a fallback OpenAI-compatible provider.

```yaml
fallback:
  enabled: true
  name: "openai"
  base_url: "https://api.openai.com/v1"
  api_key: "sk-..."
  default_model: "gpt-4o"
  primary_timeout: 30.0                # Wait up to 30s for Ollama first chunk
  stream_chunk_timeout: 180.0          # Tolerate long reasoning pauses
  timeout: 60.0
  stream_fallback: true
  model_map:
    # Map specific Ollama models to different fallback models
    "qwen3:480b": "gpt-4o"
    "deepseek-v3.1:671b": "gpt-4o"
```

Or configure via the dashboard at **Dashboard → Config → Fallback**.

Once running, your apps can hit:

| Endpoint | Purpose |
|----------|---------|
| `http://localhost:8080/v1/chat/completions` | OpenAI-compatible LLM router |
| `http://localhost:8080/v1/messages` | Anthropic-compatible proxy |
| `http://localhost:8080/tavily/search` | Tavily search (emulated) |
| `http://localhost:8080/exa/search` | Exa search (emulated) |
| `http://localhost:8080/firecrawl/scrape` | Firecrawl scrape (emulated) |
| `http://localhost:8080/brave/search` | Brave Search (emulated) |
| `http://localhost:8080/dashboard` | Web dashboard |

---

## API Reference

### LLM Router

**`POST /v1/chat/completions`** — OpenAI-compatible chat completions

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "gemma4:31b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

**`POST /v1/messages`** — Anthropic-compatible messages proxy

```bash
curl -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "gemma4:31b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 1024
  }'
```

### Search APIs

All search providers are emulated at `http://localhost:8080/<provider>/<endpoint>`:

| Provider | Endpoints | Notes |
|----------|-----------|-------|
| **Tavily** | `/tavily/search` | Tavily Search API compatible |
| **Exa** | `/exa/search`, `/exa/findSimilar` | Exa Search API compatible |
| **SearXNG** | `/searxng/search` | SearXNG API compatible |
| **Firecrawl** | `/firecrawl/scrape`, `/firecrawl/search`, `/firecrawl/crawl`, `/firecrawl/extract` | Firecrawl SDK v2 compatible |
| **Serper** | `/serper/search`, `/serper/scrape` | Serper API compatible |
| **Jina** | `/jina/search`, `/jina/rerank` | Jina API compatible |
| **Cohere** | `/cohere/rerank` | Cohere Rerank API compatible |
| **Brave** | `/brave/search` | Brave Search API compatible |

Firecrawl SDK v2 paths (`/v2/scrape`, `/v2/search`, `/v2/crawl`, `/v2/extract`) are also supported directly.

### Status & Utility Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /v1/models` | List available models |
| `GET /v1/usage` | Ollama Cloud usage/quota |
| `GET /api/ollama/status` | Ollama Cloud connectivity |
| `GET /api/ollama/models` | Full model list with metadata |

---

## Docker Deployment

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -e .
EXPOSE 8080
ENV GUANACO_CONFIG_DIR=/data
VOLUME /data
CMD ["guanaco", "start", "--host", "0.0.0.0"]
```

```bash
docker build -t guanaco .
docker run -d -p 8080:8080 \
  -e OLLAMA_API_KEY=your_key \
  -v ~/.guanaco:/data \
  guanaco
```

---

## systemd Deployment

```bash
sudo cp contrib/guanaco.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now guanaco
```

Check status:
```bash
systemctl status guanaco
journalctl -u guanaco -f
```

Edit `/etc/systemd/system/guanaco.service` to set `User`, `Group`, install directory, and venv path as appropriate for your environment.

---

## WSL Installation

Install and run Guanaco in a real WSL Linux distro such as Ubuntu.

1. In Windows PowerShell or Command Prompt, check your WSL distros:

```bash
wsl -l -v
```

2. Install Ubuntu for WSL if needed:

```bash
wsl --install -d Ubuntu
```

3. Start Ubuntu:

```bash
wsl -d Ubuntu
```

4. Inside the Ubuntu WSL distro, install prerequisites:

```bash
sudo apt update
sudo apt install -y curl bash git python3 python3-venv python3-pip
```

5. Run the Guanaco installer:

```bash
curl -sSL https://raw.githubusercontent.com/evangit2/guanaco/main/install.sh | bash
```

> **Note:** Run the installer inside a normal WSL Linux distro like Ubuntu, not a minimal helper environment that may be missing tools such as `bash` and `curl`.

## macOS Installation

1. Open Terminal — you can use the built-in Terminal app or iTerm2.

2. Install Xcode Command Line Tools:

```bash
xcode-select --install
```

3. Install Homebrew if needed:

```bash
brew --version
```

If `brew` is not installed, run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

4. Install prerequisites:

```bash
brew install git python@3.12 curl
```

5. Run the Guanaco installer:

```bash
curl -sSL https://raw.githubusercontent.com/evangit2/guanaco/main/install.sh | bash
```

> **Note:** If `python3` is still not found after installing Homebrew Python, restart Terminal or add Homebrew to your shell path first.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on the [GitHub repository](https://github.com/evangit2/guanaco).

---

## License

[MIT](LICENSE) — Copyright 2026 Guanaco Contributors
