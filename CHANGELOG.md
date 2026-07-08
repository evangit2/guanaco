# Changelog

All notable changes to Guanaco will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.7.1] - 2026-07-08

### Added
- **Cline Pass provider.** Full integration as a fourth built-in provider alongside Ollama Cloud, OpenCode Go, and UMANS. Cline Pass is a $9.99/mo subscription offering 10 open-weight models (GLM-5.2, Kimi K2.7 Code, DeepSeek V4 Pro/Flash, MiMo v2.5, MiniMax M3, Qwen3.7 Max/Plus, Kimi K2.6).
  - New `ClinePassClient` (`guanaco/cline_client.py`) with SSE streaming, reasoning delta support, static model fallback, and `test_key()`.
  - `sk_` prefix detection in `infer_provider_from_key()` (distinct from OpenCode Go's `sk-` prefix).
  - `cline/` model prefix routing for explicit provider selection.
  - Config migration adds `cline` to `provider_priority` for existing installs.
  - Install script: Cline Pass as option 4 with API key validation against `api.cline.bot`.
  - Dashboard: Accounts tab "Add Account" dropdown now shows all four providers (previously missing UMANS and Cline Pass).
  - Provider priority drag-and-drop list shows all four providers with labels and icons.
  - `/v1/models` endpoint renders `cline/` prefixed models with `owned_by: "cline"`.
  - 22 new tests covering key inference, model routing, payload normalization, config migration, and account pool handling.

## [0.7.0] - 2026-07-08

### Added
- **UMANS provider integration.** Full UMANS subscription support as a third built-in provider.
- **Provider priority system.** Drag-and-drop provider ordering in the dashboard with automatic fallback.
- **Multi-provider account management.** Per-provider account pools with usage-aware rotation.

## [0.6.3] - 2026-06-18

### Fixed
- **Streaming usage chunks.** All providers (UMANS, OpenCode Go, Ollama) now emit a final `chat.completion.chunk` with `usage` during streaming responses, matching OpenAI spec.
- **Provider priority for unprefixed models.** `_select_account` now passes `provider_priority` so models like `glm-5.2` correctly route according to configured priority.
- **`glm-5.2` alias.** Added `glm-5.2` / `glm-5-2` / `glm5.2` to `KNOWN_UMANS_MODELS`.

## [0.6.2] - 2026-06-18

### Fixed
- (yanked; superseded by 0.6.3)

### Fixed
- **Explicit provider account selection.** Model prefixes (`umans/`, `opencode-go/`) now correctly use their own provider account instead of always falling back to `provider_priority` order.
- **UMANS model prefix duplication.** Canonical UMANS model IDs no longer end up with a doubled `umans-umans-` prefix.
- **Reasoning content normalization.** Responses that leave `message.content` empty but populate `reasoning_content` or `reasoning` now surface that content so standard OpenAI clients receive a non-null reply.
- **SearXNG crash on Ollama web-search failure.** The emulator no longer returns HTTP 500 when the upstream Ollama `/api/web_search` endpoint fails.
- **Analytics migration.** Existing `request_log` databases that predate the `usage_multiplier` column are now upgraded automatically, preventing 500 errors after upgrade.
- **Config provider hint.** Accounts with an explicit `provider` field are no longer overwritten by key-pattern auto-detection, fixing UMANS keys that start with `sk-` being misidentified as OpenCode Go.

## [0.6.0] - 2026-06-18

### Added
- **UMANS provider support.** First-class provider with session-label stamping, image-limit enforcement, reasoning stripping, and retry logic. Models are prefixed as `umans/<id>` in `/v1/models`.
- **Network visibility configuration.** New dashboard Network settings consolidated into the System tab; supports `localhost`, `Tailscale`, and `0.0.0.0` binding choices via both the installer and the dashboard.
- **Provider selection in installer.** Install script now lets users choose which providers to enable and prompts for API keys only for enabled providers.
- **Reasoning effort passthrough.** `/v1/chat/completions` and `/v1/messages` now accept `reasoning_effort` and `extra_body` and forward them to upstream providers (UMANS, OpenCode Go, Ollama Cloud).

### Changed
- **Dashboard UI polish.** Removed duplicate Network card from Endpoints tab; tab bar consolidated into a single scrollable row; System tab now hosts Network Visibility.
- **Analytics performance.** Analytics DB switched to WAL mode with composite indexes for faster summary queries on large datasets; refresh times reduced from several seconds to near-instant.
- **Model list caching.** `/v1/models` responses are cached in memory for 60 seconds, reducing upstream provider calls and dashboard load.
- **Startup logging.** Lifespan prints converted to `logger.info` so all startup output flows through uvicorn / systemd journals.

### Fixed
- **Image URL conversion loop.** `_convert_image_urls_to_base64` no longer returns after the first message; all vision messages are processed.

---

## [0.5.5] - 2026-06-17

### Fixed
- **Updater no longer requires `pip` inside the virtual environment.** It detects whether the running venv has `pip`; if not, it falls back to `uv pip install --python <venv>`. This fixes self-updates on uv-created or `--without-pip` venvs.

### Yanked
- **v0.5.4 was withdrawn.** Its auto-updater failed with `No module named pip` on venvs without pip. Install v0.5.5 instead.

---

## [0.5.4] - 2026-06-17

### Fixed
- **Streaming bug with MultiProviderChatClient:** `chat_completion_stream` no longer returns a coroutine. It now yields chunks directly so callers can `async for` over the stream. This fixes the `'async for' requires an object with aiter method, got coroutine` error that broke all Ollama requests when the multi-provider client was active.

---

## [0.5.3] - 2026-06-17

### Fixed
- **OpenCode Go account detection on add/update:** accounts created through the dashboard with `sk-*` API keys are now tagged as `opencode_go` instead of defaulting to `ollama`. A provider selector is now respected and the key is tested with the matching client.
- **Config migration on load:** existing accounts whose stored provider does not match their key prefix are auto-corrected, so already-misclassified OpenCode Go accounts show Go models after restart.
- **Dashboard account list now exposes the stored provider field**, so each account shows the correct provider badge (`ollama` or `opencode_go`).

---

## [0.5.2] - 2026-06-10

### Fixed
- **Token estimation no longer silently disabled when history logging is off.** The `_history_kwargs` helper in the router now unconditionally populates `input_text`/`output_text` for cost tracking, regardless of the `save_history` toggle. Previously, disabling history logging accidentally starved the analytics pipeline of token data.
- **Ollama native eval counters now used as fallback for OpenAI-style usage.** If an upstream response lacks `prompt_tokens`/`completion_tokens`, we fall back to `prompt_eval_count` / `eval_count` before falling back to tiktoken estimation.

---

## [0.5.1] - 2026-06-10

### Fixed
- **Updater now stashes + hard-resets instead of merge-pulling.** Previously if the working tree had any local modifications (e.g. leftover version-string edits from a prior partial update), `git pull` would abort with a merge conflict and the update silently failed, leaving the old code running. The updater now unconditionally stashes any local changes (including untracked files) and resets to `origin/{branch}` before reinstalling. This makes the **Apply Update** button work reliably on every install, even if the repo is dirty.

### Changed
- Bumped version to 0.5.1 to ensure existing 0.4.2 → 0.5.x update paths hit the new updater logic immediately.

---

## [0.5.0] - 2026-06-10

### Major Release — Usage Tracking, ROI Dashboard, and Multi-Account Infrastructure

This release represents a significant milestone: Guanaco now tracks every token accurately, displays real cost analytics via a web dashboard, rotates multiple Ollama Cloud accounts, and scrapes live usage tiers from ollama.com instead of guessing.

---

### New Features

#### Token Estimation & Accurate Usage Tracking
- **skimtoken estimation fallback** (`analytics.py`): When the upstream API (OpenRouter, Ollama Cloud) omits `usage` data in the response, Guanaco now falls back to `skimtoken` for token estimation instead of logging zero tokens. Estimates are ~15% accurate — dramatically better than silently losing usage data.
- **Proper total_tokens calculation**: Fixed `total_tokens = prompt_tokens + completion_tokens` in analytics logging. Previously some code paths double-counted or omitted totals.
- **`fallback_reason` audit column**: Added to `request_log` schema. When token estimation is used instead of API-reported usage, the reason is recorded (e.g. `"api_omitted_usage"`, `"stream_missing_usage"`) for later audit.
- **Input cache-read pricing** (`roi.py`): Tracks `input_cache_read` tokens separately from `input_cache_write`, applying the correct Ollama Cloud discount rate (typically 0.25× of input price for cache hits).

#### ROI Dashboard & Per-Model Analytics
- **Web dashboard** (`dashboard/`): New FastAPI-mounted dashboard at `/dashboard/` showing:
  - Total tokens consumed (last 24h, 7d, 30d, all-time)
  - Per-model token distribution with visual bars
  - Cost estimates in USD using live OpenRouter pricing
  - **ROI configuration panel**: Slider for `cache_hit_pct` (default 70%), editable price multipliers per model
  - **Per-model value scoring**: Each model gets a "value score" based on (capability / cost) ratio, helping users pick the cheapest model for a given task
- **OpenRouter price fetcher** (`roi.py`): Scrapes current model pricing from `https://openrouter.ai/api/v1/models` with 24h caching. Falls back to hardcoded prices if fetch fails.
- **Cache-hit discount logic**: ROI calculations apply the user-configured `cache_hit_pct` to reduce effective input costs, reflecting real-world Ollama Cloud behavior where repeated prompts are cached.

#### Real Usage-Level Scraping from ollama.com
- **`_fetch_usage_level_sync()`** (`client.py`): New synchronous HTML scraper that parses `ollama.com/library/{model}` pages to determine actual GPU usage tiers:
  - Handles **top-level model badges** (`x-test-model-cost-slot-active`) for unified-tier models
  - Handles **per-tag listings** (`x-test-model-tag-cost` + `x-test-model-tag-usage-slot-active`) for models with multiple size variants
  - Returns usage level 1-4, which maps to multiplier 0.25×, 0.50×, 0.75×, 1.00×
- **`fetch_usage_levels()`**: Async parallel fetcher with **1-hour global cache**. Fetches all library pages concurrently using `asyncio.gather()` with thread-pool execution for the blocking HTTP requests.
- **Wired into API responses**: Both `/v1/models` (OpenAI-compatible) and `/api/ollama/models` (internal) now return `usage_multiplier` and `usage_level` fields based on scraped live data.
- **Fixes major heuristic errors**:
  - `gemma3:4b` and `gemma3:12b`: was 1.00×, now correctly **0.25×** (1 slot)
  - `gemma3:27b`: was 1.00×, now correctly **0.50×** (2 slots)
  - `ministral-3`: was 0.75×, now correctly **0.25×** (1 slot)
  - `qwen3-vl`: stays **0.75×** (3 slots) — heuristic accidentally got this one right
  - `deepseek-v4-pro`: stays **1.00×** (4 slots)

#### Multi-Account Ollama Cloud Rotation
- **`accounts.py`**: New module managing multiple Ollama Cloud accounts:
  - Each account has its own API key + session cookie
  - Load-balanced request routing based on usage and subscription tier
  - Quota-aware selection: least-loaded accounts preferred; new/untested accounts get priority for immediate validation
- **Premium model routing**: Models `kimi-k2.6` and `glm-5.1` restricted to Pro/Max accounts only. Free-tier accounts are skipped for these models.
- **Per-account usage tracking**: Analytics DB records which account handled each request, enabling per-account cost breakdowns.

#### Model Catalog Expansion
- Added `minimax-m3` (new MiniMax flagship)
- Added `nemotron-3-ultra` (NVIDIA enterprise model)
- Added `kimi-k2.6` with 200k context window support

#### Web Search / Scrape Emulation
- **Search provider plugins** (`search/providers/`): Modular search backend support:
  - Brave Search API
  - Cohere RAG API
  - Exa (formerly Metaphor)
  - Firecrawl (web scraping)
  - Jina AI (neural search)
  - SearXNG (self-hosted meta-search)
  - Serper (Google Search API)
  - Tavily (AI-native search)
- **Search router** (`search/base.py`): Unified interface — Guanaco presents a single `/search` endpoint regardless of which provider is configured.

### Fixes

#### Config & Install Robustness
- **Missing `UsageConfig` fields**: Added `last_plan`, `redirect_on_full`, `last_session_reset`, `last_weekly_reset`, `last_checked` to prevent `AttributeError` crashes on configs from v0.4.2 and earlier.
- **Config migration layer**: `load_config()` now auto-migrates v0.4.2 configs to v0.4.3+ schema on first load. No manual intervention needed.
- **Package rename**: Renamed PyPI package from `guanaco` → `guanaco-llm-proxy` to avoid collision with an existing `guanaco` package on PyPI.
- **Install script fixes** (`install.sh`):
  - Ollama API key validation now uses the correct env var name
  - Fixed `.env` file write pattern (was writing malformed key=value pairs)
  - Fixed `grep` pattern for detecting existing config
- **Startup version sanity check**: Detects repo/venv version mismatch on boot and logs a warning. Prevents confusing "why is `/health` showing the old version?" issues.
- **systemd service**: Fixed `WorkingDirectory` to point at the actual repo checkout. Added `GUANACO_CONFIG_DIR` env var to service file.

#### Dashboard & Analytics Fixes
- **Removed broken `usage_multiplier` column**: The analytics DB no longer tracks `usage_multiplier` per request (it was always wrong due to heuristic mismatch). Model-level multipliers are now fetched live from ollama.com.
- **Backward compat for `SearchConfig`**: Older installs missing search configuration no longer crash on startup.

### Infrastructure

#### CI/CD
- **GitHub Actions CI** (`.github/workflows/ci.yml`): Runs on every push — lint, type-check, unit tests.
- **GitHub Actions Release** (`.github/workflows/release.yml`): Automated PyPI publish on tag push.

#### Docker
- **`Dockerfile.test`**: Containerized smoke-test environment for CI.
- **`test-local.sh`**: One-command local smoke test — builds Docker image, starts server, hits `/health`, validates version string.

#### Project Hygiene
- Added `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `LICENSE` (MIT)
- Added `.gitignore` with Python/venv patterns
- Added macOS launch agent plist (`com.guanaco.start.plist`)
- Added systemd service templates (`guanaco.service`, `oct.service`)

### API Changes

#### Added Fields
- `/v1/models` response now includes:
  - `usage_multiplier` (float): cost multiplier 0.25-1.00
  - `usage_level` (int): raw level 1-4, 0 = unknown
- `/api/ollama/models` response now includes:
  - `usage_multiplier` (float)
  - `usage_level` (int)

#### Schema Changes
- `request_log` table: added `fallback_reason TEXT` column
- `request_log` table: removed `usage_multiplier` column (was unreliable)
- New `roi_config` table: stores `cache_hit_pct`, `price_multiplier`, per-model overrides

### Performance
- **Parallel library scraping**: All ollama.com library pages are fetched concurrently. For a catalog of ~50 models, total scrape time is ~3-5 seconds vs. ~60 seconds sequential.
- **1-hour cache**: Scraped usage levels are cached globally, so the 3-5 second penalty only hits once per hour.
- **ROI price cache**: OpenRouter prices cached for 24 hours. Dashboard loads instantly after first visit.

### Deprecated / Removed
- **Heuristic `_get_model_multiplier()`**: Still exists as fallback when ollama.com scraping fails, but is no longer the primary source. Returns `0.25` for ≤8B, `0.50` for ≤70B, `0.75` for ≤200B, `1.00` for larger.
- **`usage_multiplier` column in analytics DB**: Dropped. Use `/v1/models` or `/api/ollama/models` to get live multipliers.

### Known Issues
- **Dev server restart unreliable on isolated instance**: The `uvicorn` process sometimes starts without producing logs. Production (`systemctl restart guanaco.service`) is unaffected.
- **Library scraper depends on ollama.com DOM**: If ollama.com changes their HTML test attributes (`x-test-model-*`), the scraper will fall back to heuristic. Monitor `/api/ollama/models` for sudden multiplier shifts.

---

## [0.4.2] - 2026-05-15

### New Features
- Multi-account Ollama Cloud rotation with quota-aware selection
- Premium model routing (`kimi-k2.6`, `glm-5.1` → Pro/Max only)
- Per-account usage tracking

---

## [0.4.1] - 2026-05-01

### Fixes
- Rate-limit retry logic for Ollama Cloud 429 responses
- SSE streaming stability improvements

---

## [0.4.0] - 2026-04-20

### New Features
- Initial Ollama Cloud proxy support
- OpenAI-compatible `/v1/chat/completions` endpoint
- Token usage tracking with SQLite analytics DB
- Basic web dashboard

---

## [0.3.9] and earlier

See [GitHub releases](https://github.com/evangit2/guanaco/releases) for earlier versions.
