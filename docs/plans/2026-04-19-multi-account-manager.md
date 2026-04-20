# Multi-Account Manager Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Allow Guanaco to manage multiple Ollama Cloud accounts, rotating requests to the account with the most remaining usage, with a setup wizard in Settings to enter/exit multi-account mode.

**Architecture:** Add an `AccountConfig` model (name + API key + session cookie + cached usage) and an `accounts` list to `AppConfig`. An `AccountManager` class holds per-account `OllamaClient` instances, tracks usage, and selects the best account per request. The router uses the manager instead of a single client. A setup wizard in the Settings tab handles entering/exiting multi-account mode.

**Tech Stack:** Python/FastAPI backend, vanilla JS dashboard, SQLite analytics (account name logged per request)

---

## Current Architecture (what changes)

```
app.py → OllamaClient(single_key, single_cookie) → router(client, ...)
         UsageConfig(session_cookie=one) → quota check on single %
```

## New Architecture

```
app.py → AccountManager(accounts=[AccountConfig,...]) → router(manager, ...)
         Each AccountConfig has: name, api_key, session_cookie, usage cache
         AccountManager.select_account() → picks lowest-usage account
         Single-account mode: AccountManager with 1 account (backwards compat)
```

## Config YAML Shape

Single-account (default, same as today):
```yaml
ollama_api_key: "key1"
usage:
  session_cookie: "cookie1"
  ...
```

Multi-account mode:
```yaml
ollama_api_key: "key1"  # kept for backward compat, becomes "primary" named account
accounts:
  - name: "primary"
    api_key: "key1"
    session_cookie: "cookie1"
    last_session_pct: 21.8
    last_weekly_pct: 64.2
    last_session_reset: "4 hours"
    last_weekly_reset: "1 hour"
    last_plan: "pro"
    last_checked: 1776639060.0
  - name: "account2"
    api_key: "key2"
    session_cookie: "cookie2"
    last_session_pct: null
    last_weekly_pct: null
    ...
usage:
  redirect_on_full: true
  multi_account_enabled: true  # NEW: toggles multi-account mode
```

When `multi_account_enabled = true`, the `accounts` list is the source of truth.
When `false` (default), the single `ollama_api_key` + `usage.session_cookie` are used (exactly as today).

## Analytics Changes

- `request_log` gets a new `account_name TEXT` column
- In History tab, provider shows as `ollama (primary)` or `ollama (account2-3243)`
- Analytics consolidated across all accounts (no change to aggregation — just new column)

## Router Changes

- `create_router(client, ...)` becomes `create_router(account_manager, ...)`
- Before each request: `account = manager.select_account()` → returns `(name, OllamaClient)`
- The OllamaClient for the selected account is used for the actual request
- `fallback_for` field now also used when account rotates (new reason: "Account rotated: primary quota full")
- All `log_llm` calls include `account_name=account.name`

## Dashboard Changes

### Settings Tab — Multi-Account Setup
- New section: "Ollama Accounts"
- Shows current mode: "Single Account" or "Multi-Account"
- "Enter Multi-Account Mode" button → wizard:
  1. Name your primary account (pre-filled "primary")
  2. Enter name for second account
  3. Enter API key for second account
  4. Both account session cookies can be set in Status tab
- "Exit Multi-Account Mode" button → reverts to single account (keeps primary key)
- Account list with remove buttons

### Status Tab — Multi-Account Usage
- When multi-account is enabled, shows usage bars for EACH account
- Each account row: name, session/weekly %, progress bars, reset timers
- "Check All Usage" button checks all accounts in parallel

### History Tab
- Provider column shows `ollama (account_name)` instead of just `ollama`
- New "Account" filter dropdown

---

## Implementation Tasks

### Phase 1: Data Model & Config

#### Task 1: Add AccountConfig and multi_account_enabled to config.py

**Objective:** Define the data models for multi-account support.

**Files:**
- Modify: `guanaco/config.py`

**Step 1: Add AccountConfig model**

Add after the existing `UsageConfig` class:

```python
class AccountConfig(BaseModel):
    """A single Ollama Cloud account with its own API key, session cookie, and usage cache."""
    name: str = "primary"
    api_key: str = ""
    session_cookie: str = ""
    last_session_pct: Optional[float] = None
    last_weekly_pct: Optional[float] = None
    last_session_reset: Optional[str] = None
    last_weekly_reset: Optional[str] = None
    last_plan: Optional[str] = None
    last_checked: Optional[float] = None
```

**Step 2: Add fields to AppConfig**

Add to AppConfig:

```python
accounts: list[AccountConfig] = []  # Populated when multi_account_enabled=True
multi_account_enabled: bool = False
```

**Step 3: Add helper property to AppConfig**

```python
@property
def active_accounts(self) -> list[AccountConfig]:
    """Return accounts list if multi-account enabled, else synthesize single account from legacy fields."""
    if self.multi_account_enabled and self.accounts:
        return self.accounts
    # Single-account mode — synthesize from legacy fields
    return [AccountConfig(
        name="primary",
        api_key=self.ollama_api_key_resolved,
        session_cookie=self.usage.session_cookie if self.usage else "",
        last_session_pct=self.usage.last_session_pct if self.usage else None,
        last_weekly_pct=self.usage.last_weekly_pct if self.usage else None,
        last_session_reset=self.usage.last_session_reset if self.usage else None,
        last_weekly_reset=self.usage.last_weekly_reset if self.usage else None,
        last_plan=self.usage.last_plan if self.usage else None,
        last_checked=self.usage.last_checked if self.usage else None,
    )]
```

**Step 4: Verify the config loads correctly**

Run: `cd ~/projects/guanaco && source venv/bin/activate && python3 -c "from guanaco.config import load_config; c = load_config(); print('accounts:', c.accounts, 'enabled:', c.multi_account_enabled)"`

Expected: `accounts: [] enabled: False`

**Step 5: Commit**

```bash
git add guanaco/config.py
git commit -m "feat: add AccountConfig model and multi_account_enabled to AppConfig"
```

---

#### Task 2: Create AccountManager class

**Objective:** Build the class that manages per-account OllamaClient instances and selects the best account for each request.

**Files:**
- Create: `guanaco/account_manager.py`

**Step 1: Create the AccountManager**

```python
"""Multi-account manager for Ollama Cloud — rotates across accounts to maximize usage."""

import logging
from typing import Optional, Tuple

from guanaco.client import OllamaClient
from guanaco.config import AccountConfig, AppConfig

log = logging.getLogger("guanaco.accounts")


class AccountManager:
    """Manages multiple Ollama Cloud accounts and selects the best one per request."""

    def __init__(self, config: AppConfig):
        self._config = config
        self._clients: dict[str, OllamaClient] = {}
        self._rebuild_clients()

    def _rebuild_clients(self):
        """Rebuild OllamaClient instances from config."""
        self._clients.clear()
        for acct in self._config.active_accounts:
            if acct.api_key and acct.api_key not in ("***", "REPLACE_ME", "your_api_key_here"):
                self._clients[acct.name] = OllamaClient(
                    api_key=acct.api_key,
                    session_cookie=acct.session_cookie,
                )
                log.debug("Account client built: %s", acct.name)

    def refresh(self):
        """Rebuild clients after config change (e.g., account added/removed)."""
        self._rebuild_clients()

    @property
    def accounts(self) -> list[AccountConfig]:
        return self._config.active_accounts

    def get_client(self, account_name: Optional[str] = None) -> Tuple[str, OllamaClient]:
        """Get the best OllamaClient for a request.
        
        If account_name is specified, return that account's client.
        Otherwise, select the account with the lowest usage.
        
        Returns (account_name, OllamaClient).
        Raises ValueError if no accounts available.
        """
        if account_name and account_name in self._clients:
            return account_name, self._clients[account_name]

        # Select account with lowest usage
        best_name = None
        best_score = float('inf')
        for acct in self.accounts:
            if acct.name not in self._clients:
                continue
            # Score = max(session%, weekly%). Lower is better. None = unchecked = 0 (best).
            s = acct.last_session_pct if acct.last_session_pct is not None else 0
            w = acct.last_weekly_pct if acct.last_weekly_pct is not None else 0
            score = max(s, w)
            if score < best_score:
                best_score = score
                best_name = acct.name

        if best_name is None:
            # Fallback: return first available client
            if self._clients:
                best_name = next(iter(self._clients))
            else:
                raise ValueError("No Ollama accounts configured")

        return best_name, self._clients[best_name]

    def get_all_clients(self) -> dict[str, OllamaClient]:
        """Return all account clients (for usage checking etc)."""
        return dict(self._clients)

    def is_quota_full(self, account_name: Optional[str] = None) -> bool:
        """Check if a specific account (or the selected account) is quota-full."""
        if not self._config.usage.redirect_on_full:
            return False
        # Check the specific account or the best account
        target = account_name or self.get_client()[0]
        for acct in self.accounts:
            if acct.name == target:
                s = acct.last_session_pct
                w = acct.last_weekly_pct
                if s is not None and s >= 99.5:
                    return True
                if w is not None and w >= 99.5:
                    return True
                return False
        return False

    def any_account_available(self) -> bool:
        """Check if at least one account is not quota-full."""
        if not self._config.usage.redirect_on_full:
            return True
        for acct in self.accounts:
            s = acct.last_session_pct if acct.last_session_pct is not None else 0
            w = acct.last_weekly_pct if acct.last_weekly_pct is not None else 0
            if s < 99.5 and w < 99.5:
                return True
        return False

    def update_account_usage(self, account_name: str, session_pct: Optional[float],
                             weekly_pct: Optional[float], session_reset: Optional[str],
                             weekly_reset: Optional[str], plan: Optional[str]):
        """Update cached usage for an account and persist to config."""
        for acct in self._config.accounts:
            if acct.name == account_name:
                acct.last_session_pct = session_pct
                acct.last_weekly_pct = weekly_pct
                acct.last_session_reset = session_reset
                acct.last_weekly_reset = weekly_reset
                acct.last_plan = plan
                import time
                acct.last_checked = time.time()
                break
        # Persist
        try:
            from guanaco.config import save_config
            save_config(self._config)
        except Exception as e:
            log.warning("Failed to persist account usage: %s", e)

    def update_session_cookie(self, account_name: str, cookie: str):
        """Update session cookie for an account."""
        for acct in self._config.accounts:
            if acct.name == account_name:
                acct.session_cookie = cookie
                break
        if account_name in self._clients:
            self._clients[account_name]._session_cookie = cookie
        try:
            from guanaco.config import save_config
            save_config(self._config)
        except Exception as e:
            log.warning("Failed to persist session cookie: %s", e)
```

**Step 2: Test it loads**

Run: `cd ~/projects/guanaco && source venv/bin/activate && python3 -c "from guanaco.account_manager import AccountManager; print('import OK')"`

Expected: `import OK`

**Step 3: Commit**

```bash
git add guanaco/account_manager.py
git commit -m "feat: add AccountManager for multi-account rotation"
```

---

### Phase 2: Analytics Update

#### Task 3: Add account_name column to request_log

**Objective:** Track which account handled each request in analytics.

**Files:**
- Modify: `guanaco/analytics.py`

**Step 1: Add migration in `_ensure_tables`**

After the `fallback_reason` migration block, add:

```python
# Migration: add account_name column
try:
    conn.execute("ALTER TABLE request_log ADD COLUMN account_name TEXT")
except sqlite3.OperationalError:
    pass
```

**Step 2: Add account_name to log_llm signature and INSERT**

Add `account_name: Optional[str] = None` parameter to `log_llm`.

Add `account_name` to the INSERT column list and VALUES tuple.

**Step 3: Verify migration**

Run: `cd ~/projects/guanaco && source venv/bin/activate && python3 -c "import sqlite3; conn = sqlite3.connect('/home/evan/.guanaco/analytics.db'); cur = conn.cursor(); cur.execute('PRAGMA table_info(request_log)'); cols = [r[1] for r in cur.fetchall()]; print('account_name' in cols)"`

Expected: `True`

**Step 4: Commit**

```bash
git add guanaco/analytics.py
git commit -m "feat: add account_name column to request_log for multi-account tracking"
```

---

### Phase 3: Router Integration

#### Task 4: Switch router from single client to AccountManager

**Objective:** The router uses AccountManager to select the best account per request instead of a single OllamaClient.

**Files:**
- Modify: `guanaco/router/router.py`
- Modify: `guanaco/app.py`

This is the biggest task. Key changes:

**Step 1: Update app.py create_app()**

Replace:
```python
client = OllamaClient(api_key=resolved_key, session_cookie=config.usage.session_cookie)
```

With:
```python
from guanaco.account_manager import AccountManager
account_manager = AccountManager(config)
```

Pass `account_manager` instead of `client` to `create_router` and dashboard.

**Step 2: Update create_router signature**

Change from `create_router(client, ...)` to `create_router(account_manager, ...)`.

Replace `_client = client` with `_manager = account_manager`.

**Step 3: Update each request handler**

In `chat_completions` and the Anthropic endpoint, at the top:

```python
acct_name, _client = _manager.get_client()
```

This replaces the single `_client` closure. The rest of the request logic uses `_client` the same way.

**Step 4: Add account_name to all log_llm calls**

Every `_analytics.log_llm(...)` call needs `account_name=acct_name`.

**Step 5: Update _is_quota_full**

Replace:
```python
def _is_quota_full(config) -> bool:
```

With logic that uses `_manager.any_account_available()`. If any account is available, return False. If ALL accounts are full, return True (trigger fallback).

Also, when an account is full but others aren't, the manager auto-selects a different account — no fallback needed. Fallback only triggers when ALL accounts are full.

**Step 6: Update quota redirect section**

The quota-full check at line ~478 should:

1. Get the best account from the manager
2. If that account is the same as before, use it
3. If the selected account changed (rotation), use the new account's client
4. Only fall back to the external fallback if ALL accounts are full

Replace the current quota redirect block with:

```python
if _is_quota_full(_config):
    # All accounts full — go to external fallback
    ...
else:
    # Use manager-selected account (may have rotated)
    acct_name, _client = _manager.get_client()
```

**Step 7: Verify the test instance starts**

Run: `kill $(lsof -t -i:8888) 2>/dev/null; sleep 1; cd ~/projects/guanaco && source venv/bin/activate && GUANACO_ROUTER_PORT=8888 python -m uvicorn guanaco.app:create_app --factory --host 0.0.0.0 --port 8888 &`

Then: `curl -s http://localhost:8888/health`

Expected: `{"status": "ok", ...}`

**Step 8: Commit**

```bash
git add guanaco/router/router.py guanaco/app.py
git commit -m "feat: router uses AccountManager for multi-account request routing"
```

---

#### Task 5: Update dashboard.py to use AccountManager

**Objective:** Dashboard API endpoints (usage checking, session cookies, config) work with multi-account.

**Files:**
- Modify: `guanaco/dashboard/dashboard.py`

Key changes:

**Step 1: Accept account_manager parameter**

Update `create_dashboard()` to accept `account_manager` instead of (or in addition to) `client`.

**Step 2: Add multi-account API endpoints**

```python
@router.get("/api/accounts")
async def list_accounts(request: Request):
    """List all configured accounts and their usage."""
    accounts = account_manager.accounts
    return {"accounts": [a.model_dump() for a in accounts], "multi_account_enabled": config.multi_account_enabled}

@router.post("/api/accounts")
async def add_account(request: Request):
    """Add a new account in multi-account mode."""
    body = await request.json()
    name = body.get("name", "")
    api_key = body.get("api_key", "")
    if not name or not api_key:
        return {"error": "Name and API key required"}
    # Check duplicate names
    for a in config.accounts:
        if a.name == name:
            return {"error": f"Account '{name}' already exists"}
    config.accounts.append(AccountConfig(name=name, api_key=api_key))
    config.multi_account_enabled = True
    save_config(config)
    account_manager.refresh()
    return {"ok": True}

@router.delete("/api/accounts/{name}")
async def remove_account(name: str, request: Request):
    """Remove an account. If last one, disable multi-account mode."""
    config.accounts = [a for a in config.accounts if a.name != name]
    if len(config.accounts) <= 1:
        config.multi_account_enabled = False
        if config.accounts:
            # Move back to single key
            config.ollama_api_key = config.accounts[0].api_key
            config.usage.session_cookie = config.accounts[0].session_cookie
            config.accounts = []
    save_config(config)
    account_manager.refresh()
    return {"ok": True}

@router.post("/api/accounts/enable-multi")
async def enable_multi_account(request: Request):
    """Enter multi-account mode. Migrates single key to named account."""
    body = await request.json()
    primary_name = body.get("primary_name", "primary")
    config.accounts = [
        AccountConfig(
            name=primary_name,
            api_key=config.ollama_api_key_resolved,
            session_cookie=config.usage.session_cookie if config.usage else "",
            last_session_pct=config.usage.last_session_pct if config.usage else None,
            last_weekly_pct=config.usage.last_weekly_pct if config.usage else None,
            last_session_reset=config.usage.last_session_reset if config.usage else None,
            last_weekly_reset=config.usage.last_weekly_reset if config.usage else None,
            last_plan=config.usage.last_plan if config.usage else None,
            last_checked=config.usage.last_checked if config.usage else None,
        )
    ]
    config.multi_account_enabled = True
    save_config(config)
    account_manager.refresh()
    return {"ok": True, "accounts": [a.model_dump() for a in config.accounts]}

@router.post("/api/accounts/disable-multi")
async def disable_multi_account(request: Request):
    """Exit multi-account mode. Reverts to single key from first account."""
    if config.accounts:
        config.ollama_api_key = config.accounts[0].api_key
        config.usage.session_cookie = config.accounts[0].session_cookie
        config.multi_account_enabled = False
        config.accounts = []
    save_config(config)
    account_manager.refresh()
    return {"ok": True}
```

**Step 3: Update usage check endpoint**

`POST /dashboard/api/usage/check` should check ALL accounts' usage when multi-account is enabled:

```python
@router.post("/api/usage/check")
async def check_usage(request: Request):
    results = []
    for acct in account_manager.accounts:
        client = account_manager.get_all_clients().get(acct.name)
        if not client:
            results.append({"name": acct.name, "error": "No client"})
            continue
        cookie = acct.session_cookie
        if not cookie:
            results.append({"name": acct.name, "error": "No session cookie set"})
            continue
        try:
            usage = await client.get_usage(session_cookie=cookie)
            # Update account cache
            account_manager.update_account_usage(
                acct.name,
                session_pct=usage.get("session_pct"),
                weekly_pct=usage.get("weekly_pct"),
                session_reset=usage.get("session_reset"),
                weekly_reset=usage.get("weekly_reset"),
                plan=usage.get("plan"),
            )
            results.append({"name": acct.name, **usage})
        except Exception as e:
            results.append({"name": acct.name, "error": str(e)})
    # For backward compat, if single account, return flat response
    if len(results) == 1:
        return results[0]
    return {"accounts": results}
```

**Step 4: Update session cookie endpoint**

`POST /dashboard/api/usage/session-cookie` needs an `account_name` field:

```python
@router.post("/api/usage/session-cookie")
async def set_session_cookie(request: Request):
    body = await request.json()
    cookie = body.get("cookie", "")
    account_name = body.get("account_name", "primary")
    if config.multi_account_enabled:
        account_manager.update_session_cookie(account_name, cookie)
    else:
        config.usage.session_cookie = cookie
        # Also update the live client if accessible
        for client in account_manager.get_all_clients().values():
            client._session_cookie = cookie
    save_config(config)
    return {"ok": True}
```

**Step 5: Commit**

```bash
git add guanaco/dashboard/dashboard.py
git commit -m "feat: dashboard multi-account API endpoints (list, add, remove, enable/disable)"
```

---

### Phase 4: Dashboard UI

#### Task 6: Settings tab — Multi-account setup UI

**Objective:** Add the setup wizard and account management UI to Settings.

**Files:**
- Modify: `guanaco/dashboard/templates/dashboard.html`

**Step 1: Add "Ollama Accounts" section to Settings tab**

In the Settings tab, add a new card section for account management. Contains:
- Current mode badge: "Single Account" or "Multi-Account (N accounts)"
- Account list (when multi-account enabled): name, API key (masked), remove button
- "Enter Multi-Account Mode" button (when single) → opens wizard modal
- "Add Account" button (when multi) → opens add modal
- "Exit Multi-Account Mode" button (when multi) → confirms and reverts

**Step 2: Add wizard modal**

The wizard has steps:
1. "Name your primary account" — input with "primary" pre-filled
2. "Add second account" — name + API key inputs
3. "Setup complete" — shows both accounts, notes that session cookies go in Status tab

**Step 3: JS functions**

```javascript
function loadAccountConfig() {
    fetch('/dashboard/api/accounts').then(r => r.json()).then(data => {
        // Render account list and mode badge
    });
}

function enterMultiAccountMode() {
    // Show wizard modal
}

function addAccount() {
    // Show add-account modal
}

function removeAccount(name) {
    // Confirm, then DELETE /dashboard/api/accounts/{name}
}

function exitMultiAccountMode() {
    // Confirm, then POST /dashboard/api/accounts/disable-multi
}
```

**Step 4: Hook into showTab()**

When Settings tab is shown, call `loadAccountConfig()`.

**Step 5: Commit**

```bash
git add guanaco/dashboard/templates/dashboard.html
git commit -m "feat: Settings tab multi-account setup wizard UI"
```

---

#### Task 7: Status tab — Multi-account usage display

**Objective:** Show per-account usage bars when multi-account is enabled.

**Files:**
- Modify: `guanaco/dashboard/templates/dashboard.html`

**Step 1: Update usage check JS**

When multi-account is enabled, `checkUsage()` should handle the `accounts` array response and render a row per account.

Each row: account name, session % bar, weekly % bar, reset timers, "Check" button.

**Step 2: Update session cookie section**

When multi-account, show a dropdown to select which account's cookie to set, plus the cookie input and save button.

**Step 3: Commit**

```bash
git add guanaco/dashboard/templates/dashboard.html
git commit -m "feat: Status tab multi-account usage display with per-account bars"
```

---

#### Task 8: History tab — Account name in provider column

**Objective:** Show which account handled each request in the History list and modal.

**Files:**
- Modify: `guanaco/dashboard/templates/dashboard.html`

**Step 1: Update list rendering**

Change the provider display from `🏭 ${r.provider || 'ollama'}` to:

```javascript
let providerDisplay = r.provider || 'ollama';
if (providerDisplay === 'ollama' && r.account_name) {
    providerDisplay = `ollama (${escapeHtml(r.account_name)})`;
}
```

**Step 2: Update modal**

In the detail modal's metadata grid, add:

```html
<div><strong>Account:</strong> ${data.account_name || '—'}</div>
```

**Step 3: Add account filter dropdown**

Add a new filter dropdown in the History tab header for filtering by account name.

**Step 4: Commit**

```bash
git add guanaco/dashboard/templates/dashboard.html
git commit -m "feat: History tab shows account name in provider column, account filter"
```

---

### Phase 5: Config Persistence

#### Task 9: Add save_config function and ensure accounts persist

**Objective:** Ensure the `accounts` list and `multi_account_enabled` field survive config.yaml round-trips.

**Files:**
- Modify: `guanaco/config.py`

**Step 1: Verify save_config exists**

Check that `save_config` writes all Pydantic model fields including `accounts` and `multi_account_enabled` to `config.yaml`. If it doesn't exist, add it.

**Step 2: Test round-trip**

```python
from guanaco.config import load_config, save_config
c = load_config()
c.multi_account_enabled = True
c.accounts = [AccountConfig(name="test", api_key="key123")]
save_config(c)
c2 = load_config()
assert c2.multi_account_enabled == True
assert len(c2.accounts) == 1
assert c2.accounts[0].name == "test"
```

**Step 3: Commit**

```bash
git add guanaco/config.py
git commit -m "feat: config persistence for accounts and multi_account_enabled"
```

---

### Phase 6: Testing & Polish

#### Task 10: End-to-end test on port 8888

**Objective:** Verify the full multi-account flow works.

**Step 1: Start test instance**

```bash
cd ~/projects/guanaco && source venv/bin/activate && GUANACO_ROUTER_PORT=8888 python -m uvicorn guanaco.app:create_app --factory --host 0.0.0.0 --port 8888 &
```

**Step 2: Test single-account mode (default)**

- `curl http://localhost:8888/dashboard/api/accounts` → `{"accounts": [...], "multi_account_enabled": false}`
- Send a request, verify it works
- Check analytics: `account_name` should be "primary"

**Step 3: Test entering multi-account mode**

- `curl -X POST http://localhost:8888/dashboard/api/accounts/enable-multi -H 'Content-Type: application/json' -d '{"primary_name":"main"}'`
- Verify accounts list now has one account named "main"

**Step 4: Test adding second account**

- `curl -X POST http://localhost:8888/dashboard/api/accounts -H 'Content-Type: application/json' -d '{"name":"backup","api_key":"test-key"}'`
- Verify accounts list has two entries

**Step 5: Test account rotation**

- Send multiple requests, verify the account with lower usage is selected
- Check analytics for `account_name` values

**Step 6: Test removing account**

- `curl -X DELETE http://localhost:8888/dashboard/api/accounts/backup`
- Verify single account remains

**Step 7: Test exiting multi-account mode**

- `curl -X POST http://localhost:8888/dashboard/api/accounts/disable-multi`
- Verify `multi_account_enabled = false` and `accounts = []`

**Step 8: Commit**

```bash
git commit -m "test: end-to-end multi-account verification"
```

---

## Key Design Decisions

1. **Backward compatible**: Single-account mode (default) works exactly as today. `multi_account_enabled=false` means `accounts` list is ignored, legacy `ollama_api_key` + `usage.session_cookie` are used.

2. **AccountManager abstracts the client**: The router doesn't need to know about multi-account. It calls `manager.get_client()` and gets back a `(name, OllamaClient)` pair. The manager handles selection.

3. **Usage-based rotation**: The manager picks the account with the lowest `max(session%, weekly%)`. Unchecked accounts (None values) score 0, so they get tried first (then their usage gets populated).

4. **Analytics include account_name**: New column, shown in History as `ollama (account_name)`. Consolidated stats work the same — you can filter by account.

5. **Fallback still works**: If ALL accounts are quota-full, the external fallback triggers. Individual account rotation happens first.

6. **Session cookies per account**: Each account has its own cookie, set via the Status tab. The wizard tells the user to set cookies there after setup.

7. **Wizard flow**: "Enter Multi-Account Mode" → name primary → name + key for second account → done. User can add more later. "Exit" reverts to single key from first account.

## Files Changed Summary

| File | Change |
|---|---|
| `guanaco/config.py` | Add `AccountConfig`, `multi_account_enabled`, `active_accounts` property, ensure `save_config` works |
| `guanaco/account_manager.py` | NEW — AccountManager class |
| `guanaco/analytics.py` | Add `account_name` column migration + log_llm param |
| `guanaco/router/router.py` | Use AccountManager instead of single client, pass account_name to log_llm |
| `guanaco/app.py` | Create AccountManager, pass it to router + dashboard |
| `guanaco/dashboard/dashboard.py` | Multi-account API endpoints, usage check per account |
| `guanaco/dashboard/templates/dashboard.html` | Settings wizard, Status per-account bars, History account column |