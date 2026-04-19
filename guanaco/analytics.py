"""Request logging and analytics engine.

Tracks every API call with: timestamp, model, prompt/completion tokens,
TPS, TTFT, duration, provider, endpoint, and any errors.
Also tracks Ollama Cloud usage/quota and system status events.
Persists to SQLite for long-term analytics.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional


def _default_db_path() -> Path:
    from guanaco.config import get_default_config_dir
    return get_default_config_dir() / "analytics.db"


def _normalize_model_name(model: str) -> str:
    """Strip routing suffixes (:cloud, :local) for analytics grouping."""
    if model and ":" in model:
        suffix = model.split(":")[-1]
        if suffix in ("cloud", "local"):
            return model.rsplit(":", 1)[0]
    return model


class AnalyticsLogger:
    """SQLite-backed request logger and analytics engine."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS request_log (
                    id TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    type TEXT NOT NULL,          -- 'llm' or 'search'
                    model TEXT,                    -- model name (for LLM calls)
                    provider TEXT,                -- 'ollama', 'fallback', or search provider
                    endpoint TEXT,                -- full endpoint path
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    tps REAL,                     -- tokens per second (output)
                    prompt_tps REAL,              -- prompt tokens per second
                    ttft_seconds REAL,            -- time to first token
                    total_duration_seconds REAL,
                    load_duration_seconds REAL,
                    error TEXT,                   -- error message if failed
                    request_id TEXT,
                    fallback_for TEXT,            -- original model name if this was a fallback call
                    extra TEXT                    -- JSON blob for additional data
                )
            """)
            # Migration: add provider column if upgrading from older schema
            try:
                conn.execute("ALTER TABLE request_log ADD COLUMN provider TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists
            # Migration: add fallback_for column if upgrading from older schema
            try:
                conn.execute("ALTER TABLE request_log ADD COLUMN fallback_for TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists
            # Migration: add caller info and content columns for full history
            for col in ["source_ip TEXT", "source_port INTEGER", "user_agent TEXT", "input_text TEXT", "output_text TEXT"]:
                try:
                    conn.execute(f"ALTER TABLE request_log ADD COLUMN {col}")
                except sqlite3.OperationalError:
                    pass  # column already exists
            # Migration: add fallback_reason column
            try:
                conn.execute("ALTER TABLE request_log ADD COLUMN fallback_reason TEXT")
            except sqlite3.OperationalError:
                pass
            conn.execute("""
                CREATE TABLE IF NOT EXISTS status_events (
                    id TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    level TEXT NOT NULL,          -- 'info', 'warning', 'error'
                    source TEXT NOT NULL,         -- 'ollama', 'router', 'search', 'system'
                    message TEXT NOT NULL,
                    details TEXT                  -- JSON blob for extra info
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_snapshots (
                    id TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    session_pct REAL,            -- session usage percentage
                    weekly_pct REAL,             -- weekly usage percentage
                    plan TEXT,                   -- subscription plan
                    source TEXT                  -- 'api' or 'scrape'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_log_ts ON request_log(ts)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_log_model ON request_log(model)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_log_type ON request_log(type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status_ts ON status_events(ts)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status_level ON status_events(level)
            """)
            # Migration: normalize :cloud/:local model names in existing data
            try:
                conn.execute("UPDATE request_log SET model = REPLACE(REPLACE(model, ':cloud', ''), ':local', '') WHERE model LIKE '%:cloud%' OR model LIKE '%:local%'")
                conn.execute("UPDATE request_log SET fallback_for = REPLACE(REPLACE(fallback_for, ':cloud', ''), ':local', '') WHERE fallback_for IS NOT NULL AND (fallback_for LIKE '%:cloud%' OR fallback_for LIKE '%:local%')")
            except Exception:
                pass

    def log_llm(
        self,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        tps: Optional[float] = None,
        prompt_tps: Optional[float] = None,
        ttft_seconds: Optional[float] = None,
        total_duration_seconds: Optional[float] = None,
        load_duration_seconds: Optional[float] = None,
        error: Optional[str] = None,
        request_id: Optional[str] = None,
        provider: Optional[str] = None,
        fallback_for: Optional[str] = None,
        extra: Optional[dict] = None,
        # Full history fields (optional, requires opt-in)
        source_ip: Optional[str] = None,
        source_port: Optional[int] = None,
        user_agent: Optional[str] = None,
        input_text: Optional[str] = None,
        output_text: Optional[str] = None,
        fallback_reason: Optional[str] = None,
    ) -> str:
        """Log an LLM request. Returns the log entry ID."""
        # Normalize model name so glm-5.1:cloud and glm-5.1 are grouped together
        model = _normalize_model_name(model)
        fallback_for = _normalize_model_name(fallback_for) if fallback_for else fallback_for
        entry_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO request_log
                   (id, ts, type, model, prompt_tokens, completion_tokens, total_tokens,
                    tps, prompt_tps, ttft_seconds, total_duration_seconds,
                    load_duration_seconds, error, request_id, provider, fallback_for,
                    source_ip, source_port, user_agent, input_text, output_text, fallback_reason)
                   VALUES (?, ?, 'llm', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (entry_id, time.time(), model, prompt_tokens, completion_tokens,
                 total_tokens, tps, prompt_tps, ttft_seconds, total_duration_seconds,
                 load_duration_seconds, error, request_id, provider, fallback_for,
                 source_ip, source_port, user_agent, input_text, output_text, fallback_reason),
            )
        
        # Write plaintext log file if configured
        if input_text or output_text:
            try:
                from guanaco.config import get_config
                _cfg = get_config()
                if _cfg.history.log_to_files:
                    self._write_log_file(entry_id, model, provider, source_ip, input_text, output_text, error, _cfg.history)
            except Exception:
                pass  # Don't break the request if log file writing fails
        
        return entry_id

    def _write_log_file(self, entry_id: str, model: str, provider: Optional[str],
                         source_ip: Optional[str], input_text: Optional[str],
                         output_text: Optional[str], error: Optional[str],
                         history_config=None):
        """Write a plaintext log file for this request."""
        try:
            log_dir = history_config.get_log_dir() if history_config else None
            if not log_dir:
                return
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            # One file per request: <timestamp>_<model>_<short_id>.log
            safe_model = model.replace("/", "_").replace(":", "_").replace(" ", "_")
            filename = f"{ts}_{safe_model}_{entry_id[:8]}.log"
            filepath = log_dir / filename
            
            lines = []
            lines.append(f"=== Guanaco Request Log ===")
            lines.append(f"ID:       {entry_id}")
            lines.append(f"Time:     {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            lines.append(f"Model:    {model}")
            lines.append(f"Provider: {provider or 'ollama'}")
            lines.append(f"Caller:   {source_ip or 'unknown'}")
            if error:
                lines.append(f"Error:    {error}")
            lines.append(f"")
            if input_text:
                lines.append(f"--- INPUT ---")
                lines.append(input_text)
                lines.append(f"")
            if output_text:
                lines.append(f"--- OUTPUT ---")
                lines.append(output_text)
                lines.append(f"")
            lines.append(f"=== END ===")
            
            filepath.write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            pass  # Don't break the request if log file writing fails

    def log_search(
        self,
        provider: str,
        endpoint: str,
        duration_seconds: Optional[float] = None,
        result_count: int = 0,
        error: Optional[str] = None,
    ) -> str:
        """Log a search/scrape request. Returns the log entry ID."""
        entry_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO request_log
                   (id, ts, type, provider, endpoint, total_duration_seconds, error, extra)
                   VALUES (?, ?, 'search', ?, ?, ?, ?, ?)""",
                (entry_id, time.time(), provider, endpoint,
                 duration_seconds, error, json.dumps({"result_count": result_count})),
            )
        return entry_id

    def log_status(
        self,
        level: str,
        source: str,
        message: str,
        details: Optional[dict] = None,
    ) -> str:
        """Log a status event (info, warning, error)."""
        entry_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO status_events (id, ts, level, source, message, details)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (entry_id, time.time(), level, source, message,
                 json.dumps(details) if details else None),
            )
        return entry_id

    def log_usage_snapshot(
        self,
        session_pct: Optional[float] = None,
        weekly_pct: Optional[float] = None,
        plan: Optional[str] = None,
        source: str = "api",
    ) -> str:
        """Log a usage/quota snapshot."""
        entry_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO usage_snapshots (id, ts, session_pct, weekly_pct, plan, source)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (entry_id, time.time(), session_pct, weekly_pct, plan, source),
            )
        return entry_id

    def get_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        type_filter: Optional[str] = None,
        model_filter: Optional[str] = None,
    ) -> list[dict]:
        """Get recent log entries."""
        query = "SELECT * FROM request_log WHERE 1=1"
        params = []
        if type_filter:
            query += " AND type = ?"
            params.append(type_filter)
        if model_filter:
            query += " AND model = ?"
            params.append(model_filter)
        query += " ORDER BY ts DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_status_events(
        self,
        limit: int = 100,
        level: Optional[str] = None,
        source: Optional[str] = None,
    ) -> list[dict]:
        """Get recent status events."""
        query = "SELECT * FROM status_events WHERE 1=1"
        params = []
        if level:
            query += " AND level = ?"
            params.append(level)
        if source:
            query += " AND source = ?"
            params.append(source)
        query += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_summary(self) -> dict:
        """Get aggregate analytics summary."""
        with sqlite3.connect(self.db_path) as conn:
            # Total counts
            total = conn.execute("SELECT COUNT(*) FROM request_log").fetchone()[0]
            llm_calls = conn.execute("SELECT COUNT(*) FROM request_log WHERE type='llm'").fetchone()[0]
            search_calls = conn.execute("SELECT COUNT(*) FROM request_log WHERE type='search'").fetchone()[0]
            errors = conn.execute("SELECT COUNT(*) FROM request_log WHERE error IS NOT NULL").fetchone()[0]

            # Token totals
            row = conn.execute(
                "SELECT COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0), "
                "COALESCE(SUM(total_tokens),0) FROM request_log WHERE type='llm'"
            ).fetchone()
            prompt_tokens, completion_tokens, total_tokens = row

            # Average TPS — based on most recent rows covering ~10k completion tokens
            tps_rows = conn.execute(
                "SELECT tps, COALESCE(completion_tokens,0) FROM request_log "
                "WHERE type='llm' AND tps IS NOT NULL ORDER BY ts DESC"
            ).fetchall()
            recent_tps = []
            token_budget = 10000
            for tps_val, ct in tps_rows:
                if token_budget <= 0:
                    break
                recent_tps.append(tps_val)
                token_budget -= ct
            avg_tps = round(sum(recent_tps) / len(recent_tps), 2) if recent_tps else 0

            # Average TTFT — same 10k token window
            ttft_rows = conn.execute(
                "SELECT ttft_seconds, COALESCE(completion_tokens,0) FROM request_log "
                "WHERE type='llm' AND ttft_seconds IS NOT NULL ORDER BY ts DESC"
            ).fetchall()
            recent_ttft = []
            token_budget = 10000
            for ttft_val, ct in ttft_rows:
                if token_budget <= 0:
                    break
                recent_ttft.append(ttft_val)
                token_budget -= ct
            avg_ttft = round(sum(recent_ttft) / len(recent_ttft), 3) if recent_ttft else 0

            # Per-model stats — TPS/TTFT from most recent 10k completion tokens per model
            model_rows = conn.execute(
                """SELECT model, COUNT(*), SUM(prompt_tokens), SUM(completion_tokens),
                   MAX(ts)
                   FROM request_log WHERE type='llm' GROUP BY model ORDER BY MAX(ts) DESC"""
            ).fetchall()
            models = []
            for row in model_rows:
                model_name = row[0]
                # Get recent TPS for this model
                m_tps_rows = conn.execute(
                    "SELECT tps, COALESCE(completion_tokens,0) FROM request_log "
                    "WHERE type='llm' AND model=? AND tps IS NOT NULL ORDER BY ts DESC",
                    (model_name,)
                ).fetchall()
                recent_m_tps = []
                budget = 10000
                for tps_val, ct in m_tps_rows:
                    if budget <= 0:
                        break
                    recent_m_tps.append(tps_val)
                    budget -= ct
                m_avg_tps = round(sum(recent_m_tps) / len(recent_m_tps), 2) if recent_m_tps else 0

                # Get recent TTFT for this model
                m_ttft_rows = conn.execute(
                    "SELECT ttft_seconds, COALESCE(completion_tokens,0) FROM request_log "
                    "WHERE type='llm' AND model=? AND ttft_seconds IS NOT NULL ORDER BY ts DESC",
                    (model_name,)
                ).fetchall()
                recent_m_ttft = []
                budget = 10000
                for ttft_val, ct in m_ttft_rows:
                    if budget <= 0:
                        break
                    recent_m_ttft.append(ttft_val)
                    budget -= ct
                m_avg_ttft = round(sum(recent_m_ttft) / len(recent_m_ttft), 3) if recent_m_ttft else 0

                models.append({
                    "model": model_name, "requests": row[1],
                    "prompt_tokens": row[2] or 0, "completion_tokens": row[3] or 0,
                    "avg_tps": m_avg_tps,
                    "avg_ttft": m_avg_ttft,
                    "last_used": row[4],
                })

            # Per-provider stats (for search calls)
            provider_rows = conn.execute(
                """SELECT provider, COUNT(*), MAX(ts) FROM request_log
                   WHERE type='search' GROUP BY provider ORDER BY MAX(ts) DESC"""
            ).fetchall()
            providers = []
            for row in provider_rows:
                providers.append({
                    "provider": row[0], "requests": row[1], "last_used": row[2],
                })

            # Per-provider LLM stats — TPS/TTFT from most recent 10k tokens per provider
            llm_provider_rows = conn.execute(
                """SELECT provider, COUNT(*), SUM(prompt_tokens), SUM(completion_tokens),
                   MAX(ts)
                   FROM request_log WHERE type='llm' GROUP BY provider ORDER BY MAX(ts) DESC"""
            ).fetchall()
            llm_providers = []
            for row in llm_provider_rows:
                prov_name = row[0]
                # Get recent TPS for this provider
                p_tps_rows = conn.execute(
                    "SELECT tps, COALESCE(completion_tokens,0) FROM request_log "
                    "WHERE type='llm' AND provider=? AND tps IS NOT NULL ORDER BY ts DESC",
                    (prov_name,)
                ).fetchall()
                recent_p_tps = []
                budget = 10000
                for tps_val, ct in p_tps_rows:
                    if budget <= 0:
                        break
                    recent_p_tps.append(tps_val)
                    budget -= ct
                p_avg_tps = round(sum(recent_p_tps) / len(recent_p_tps), 2) if recent_p_tps else 0

                # Get recent TTFT for this provider
                p_ttft_rows = conn.execute(
                    "SELECT ttft_seconds, COALESCE(completion_tokens,0) FROM request_log "
                    "WHERE type='llm' AND provider=? AND ttft_seconds IS NOT NULL ORDER BY ts DESC",
                    (prov_name,)
                ).fetchall()
                recent_p_ttft = []
                budget = 10000
                for ttft_val, ct in p_ttft_rows:
                    if budget <= 0:
                        break
                    recent_p_ttft.append(ttft_val)
                    budget -= ct
                p_avg_ttft = round(sum(recent_p_ttft) / len(recent_p_ttft), 3) if recent_p_ttft else 0

                llm_providers.append({
                    "provider": prov_name, "requests": row[1],
                    "prompt_tokens": row[2] or 0, "completion_tokens": row[3] or 0,
                    "avg_tps": p_avg_tps,
                    "avg_ttft": p_avg_ttft,
                    "last_used": row[4],
                })

            # Fallback stats
            fallback_count = conn.execute(
                "SELECT COUNT(*) FROM request_log WHERE fallback_for IS NOT NULL"
            ).fetchone()[0]
            fallback_rows = conn.execute(
                """SELECT fallback_for, COUNT(*), MAX(ts) FROM request_log
                   WHERE fallback_for IS NOT NULL GROUP BY fallback_for ORDER BY MAX(ts) DESC"""
            ).fetchall()
            fallbacks = []
            for row in fallback_rows:
                fallbacks.append({
                    "original_model": row[0], "fallback_count": row[1], "last_used": row[2],
                })
            
            # Fallback rate (24h window) - percentage of requests routed to fallback
            cutoff_24h = time.time() - (24 * 3600)
            fb_24h = conn.execute(
                "SELECT COUNT(*) FROM request_log WHERE type='llm' AND fallback_for IS NOT NULL AND ts > ?",
                (cutoff_24h,)
            ).fetchone()[0]
            main_24h = conn.execute(
                "SELECT COUNT(*) FROM request_log WHERE type='llm' AND (provider='ollama' OR provider IS NULL) AND fallback_for IS NULL AND ts > ?",
                (cutoff_24h,)
            ).fetchone()[0]
            total_24h = fb_24h + main_24h
            fallback_rate = round((fb_24h / total_24h) * 100, 1) if total_24h > 0 else 0.0

            # Recent errors
            error_rows = conn.execute(
                """SELECT ts, type, model, provider, endpoint, error
                   FROM request_log WHERE error IS NOT NULL ORDER BY ts DESC LIMIT 20"""
            ).fetchall()
            recent_errors = []
            for row in error_rows:
                recent_errors.append({
                    "ts": row[0], "type": row[1], "model": row[2],
                    "provider": row[3], "endpoint": row[4], "error": row[5],
                })

            # Status event counts
            status_error_count = conn.execute(
                "SELECT COUNT(*) FROM status_events WHERE level='error'"
            ).fetchone()[0]
            status_warning_count = conn.execute(
                "SELECT COUNT(*) FROM status_events WHERE level='warning'"
            ).fetchone()[0]

            # Latest usage snapshot
            usage_row = conn.execute(
                "SELECT session_pct, weekly_pct, plan, ts FROM usage_snapshots ORDER BY ts DESC LIMIT 1"
            ).fetchone()

            return {
                "total_requests": total,
                "llm_calls": llm_calls,
                "search_calls": search_calls,
                "errors": errors,
                "prompt_tokens": prompt_tokens or 0,
                "completion_tokens": completion_tokens or 0,
                "total_tokens": total_tokens or 0,
                "avg_tps": avg_tps,
                "avg_ttft": avg_ttft,
                "models": models,
                "llm_providers": llm_providers,
                "providers": providers,
                "fallbacks": fallbacks,
                "fallback_count": fallback_count,
                "recent_errors": recent_errors,
                "status_errors": status_error_count,
                "status_warnings": status_warning_count,
                "fallback_rate": fallback_rate,
                "usage": {
                    "session_pct": usage_row[0] if usage_row else None,
                    "weekly_pct": usage_row[1] if usage_row else None,
                    "plan": usage_row[2] if usage_row else None,
                    "last_checked": usage_row[3] if usage_row else None,
                } if usage_row else None,
            }

    def get_timeseries(self, hours: int = 24, bucket_minutes: int = 60) -> list[dict]:
        """Get request count timeseries data."""
        cutoff = time.time() - (hours * 3600)
        bucket_sec = bucket_minutes * 60

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT ts, type, model, total_tokens FROM request_log WHERE ts > ? ORDER BY ts",
                (cutoff,),
            ).fetchall()

        buckets = {}
        for ts, rtype, model, tokens in rows:
            bucket = int(ts // bucket_sec) * bucket_sec
            key = bucket
            if key not in buckets:
                buckets[key] = {"ts": bucket, "llm": 0, "search": 0, "tokens": 0}
            if rtype == "llm":
                buckets[key]["llm"] += 1
                buckets[key]["tokens"] += (tokens or 0)
            else:
                buckets[key]["search"] += 1

        return sorted(buckets.values(), key=lambda x: x["ts"])

    def get_model_history(self, model: str, limit: int = 50) -> list[dict]:
        """Get detailed history for a specific model."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM request_log
                   WHERE model = ? AND type = 'llm'
                   ORDER BY ts DESC LIMIT ?""",
                (model, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_fallback_rate(self, hours: int = 24) -> dict:
        """Calculate fallback routing rate for the specified time window.
        
        Returns the percentage of requests that were routed to fallback provider
        due to main provider failures (timeout, error, quota full).
        """
        cutoff = time.time() - (hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            fallback_count = conn.execute(
                "SELECT COUNT(*) FROM request_log WHERE type='llm' AND fallback_for IS NOT NULL AND ts > ?",
                (cutoff,)
            ).fetchone()[0]
            main_count = conn.execute(
                "SELECT COUNT(*) FROM request_log WHERE type='llm' AND (provider='ollama' OR provider IS NULL) AND fallback_for IS NULL AND ts > ?",
                (cutoff,)
            ).fetchone()[0]
            total = fallback_count + main_count
            rate = round((fallback_count / total) * 100, 1) if total > 0 else 0.0
            return {
                "rate": rate,
                "fallback_count": fallback_count,
                "main_count": main_count,
                "total": total,
                "hours": hours,
            }


    def get_history(
        self,
        limit: int = 100,
        offset: int = 0,
        model_filter: Optional[str] = None,
        provider_filter: Optional[str] = None,
        has_content: Optional[bool] = None,
        errors_only: bool = False,
        include_content: bool = False,
    ) -> list[dict]:
        """Get paginated request history with optional filters.
        
        Args:
            limit: Max results to return
            offset: Skip this many results (pagination)
            model_filter: Filter by model name
            provider_filter: Filter by provider
            has_content: Filter to only requests with/without saved content
            errors_only: Filter to only failed requests (error IS NOT NULL)
            include_content: Include input_text/output_text in results
        """
        query = "SELECT * FROM request_log WHERE type='llm'"
        params = []
        
        if model_filter:
            query += " AND model = ?"
            params.append(model_filter)
        if provider_filter:
            query += " AND provider = ?"
            params.append(provider_filter)
        if errors_only:
            query += " AND error IS NOT NULL AND error != ''"
        elif has_content is True:
            query += " AND input_text IS NOT NULL"
        elif has_content is False:
            query += " AND input_text IS NULL"
        
        query += " ORDER BY ts DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                # Add has_content flag for badge rendering without needing full text
                has_input = bool(d.get("input_text"))
                has_output = bool(d.get("output_text"))
                d["has_content"] = has_input or has_output
                # Don't include content unless requested (can be large)
                if not include_content:
                    d.pop("input_text", None)
                    d.pop("output_text", None)
                # Format timestamp
                d["ts_formatted"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(d["ts"]))
                results.append(d)
            return results

    def get_request_detail(self, request_id: str) -> Optional[dict]:
        """Get full details of a single request including content."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM request_log WHERE id = ?",
                (request_id,)
            ).fetchone()
            if row:
                d = dict(row)
                d["ts_formatted"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(d["ts"]))
                return d
            return None

    def get_history_stats(self) -> dict:
        """Get stats about history logging."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM request_log WHERE type='llm'").fetchone()[0]
            with_content = conn.execute(
                "SELECT COUNT(*) FROM request_log WHERE type='llm' AND input_text IS NOT NULL"
            ).fetchone()[0]
            oldest = conn.execute(
                "SELECT MIN(ts) FROM request_log WHERE type='llm'"
            ).fetchone()[0]
            newest = conn.execute(
                "SELECT MAX(ts) FROM request_log WHERE type='llm'"
            ).fetchone()[0]
            
            # Storage size estimate
            content_size = conn.execute(
                "SELECT COALESCE(SUM(LENGTH(input_text) + LENGTH(output_text)), 0) FROM request_log WHERE input_text IS NOT NULL OR output_text IS NOT NULL"
            ).fetchone()[0]
            
            # Error count
            error_count = conn.execute(
                "SELECT COUNT(*) FROM request_log WHERE type='llm' AND error IS NOT NULL AND error != ''"
            ).fetchone()[0]
            
            return {
                "total_requests": total,
                "requests_with_content": with_content,
                "error_count": error_count,
                "oldest_ts": oldest,
                "newest_ts": newest,
                "content_size_bytes": content_size,
                "content_size_mb": round(content_size / (1024 * 1024), 2),
            }

    def clear(self):
        """Clear all analytics data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM request_log")
            conn.execute("DELETE FROM status_events")
            conn.execute("DELETE FROM usage_snapshots")

    def cleanup_old_log_files(self, history_config=None):
        """Delete log files older than retention_days from the history_logs directory."""
        if not history_config or not history_config.log_to_files:
            return 0
        retention_days = history_config.retention_days
        if retention_days <= 0:
            return 0  # 0 means keep forever
        try:
            log_dir = history_config.get_log_dir()
            if not log_dir.exists():
                return 0
            cutoff = time.time() - (retention_days * 86400)
            deleted = 0
            for f in log_dir.glob("*.log"):
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    deleted += 1
            return deleted
        except Exception:
            return 0