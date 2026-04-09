"""CLI entry point for Guanaco."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="guanaco",
        description="🦙 Guanaco — maximize your Ollama Cloud subscription",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── start ──
    start_parser = subparsers.add_parser("start", help="Start all services")
    start_parser.add_argument("--host", default=None, help="Bind host (default: 127.0.0.1)")
    start_parser.add_argument("--port", type=int, default=None, help="Port (default: 8080)")
    start_parser.add_argument("--tailscale", action="store_true", help="Use Tailscale IP for endpoint URLs")

    # ── setup ──
    subparsers.add_parser("setup", help="Interactive setup wizard")

    # ── key ──
    key_parser = subparsers.add_parser("key", help="Manage API keys")
    key_parser.add_argument("action", choices=["generate", "list", "revoke"], help="Action")
    key_parser.add_argument("--provider", default="general", help="Provider for key")
    key_parser.add_argument("--name", default="", help="Key name")

    # ── models ──
    models_parser = subparsers.add_parser("models", help="List available Ollama Cloud models")
    models_parser.add_argument("--refresh", action="store_true", help="Force refresh from Ollama API")
    models_parser.add_argument("--json", action="store_true", help="Output as JSON")
    models_parser.add_argument("--capabilities", action="store_true", help="Show model capabilities")

    # ── usage ──
    subparsers.add_parser("usage", help="Check Ollama Cloud usage/quota")

    # ── status ──
    status_parser = subparsers.add_parser("status", help="Show service status and Ollama connectivity")
    status_parser.add_argument("--json", action="store_true", help="Output as JSON")
    status_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed info")

    # ── analytics ──
    analytics_parser = subparsers.add_parser("analytics", help="View request analytics")
    analytics_parser.add_argument("--model", default=None, help="Filter by model")
    analytics_parser.add_argument("--limit", type=int, default=20, help="Number of entries")
    analytics_parser.add_argument("--summary", action="store_true", help="Show summary only")
    analytics_parser.add_argument("--errors", action="store_true", help="Show recent errors")

    # ── config ──
    config_parser = subparsers.add_parser("config", help="View or modify configuration")
    config_parser.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"), help="Set a config value")
    config_parser.add_argument("--show", action="store_true", help="Show current config")

    # ── version ──
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "version":
        from guanaco import __version__
        print(f"🦙 guanaco v{__version__}")
        return

    if args.command == "setup":
        _run_setup()
        return

    if args.command == "start":
        _run_start(args)
        return

    if args.command == "key":
        _run_key(args)
        return

    if args.command == "models":
        _run_models(args)
        return

    if args.command == "usage":
        _run_usage()
        return

    if args.command == "status":
        _run_status(args)
        return

    if args.command == "analytics":
        _run_analytics(args)
        return

    if args.command == "config":
        _run_config(args)
        return


def _run_setup():
    """Interactive setup wizard."""
    from guanaco.config import AppConfig, save_config, get_default_config_path

    print("🦙 Guanaco — Setup Wizard\n")

    api_key = os.environ.get("OLLAMA_API_KEY", "")
    if not api_key:
        api_key = input("Enter your Ollama API key: ").strip()
    else:
        print(f"Found OLLAMA_API_KEY in environment")
        use_env = input("Use environment variable? [Y/n]: ").strip().lower()
        if use_env == "n":
            api_key = input("Enter your Ollama API key: ").strip()

    # Auto-detect Tailscale for smarter default
    ts_ip = ""
    try:
        import subprocess
        r = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=3)
        if r.returncode == 0:
            ts_ip = r.stdout.strip()
    except Exception:
        pass

    default_host = "0.0.0.0" if ts_ip else "127.0.0.1"
    host = input(f"Bind host [{default_host}]: ").strip() or default_host
    port = int(input("Port [8080]: ").strip() or "8080")
    use_tailscale = False
    if ts_ip and host != "127.0.0.1":
        print(f"   🌐 Tailscale detected at {ts_ip} — dashboard will be at http://{ts_ip}:{port}/dashboard/")
        use_tailscale = True
    elif host == "0.0.0.0":
        use_tailscale = input("Use Tailscale IP for base URL? [y/N]: ").strip().lower() == "y"

    # LLM config
    print("\n📡 LLM Configuration")
    print("   Available Ollama Cloud models: qwen3:480b, gpt-oss:120b, deepseek-v3.1, oss120b")
    print("   Also: qwen3.5:122b, glm-5.1, minimax-m2.7, llama4:109b, etc.")
    reranker = input("Reranker model [oss120b]: ").strip() or "oss120b"
    scraper = input("Scraper model [qwen3:480b]: ").strip() or "qwen3:480b"
    summary = input("Summary model [qwen3:480b]: ").strip() or "qwen3:480b"
    default_model = input("Default chat model [qwen3:480b]: ").strip() or "qwen3:480b"
    emulate_anthropic = input("Enable Anthropic /v1/messages emulation? [Y/n]: ").strip().lower() != "n"
    emulate_openai = input("Enable OpenAI /v1/chat/completions? [Y/n]: ").strip().lower() != "n"

    config = AppConfig(
        ollama_api_key=api_key,
        router={"host": host, "port": port, "use_tailscale": use_tailscale},
        llm={
            "reranker_model": reranker,
            "scraper_model": scraper,
            "summary_model": summary,
            "default_model": default_model,
            "emulate_anthropic": emulate_anthropic,
            "emulate_openai": emulate_openai,
        },
    )

    config_path = get_default_config_path()
    save_config(config, config_path)
    print(f"\n✅ Config saved to {config_path}")
    print(f"\nEndpoints:")
    print(f"   LLM Router:     http://{host}:{port}/v1/chat/completions")
    if emulate_anthropic:
        print(f"   Anthropic:       http://{host}:{port}/v1/messages")
    print(f"   Search APIs:     http://{host}:{port}/<provider>/...")
    print(f"   Dashboard:       http://{host}:{port}/dashboard")
    print(f"\nRun 'guanaco start' to begin!")


def _run_start(args):
    """Start all services using uvicorn."""
    from guanaco.config import load_config, save_config

    config = load_config()

    if args.host:
        config.router.host = args.host
    if args.port:
        config.router.port = args.port
    if args.tailscale:
        config.router.use_tailscale = True

    save_config(config)

    port = config.router.port
    print("🦙 Starting Guanaco...")
    print(f"   Host: {config.router.host}")
    print(f"   Port: {port}")
    print(f"   Tailscale: {'Yes' if config.router.use_tailscale else 'No'}")
    print(f"   Anthropic: {'Yes' if config.llm.emulate_anthropic else 'No'}")
    print(f"   OpenAI: {'Yes' if config.llm.emulate_openai else 'No'}")
    print(f"   Default model: {config.llm.default_model}")
    print(f"   Reranker: {config.llm.reranker_model}")
    print()

    try:
        import uvicorn
        from guanaco.app import create_app
        app = create_app(config)
        uvicorn.run(app, host=config.router.host, port=port, log_level="info")
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install -e .")
        sys.exit(1)


def _run_key(args):
    """Manage API keys."""
    from guanaco.config import get_default_config_dir
    from guanaco.utils.api_keys import ApiKeyManager

    km = ApiKeyManager(get_default_config_dir())

    if args.action == "generate":
        key = km.generate_key(provider=args.provider, name=args.name)
        print(f"🔑 Generated key for {args.provider}:")
        print(f"   {key}")
        print(f"\n⚠️  Save this key now — it won't be shown again!")
    elif args.action == "list":
        keys = km.list_keys()
        if not keys:
            print("No API keys found.")
        else:
            print(f"{'Provider':<12} {'Name':<20} {'Prefix':<20} {'Created'}")
            print("-" * 72)
            for k in keys:
                from datetime import datetime
                created = datetime.fromtimestamp(k['created_at']).strftime('%Y-%m-%d %H:%M')
                print(f"{k['provider']:<12} {k['name']:<20} {k['prefix']:<20} {created}")
    elif args.action == "revoke":
        prefix = input("Enter key prefix to revoke: ").strip()
        if km.revoke_by_prefix(prefix):
            print("✅ Key revoked.")
        else:
            print("❌ Key not found.")


def _run_models(args):
    """List available Ollama Cloud models."""
    from guanaco.config import load_config
    from guanaco.client import OllamaClient, KNOWN_CLOUD_MODELS

    config = load_config()
    api_key = config.ollama_api_key_resolved

    if not api_key:
        print("❌ OLLAMA_API_KEY not set. Run 'guanaco setup' first.")
        return

    client = OllamaClient(api_key=api_key)

    async def fetch():
        try:
            if args.refresh:
                models = await client.list_models(force_refresh=True)
            else:
                models = await client.get_cloud_models()
            await client.close()
            return models
        except Exception as e:
            await client.close()
            print(f"❌ Error fetching models: {e}")
            return []

    models = asyncio.run(fetch())
    if not models:
        print("No models found.")
        return

    if args.json:
        import json
        print(json.dumps(models, indent=2))
        return

    print(f"🦙 Available Ollama Cloud Models ({len(models)}):\n")

    if args.capabilities:
        print(f"{'Model':<28} {'Size':>8} {'Family':<14} {'Capabilities'}")
        print("─" * 80)
        for m in models:
            name = m.get("display_name", m.get("name", ""))
            size = m.get("parameter_size", "")
            family = m.get("family", "")
            caps = m.get("capabilities", ["cloud"])
            caps_str = " ".join(f"[{c}]" for c in caps)
            print(f"{name:<28} {size:>8} {family:<14} {caps_str}")
    else:
        print(f"{'Model':<28} {'Size':>8} {'Family':<14} {'Quant':<10} {'Modified'}")
        print("─" * 80)
        for m in models:
            name = m.get("display_name", m.get("name", ""))
            size = m.get("parameter_size", "")
            family = m.get("family", "")
            quant = m.get("quantization", "")
            modified = m.get("modified_at", "")[:10] if m.get("modified_at") else ""
            print(f"{name:<28} {size:>8} {family:<14} {quant:<10} {modified}")

    # Show current config
    print(f"\n📡 Current model config:")
    print(f"   Default:     {config.llm.default_model}")
    print(f"   Reranker:     {config.llm.reranker_model}")
    print(f"   Scraper:      {config.llm.scraper_model}")
    print(f"   Summary:      {config.llm.summary_model}")
    print(f"   Fallback:     {config.llm.fallback_model}")


def _run_usage():
    """Check Ollama Cloud usage/quota."""
    from guanaco.config import load_config
    from guanaco.client import OllamaClient

    config = load_config()
    api_key = config.ollama_api_key_resolved
    session_cookie = config.usage.session_cookie

    if not session_cookie:
        print("⚠️  No session cookie configured.")
        print("   Paste your __Secure-session cookie from ollama.com in the dashboard Status tab,")
        print("   or set it in ~/.guanaco/config.yaml under usage.session_cookie")
        return

    client = OllamaClient(api_key=api_key, session_cookie=session_cookie)

    async def check():
        try:
            usage = await client.get_usage(session_cookie=session_cookie)
            await client.close()
            return usage
        except Exception as e:
            await client.close()
            print(f"❌ Error checking usage: {e}")
            return None

    usage = asyncio.run(check())
    if not usage:
        return

    source = usage.get("source", "unknown")
    if source in ("unavailable", "error"):
        print(f"❌ {usage.get('error', 'Could not retrieve usage information.')}")
        return

    plan = usage.get("plan", "—")
    print(f"🦙 Ollama Cloud Usage ({plan})\n")

    if usage.get("session_pct") is not None:
        reset = usage.get("session_reset", "")
        reset_str = f" (resets in {reset})" if reset else ""
        print(f"   Session:  {usage['session_pct']}%{reset_str}")
    if usage.get("weekly_pct") is not None:
        reset = usage.get("weekly_reset", "")
        reset_str = f" (resets in {reset})" if reset else ""
        print(f"   Weekly:   {usage['weekly_pct']}%{reset_str}")


def _run_status(args):
    """Show service status and Ollama connectivity."""
    import json as json_mod
    from guanaco.config import load_config, get_base_url
    from guanaco.client import OllamaClient
    from guanaco.analytics import AnalyticsLogger

    config = load_config()
    base_url = get_base_url(config)
    port = config.router.port

    results = {}

    # Check if service is running
    import httpx
    try:
        resp = httpx.get(f"http://{config.router.host}:{port}/health", timeout=2)
        if resp.status_code == 200:
            results["service"] = "running"
            results["version"] = resp.json().get("version", "unknown")
        else:
            results["service"] = "error"
    except Exception:
        results["service"] = "not_running"

    # Check Ollama Cloud connectivity
    api_key = config.ollama_api_key_resolved
    if api_key:
        client = OllamaClient(api_key=api_key)

        async def check_ollama():
            health = await client.health_check()
            await client.close()
            return health

        ollama_health = asyncio.run(check_ollama())
        results["ollama"] = ollama_health
    else:
        results["ollama"] = {"status": "no_api_key"}

    # Local analytics
    analytics = AnalyticsLogger()
    summary = analytics.get_summary()
    results["analytics"] = {
        "total_requests": summary["total_requests"],
        "errors": summary["errors"],
        "status_errors": summary["status_errors"],
        "status_warnings": summary["status_warnings"],
    }

    if args.json:
        print(json_mod.dumps(results, indent=2))
        return

    # Human-readable output
    service = results["service"]
    if service == "running":
        print("🟢 Guanaco is running")
        print(f"   Version: {results.get('version', 'unknown')}")
        print(f"   Dashboard: {base_url}:{port}/dashboard")
    elif service == "error":
        print("🔴 Guanaco returned error")
    else:
        print("⚪ Guanaco is not running")
        print("   Run 'guanaco start' to begin")

    print()

    # Ollama Cloud status
    ollama = results.get("ollama", {})
    ollama_status = ollama.get("status", "unknown")
    if ollama_status == "connected":
        print(f"🟢 Ollama Cloud: Connected ({ollama.get('model_count', '?')} models, {ollama.get('latency_ms', '?')}ms)")
    elif ollama_status == "auth_error":
        print("🔴 Ollama Cloud: Invalid/expired API key")
    elif ollama_status == "rate_limited":
        print("🟡 Ollama Cloud: Rate limited")
    elif ollama_status == "no_api_key":
        print("⚪ Ollama Cloud: No API key configured")
    else:
        print(f"🔴 Ollama Cloud: {ollama.get('message', ollama_status)}")

    # Analytics summary
    an = results.get("analytics", {})
    print(f"\n📊 Analytics:")
    print(f"   Total requests: {an.get('total_requests', 0)}")
    print(f"   Errors: {an.get('errors', 0)}")
    print(f"   Status events: {an.get('status_errors', 0)} errors, {an.get('status_warnings', 0)} warnings")

    if args.verbose:
        print(f"\n📡 Endpoints:")
        print(f"   OpenAI:   {base_url}:{port}/v1/chat/completions")
        if config.llm.emulate_anthropic:
            print(f"   Anthropic: {base_url}:{port}/v1/messages")
        print(f"   Models:    {base_url}:{port}/v1/models")
        print(f"   Usage:     {base_url}:{port}/v1/usage")
        print(f"   Health:    {base_url}:{port}/health")
        print(f"\n📡 Model Config:")
        print(f"   Default:     {config.llm.default_model}")
        print(f"   Reranker:     {config.llm.reranker_model}")
        print(f"   Scraper:      {config.llm.scraper_model}")
        print(f"   Summary:      {config.llm.summary_model}")
        print(f"   Fallback:     {config.llm.fallback_model}")
        print(f"   Anthropic:    {'enabled' if config.llm.emulate_anthropic else 'disabled'}")
        print(f"   OpenAI:       {'enabled' if config.llm.emulate_openai else 'disabled'}")


def _run_analytics(args):
    """View request analytics."""
    from guanaco.analytics import AnalyticsLogger

    analytics = AnalyticsLogger()

    if args.errors:
        events = analytics.get_status_events(limit=args.limit, level="error")
        if not events:
            print("✅ No errors found!")
            return
        print(f"⚠️  Recent Errors ({len(events)}):\n")
        from datetime import datetime
        for e in events:
            ts = datetime.fromtimestamp(e["ts"]).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  [{ts}] [{e['source']}] {e['message']}")
            if e.get("details"):
                print(f"    Details: {e['details']}")
        return

    if args.model:
        entries = analytics.get_model_history(args.model, limit=args.limit)
        if not entries:
            print(f"No entries for model '{args.model}'")
            return
        print(f"📊 History for {args.model} ({len(entries)} entries):\n")
        from datetime import datetime
        for e in entries[:args.limit]:
            ts = datetime.fromtimestamp(e["ts"]).strftime("%H:%M:%S")
            tokens = e.get("total_tokens", 0)
            tps = e.get("tps") or "—"
            ttft = f"{(e.get('ttft_seconds') or 0) * 1000:.0f}ms" if e.get("ttft_seconds") else "—"
            err = f" ERR: {e['error'][:40]}" if e.get("error") else ""
            print(f"  [{ts}] tok={tokens} tps={tps} ttft={ttft}{err}")
        return

    summary = analytics.get_summary()
    if args.summary or True:
        print("📊 Analytics Summary\n")
        print(f"  Total requests:  {summary['total_requests']}")
        print(f"  LLM calls:       {summary['llm_calls']}")
        print(f"  Search calls:    {summary['search_calls']}")
        print(f"  Errors:           {summary['errors']}")
        print(f"  Prompt tokens:    {summary['prompt_tokens']:,}")
        print(f"  Completion tokens:{summary['completion_tokens']:,}")
        print(f"  Total tokens:     {summary['total_tokens']:,}")
        print(f"  Avg TPS:          {summary['avg_tps']}")
        print(f"  Avg TTFT:         {summary['avg_ttft']*1000:.0f}ms" if summary['avg_ttft'] else "  Avg TTFT:         —")

        if summary.get("models"):
            print(f"\n📡 Per-Model Stats:")
            print(f"  {'Model':<28} {'Reqs':>6} {'PTok':>10} {'CTok':>10} {'TPS':>8} {'TTFT':>8}")
            print(f"  {'─'*28} {'─'*6} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")
            for m in summary["models"][:10]:
                ttft = f"{m['avg_ttft']*1000:.0f}ms" if m.get("avg_ttft") else "—"
                print(f"  {m['model']:<28} {m['requests']:>6} {m['prompt_tokens']:>10,} {m['completion_tokens']:>10,} {m.get('avg_tps', '—'):>8} {ttft:>8}")

        if summary.get("usage"):
            u = summary["usage"]
            print(f"\n📈 Ollama Cloud Usage:")
            if u.get("plan"):
                print(f"  Plan: {u['plan']}")
            if u.get("session_pct") is not None:
                print(f"  Session: {u['session_pct']}%")
            if u.get("weekly_pct") is not None:
                print(f"  Weekly: {u['weekly_pct']}%")


def _run_config(args):
    """View or modify configuration."""
    from guanaco.config import load_config, save_config

    config = load_config()

    if args.set:
        key, value = args.set
        # Navigate dot-notation config key
        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part, None)
            if obj is None:
                print(f"❌ Unknown config key: {key}")
                return
        last_key = parts[-1]
        if not hasattr(obj, last_key):
            print(f"❌ Unknown config key: {key}")
            return

        # Type coercion
        current = getattr(obj, last_key)
        if isinstance(current, bool):
            value = value.lower() in ("true", "1", "yes", "on")
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)

        setattr(obj, last_key, value)
        save_config(config)
        print(f"✅ Set {key} = {value}")
        return

    # Show current config
    import json
    print("🦙 Current Configuration\n")
    print(f"  API Key: {'*' * 8}{config.ollama_api_key_resolved[-4:]}" if config.ollama_api_key_resolved else "  API Key: (not set)")
    print(f"\n  Router:")
    print(f"    Host: {config.router.host}")
    print(f"    Port: {config.router.port}")
    print(f"    Tailscale: {config.router.use_tailscale}")
    print(f"\n  LLM:")
    print(f"    Default model:     {config.llm.default_model}")
    print(f"    Reranker model:    {config.llm.reranker_model}")
    print(f"    Scraper model:     {config.llm.scraper_model}")
    print(f"    Summary model:     {config.llm.summary_model}")
    print(f"    Fallback model:    {config.llm.fallback_model}")
    print(f"    Emulate Anthropic: {config.llm.emulate_anthropic}")
    print(f"    Emulate OpenAI:    {config.llm.emulate_openai}")
    print(f"    Available models:  {', '.join(config.llm.available_models)}")
    print(f"\n  Providers:")
    for name, prov in config.providers.model_dump().items():
        en = "✅" if prov.get("enabled", True) else "❌"
        key_status = "🔑" if prov.get("require_api_key") else ""
        print(f"    {en} {name} {key_status}")


if __name__ == "__main__":
    main()