"""Command Code Go API client — direct integration (no external proxy needed).

Command Code (commandcode.ai) offers a $1/mo Go plan with CLI access to 20+ models.
The Go plan does NOT include the official OpenAI API endpoint ($15/mo Provider plan
required), but the CLI's internal /alpha/generate endpoint can be called directly
with the right headers and body structure.

This client talks DIRECTLY to https://api.commandcode.ai/alpha/generate, handling:
  - CLI header mimicry (x-session-id, x-command-code-version, x-cmd-zdr, etc.)
  - OpenAI → Command Code request body translation (memory, params, config fields)
  - SSE response translation (text-delta/reasoning-delta/finish → OpenAI chunks)
  - Zero Data Retention (ZDR) mode via x-cmd-zdr header

No external proxy process is needed — everything is self-contained in this client,
matching the architecture of ClinePassClient, UmansClient, and OpenCodeGoClient.

Auth: Bearer <API_KEY> (user_... prefix, from ~/.commandcode/auth.json or env var)
Models: 20+ open-weight models, zero per-token cost ($1/mo flat rate)
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import time
import uuid
from typing import Any, AsyncGenerator, Optional

import httpx

from guanaco.providers.base import BaseProvider, ProviderMetrics

logger = logging.getLogger(__name__)

CMDCODE_API_BASE = "https://api.commandcode.ai"
CMDCODE_GENERATE_URL = f"{CMDCODE_API_BASE}/alpha/generate"
CMDCODE_USAGE_URL = f"{CMDCODE_API_BASE}/alpha/usage/summary"
CMDCODE_CREDITS_URL = f"{CMDCODE_API_BASE}/alpha/billing/credits"
CMDCODE_SUBSCRIPTION_URL = f"{CMDCODE_API_BASE}/alpha/billing/subscriptions"

# NPM registry endpoint for latest command-code CLI version
CMDCODE_NPM_URL = "https://registry.npmjs.org/command-code/latest"

# Fallback version used if the npm registry is unreachable on first load.
# Updated automatically at runtime by _get_cli_version() with a 6-hour TTL.
CMDCODE_CLI_VERSION_FALLBACK = "0.50.0"

# Module-level cache for the dynamically fetched CLI version.
_cli_version_cache: str | None = None
_cli_version_fetched_at: float = 0.0
_CLI_VERSION_TTL = 6 * 3600  # 6 hours


def _get_cli_version() -> str:
    """Return the latest Command Code CLI version, cached with a TTL.

    Fetches from the npm registry on first call and every 6 hours thereafter.
    If the fetch fails, falls back to the last known good version (or the
    hardcoded fallback if no successful fetch has ever occurred).
    """
    global _cli_version_cache, _cli_version_fetched_at
    now = time.time()
    if _cli_version_cache and (now - _cli_version_fetched_at) < _CLI_VERSION_TTL:
        return _cli_version_cache

    # Try to fetch the latest version from npm registry (quick, non-blocking feel)
    try:
        resp = httpx.get(CMDCODE_NPM_URL, timeout=5.0)
        resp.raise_for_status()
        version = resp.json().get("version")
        if version:
            _cli_version_cache = version
            _cli_version_fetched_at = now
            logger.debug("CmdCode CLI version fetched from npm: %s", version)
            return version
    except Exception as e:
        logger.debug("CmdCode CLI version fetch failed, using fallback: %s", e)

    # Use last known good version if available, otherwise the hardcoded fallback
    return _cli_version_cache or CMDCODE_CLI_VERSION_FALLBACK


# ── DSML tool-call parsing ──
# DeepSeek V4 models emit tool calls in a proprietary DSML (DeepSeek Markup
# Language) format embedded in the text content stream when the backend doesn't
# translate them into structured tool_calls.  We parse two variants:
#
#   1. invoke/parameter form:
#       <｜DSML｜tool_calls>
#       <｜DSML｜invoke name="bash">
#       <｜DSML｜parameter name="command" string="true">ls -la</｜DSML｜parameter>
#       </｜DSML｜invoke>
#       </｜DSML｜tool_calls>
#
#   2. name/parameters form (simpler):
#       <｜DSML｜tool_calls>
#       <name>search</name>
#       <parameters>{"query": "hello"}</parameters>
#       </｜DSML｜tool_calls>
#
# The Unicode fullwidth pipe '｜' (U+FF5C) is the canonical delimiter, but
# some backends emit ASCII '|' instead — we accept both.

import re as _re

# DSML delimiters can be:
#   ｜  = U+FF5C fullwidth pipe (canonical)
#   |  = U+007C ASCII pipe
#   ｜｜ = double fullwidth pipes (some models emit these)
#   ||  = double ASCII pipes
# We use [｜|]+ to match one or more of either char.
_DSML_PIPE = r'[｜|]+'

# Match the opening tool_calls block
_DSML_TC_OPEN = _re.compile(rf'<{_DSML_PIPE}DSML{_DSML_PIPE}tool_calls>')
# Match the closing tool_calls block (may have </ or just ] prefix)
_DSML_TC_CLOSE = _re.compile(rf'</?{_DSML_PIPE}DSML{_DSML_PIPE}tool_calls>')

# Match invoke open: <｜DSML｜invoke name="function_name">
_DSML_INVOKE = _re.compile(rf'<{_DSML_PIPE}DSML{_DSML_PIPE}invoke\s+name="([^"]+)">')

# Match invoke close: </｜DSML｜invoke>
_DSML_INVOKE_CLOSE = _re.compile(rf'</?{_DSML_PIPE}DSML{_DSML_PIPE}invoke>')

# Match parameter: <｜DSML｜parameter name="key" string="true">value</｜DSML｜parameter>
_DSML_PARAM = _re.compile(
    rf'<{_DSML_PIPE}DSML{_DSML_PIPE}parameter\s+name="([^"]+)"\s+string="([^"]*)">(.*?)</?{_DSML_PIPE}DSML{_DSML_PIPE}parameter>',
    _re.DOTALL,
)

# Match name/parameters form: <name>func</name> and <parameters>{...}</parameters>
_DSML_NAME = _re.compile(r'<name>(.*?)</name>', _re.DOTALL)
_DSML_PARAMS = _re.compile(r'<parameters>(.*?)</parameters>', _re.DOTALL)

# Match ANY DSML tag (opening or closing), including double-pipe variants
# and tags with attributes (e.g. <｜DSML｜invoke name="terminal">).
# Used for _contains_dsml and _strip_dsml_from_content.
_DSML_ANY_TAG = _re.compile(rf'</?{_DSML_PIPE}DSML{_DSML_PIPE}\w+[^>]*>')

# Detect bare DSML tag fragments where the model omitted the <｜DSML｜ prefix.
# These appear alongside proper DSML closing tags in malformed output.
# Examples: 'invoke name="terminal">'  or  'parameter name="cmd" string="true">'
# Match the FULL element: opening fragment + value + closing tag (if present).
_BARE_INVOKE = _re.compile(
    r'invoke\s+name="[^"]*">.*?(?=</?[｜|]+DSML[｜|]+(?:invoke|parameter|tool_calls)|\Z)',
    _re.DOTALL,
)
_BARE_PARAMETER = _re.compile(
    r'parameter\s+name="[^"]*"\s+string="[^"]*">.*?(?=</?[｜|]+DSML[｜|]+(?:invoke|parameter|tool_calls)|\Z)',
    _re.DOTALL,
)

# Catch <dsml_ignore>...</dsml_ignore> blocks — a non-pipe DSML-adjacent format
# some models emit for commentary around tool calls.
_DSML_IGNORE_BLOCK = _re.compile(r'<dsml_ignore>.*?</dsml_ignore>', _re.DOTALL | _re.IGNORECASE)
_DSML_IGNORE_TAG = _re.compile(r'</?dsml_ignore\s*>', _re.IGNORECASE)


# Catch hallucinated non-DSML tool call tags that some models emit in text.
# These are always formatting artifacts, never user-facing content.
# Matches things like </aktool_calls>, <tool_call>, </tool_calls>, etc.
_HALLUCINATED_TC_TAGS = _re.compile(r'</?[a-zA-Z_]*tool[a-zA-Z_]*call[s]?>', _re.IGNORECASE)


def _strip_hallucinated_tags(text: str) -> str:
    """Remove hallucinated tool call tags from text output."""
    return _HALLUCINATED_TC_TAGS.sub('', text)


def _contains_dsml(text: str) -> bool:
    """Check if text contains any DSML tool call markers (opening or closing)."""
    return bool(
        _DSML_ANY_TAG.search(text)
        or _BARE_INVOKE.search(text)
        or _BARE_PARAMETER.search(text)
        or _DSML_IGNORE_BLOCK.search(text)
        or _DSML_IGNORE_TAG.search(text)
        or '<｜DSML｜' in text
        or '<|DSML|' in text
    )


def _parse_dsml_tool_calls(text: str) -> list[dict[str, Any]] | None:
    """Parse DSML tool_calls blocks from text into OpenAI tool_calls format.

    Returns a list of OpenAI-format tool_call dicts, or None if no DSML found.

    Each dict looks like:
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": "{\"command\": \"ls -la\"}"
            }
        }
    """
    if not _contains_dsml(text):
        return None

    # Extract the content inside <｜DSML｜tool_calls>...</｜DSML｜tool_calls>
    # Find all tool_calls blocks (there could be multiple)
    tool_calls: list[dict[str, Any]] = []

    # Find each tool_calls block
    blocks = []
    pos = 0
    while True:
        open_match = _DSML_TC_OPEN.search(text, pos)
        if not open_match:
            break
        close_match = _DSML_TC_CLOSE.search(text, open_match.end())
        if not close_match:
            # Block not yet closed — incomplete, return None to wait for more
            return None
        blocks.append(text[open_match.end():close_match.start()])
        pos = close_match.end()

    for block in blocks:
        # Try invoke/parameter form first
        invokes = _DSML_INVOKE.findall(block)
        if invokes:
            # Split the block into individual invoke sections
            invoke_blocks = _re.split(r'</[｜|]DSML[｜|]invoke>', block)
            for invoke_block in invoke_blocks:
                name_match = _DSML_INVOKE.search(invoke_block)
                if not name_match:
                    continue
                func_name = name_match.group(1)
                # Extract all parameters
                params = {}
                for p_match in _DSML_PARAM.finditer(invoke_block):
                    p_name = p_match.group(1)
                    p_is_string = p_match.group(2) == "true"
                    p_value = p_match.group(3)
                    if p_is_string:
                        params[p_name] = p_value
                    else:
                        # Non-string: parse as JSON
                        try:
                            params[p_name] = json.loads(p_value)
                        except json.JSONDecodeError:
                            params[p_name] = p_value

                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(params, ensure_ascii=False),
                    },
                })
        else:
            # Try name/parameters form
            name_match = _DSML_NAME.search(block)
            params_match = _DSML_PARAMS.search(block)
            if name_match:
                func_name = name_match.group(1).strip()
                args_str = "{}"
                if params_match:
                    raw_params = params_match.group(1).strip()
                    # Validate it's parseable JSON
                    try:
                        parsed = json.loads(raw_params)
                        args_str = json.dumps(parsed, ensure_ascii=False)
                    except json.JSONDecodeError:
                        args_str = raw_params

                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": args_str,
                    },
                })

    return tool_calls if tool_calls else None


def _fuzzy_parse_dsml(text: str) -> list[dict[str, Any]] | None:
    """Best-effort parse of malformed/incomplete DSML blocks.
    
    Used when the DSML buffer overflows without a clean close tag.
    Tries to extract invoke names and parameters even with broken closing tags.
    """
    tool_calls: list[dict[str, Any]] = []
    
    # Find all invoke openings — even without proper closes
    invoke_starts = list(_DSML_INVOKE.finditer(text))
    if not invoke_starts:
        return None
    
    # Split on invoke openings
    for i, inv_match in enumerate(invoke_starts):
        func_name = inv_match.group(1)
        start = inv_match.end()
        # Block ends at next invoke start, or end of text
        end = invoke_starts[i + 1].start() if i + 1 < len(invoke_starts) else len(text)
        block = text[start:end]
        
        # Extract parameters — match even without proper closing tags
        params = {}
        for p_match in _DSML_PARAM.finditer(block):
            p_name = p_match.group(1)
            p_is_string = p_match.group(2) == "true"
            p_value = p_match.group(3)
            if p_is_string:
                params[p_name] = p_value
            else:
                try:
                    params[p_name] = json.loads(p_value)
                except (json.JSONDecodeError, ValueError):
                    params[p_name] = p_value
        
        # Also try to catch parameters with broken closing tags
        # Pattern: <｜DSML｜parameter name="key" string="true">value  (no close)
        broken_param_re = _re.compile(
            rf'<{_DSML_PIPE}DSML{_DSML_PIPE}parameter\s+name="([^"]+)"\s+string="([^"]*)">([^<]*)',
            _re.DOTALL,
        )
        for p_match in broken_param_re.finditer(block):
            p_name = p_match.group(1)
            p_is_string = p_match.group(2) == "true"
            p_value = p_match.group(3).strip()
            if p_name not in params:  # Don't overwrite properly parsed ones
                if p_is_string:
                    params[p_name] = p_value
                else:
                    try:
                        params[p_name] = json.loads(p_value)
                    except (json.JSONDecodeError, ValueError):
                        params[p_name] = p_value
        
        if func_name:
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(params, ensure_ascii=False),
                },
            })
    
    return tool_calls if tool_calls else None


def _strip_dsml_from_content(text: str) -> str:
    """Remove all DSML tags and blocks from content text.
    
    Handles:
    - Complete <｜DSML｜tool_calls>...</｜DSML｜tool_calls> blocks
    - Incomplete blocks (open but no close)
    - Stray closing tags (e.g. </｜DSML｜parameter>)
    - Double-pipe variants (</｜｜DSML｜｜tool_calls>)
    - Solo invoke/parameter tags without tool_calls wrapper
    - Bare tag fragments where model omitted <｜DSML｜ prefix
      (e.g. 'invoke name="terminal">' without the DSML prefix)
    - <dsml_ignore>...</dsml_ignore> commentary blocks
    - Stray '>' characters left from partially stripped tags
    """
    if not _contains_dsml(text):
        return text
    # First remove <dsml_ignore>...</dsml_ignore> blocks
    result = _DSML_IGNORE_BLOCK.sub('', text)
    # Remove complete tool_calls blocks
    while True:
        open_match = _DSML_TC_OPEN.search(result)
        if not open_match:
            break
        close_match = _DSML_TC_CLOSE.search(result, open_match.end())
        if not close_match:
            # Incomplete block — strip from the opening tag to end
            result = result[:open_match.start()]
            break
        result = result[:open_match.start()] + result[close_match.end():]
    # Strip any remaining stray DSML tags (closing tags, solo invokes, etc.)
    result = _DSML_ANY_TAG.sub('', result)
    # Strip bare DSML tag fragments where the model omitted the <｜DSML｜ prefix.
    # These match the full element: opening fragment + value + closing tag.
    result = _BARE_INVOKE.sub('', result)
    result = _BARE_PARAMETER.sub('', result)
    # Strip stray <dsml_ignore> tags (in case block regex didn't match)
    result = _DSML_IGNORE_TAG.sub('', result)
    # Clean up stray '>' characters on their own line (leftover from partial tag stripping)
    result = _re.sub(r'\n>\s*\n', '\n', result)
    result = _re.sub(r'^>\s*\n', '', result)
    return result.strip()


# Static model list — Command Code Go plan offers 20+ models with ZDR support.
CMDCODE_MODELS: dict[str, dict[str, Any]] = {
    "deepseek-v4-pro": {
        "family": "deepseek", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "deepseek-v4-flash": {
        "family": "deepseek", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "kimi-k2.7-code": {
        "family": "kimi", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "kimi-k2.7-code-highspeed": {
        "family": "kimi", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "kimi-k2.6": {
        "family": "kimi", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "kimi-k2.5": {
        "family": "kimi", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "glm-5.2": {
        "family": "glm", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "glm-5.2-fast": {
        "family": "glm", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "glm-5.1": {
        "family": "glm", "supports_vision": False, "supports_tools": True,
        "supports_thinking": False, "usage_multiplier": 0.0,
    },
    "glm-5": {
        "family": "glm", "supports_vision": False, "supports_tools": True,
        "supports_thinking": False, "usage_multiplier": 0.0,
    },
    "minimax-m3": {
        "family": "minimax", "supports_vision": False, "supports_tools": True,
        "supports_thinking": False, "usage_multiplier": 0.0,
    },
    "minimax-m2.7": {
        "family": "minimax", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "minimax-m2.5": {
        "family": "minimax", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "mimo-v2.5-pro": {
        "family": "mimo", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "mimo-v2.5": {
        "family": "mimo", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "qwen3.7-plus": {
        "family": "qwen", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "qwen3.6-plus": {
        "family": "qwen", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "tencent-hy3": {
        "family": "tencent", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "nemotron-3-ultra": {
        "family": "nvidia", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
    "step-3.5-flash": {
        "family": "stepfun", "supports_vision": False, "supports_tools": True,
        "supports_thinking": True, "usage_multiplier": 0.0,
    },
}

# Model name mapping: OpenAI-style short names → Command Code full model IDs
MODEL_MAP: dict[str, str] = {
    "deepseek-v4-pro": "deepseek/deepseek-v4-pro",
    "deepseek-v4-flash": "deepseek/deepseek-v4-flash",
    "kimi-k2.7-code": "moonshotai/Kimi-K2.7-Code",
    "kimi-k2.7-code-highspeed": "moonshotai/Kimi-K2.7-Code-Highspeed",
    "kimi-k2.6": "moonshotai/Kimi-K2.6",
    "kimi-k2.5": "moonshotai/Kimi-K2.5",
    "glm-5.2": "zai-org/GLM-5.2",
    "glm-5.2-fast": "zai-org/GLM-5.2-Fast",
    "glm-5.1": "zai-org/GLM-5.1",
    "glm-5": "zai-org/GLM-5",
    "minimax-m3": "MiniMaxAI/MiniMax-M3",
    "minimax-m2.7": "MiniMaxAI/MiniMax-M2.7",
    "minimax-m2.5": "MiniMaxAI/MiniMax-M2.5",
    "mimo-v2.5-pro": "xiaomi/mimo-v2.5-pro",
    "mimo-v2.5": "xiaomi/mimo-v2.5",
    "qwen3.6-plus": "Qwen/Qwen3.6-Plus",
    "qwen3.7-plus": "Qwen/Qwen3.7-Plus",
    "tencent-hy3": "tencent/Hy3",
    "nemotron-3-ultra": "nvidia/Nemotron-3-Ultra",
    "step-3.5-flash": "stepfun/Step-3.5-Flash",
}


def _strip_cmdcode_prefix(model: str) -> str:
    """Return the model id without the cmdcode/ prefix."""
    model = model.strip()
    lower = model.lower()
    if lower.startswith("cmdcode/"):
        model = model[len("cmdcode/"):]
    return model


def _resolve_model(model_id: str) -> str:
    """Resolve short model name to full Command Code model ID."""
    if model_id in MODEL_MAP:
        return MODEL_MAP[model_id]
    if "/" in model_id:
        return model_id  # already a full ID
    for k, v in MODEL_MAP.items():
        if k.lower() == model_id.lower():
            return v
    return model_id  # pass through


class CmdCodeClient(BaseProvider):
    """Async client for Command Code Go plan — direct API integration.

    Calls https://api.commandcode.ai/alpha/generate directly, translating
    OpenAI chat completion requests to Command Code's internal format and
    translating the SSE response back to OpenAI-compatible chunks.

    No external proxy process is needed. This is self-contained, matching
    the architecture of ClinePassClient, UmansClient, and OpenCodeGoClient.
    """

    provider_name = "cmdcode"
    supports_streaming = True
    supports_vision = False
    supports_thinking = True

    def __init__(self, api_key: str = "", timeout: float = 300.0, base_url: str = ""):
        # base_url is accepted for config compatibility but ignored — we always
        # talk directly to api.commandcode.ai
        super().__init__(api_key=api_key, timeout=timeout, base_url=CMDCODE_API_BASE)

    # ── Header / body builders ──

    def _build_headers(self) -> dict[str, str]:
        """Build headers that mimic the Command Code CLI."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "x-command-code-version": _get_cli_version(),
            "x-cli-environment": "cli",
            "x-session-id": str(uuid.uuid4()),
            "x-project-slug": "command-code",
            "x-taste-learning": "false",
            "x-taste-usage": "false",
            "x-cmd-zdr": "1",  # Zero Data Retention
            "User-Agent": "cli",
            "Accept": "application/json",
        }

    def _build_generate_body(self, openai_request: dict) -> dict:
        """Convert an OpenAI chat completion request to Command Code /alpha/generate format."""
        messages = openai_request.get("messages", [])
        model = _resolve_model(openai_request.get("model", "deepseek/deepseek-v4-flash"))
        max_tokens = openai_request.get("max_tokens", 64000)
        tools = openai_request.get("tools", [])

        # Build system prompt from system messages
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        system_text = "\n".join(system_parts) if system_parts else "You are a helpful assistant."

        # Non-system messages — convert tool role and assistant tool_calls
        conv_messages: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "system":
                continue
            role = m.get("role", "user")

            # Convert tool results (role=tool) to user messages with tool_result wrapper
            if role == "tool":
                tool_call_id = m.get("tool_call_id", "")
                content = m.get("content", "")
                conv_messages.append({
                    "role": "user",
                    "content": f"<tool_result>{content}</tool_result>",
                })
                continue

            # Handle assistant messages with tool_calls
            if role == "assistant" and m.get("tool_calls"):
                tc_parts = []
                for tc in m["tool_calls"]:
                    func = tc.get("function", {})
                    func_name = func.get("name", "")
                    args_str = func.get("arguments", "{}")
                    try:
                        args = json.loads(args_str)
                    except (json.JSONDecodeError, TypeError):
                        args = {}

                    # Guard: args could be a non-dict JSON value (str, list, int)
                    if not isinstance(args, dict):
                        args = {"value": args}

                    # Render as DSML invoke/parameter format
                    param_lines = []
                    for k, v in args.items():
                        is_str = isinstance(v, str)
                        val = v if is_str else json.dumps(v, ensure_ascii=False)
                        param_lines.append(
                            f'<｜DSML｜parameter name="{k}" string="{"true" if is_str else "false"}">{val}</｜DSML｜parameter>'
                        )
                    params_xml = "\n".join(param_lines)
                    tc_parts.append(
                        f'<｜DSML｜invoke name="{func_name}">\n{params_xml}\n</｜DSML｜invoke>'
                    )

                tc_block = "<｜DSML｜tool_calls>\n" + "\n".join(tc_parts) + "\n</｜DSML｜tool_calls>"
                content = m.get("content", "") or ""
                conv_messages.append({
                    "role": "assistant",
                    "content": content + "\n\n" + tc_block if content else tc_block,
                })
                continue

            # Normal message
            content = m.get("content", "")
            if isinstance(content, list):
                # Handle content blocks (multimodal)
                content = " ".join(
                    block.get("text", "") for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            conv_messages.append({"role": role, "content": content or ""})

        if not conv_messages:
            conv_messages = [{"role": "user", "content": ""}]

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Convert OpenAI tools to Command Code format
        # CC uses Anthropic-style format: name, description, input_schema (not parameters)
        cc_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                cc_tools.append({
                    "type": "function",
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })

        return {
            "model": model,
            "messages": conv_messages,
            "max_tokens": max_tokens,
            "stream": True,  # ALWAYS stream from backend — collect for non-streaming clients
            "memory": "",
            "params": {
                "model": model,
                "messages": conv_messages,
                "tools": cc_tools,
                "system": system_text,
                "max_tokens": max_tokens,
                "stream": True,
            },
            "config": {
                "workingDir": os.getcwd(),
                "date": now,
                "environment": "cli",
                "structure": [],
                "isGitRepo": False,
                "currentBranch": "",
                "mainBranch": "",
                "gitStatus": "",
                "recentCommits": [],
            },
            "taste": None,
            "skills": None,
        }

    @staticmethod
    def _parse_cc_sse_line(line: str) -> dict | None:
        """Parse a single Command Code SSE line into an event dict.

        CC sends newline-delimited JSON objects (not SSE data: prefix):
        {"type":"start"}
        {"type":"start-step","request":{...}}
        {"type":"reasoning-start","id":"reasoning-0"}
        {"type":"reasoning-delta","id":"reasoning-0","text":"..."}
        {"type":"reasoning-end","id":"reasoning-0"}
        {"type":"text-delta","text":"..."}
        {"type":"finish-step","finishReason":"stop","usage":{"inputTokens":N,"outputTokens":N,...}}
        {"type":"finish","finishReason":"stop","totalUsage":{"inputTokens":N,"outputTokens":N,...}}
        {"type":"provider-metadata","providerMetadata":{...}}
        {"type":"error","error":{"message":"..."}}
        """
        line = line.strip()
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_usage(event: dict) -> dict:
        """Extract usage dict from either old or new Command Code SSE format.

        New format (v0.18.10+):
          - finish-step event has "usage" with inputTokens/outputTokens
          - finish event has "totalUsage" with inputTokens/outputTokens
        Old format:
          - finish event has "usage" with promptTokens/completionTokens
        """
        # New format: finish-step has usage with inputTokens
        usage = event.get("usage")
        if usage and "inputTokens" in usage:
            return {
                "promptTokens": usage.get("inputTokens", 0),
                "completionTokens": usage.get("outputTokens", 0),
                "totalTokens": usage.get("totalTokens", 0),
            }
        # New format: finish event has totalUsage with inputTokens
        total_usage = event.get("totalUsage")
        if total_usage and "inputTokens" in total_usage:
            return {
                "promptTokens": total_usage.get("inputTokens", 0),
                "completionTokens": total_usage.get("outputTokens", 0),
                "totalTokens": total_usage.get("totalTokens", 0),
            }
        # Old format: finish event has usage with promptTokens
        if usage and "promptTokens" in usage:
            return usage
        return {}

    @staticmethod
    def _make_openai_chunk(
        model: str, content: str = "", reasoning: str = "", finish_reason: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        chunk_id: str | None = None,
    ) -> str:
        """Build an OpenAI-compatible streaming chunk (SSE format)."""
        delta: dict[str, Any] = {}
        if reasoning:
            delta["reasoning_content"] = reasoning
        if content:
            delta["content"] = content
        if tool_calls:
            delta["tool_calls"] = tool_calls
        if finish_reason:
            delta = {}

        chunk = {
            "id": chunk_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    @staticmethod
    def _make_openai_response(model: str, content: str, reasoning: str, usage: dict, finish_reason: str, tool_calls: list[dict[str, Any]] | None = None) -> dict:
        """Build an OpenAI-compatible non-streaming response."""
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": usage.get("promptTokens", 0),
                "completion_tokens": usage.get("completionTokens", 0),
                "total_tokens": usage.get("totalTokens", 0),
            } if usage else {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    # ── Model listing ──

    async def list_models(self, force_refresh: bool = False, api_key: Optional[str] = None) -> list[dict]:
        """List available Command Code models.

        Command Code doesn't have a /models endpoint on /alpha/generate, so we
        always return the static list. This matches the Cline pattern where
        the static list is the source of truth for capability hints.
        """
        now = time.time()
        if not force_refresh and not api_key and self._models_cache and (now - self._models_cache_time) < self._models_cache_ttl:
            return self._models_cache

        models = self._static_models()
        self._models_cache = models
        self._models_cache_time = now
        return models

    def _static_models(self) -> list[dict]:
        """Return static model list."""
        return [
            {"id": mid, "name": mid, "model": mid, "display_name": mid, "details": {}}
            for mid in CMDCODE_MODELS
        ]

    async def test_key(self, api_key: Optional[str] = None) -> dict:
        """Test the API key by making a minimal generate request."""
        key = api_key or self.api_key
        if not key:
            return {"ok": False, "error": "No Command Code API key configured"}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "x-command-code-version": _get_cli_version(),
            "x-cli-environment": "cli",
            "x-session-id": str(uuid.uuid4()),
            "x-project-slug": "command-code",
            "x-taste-learning": "false",
            "x-taste-usage": "false",
            "x-cmd-zdr": "1",
            "User-Agent": "cli",
            "Accept": "application/json",
        }
        body = self._build_generate_body({
            "model": "deepseek/deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1,
        })
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(CMDCODE_GENERATE_URL, json=body, headers=headers)
                if resp.status_code == 200:
                    return {"ok": True, "error": None, "model_count": len(CMDCODE_MODELS)}
                if resp.status_code == 401:
                    return {"ok": False, "error": "Invalid or expired Command Code API key"}
                return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        except Exception as e:
            logger.warning("CmdCode key test failed: %s", e)
            return {"ok": False, "error": str(e)[:200]}

    # ── Usage / Billing ──

    async def fetch_usage(self) -> dict:
        """Fetch monthly usage summary from Command Code.

        Returns:
            {
                "total_requests": int,
                "completed": int,
                "failed": int,
                "success_rate": float,
                "tokens_in": int,
                "tokens_out": int,
                "total_tokens": int,
                "credits_used": float,
                "monthly_credits_used": float,
                "remaining_credits": float,
                "five_hour_used": float,
                "five_hour_cap": float,
                "weekly_used": float,
                "weekly_cap": float,
                "weekly_reset_at": str | None,
                "plan": str | None,
                "period_start": str | None,
                "period_end": str | None,
                "subscription_status": str | None,
            }
        """
        if not self.api_key:
            return {}
        headers = self._build_headers()
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                # Fetch usage summary, credits, and subscription in parallel
                usage_resp, credits_resp, sub_resp = await asyncio.gather(
                    client.get(CMDCODE_USAGE_URL, headers=headers),
                    client.get(CMDCODE_CREDITS_URL, headers=headers),
                    client.get(CMDCODE_SUBSCRIPTION_URL, headers=headers),
                    return_exceptions=True,
                )

            result: dict[str, Any] = {}

            if isinstance(usage_resp, httpx.Response) and usage_resp.status_code == 200:
                data = usage_resp.json()
                result.update({
                    "total_requests": data.get("totalCount", 0),
                    "completed": data.get("completedCount", 0),
                    "failed": data.get("failedCount", 0),
                    "success_rate": data.get("successRate", 0),
                    "tokens_in": data.get("totalTokensIn", 0),
                    "tokens_out": data.get("totalTokensOut", 0),
                    "total_tokens": data.get("totalTokens", 0),
                    "credits_used": data.get("totalCredits", 0),
                    "monthly_credits_used": data.get("totalMonthlyCredits", 0),
                })

            if isinstance(credits_resp, httpx.Response) and credits_resp.status_code == 200:
                data = credits_resp.json()
                credits = data.get("credits", {})
                window = data.get("windowLimits", {})
                five_hour = window.get("fiveHour", {})
                weekly = window.get("weekly", {})
                result.update({
                    "remaining_credits": credits.get("monthlyCredits", 0),
                    "five_hour_used": five_hour.get("used", 0),
                    "five_hour_cap": five_hour.get("cap", 0),
                    "weekly_used": weekly.get("used", 0),
                    "weekly_cap": weekly.get("cap", 0),
                    "weekly_reset_at": (
                        datetime.datetime.fromtimestamp(
                            weekly.get("resetAt", 0) / 1000,
                            tz=datetime.timezone.utc
                        ).isoformat() if weekly.get("resetAt") else None
                    ),
                })

            if isinstance(sub_resp, httpx.Response) and sub_resp.status_code == 200:
                data = sub_resp.json()
                sub = data.get("data", {})
                result.update({
                    "plan": sub.get("planId"),
                    "period_start": sub.get("currentPeriodStart"),
                    "period_end": sub.get("currentPeriodEnd"),
                    "subscription_status": sub.get("status"),
                })

            return result
        except Exception as e:
            logger.warning("CmdCode usage fetch failed: %s", e)
            return {}

    # ── Capabilities ──

    def _get_model_capabilities(self, model: str) -> dict:
        """Return capability dict for a Command Code model."""
        canonical = _strip_cmdcode_prefix(model)
        caps = CMDCODE_MODELS.get(canonical, {})
        return {
            "supports_vision": bool(caps.get("supports_vision", False)),
            "supports_tools": bool(caps.get("supports_tools", True)),
            "supports_thinking": bool(caps.get("supports_thinking", False)),
            "family": caps.get("family", canonical.split("-")[0] if "-" in canonical else "unknown"),
            "usage_multiplier": 0.0,  # $1/mo flat rate — zero per-token cost
            "provider": "cmdcode",
        }

    # ── Payload normalization ──

    def _prepare_payload(self, payload: dict) -> dict:
        """Strip cmdcode/ prefix and normalize payload."""
        payload = dict(payload)
        model = payload.get("model", "")
        payload["model"] = _strip_cmdcode_prefix(model)
        # Strip reasoning_content from assistant messages (same as Cline/UMANS)
        msgs = payload.get("messages")
        if isinstance(msgs, list):
            for m in msgs:
                if m.get("role") == "assistant":
                    m.pop("reasoning_content", None)
                    m.pop("reasoningContent", None)
        return payload

    # ── Chat completions ──

    async def chat_completion(self, payload: dict, api_key: Optional[str] = None) -> dict:
        """Non-streaming chat completion.

        Always streams from the Command Code backend, collects the full response,
        then returns a single OpenAI-compatible JSON response.
        """
        payload = self._prepare_payload(payload)
        client_model = payload.get("model", "")
        cc_body = self._build_generate_body(payload)

        key = api_key or self.api_key
        headers = self._build_headers()
        if api_key and api_key != self.api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        start = time.time()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict = {}
        finish_reason = "stop"
        # Tool call accumulation state
        _tool_calls: list[dict[str, Any]] = []
        _tc_arg_buffers: dict[str, str] = {}  # id → accumulated argument delta

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as http_client:
                async with http_client.stream("POST", CMDCODE_GENERATE_URL, json=cc_body, headers=headers) as resp:
                    if resp.status_code != 200:
                        body_preview = ""
                        try:
                            body_preview = (await resp.aread()).decode(errors="replace")[:500]
                        except Exception:
                            pass
                        logger.error("CmdCode non-stream HTTP %d — body: %s", resp.status_code, body_preview)
                        raise httpx.HTTPStatusError(
                            f"Command Code HTTP {resp.status_code}: {body_preview}",
                            request=resp.request,
                            response=resp,
                        )
                    async for line in resp.aiter_lines():
                        event = self._parse_cc_sse_line(line)
                        if event is None:
                            continue
                        etype = event.get("type", "")
                        if etype == "text-delta":
                            content_parts.append(event.get("text", ""))
                        elif etype == "reasoning-delta":
                            reasoning_parts.append(event.get("text", ""))
                        elif etype == "tool-input-start":
                            # Start of a tool call — create placeholder
                            tc_id = event.get("id", f"call_{uuid.uuid4().hex[:24]}")
                            tc_name = event.get("toolName", "")
                            _tc_arg_buffers[tc_id] = ""
                            _tool_calls.append({
                                "id": tc_id,
                                "type": "function",
                                "function": {"name": tc_name, "arguments": ""},
                            })
                        elif etype == "tool-input-delta":
                            # Accumulate argument fragments
                            tc_id = event.get("id", "")
                            delta = event.get("delta", "")
                            if tc_id in _tc_arg_buffers:
                                _tc_arg_buffers[tc_id] += delta
                        elif etype == "tool-input-end":
                            # Tool input complete — finalize arguments
                            tc_id = event.get("id", "")
                            if tc_id in _tc_arg_buffers:
                                for tc in _tool_calls:
                                    if tc["id"] == tc_id:
                                        tc["function"]["arguments"] = _tc_arg_buffers[tc_id]
                                        break
                        elif etype == "tool-call":
                            # Complete tool call (may arrive after tool-input-end)
                            tc_id = event.get("toolCallId", "")
                            tc_name = event.get("toolName", "")
                            tc_input = event.get("input", {})
                            # Update or append
                            found = False
                            for tc in _tool_calls:
                                if tc["id"] == tc_id:
                                    tc["function"]["name"] = tc_name
                                    tc["function"]["arguments"] = json.dumps(tc_input, ensure_ascii=False)
                                    found = True
                                    break
                            if not found:
                                _tool_calls.append({
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {
                                        "name": tc_name,
                                        "arguments": json.dumps(tc_input, ensure_ascii=False),
                                    },
                                })
                        elif etype == "finish-step":
                            u = self._extract_usage(event)
                            if u:
                                usage = u
                            fr = event.get("finishReason")
                            if fr:
                                finish_reason = fr
                        elif etype == "finish":
                            finish_reason = event.get("finishReason", "stop")
                            u = self._extract_usage(event)
                            if u:
                                usage = u
                        elif etype == "error":
                            msg = event.get("error", {}).get("message", "unknown error")
                            raise httpx.HTTPStatusError(
                                f"Command Code error: {msg}",
                                request=resp.request,
                                response=resp,
                            )
        except httpx.HTTPStatusError:
            raise
        except Exception as e:
            logger.error("CmdCode non-stream error: %s", e)
            raise

        elapsed = time.time() - start
        full_content = "".join(content_parts)

        # Strip hallucinated tool call tags from content (e.g. </aktool_calls>)
        full_content = _strip_hallucinated_tags(full_content)

        # Use structured tool_calls from SSE events if available;
        # otherwise fall back to DSML parsing from text content
        if _tool_calls:
            parsed_tool_calls = _tool_calls
            finish_reason = "tool_calls"
        else:
            parsed_tool_calls = _parse_dsml_tool_calls(full_content)
            if parsed_tool_calls:
                full_content = _strip_dsml_from_content(full_content)
                finish_reason = "tool_calls"
            elif _contains_dsml(full_content):
                # Incomplete DSML block (no close tag) — try fuzzy parse,
                # then strip any DSML tags so they don't leak as content.
                parsed_tool_calls = _fuzzy_parse_dsml(full_content)
                if parsed_tool_calls:
                    full_content = _strip_dsml_from_content(full_content)
                    finish_reason = "tool_calls"
                else:
                    full_content = _strip_dsml_from_content(full_content)

        result = self._make_openai_response(
            model=client_model,
            content=full_content,
            reasoning="".join(reasoning_parts),
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=parsed_tool_calls,
        )

        u = result.get("usage", {})
        metrics = {
            "elapsed_seconds": elapsed,
            "prompt_eval_count": u.get("prompt_tokens", 0),
            "eval_count": u.get("completion_tokens", 0),
        }
        if metrics["eval_count"] and elapsed > 0:
            metrics["tps"] = round(min(metrics["eval_count"] / elapsed, 1000.0), 2)
        if elapsed > 0:
            metrics["ttft_seconds"] = round(elapsed, 3)
        result["_oct_metrics"] = metrics
        return result

    async def chat_completion_stream(self, payload: dict, api_key: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Streaming chat completion.

        Translates Command Code's SSE format (text-delta/reasoning-delta/finish
        JSON events) into OpenAI-compatible SSE chunks (data: {chat.completion.chunk}).
        """
        payload = self._prepare_payload(payload)
        client_model = payload.get("model", "")
        cc_body = self._build_generate_body(payload)

        key = api_key or self.api_key
        headers = self._build_headers()
        if api_key and api_key != self.api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        first_token_time: Optional[float] = None
        content_chars = 0
        reasoning_chars = 0
        prompt_tokens = 0
        completion_tokens = 0
        start = time.time()

        # Single consistent ID for all chunks in this stream
        _stream_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        # DSML streaming state — buffer text and detect tool call blocks
        _dsml_buffer = ""
        _in_dsml = False
        _tool_calls_emitted = False

        # Structured tool call streaming state (from tool-input-* / tool-call events)
        _tool_calls_list: list[dict[str, Any]] = []
        _tc_arg_buffers: dict[str, str] = {}

        # Markers for partial-match detection (handle split-across-deltas)
        _DSML_OPEN_U = "<｜DSML｜tool_calls>"
        _DSML_OPEN_A = "<|DSML|tool_calls>"
        # Also detect partial solo invoke markers
        _DSML_INVOKE_U = "<｜DSML｜invoke"
        _DSML_INVOKE_A = "<|DSML|invoke"

        def _partial_dsml_open_at_end(text: str) -> int:
            """If text ends with a prefix of a DSML open marker, return the start index."""
            for marker in (_DSML_OPEN_U, _DSML_OPEN_A, _DSML_INVOKE_U, _DSML_INVOKE_A):
                max_check = min(len(marker), len(text))
                for length in range(max_check, 0, -1):
                    if marker.startswith(text[-length:]):
                        return len(text) - length
            return -1

        def _check_dsml_enter(combined: str) -> int | None:
            """Check if combined text contains a DSML entry point.
            
            Returns the index where the DSML block starts, or None.
            Detects both <｜DSML｜tool_calls> wrapper and solo <｜DSML｜invoke>.
            """
            # Check for tool_calls wrapper
            for marker in (_DSML_OPEN_U, _DSML_OPEN_A):
                idx = combined.find(marker)
                if idx >= 0:
                    return idx
            # Check for solo invoke (no tool_calls wrapper)
            m = _DSML_INVOKE.search(combined)
            if m:
                return m.start()
            return None

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as http_client:
                async with http_client.stream("POST", CMDCODE_GENERATE_URL, json=cc_body, headers=headers) as resp:
                    if resp.status_code != 200:
                        body_preview = ""
                        try:
                            body_preview = (await resp.aread()).decode(errors="replace")[:500]
                        except Exception:
                            pass
                        logger.error("CmdCode stream HTTP %d — body: %s", resp.status_code, body_preview)
                        raise httpx.HTTPStatusError(
                            f"Command Code HTTP {resp.status_code}: {body_preview}",
                            request=resp.request,
                            response=resp,
                        )
                    async for line in resp.aiter_lines():
                        event = self._parse_cc_sse_line(line)
                        if event is None:
                            continue

                        etype = event.get("type", "")

                        if etype == "text-delta":
                            text = event.get("text", "")
                            if not text:
                                continue
                            # Strip hallucinated tool call tags (e.g. </aktool_calls>)
                            text = _strip_hallucinated_tags(text)
                            if not text:
                                continue
                            if not first_token_time:
                                first_token_time = time.time()
                            content_chars += len(text)

                            if _in_dsml:
                                # Already inside a DSML block — buffer until close
                                _dsml_buffer += text
                                # Check if the block is now complete
                                parsed = _parse_dsml_tool_calls(_dsml_buffer)
                                if parsed is not None:
                                    # Block complete — emit tool_calls chunk
                                    yield self._make_openai_chunk(client_model, chunk_id=_stream_id,
                                        tool_calls=parsed
                                    )
                                    _tool_calls_emitted = True
                                    _dsml_buffer = ""
                                    _in_dsml = False
                                elif len(_dsml_buffer) > 8000:
                                    # Buffer overflow — model emitted malformed/incomplete DSML.
                                    # Try a fuzzy parse first; if that fails, flush as stripped content.
                                    logger.warning("DSML buffer overflow (%d chars) — attempting fuzzy parse", len(_dsml_buffer))
                                    parsed = _fuzzy_parse_dsml(_dsml_buffer)
                                    if parsed:
                                        yield self._make_openai_chunk(client_model, chunk_id=_stream_id,
                                            tool_calls=parsed
                                        )
                                        _tool_calls_emitted = True
                                    else:
                                        cleaned = _strip_dsml_from_content(_dsml_buffer)
                                        if cleaned:
                                            yield self._make_openai_chunk(client_model, chunk_id=_stream_id, content=cleaned)
                                    _dsml_buffer = ""
                                    _in_dsml = False
                            else:
                                # Not in DSML — check if this chunk starts one
                                combined = _dsml_buffer + text
                                dsml_idx = _check_dsml_enter(combined)
                                if dsml_idx is not None:
                                    # DSML block started — flush pre-DSML content
                                    pre_dsml = combined[:dsml_idx]
                                    if pre_dsml:
                                        yield self._make_openai_chunk(client_model, chunk_id=_stream_id, content=pre_dsml)
                                    _dsml_buffer = combined[dsml_idx:]
                                    _in_dsml = True
                                    # Check if already complete in this same chunk
                                    parsed = _parse_dsml_tool_calls(_dsml_buffer)
                                    if parsed is not None:
                                        yield self._make_openai_chunk(client_model, chunk_id=_stream_id,
                                            tool_calls=parsed
                                        )
                                        _tool_calls_emitted = True
                                        _dsml_buffer = ""
                                        _in_dsml = False
                                else:
                                    # No DSML marker — but the end of text might be
                                    # a partial marker. Check and split if needed.
                                    partial_idx = _partial_dsml_open_at_end(combined)
                                    if partial_idx >= 0 and partial_idx < len(combined):
                                        # Flush the safe part, buffer the partial marker
                                        safe = combined[:partial_idx]
                                        _dsml_buffer = combined[partial_idx:]
                                        if safe:
                                            yield self._make_openai_chunk(client_model, chunk_id=_stream_id, content=safe)
                                    else:
                                        # Completely safe — flush everything
                                        _dsml_buffer = ""
                                        # Safety net: strip any stray DSML tags that
                                        # might have slipped through (closing tags, etc.)
                                        if _contains_dsml(text):
                                            text = _strip_dsml_from_content(text)
                                        # Suppress stray '>' on its own line —
                                        # leftover from a stripped DSML tag.
                                        # Only matches a line that is just '>' + whitespace.
                                        if text.strip() == '>':
                                            text = ''
                                        if text:
                                            yield self._make_openai_chunk(client_model, chunk_id=_stream_id, content=text)

                        elif etype == "reasoning-delta":
                            text = event.get("text", "")
                            if text:
                                reasoning_chars += len(text)
                                if not first_token_time:
                                    first_token_time = time.time()
                                yield self._make_openai_chunk(client_model, chunk_id=_stream_id, reasoning=text)

                        elif etype == "tool-input-start":
                            # Start of a structured tool call from the API
                            tc_id = event.get("id", f"call_{uuid.uuid4().hex[:24]}")
                            tc_name = event.get("toolName", "")
                            _tc_arg_buffers[tc_id] = ""
                            _tool_calls_list.append({
                                "id": tc_id,
                                "index": 0,
                                "type": "function",
                                "function": {"name": tc_name, "arguments": ""},
                            })
                            if not first_token_time:
                                first_token_time = time.time()

                        elif etype == "tool-input-delta":
                            tc_id = event.get("id", "")
                            delta = event.get("delta", "")
                            if tc_id in _tc_arg_buffers:
                                _tc_arg_buffers[tc_id] += delta

                        elif etype == "tool-input-end":
                            tc_id = event.get("id", "")
                            if tc_id in _tc_arg_buffers:
                                for tc in _tool_calls_list:
                                    if tc["id"] == tc_id:
                                        tc["function"]["arguments"] = _tc_arg_buffers[tc_id]
                                        break

                        elif etype == "tool-call":
                            # Complete tool call event
                            tc_id = event.get("toolCallId", "")
                            tc_name = event.get("toolName", "")
                            tc_input = event.get("input", {})
                            args_str = json.dumps(tc_input, ensure_ascii=False)
                            found = False
                            for tc in _tool_calls_list:
                                if tc["id"] == tc_id:
                                    tc["function"]["name"] = tc_name
                                    tc["function"]["arguments"] = args_str
                                    found = True
                                    break
                            if not found:
                                _tool_calls_list.append({
                                    "id": tc_id,
                                    "index": len(_tool_calls_list),
                                    "type": "function",
                                    "function": {"name": tc_name, "arguments": args_str},
                                })
                            # Emit the tool call as an OpenAI chunk
                            _tc_idx = next((i for i, t in enumerate(_tool_calls_list) if t["id"] == tc_id), 0)
                            yield self._make_openai_chunk(client_model, chunk_id=_stream_id,
                                tool_calls=[{
                                    "index": _tc_idx,
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {"name": tc_name, "arguments": args_str},
                                }],
                            )
                            _tool_calls_emitted = True

                        elif etype == "finish-step":
                            # New format: per-step usage with inputTokens/outputTokens
                            u = self._extract_usage(event)
                            if u:
                                prompt_tokens = u.get("promptTokens", prompt_tokens)
                                completion_tokens = u.get("completionTokens", completion_tokens)

                        elif etype == "finish":
                            finish_reason = event.get("finishReason", "stop")
                            u = self._extract_usage(event)
                            if u:
                                prompt_tokens = u.get("promptTokens", prompt_tokens)
                                completion_tokens = u.get("completionTokens", completion_tokens)
                            # If we emitted tool_calls, override finish_reason
                            if _tool_calls_emitted:
                                finish_reason = "tool_calls"
                            # Flush any remaining buffered content (incomplete DSML or trailing text)
                            if _dsml_buffer and not _tool_calls_emitted:
                                # Try fuzzy parse first — the model may have emitted
                                # an incomplete DSML block (no close tag) before the
                                # stream ended.  This extracts any valid tool calls
                                # and prevents raw DSML tags from leaking as content.
                                if _contains_dsml(_dsml_buffer):
                                    logger.warning("Incomplete DSML at stream finish (%d chars) — fuzzy parsing", len(_dsml_buffer))
                                    parsed = _fuzzy_parse_dsml(_dsml_buffer)
                                    if parsed:
                                        yield self._make_openai_chunk(client_model, chunk_id=_stream_id,
                                            tool_calls=parsed)
                                        _tool_calls_emitted = True
                                        finish_reason = "tool_calls"
                                    else:
                                        # Fuzzy parse found nothing — strip DSML tags
                                        # and emit only the cleaned text (if any)
                                        cleaned = _strip_dsml_from_content(_dsml_buffer)
                                        if cleaned:
                                            yield self._make_openai_chunk(client_model, chunk_id=_stream_id, content=cleaned)
                                else:
                                    yield self._make_openai_chunk(client_model, chunk_id=_stream_id, content=_dsml_buffer)
                                _dsml_buffer = ""
                            # Send final chunk with finish_reason
                            yield self._make_openai_chunk(client_model, chunk_id=_stream_id, finish_reason=finish_reason)

                        elif etype == "error":
                            msg = event.get("error", {}).get("message", "unknown error")
                            logger.error("Command Code stream error: %s", msg)
                            # Send error as content then stop
                            yield self._make_openai_chunk(client_model, chunk_id=_stream_id, content=f"[Error: {msg}]")
                            yield self._make_openai_chunk(client_model, chunk_id=_stream_id, finish_reason="error")

                    # Compute metrics
                    estimated_content_tokens = max(1, content_chars // 4) if content_chars else 0
                    estimated_reasoning_tokens = max(1, reasoning_chars // 4) if reasoning_chars else 0
                    final_tokens = completion_tokens or (estimated_content_tokens + estimated_reasoning_tokens)
                    elapsed = time.time() - start
                    ttft = (first_token_time - start) if first_token_time else None
                    _MIN_GENERATION_TIME = 0.05
                    if ttft is not None and (elapsed - ttft) > _MIN_GENERATION_TIME:
                        generation_time = elapsed - ttft
                    else:
                        generation_time = elapsed

                    metrics = {
                        "eval_count": final_tokens,
                        "prompt_eval_count": prompt_tokens,
                        "reasoning_tokens": estimated_reasoning_tokens,
                        "elapsed_seconds": round(elapsed, 3),
                        "ttft_seconds": round(ttft, 3) if ttft else None,
                    }
                    if final_tokens and generation_time > 0:
                        raw_tps = final_tokens / generation_time
                        metrics["tps"] = round(min(raw_tps, 1000.0), 2)

                    yield self._build_usage_chunk(
                        client_model,
                        metrics.get("prompt_eval_count", 0),
                        metrics.get("eval_count", 0),
                        metrics.get("reasoning_tokens", 0),
                        chunk_id=_stream_id,
                    )
                    yield "data: [DONE]\n\n"
                    yield f"__oct_metrics__:{json.dumps(metrics)}\n\n"
        except httpx.HTTPStatusError as e:
            # Log the actual response body for debugging, then re-raise so the
            # router's failover logic can catch it and try the next provider.
            body_preview = ""
            try:
                body_preview = e.response.text[:500]
            except Exception:
                pass
            logger.error("Command Code HTTP %d: %s — body: %s", e.response.status_code, e, body_preview)
            raise