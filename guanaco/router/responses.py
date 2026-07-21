"""OpenAI Responses API (/v1/responses) endpoint.

Translates between the Responses API format and Guanaco's existing
Chat Completions pipeline, so all providers (Ollama, Cline, UMANS,
CmdCode, OpenCode Go, custom) work transparently.

Key design:
- Input conversion: Responses `input` → Chat Completions `messages`
- Output conversion: Chat Completions response → Responses `output` array
  with proper reasoning items (GLM-5.2's `reasoning_content` → reasoning items)
- Streaming: translates OpenAI SSE chunks into Responses API events:
  response.created, response.output_item.added, response.output_text.delta,
  response.output_text.done, response.reasoning_summary_part.added,
  response.reasoning_summary_text.delta, response.reasoning_summary_text.done,
  response.output_item.done, response.completed

This makes Guanaco a drop-in for tools that only speak Responses API
(notably the Codex CLI).
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional

from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


log = __import__("logging").getLogger("guanaco.router.responses")


# ── Request model ──

class ResponsesRequest(BaseModel):
    """POST /v1/responses request body.

    Only the fields Guanaco needs are modeled; unknown fields are accepted
    (Pydantic default) and ignored.  This matches how the Chat Completions
    endpoint already handles `extra_body`.
    """
    model: str
    input: str | list[Any]
    instructions: Optional[str | list[Any]] = None
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[str | dict] = None
    reasoning: Optional[dict] = None  # {"effort": "low"|"medium"|"high"|"minimal"}
    metadata: Optional[dict] = None
    store: Optional[bool] = None
    previous_response_id: Optional[str] = None
    # Allow any extra fields (service_tier, user, etc.) — they're ignored
    model_config = {"extra": "allow"}


# ── Input → Chat Completions conversion ──

def _input_to_messages(body: ResponsesRequest) -> list[dict]:
    """Convert Responses API `input` + `instructions` to OpenAI chat messages."""
    messages: list[dict] = []

    # instructions → system message
    if body.instructions:
        if isinstance(body.instructions, str):
            messages.append({"role": "system", "content": body.instructions})
        elif isinstance(body.instructions, list):
            # Developer/system instructions can be content arrays
            text = _extract_content_text(body.instructions)
            if text:
                messages.append({"role": "system", "content": text})

    # input can be a simple string or an array of input items
    if isinstance(body.input, str):
        messages.append({"role": "user", "content": body.input})
    elif isinstance(body.input, list):
        for item in body.input:
            if not isinstance(item, dict):
                # Bare string in the array → user message
                messages.append({"role": "user", "content": str(item)})
                continue

            item_type = item.get("type", "message")

            if item_type == "message":
                role = item.get("role", "user")
                content = item.get("content")
                if isinstance(content, str):
                    messages.append({"role": role, "content": content})
                elif isinstance(content, list):
                    # Content array: extract text from input_text/output_text blocks
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict):
                            part_type = part.get("type", "")
                            if part_type in ("input_text", "output_text", "text"):
                                text_parts.append(part.get("text", ""))
                            elif part_type == "input_image":
                                # Pass through as image_url for vision
                                if part.get("image_url"):
                                    messages.append({"role": role, "content": [part]})
                                elif part.get("image_base64"):
                                    # Convert to image_url format
                                    b64 = part["image_base64"]
                                    mime = part.get("media_type", "image/png")
                                    messages.append({"role": role, "content": [
                                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                                    ]})
                            elif part_type == "output_image":
                                # Skip images in output for now
                                pass
                        elif isinstance(part, str):
                            text_parts.append(part)
                    if text_parts:
                        text = "\n".join(text_parts)
                        messages.append({"role": role, "content": text})
                elif content is None:
                    # Skip empty messages
                    pass
                else:
                    messages.append({"role": role, "content": str(content)})

            elif item_type == "function_call":
                # Tool call from a previous turn — represent as an assistant message
                # with tool_calls so the model sees the conversation history
                func = item.get("name", "")
                args = item.get("arguments", "{}")
                call_id = item.get("call_id", item.get("id", f"call_{uuid.uuid4().hex[:24]}"))
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {"name": func, "arguments": args},
                    }],
                })

            elif item_type == "function_call_output":
                # Tool result — represent as a tool message
                call_id = item.get("call_id", "")
                output = item.get("output", "")
                if isinstance(output, dict):
                    output = json.dumps(output)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": str(output),
                })

            elif item_type in ("reasoning",):
                # Skip reasoning items in input — they're context from a previous
                # turn that the model doesn't need repeated
                pass

            else:
                # Unknown item type — try to extract as a message
                role = item.get("role", "user")
                content = item.get("content", item.get("text", ""))
                if content:
                    messages.append({"role": role, "content": str(content)})

    return messages


def _extract_content_text(content: list) -> str:
    """Extract plain text from a Responses content array."""
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            text = part.get("text", part.get("content", ""))
            if text:
                parts.append(text)
    return "\n".join(parts)


def _build_chat_payload(body: ResponsesRequest, resolved_model: str) -> dict:
    """Build an OpenAI Chat Completions payload from a Responses request."""
    messages = _input_to_messages(body)
    payload: dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "stream": body.stream,
    }
    if body.temperature is not None:
        payload["temperature"] = body.temperature
    if body.top_p is not None:
        payload["top_p"] = body.top_p
    if body.max_output_tokens is not None:
        payload["max_tokens"] = body.max_output_tokens

    # Reasoning effort
    if body.reasoning and isinstance(body.reasoning, dict):
        effort = body.reasoning.get("effort")
        if effort:
            payload["reasoning_effort"] = effort

    # Tools — Responses API tools have a slightly different shape
    if body.tools:
        openai_tools = []
        for tool in body.tools:
            if not isinstance(tool, dict):
                continue
            t_type = tool.get("type", "function")
            if t_type == "function":
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", tool.get("input_schema", {})),
                    },
                })
            elif t_type == "file_search" or t_type == "web_search":
                # Skip non-function tools — Guanaco doesn't implement them
                pass
            else:
                # Skip non-standard tool types (e.g. Codex's "namespace",
                # "web_search" with external_web_access, etc.) — upstream
                # providers reject them with 400. Only forward type: "function".
                pass
        if openai_tools:
            payload["tools"] = openai_tools

    if body.tool_choice is not None:
        # Responses tool_choice uses the same values as Chat Completions
        payload["tool_choice"] = body.tool_choice

    return payload


# ── Chat Completions → Responses conversion (non-streaming) ──

def _chat_response_to_responses(
    chat_resp: dict,
    request_model: str,
    response_id: str,
) -> dict:
    """Convert an OpenAI Chat Completions response to a Responses API response."""
    choices = chat_resp.get("choices", [])
    usage = chat_resp.get("usage", {})

    output: list[dict] = []
    status = "completed"

    if choices:
        choice = choices[0]
        msg = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")

        # Build reasoning item if reasoning_content present
        reasoning_text = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if reasoning_text and isinstance(reasoning_text, str) and reasoning_text.strip():
            reasoning_id = f"rs_{uuid.uuid4().hex[:24]}"
            output.append({
                "id": reasoning_id,
                "type": "reasoning",
                "summary": [
                    {"type": "summary_text", "text": reasoning_text},
                ],
            })

        # Build message item
        content_text = msg.get("content", "")
        tool_calls = msg.get("tool_calls")

        content_blocks: list[dict] = []
        if content_text and isinstance(content_text, str) and content_text.strip():
            content_blocks.append({
                "type": "output_text",
                "text": content_text,
                "annotations": [],
            })
        elif content_text and isinstance(content_text, list):
            # Already content blocks (shouldn't happen with our providers, but handle it)
            for block in content_text:
                if isinstance(block, dict) and block.get("type") in ("text", "output_text"):
                    content_blocks.append({
                        "type": "output_text",
                        "text": block.get("text", ""),
                        "annotations": [],
                    })

        # Tool calls → function_call output items
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                call_id = tc.get("id", f"call_{uuid.uuid4().hex[:24]}")
                output.append({
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "type": "function_call",
                    "call_id": call_id,
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", "{}"),
                })

        # Always include a message item (even if empty) unless we only have tool calls
        if content_blocks or not tool_calls:
            msg_id = f"msg_{uuid.uuid4().hex[:24]}"
            message_item = {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": content_blocks if content_blocks else [{"type": "output_text", "text": "", "annotations": []}],
            }
            # Insert message after reasoning but before function calls
            if tool_calls:
                # Insert message before function_call items
                insert_at = len(output)
                # Find first function_call and insert before it
                for i, item in enumerate(output):
                    if item.get("type") == "function_call":
                        insert_at = i
                        break
                output.insert(insert_at, message_item)
            else:
                output.append(message_item)

        # Map finish_reason to status
        if finish_reason == "length":
            status = "incomplete"
        elif finish_reason == "content_filter":
            status = "incomplete"

    # Ensure at least an empty message item
    if not output:
        output.append({
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "", "annotations": []}],
        })

    # Build usage with reasoning token details if available
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

    response = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": request_model,
        "output": output,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
    }

    # Include any extra fields from the chat response that map to Responses
    # (e.g. system_fingerprint)
    if chat_resp.get("system_fingerprint"):
        response["system_fingerprint"] = chat_resp["system_fingerprint"]

    return response


def _next_output_index(reasoning_started: bool, message_started: bool) -> int:
    """Compute the next output_index for a new output item.

    Items are emitted in order: reasoning (0) → message (1) → tool calls (2+).
    Returns the index for the next item based on what has already been started.
    """
    idx = 0
    if reasoning_started:
        idx += 1
    if message_started:
        idx += 1
    return idx


# ── Streaming: SSE event translator ──

def _sse_event(event_type: str, data: dict) -> str:
    """Format a single Responses API SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def _stream_responses(
    client,
    payload: dict,
    model: str,
    analytics,
    start_time: float,
    config=None,
    history_kwargs: dict | None = None,
):
    """Stream Responses API SSE events by translating from Chat Completions stream.

    Event sequence:
    1. response.created          — initial response object (status: in_progress)
    2. response.output_item.added — reasoning item added (if reasoning starts)
    3. response.reasoning_summary_part.added
    4. response.reasoning_summary_text.delta (×N)
    5. response.reasoning_summary_text.done
    6. response.output_item.done   — reasoning item done
    7. response.output_item.added  — message item added
    8. response.content_part.added
    9. response.output_text.delta (×N)
    10. response.output_text.done
    11. response.output_item.done  — message item done
    12. response.completed         — final response with full output + usage
    """
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    reasoning_id = f"rs_{uuid.uuid4().hex[:24]}"
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())

    # Track accumulated content for the final response
    accumulated_content: list[str] = []
    accumulated_reasoning: list[str] = []
    stream_metrics: dict = {}
    finish_reason: str | None = None
    final_usage: dict = {}

    # Output items accumulated during streaming (used in response.completed)
    output: list[dict] = []

    # State: what output items have been started
    reasoning_item_started = False
    reasoning_summary_part_started = False
    message_item_started = False
    content_part_started = False
    message_item_closed = False  # Set when tool call handler closes message inline

    # Tool call tracking: {index: {"id": call_id, "name": name, "arguments": ""}}
    tool_calls_state: dict[int, dict] = {}
    # Order in which tool calls first appeared (for output_index assignment)
    tool_call_indices: list[int] = []

    async def generate():
        nonlocal reasoning_item_started, reasoning_summary_part_started
        nonlocal message_item_started, content_part_started, message_item_closed
        nonlocal finish_reason, final_usage, stream_metrics, output

        # 1. response.created
        base_response = {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "in_progress",
            "model": model,
            "output": [],
        }
        yield _sse_event("response.created", {
            "type": "response.created",
            "response": base_response,
        })

        try:
            async for chunk in client.chat_completion_stream(payload):
                # Capture internal metrics
                if chunk.startswith("__oct_metrics__:"):
                    try:
                        stream_metrics = json.loads(chunk.split(":", 1)[1])
                    except (json.JSONDecodeError, ValueError):
                        pass
                    continue

                # Parse OpenAI SSE chunk
                if not chunk.startswith("data: "):
                    continue
                data_str = chunk[6:].strip()
                if data_str == "[DONE]":
                    continue

                try:
                    data = json.loads(data_str)
                except (json.JSONDecodeError, ValueError):
                    continue

                choices = data.get("choices", [])
                if not choices:
                    # Check for usage in the final chunk
                    if data.get("usage"):
                        final_usage = data["usage"]
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})

                # Check for finish_reason
                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = fr

                # Capture usage from final chunk
                if data.get("usage"):
                    final_usage = data["usage"]

                # ── Reasoning content ──
                reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if reasoning_delta and isinstance(reasoning_delta, str):
                    if not reasoning_item_started:
                        # Start reasoning output item
                        yield _sse_event("response.output_item.added", {
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {
                                "id": reasoning_id,
                                "type": "reasoning",
                                "summary": [],
                            },
                        })
                        reasoning_item_started = True

                    if not reasoning_summary_part_started:
                        yield _sse_event("response.reasoning_summary_part.added", {
                            "type": "response.reasoning_summary_part.added",
                            "output_index": 0,
                            "summary_index": 0,
                            "part": {
                                "type": "summary_text",
                                "text": "",
                            },
                        })
                        reasoning_summary_part_started = True

                    accumulated_reasoning.append(reasoning_delta)
                    yield _sse_event("response.reasoning_summary_text.delta", {
                        "type": "response.reasoning_summary_text.delta",
                        "output_index": 0,
                        "summary_index": 0,
                        "delta": reasoning_delta,
                    })

                # ── Content text ──
                content_delta = delta.get("content", "")
                if content_delta and isinstance(content_delta, str):
                    # If reasoning was in progress, close it out and start message item
                    if reasoning_item_started and not message_item_started:
                        # Close reasoning summary text
                        full_reasoning = "".join(accumulated_reasoning)
                        yield _sse_event("response.reasoning_summary_text.done", {
                            "type": "response.reasoning_summary_text.done",
                            "output_index": 0,
                            "summary_index": 0,
                            "text": full_reasoning,
                        })
                        # Close reasoning output item
                        yield _sse_event("response.output_item.done", {
                            "type": "response.output_item.done",
                            "output_index": 0,
                            "item": {
                                "id": reasoning_id,
                                "type": "reasoning",
                                "summary": [{"type": "summary_text", "text": full_reasoning}],
                            },
                        })

                    if not message_item_started:
                        # Start message output item
                        output_index = 1 if reasoning_item_started else 0
                        yield _sse_event("response.output_item.added", {
                            "type": "response.output_item.added",
                            "output_index": output_index,
                            "item": {
                                "id": msg_id,
                                "type": "message",
                                "role": "assistant",
                                "status": "in_progress",
                                "content": [],
                            },
                        })
                        message_item_started = True

                        # Start content part
                        yield _sse_event("response.content_part.added", {
                            "type": "response.content_part.added",
                            "output_index": output_index,
                            "content_index": 0,
                            "part": {
                                "type": "output_text",
                                "text": "",
                                "annotations": [],
                            },
                        })
                        content_part_started = True

                    accumulated_content.append(content_delta)
                    output_index = 1 if reasoning_item_started else 0
                    yield _sse_event("response.output_text.delta", {
                        "type": "response.output_text.delta",
                        "output_index": output_index,
                        "content_index": 0,
                        "delta": content_delta,
                    })

                # ── Tool calls in delta ──
                tool_calls_delta = delta.get("tool_calls")
                if tool_calls_delta:
                    for tc_delta in tool_calls_delta:
                        if not isinstance(tc_delta, dict):
                            continue
                        tc_index = tc_delta.get("index", 0)

                        # First chunk for this tool call: has id + function.name
                        if tc_index not in tool_calls_state:
                            tc_id = tc_delta.get("id", f"call_{uuid.uuid4().hex[:24]}")
                            tc_func = tc_delta.get("function", {})
                            tc_name = tc_func.get("name", "")
                            tool_calls_state[tc_index] = {
                                "id": tc_id,
                                "name": tc_name,
                                "arguments": "",
                            }
                            tool_call_indices.append(tc_index)

                            # Close content/reasoning items before starting tool calls
                            if reasoning_item_started and not message_item_started:
                                full_reasoning = "".join(accumulated_reasoning)
                                yield _sse_event("response.reasoning_summary_text.done", {
                                    "type": "response.reasoning_summary_text.done",
                                    "output_index": 0,
                                    "summary_index": 0,
                                    "text": full_reasoning,
                                })
                                yield _sse_event("response.output_item.done", {
                                    "type": "response.output_item.done",
                                    "output_index": 0,
                                    "item": {
                                        "id": reasoning_id,
                                        "type": "reasoning",
                                        "summary": [{"type": "summary_text", "text": full_reasoning}],
                                    },
                                })

                            if message_item_started and content_part_started:
                                # Close message content before tool calls
                                full_content = "".join(accumulated_content)
                                msg_output_index = 1 if reasoning_item_started else 0
                                yield _sse_event("response.output_text.done", {
                                    "type": "response.output_text.done",
                                    "output_index": msg_output_index,
                                    "content_index": 0,
                                    "text": full_content,
                                })
                                yield _sse_event("response.content_part.done", {
                                    "type": "response.content_part.done",
                                    "output_index": msg_output_index,
                                    "content_index": 0,
                                    "part": {
                                        "type": "output_text",
                                        "text": full_content,
                                        "annotations": [],
                                    },
                                })
                                yield _sse_event("response.output_item.done", {
                                    "type": "response.output_item.done",
                                    "output_index": msg_output_index,
                                    "item": {
                                        "id": msg_id,
                                        "type": "message",
                                        "role": "assistant",
                                        "status": "completed",
                                        "content": [{"type": "output_text", "text": full_content, "annotations": []}],
                                    },
                                })
                                message_item_closed = True
                                # Add closed message to output array
                                output.append({
                                    "id": msg_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "completed",
                                    "content": [{"type": "output_text", "text": full_content, "annotations": []}],
                                })

                            # Compute output_index for this tool call
                            fc_output_index = _next_output_index(
                                reasoning_item_started, message_item_started,
                            ) + len(tool_call_indices) - 1

                            fc_item_id = f"fc_{uuid.uuid4().hex[:24]}"
                            tool_calls_state[tc_index]["item_id"] = fc_item_id
                            tool_calls_state[tc_index]["output_index"] = fc_output_index

                            yield _sse_event("response.output_item.added", {
                                "type": "response.output_item.added",
                                "output_index": fc_output_index,
                                "item": {
                                    "id": fc_item_id,
                                    "type": "function_call",
                                    "call_id": tc_id,
                                    "name": tc_name,
                                    "arguments": "",
                                    "status": "in_progress",
                                },
                            })

                        # Argument fragment (may come in same chunk or subsequent chunks)
                        arg_delta = tc_delta.get("function", {}).get("arguments", "")
                        if arg_delta:
                            tool_calls_state[tc_index]["arguments"] += arg_delta
                            fc_output_index = tool_calls_state[tc_index]["output_index"]
                            yield _sse_event("response.function_call_arguments.delta", {
                                "type": "response.function_call_arguments.delta",
                                "output_index": fc_output_index,
                                "item_id": tool_calls_state[tc_index]["item_id"],
                                "delta": arg_delta,
                            })

        except Exception as e:
            # Emit error event
            error_msg = str(e)
            yield _sse_event("response.failed", {
                "type": "response.failed",
                "response": {
                    "id": response_id,
                    "status": "failed",
                    "error": {"message": error_msg, "type": "server_error"},
                },
            })
            # Log analytics
            if analytics:
                _hist_kw = dict(history_kwargs) if history_kwargs else {}
                analytics.log_llm(
                    model=model,
                    error=error_msg,
                    total_duration_seconds=time.time() - start_time,
                    **_hist_kw,
                )
            return

        # ── Close out streaming items ──
        full_content = "".join(accumulated_content)
        full_reasoning = "".join(accumulated_reasoning)

        # Add reasoning item to output if reasoning was emitted
        if reasoning_item_started and full_reasoning:
            output.append({
                "id": reasoning_id,
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": full_reasoning}],
            })

        # Close reasoning if it was started and not already closed (no content followed)
        if reasoning_item_started and not message_item_started:
            # Reasoning was the only output (no content followed)
            yield _sse_event("response.reasoning_summary_text.done", {
                "type": "response.reasoning_summary_text.done",
                "output_index": 0,
                "summary_index": 0,
                "text": full_reasoning,
            })
            yield _sse_event("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": reasoning_id,
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": full_reasoning}],
                },
            })

        # Close message item if it was started and not already closed by tool call handler
        if message_item_started and not message_item_closed:
            output_index = 1 if reasoning_item_started else 0
            # Close content part
            yield _sse_event("response.output_text.done", {
                "type": "response.output_text.done",
                "output_index": output_index,
                "content_index": 0,
                "text": full_content,
            })
            yield _sse_event("response.content_part.done", {
                "type": "response.content_part.done",
                "output_index": output_index,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": full_content,
                    "annotations": [],
                },
            })
            yield _sse_event("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": full_content, "annotations": []}],
                },
            })
            output.append({
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": full_content, "annotations": []}],
            })

        # Close tool call items and add to output
        for tc_idx in tool_call_indices:
            tc = tool_calls_state[tc_idx]
            fc_output_index = tc["output_index"]
            fc_item_id = tc["item_id"]
            full_args = tc["arguments"]

            yield _sse_event("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "output_index": fc_output_index,
                "item_id": fc_item_id,
                "arguments": full_args,
            })
            yield _sse_event("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": fc_output_index,
                "item": {
                    "id": fc_item_id,
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": tc["name"],
                    "arguments": full_args,
                    "status": "completed",
                },
            })
            output.append({
                "id": fc_item_id,
                "type": "function_call",
                "call_id": tc["id"],
                "name": tc["name"],
                "arguments": full_args,
                "status": "completed",
            })

        # If neither reasoning nor content was emitted, emit an empty message
        if not output:
            yield _sse_event("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
            })
            yield _sse_event("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "", "annotations": []}],
                },
            })
            output.append({
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "", "annotations": []}],
            })

        # ── Build final usage ──
        input_tokens = final_usage.get("prompt_tokens", stream_metrics.get("prompt_eval_count", 0))
        output_tokens = final_usage.get("completion_tokens", stream_metrics.get("eval_count", 0))
        if not output_tokens and accumulated_content:
            output_tokens = max(1, len(full_content) // 4)
        if not input_tokens and stream_metrics.get("prompt_eval_count"):
            input_tokens = stream_metrics["prompt_eval_count"]
        total_tokens = final_usage.get("total_tokens", input_tokens + output_tokens)

        status = "completed"
        if finish_reason == "length":
            status = "incomplete"

        # 12. response.completed
        final_response = {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": status,
            "model": model,
            "output": output,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
        }
        yield _sse_event("response.completed", {
            "type": "response.completed",
            "response": final_response,
        })

        # ── Analytics logging ──
        if analytics:
            _hist_kw = dict(history_kwargs) if history_kwargs else {}
            if (accumulated_content or accumulated_reasoning) and config and config.history.enabled and config.history.save_output:
                parts = []
                if accumulated_content:
                    parts.append("".join(accumulated_content))
                if accumulated_reasoning:
                    parts.append(f"<thinking>\n{''.join(accumulated_reasoning)}\n</thinking>")
                output_text = "\n".join(parts)
                if len(output_text) > config.history.max_content_size:
                    output_text = output_text[:config.history.max_content_size] + "\n...[truncated]"
                _hist_kw["output_text"] = output_text
            analytics.log_llm(
                model=model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=total_tokens,
                tps=stream_metrics.get("tps"),
                ttft_seconds=stream_metrics.get("ttft_seconds"),
                total_duration_seconds=stream_metrics.get("elapsed_seconds", time.time() - start_time),
                **_hist_kw,
            )

    return StreamingResponse(generate(), media_type="text/event-stream")
