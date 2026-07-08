"""Pydantic request/response models for the Guanaco router.

Extracted from router.py for modularity. All models are re-exported from
``guanaco.router.router`` for backward compatibility.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str | list | None = None
    name: Optional[str] = None
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[list[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[str | dict] = None
    response_format: Optional[dict] = None
    reasoning_effort: Optional[str] = None
    extra_body: Optional[dict] = None


# ── Anthropic Request Models ──

class AnthropicMessage(BaseModel):
    role: str
    content: str | list


class AnthropicRequest(BaseModel):
    model: str
    max_tokens: int = 4096
    messages: list[AnthropicMessage]
    system: Optional[str | list] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    stop_sequences: Optional[list[str]] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[dict] = None
    reasoning_effort: Optional[str] = None
    extra_body: Optional[dict] = None
