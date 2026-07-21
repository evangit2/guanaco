"""Tests for the OpenAI Responses API (/v1/responses) endpoint.

Tests the translation layer between Responses API format and Guanaco's
Chat Completions pipeline: input conversion, output conversion, streaming
event translation, and reasoning item handling.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from guanaco.router.responses import (
    ResponsesRequest,
    _input_to_messages,
    _build_chat_payload,
    _chat_response_to_responses,
    _stream_responses,
    _sse_event,
)


# ── Input conversion tests ──

class TestInputConversion:
    """Test Responses API input → Chat Completions messages conversion."""

    def test_simple_string_input(self):
        body = ResponsesRequest(model="glm-5.2", input="Hello, world!")
        messages = _input_to_messages(body)
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello, world!"}

    def test_string_input_with_instructions(self):
        body = ResponsesRequest(
            model="glm-5.2",
            input="What is 2+2?",
            instructions="You are a helpful math tutor.",
        )
        messages = _input_to_messages(body)
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are a helpful math tutor."}
        assert messages[1] == {"role": "user", "content": "What is 2+2?"}

    def test_message_array_input(self):
        body = ResponsesRequest(
            model="glm-5.2",
            input=[
                {"type": "message", "role": "user", "content": "Hi"},
                {"type": "message", "role": "assistant", "content": "Hello!"},
                {"type": "message", "role": "user", "content": "How are you?"},
            ],
        )
        messages = _input_to_messages(body)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hi"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "How are you?"

    def test_content_array_with_input_text(self):
        body = ResponsesRequest(
            model="glm-5.2",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What's in this image?"},
                        {"type": "input_text", "text": "Please describe it."},
                    ],
                },
            ],
        )
        messages = _input_to_messages(body)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "What's in this image?" in messages[0]["content"]
        assert "Please describe it." in messages[0]["content"]

    def test_function_call_in_input(self):
        body = ResponsesRequest(
            model="glm-5.2",
            input=[
                {"type": "message", "role": "user", "content": "What's the weather?"},
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "get_weather",
                    "arguments": '{"city": "NYC"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": "Sunny, 72°F",
                },
            ],
        )
        messages = _input_to_messages(body)
        # user message + assistant tool_call + tool result
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert messages[2]["role"] == "tool"
        assert messages[2]["content"] == "Sunny, 72°F"

    def test_reasoning_items_skipped_in_input(self):
        body = ResponsesRequest(
            model="glm-5.2",
            input=[
                {"type": "message", "role": "user", "content": "Hi"},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking..."}]},
                {"type": "message", "role": "assistant", "content": "Hello!"},
            ],
        )
        messages = _input_to_messages(body)
        # reasoning item should be skipped — only 2 messages
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_bare_string_in_input_array(self):
        body = ResponsesRequest(
            model="glm-5.2",
            input=["Just a string"],
        )
        messages = _input_to_messages(body)
        assert len(messages) == 1
        assert messages[0]["content"] == "Just a string"


# ── Payload building tests ──

class TestPayloadBuilding:
    """Test that the Chat Completions payload is built correctly."""

    def test_basic_payload(self):
        body = ResponsesRequest(model="glm-5.2", input="Hello")
        payload = _build_chat_payload(body, "glm-5.2")
        assert payload["model"] == "glm-5.2"
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]
        assert payload["stream"] is False

    def test_reasoning_effort(self):
        body = ResponsesRequest(
            model="glm-5.2",
            input="Think carefully",
            reasoning={"effort": "high"},
        )
        payload = _build_chat_payload(body, "glm-5.2")
        assert payload["reasoning_effort"] == "high"

    def test_max_output_tokens_mapped(self):
        body = ResponsesRequest(
            model="glm-5.2",
            input="Hello",
            max_output_tokens=1024,
        )
        payload = _build_chat_payload(body, "glm-5.2")
        assert payload["max_tokens"] == 1024

    def test_tools_conversion(self):
        body = ResponsesRequest(
            model="glm-5.2",
            input="What's the weather?",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            ],
        )
        payload = _build_chat_payload(body, "glm-5.2")
        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["type"] == "function"
        assert payload["tools"][0]["function"]["name"] == "get_weather"
        assert payload["tools"][0]["function"]["parameters"]["properties"]["city"]["type"] == "string"

    def test_temperature_and_top_p(self):
        body = ResponsesRequest(
            model="glm-5.2",
            input="Hello",
            temperature=0.7,
            top_p=0.9,
        )
        payload = _build_chat_payload(body, "glm-5.2")
        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.9


# ── Output conversion tests ──

class TestOutputConversion:
    """Test Chat Completions response → Responses API response conversion."""

    def test_basic_text_response(self):
        chat_resp = {
            "id": "chatcmpl-123",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello! How can I help you?"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }
        result = _chat_response_to_responses(chat_resp, "glm-5.2", "resp_abc")

        assert result["id"] == "resp_abc"
        assert result["object"] == "response"
        assert result["status"] == "completed"
        assert result["model"] == "glm-5.2"

        # Should have one message item with output_text
        assert len(result["output"]) == 1
        msg_item = result["output"][0]
        assert msg_item["type"] == "message"
        assert msg_item["role"] == "assistant"
        assert len(msg_item["content"]) == 1
        assert msg_item["content"][0]["type"] == "output_text"
        assert msg_item["content"][0]["text"] == "Hello! How can I help you?"

        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 8
        assert result["usage"]["total_tokens"] == 18

    def test_reasoning_content_becomes_reasoning_item(self):
        """GLM-5.2 emits reasoning_content — it should become a reasoning item."""
        chat_resp = {
            "id": "chatcmpl-123",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The answer is 4.",
                    "reasoning_content": "Let me think... 2+2=4",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        result = _chat_response_to_responses(chat_resp, "glm-5.2", "resp_abc")

        # Should have reasoning item + message item
        assert len(result["output"]) == 2

        # Reasoning item first
        reasoning_item = result["output"][0]
        assert reasoning_item["type"] == "reasoning"
        assert len(reasoning_item["summary"]) == 1
        assert reasoning_item["summary"][0]["type"] == "summary_text"
        assert reasoning_item["summary"][0]["text"] == "Let me think... 2+2=4"

        # Message item second
        msg_item = result["output"][1]
        assert msg_item["type"] == "message"
        assert msg_item["content"][0]["text"] == "The answer is 4."

    def test_reasoning_only_no_content(self):
        """When the model only emits reasoning_content (no content), still produce valid output."""
        chat_resp = {
            "id": "chatcmpl-123",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Just thinking, no answer yet.",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        result = _chat_response_to_responses(chat_resp, "glm-5.2", "resp_abc")

        # Should have reasoning + empty message
        assert len(result["output"]) == 2
        assert result["output"][0]["type"] == "reasoning"
        assert result["output"][1]["type"] == "message"

    def test_tool_calls_in_response(self):
        chat_resp = {
            "id": "chatcmpl-123",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_456",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = _chat_response_to_responses(chat_resp, "glm-5.2", "resp_abc")

        # Should have a message item (empty) + function_call item
        types = [item["type"] for item in result["output"]]
        assert "function_call" in types
        fc_item = [item for item in result["output"] if item["type"] == "function_call"][0]
        assert fc_item["name"] == "get_weather"
        assert fc_item["arguments"] == '{"city": "NYC"}'
        assert fc_item["call_id"] == "call_456"

    def test_empty_response(self):
        chat_resp = {
            "id": "chatcmpl-123",
            "choices": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        result = _chat_response_to_responses(chat_resp, "glm-5.2", "resp_abc")

        # Should have at least an empty message item
        assert len(result["output"]) >= 1
        assert result["output"][0]["type"] == "message"

    def test_length_finish_reason_maps_to_incomplete(self):
        chat_resp = {
            "id": "chatcmpl-123",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Truncated..."},
                "finish_reason": "length",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 100, "total_tokens": 110},
        }
        result = _chat_response_to_responses(chat_resp, "glm-5.2", "resp_abc")
        assert result["status"] == "incomplete"


# ── Streaming event tests ──

class TestStreamingEvents:
    """Test that streaming produces correct Responses API SSE events."""

    def test_sse_event_format(self):
        evt = _sse_event("response.created", {"type": "response.created", "response": {"id": "resp_123"}})
        assert evt.startswith("event: response.created\n")
        assert "data: " in evt
        assert evt.endswith("\n\n")
        data = json.loads(evt.split("data: ", 1)[1].strip())
        assert data["type"] == "response.created"
        assert data["response"]["id"] == "resp_123"

    @pytest.mark.asyncio
    async def test_streaming_text_only(self):
        """Test streaming with content only (no reasoning)."""
        # Mock client that yields OpenAI SSE chunks
        chunks = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":", world!"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            'data: {"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}',
            'data: [DONE]',
        ]

        async def mock_stream(payload):
            for c in chunks:
                yield c

        mock_client = MagicMock()
        mock_client.chat_completion_stream = mock_stream

        mock_analytics = MagicMock()
        mock_config = MagicMock()
        mock_config.history.enabled = False

        response = await _stream_responses(
            mock_client, {"model": "glm-5.2"}, "glm-5.2",
            mock_analytics, 0.0, config=mock_config,
        )

        # Collect all events
        events = []
        async for chunk in response.body_iterator:
            for line in chunk.split("\n"):
                if line.startswith("event: "):
                    events.append(line[7:])

        # Verify event sequence
        assert "response.created" in events
        assert "response.output_item.added" in events
        assert "response.output_text.delta" in events
        assert "response.output_text.done" in events
        assert "response.output_item.done" in events
        assert "response.completed" in events

        # Verify order: created comes first, completed comes last
        assert events[0] == "response.created"
        assert events[-1] == "response.completed"

    @pytest.mark.asyncio
    async def test_streaming_with_reasoning(self):
        """Test streaming with reasoning_content (GLM-5.2 pattern)."""
        chunks = [
            'data: {"choices":[{"delta":{"reasoning_content":"Let me think..."}}]}',
            'data: {"choices":[{"delta":{"reasoning_content":"2+2=4"}}]}',
            'data: {"choices":[{"delta":{"content":"The answer is 4."}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            'data: [DONE]',
        ]

        async def mock_stream(payload):
            for c in chunks:
                yield c

        mock_client = MagicMock()
        mock_client.chat_completion_stream = mock_stream

        mock_analytics = MagicMock()
        mock_config = MagicMock()
        mock_config.history.enabled = False

        response = await _stream_responses(
            mock_client, {"model": "glm-5.2"}, "glm-5.2",
            mock_analytics, 0.0, config=mock_config,
        )

        events = []
        event_data = []
        async for chunk in response.body_iterator:
            lines = chunk.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("event: "):
                    events.append(line[7:])
                if line.startswith("data: "):
                    try:
                        event_data.append(json.loads(line[6:].strip()))
                    except json.JSONDecodeError:
                        pass

        # Verify reasoning events are present
        assert "response.created" in events
        assert "response.output_item.added" in events
        assert "response.reasoning_summary_part.added" in events
        assert "response.reasoning_summary_text.delta" in events
        assert "response.reasoning_summary_text.done" in events
        assert "response.output_text.delta" in events
        assert "response.output_text.done" in events
        assert "response.completed" in events

        # Verify reasoning deltas
        reasoning_deltas = [
            d for d in event_data
            if d.get("type") == "response.reasoning_summary_text.delta"
        ]
        assert len(reasoning_deltas) == 2
        assert reasoning_deltas[0]["delta"] == "Let me think..."
        assert reasoning_deltas[1]["delta"] == "2+2=4"

        # Verify content deltas
        content_deltas = [
            d for d in event_data
            if d.get("type") == "response.output_text.delta"
        ]
        assert len(content_deltas) == 1
        assert content_deltas[0]["delta"] == "The answer is 4."

        # Verify final response has both reasoning and message items
        completed = [d for d in event_data if d.get("type") == "response.completed"]
        assert len(completed) == 1
        final_output = completed[0]["response"]["output"]
        types = [item["type"] for item in final_output]
        assert "reasoning" in types
        assert "message" in types

    @pytest.mark.asyncio
    async def test_streaming_reasoning_only(self):
        """Test streaming where only reasoning is emitted (no content)."""
        chunks = [
            'data: {"choices":[{"delta":{"reasoning_content":"Just thinking..."}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            'data: [DONE]',
        ]

        async def mock_stream(payload):
            for c in chunks:
                yield c

        mock_client = MagicMock()
        mock_client.chat_completion_stream = mock_stream

        response = await _stream_responses(
            mock_client, {"model": "glm-5.2"}, "glm-5.2",
            None, 0.0, config=MagicMock(),
        )

        events = []
        event_data = []
        async for chunk in response.body_iterator:
            lines = chunk.split("\n")
            for line in lines:
                if line.startswith("event: "):
                    events.append(line[7:])
                if line.startswith("data: "):
                    try:
                        event_data.append(json.loads(line[6:].strip()))
                    except json.JSONDecodeError:
                        pass

        # Should have reasoning events
        assert "response.reasoning_summary_text.delta" in events
        assert "response.reasoning_summary_text.done" in events

        # Final response should have reasoning item
        completed = [d for d in event_data if d.get("type") == "response.completed"]
        assert len(completed) == 1
        output = completed[0]["response"]["output"]
        assert any(item["type"] == "reasoning" for item in output)

    @pytest.mark.asyncio
    async def test_streaming_tool_calls(self):
        """Test streaming with tool calls — the bug that was fixed."""
        # Simulates OpenAI streaming format: first chunk has id+name, then argument fragments
        chunks = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"write_file","arguments":""}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"path\\""}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":": \\"test"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":".py\\""}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":", \\"content\\": \\"print(1)\\"}"}}]}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            'data: [DONE]',
        ]

        async def mock_stream(payload):
            for c in chunks:
                yield c

        mock_client = MagicMock()
        mock_client.chat_completion_stream = mock_stream

        response = await _stream_responses(
            mock_client, {"model": "glm-5.2"}, "glm-5.2",
            None, 0.0, config=MagicMock(),
        )

        events = []
        event_data = []
        async for chunk in response.body_iterator:
            lines = chunk.split("\n")
            for line in lines:
                if line.startswith("event: "):
                    events.append(line[7:])
                if line.startswith("data: "):
                    try:
                        event_data.append(json.loads(line[6:].strip()))
                    except json.JSONDecodeError:
                        pass

        # Verify function_call events are present
        assert "response.output_item.added" in events
        assert "response.function_call_arguments.delta" in events
        assert "response.function_call_arguments.done" in events
        assert "response.output_item.done" in events
        assert "response.completed" in events

        # Verify the function_call item was added with correct type
        added_items = [d for d in event_data if d.get("type") == "response.output_item.added"]
        fc_added = [d for d in added_items if d.get("item", {}).get("type") == "function_call"]
        assert len(fc_added) == 1
        assert fc_added[0]["item"]["name"] == "write_file"
        assert fc_added[0]["item"]["call_id"] == "call_abc"

        # Verify argument deltas
        arg_deltas = [d for d in event_data if d.get("type") == "response.function_call_arguments.delta"]
        assert len(arg_deltas) >= 4  # Multiple argument fragments

        # Verify arguments.done has full arguments
        args_done = [d for d in event_data if d.get("type") == "response.function_call_arguments.done"]
        assert len(args_done) == 1
        full_args = args_done[0]["arguments"]
        assert '"path"' in full_args
        assert "test.py" in full_args
        assert "print(1)" in full_args

        # Verify final response has function_call in output
        completed = [d for d in event_data if d.get("type") == "response.completed"]
        assert len(completed) == 1
        output = completed[0]["response"]["output"]
        fc_items = [item for item in output if item["type"] == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0]["name"] == "write_file"
        assert fc_items[0]["call_id"] == "call_abc"
        assert "test.py" in fc_items[0]["arguments"]

    @pytest.mark.asyncio
    async def test_streaming_reasoning_content_then_tool_call(self):
        """Test streaming with reasoning + content + tool call — full Codex pattern."""
        chunks = [
            'data: {"choices":[{"delta":{"reasoning_content":"I need to create a file."}}]}',
            'data: {"choices":[{"delta":{"content":"Creating file..."}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_xyz","type":"function","function":{"name":"write_file","arguments":""}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"path\\":\\"a.py\\",\\"content\\":\\"x\\"}"}}]}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            'data: [DONE]',
        ]

        async def mock_stream(payload):
            for c in chunks:
                yield c

        mock_client = MagicMock()
        mock_client.chat_completion_stream = mock_stream

        response = await _stream_responses(
            mock_client, {"model": "glm-5.2"}, "glm-5.2",
            None, 0.0, config=MagicMock(),
        )

        event_data = []
        async for chunk in response.body_iterator:
            lines = chunk.split("\n")
            for line in lines:
                if line.startswith("data: "):
                    try:
                        event_data.append(json.loads(line[6:].strip()))
                    except json.JSONDecodeError:
                        pass

        completed = [d for d in event_data if d.get("type") == "response.completed"]
        assert len(completed) == 1
        output = completed[0]["response"]["output"]
        types = [item["type"] for item in output]
        assert "reasoning" in types
        assert "message" in types
        assert "function_call" in types

    @pytest.mark.asyncio
    async def test_streaming_multiple_tool_calls(self):
        """Test streaming with multiple tool calls in one response."""
        chunks = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"write_file","arguments":""}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"path\\":\\"a.py\\"}"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"read_file","arguments":""}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\\"path\\":\\"b.py\\"}"}}]}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            'data: [DONE]',
        ]

        async def mock_stream(payload):
            for c in chunks:
                yield c

        mock_client = MagicMock()
        mock_client.chat_completion_stream = mock_stream

        response = await _stream_responses(
            mock_client, {"model": "glm-5.2"}, "glm-5.2",
            None, 0.0, config=MagicMock(),
        )

        event_data = []
        async for chunk in response.body_iterator:
            lines = chunk.split("\n")
            for line in lines:
                if line.startswith("data: "):
                    try:
                        event_data.append(json.loads(line[6:].strip()))
                    except json.JSONDecodeError:
                        pass

        completed = [d for d in event_data if d.get("type") == "response.completed"]
        assert len(completed) == 1
        output = completed[0]["response"]["output"]
        fc_items = [item for item in output if item["type"] == "function_call"]
        assert len(fc_items) == 2
        names = [fc["name"] for fc in fc_items]
        assert "write_file" in names
        assert "read_file" in names


# ── Request model tests ──

class TestResponsesRequest:
    """Test the ResponsesRequest Pydantic model."""

    def test_minimal_request(self):
        body = ResponsesRequest(model="glm-5.2", input="Hello")
        assert body.model == "glm-5.2"
        assert body.input == "Hello"
        assert body.stream is False

    def test_stream_flag(self):
        body = ResponsesRequest(model="glm-5.2", input="Hello", stream=True)
        assert body.stream is True

    def test_extra_fields_accepted(self):
        """Extra fields like service_tier should be accepted, not rejected."""
        body = ResponsesRequest(
            model="glm-5.2",
            input="Hello",
            service_tier="default",
            user="user_123",
        )
        assert body.model == "glm-5.2"
        # Extra fields should be accessible via model_dump
        dumped = body.model_dump(exclude_none=True)
        assert dumped.get("service_tier") == "default"
        assert dumped.get("user") == "user_123"

    def test_instructions_as_list(self):
        body = ResponsesRequest(
            model="glm-5.2",
            input="Hello",
            instructions=[{"type": "input_text", "text": "Be concise."}],
        )
        # Instructions list should be accepted
        assert body.instructions is not None

    def test_reasoning_dict(self):
        body = ResponsesRequest(
            model="glm-5.2",
            input="Think",
            reasoning={"effort": "high"},
        )
        assert body.reasoning == {"effort": "high"}
