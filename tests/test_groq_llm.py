import json
from types import SimpleNamespace
import httpx
import pytest

from support_agent.llm_client import (
    DEFAULT_GROQ_MODEL,
    OpenAICompatibleClient,
    get_llm_client,
    to_openai_messages,
    to_openai_tools,
)
from support_agent.settings import AgentSettings


def test_to_openai_tools():
    anthropic_tools = [
        {
            "name": "search_help_articles",
            "description": "Search the help centre",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]
    converted = to_openai_tools(anthropic_tools)
    assert len(converted) == 1
    assert converted[0]["type"] == "function"
    assert converted[0]["function"]["name"] == "search_help_articles"
    assert converted[0]["function"]["parameters"]["properties"]["query"]["type"] == "string"


def test_to_openai_messages():
    messages = [
        {"role": "user", "content": "Hello store"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check that."},
                {"type": "tool_use", "id": "t_1", "name": "search_help_articles", "input": {"query": "returns"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t_1", "content": '{"articles": ["7-day return policy"]}'}
            ],
        },
    ]
    oa_msgs = to_openai_messages("You are a helpful assistant.", messages)
    assert oa_msgs[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert oa_msgs[1] == {"role": "user", "content": "Hello store"}
    assert oa_msgs[2]["role"] == "assistant"
    assert oa_msgs[2]["content"] == "Let me check that."
    assert oa_msgs[2]["tool_calls"][0]["id"] == "t_1"
    assert json.loads(oa_msgs[2]["tool_calls"][0]["function"]["arguments"]) == {"query": "returns"}
    assert oa_msgs[3] == {
        "role": "tool",
        "tool_call_id": "t_1",
        "content": '{"articles": ["7-day return policy"]}',
    }


def test_openai_compatible_client_tool_response():
    def mock_handler(request: httpx.Request):
        req_data = json.loads(request.content)
        assert req_data["model"] == "llama-3.3-70b-versatile"
        assert req_data["messages"][0]["role"] == "system"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {
                                        "name": "search_help_articles",
                                        "arguments": '{"query": "shipping"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(mock_handler))
    client = OpenAICompatibleClient(api_key="gsk_test", http=http)
    resp = client.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1000,
        system="Test system",
        tools=[{"name": "search_help_articles", "input_schema": {}}],
        messages=[{"role": "user", "content": "shipping policy"}],
    )
    assert resp.stop_reason == "tool_use"
    assert len(resp.content) == 1
    call = resp.content[0]
    assert call.type == "tool_use"
    assert call.id == "call_123"
    assert call.name == "search_help_articles"
    assert call.input == {"query": "shipping"}


def test_openai_compatible_client_text_response():
    def mock_handler(request: httpx.Request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Orders arrive within 1-3 days.",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(mock_handler))
    client = OpenAICompatibleClient(api_key="gsk_test", http=http)
    resp = client.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1000,
        system="Test system",
        tools=[],
        messages=[{"role": "user", "content": "shipping"}],
    )
    assert resp.stop_reason == "end_turn"
    assert len(resp.content) == 1
    assert resp.content[0].type == "text"
    assert resp.content[0].text == "Orders arrive within 1-3 days."


def test_get_llm_client_groq_autodetect(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env_key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = AgentSettings()
    client = get_llm_client(s)
    assert isinstance(client, OpenAICompatibleClient)
    assert client.api_key == "gsk_env_key"
    assert client.base_url == "https://api.groq.com/openai/v1"
    assert s.model == "llama-3.3-70b-versatile"
