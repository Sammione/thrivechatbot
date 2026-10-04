"""
LLM Client adapter supporting Groq, OpenAI-compatible APIs (DeepSeek, OpenRouter), and Anthropic.
"""
from __future__ import annotations

import json
import os
import uuid
from types import SimpleNamespace
from typing import Optional

import httpx

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-latest"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


def to_openai_tools(tools: list) -> list:
    """Convert Anthropic tool schemas to OpenAI/Groq function tool format."""
    out = []
    for t in tools:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return out


def to_openai_messages(system_prompt: str, messages: list) -> list:
    """Convert internal session messages to OpenAI/Groq chat format."""
    out = []
    if system_prompt:
        out.append({"role": "system", "content": system_prompt})

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
            elif isinstance(content, list):
                # Anthropic tool results: list of {"type": "tool_result", "tool_use_id": ..., "content": ...}
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        out.append({
                            "role": "tool",
                            "tool_call_id": item.get("tool_use_id"),
                            "content": item.get("content", ""),
                        })
                    else:
                        out.append({"role": "user", "content": str(item)})
        elif role == "assistant":
            if isinstance(content, str):
                out.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                text_parts = []
                tool_calls = []
                for b in content:
                    b_type = b.get("type")
                    if b_type == "text" and b.get("text"):
                        text_parts.append(b["text"])
                    elif b_type == "tool_use":
                        tool_calls.append({
                            "id": b.get("id"),
                            "type": "function",
                            "function": {
                                "name": b.get("name"),
                                "arguments": json.dumps(b.get("input") or {}),
                            },
                        })
                msg_dict = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                }
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls
                out.append(msg_dict)
        elif role == "system":
            out.append({"role": "system", "content": str(content)})

    return out


class OpenAICompatibleClient:
    """Adapter for Groq, DeepSeek, OpenRouter, and any OpenAI-compatible provider."""

    def __init__(self, api_key: str, base_url: str = "https://api.groq.com/openai/v1", http: Optional[httpx.Client] = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=45.0)
        self.messages = self

    def create(self, model: str, max_tokens: int, system: str, tools: list, messages: list):
        oa_messages = to_openai_messages(system, messages)
        payload = {
            "model": model,
            "messages": oa_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = to_openai_tools(tools)
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        resp = self.http.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
        if not resp.is_success:
            try:
                err_json = resp.json()
                err_msg = err_json.get("error", {}).get("message") or resp.text
            except Exception:
                err_msg = resp.text
            raise RuntimeError(f"API Error ({resp.status_code}): {err_msg}")

        data = resp.json()
        choice = data["choices"][0]
        msg = choice.get("message", {})

        blocks = []
        if msg.get("content"):
            blocks.append(SimpleNamespace(type="text", text=msg["content"]))

        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except Exception:
                args = {}
            blocks.append(SimpleNamespace(
                type="tool_use",
                id=tc.get("id", uuid.uuid4().hex[:8]),
                name=name,
                input=args,
            ))

        stop_reason = "tool_use" if tool_calls else "end_turn"
        return SimpleNamespace(content=blocks, stop_reason=stop_reason)


def get_llm_client(settings):
    """
    Factory creating either an OpenAICompatibleClient (for Groq/DeepSeek/OpenRouter)
    or anthropic.Anthropic client based on settings and environment.
    """
    provider = getattr(settings, "provider", "auto")
    groq_key = os.getenv("GROQ_API_KEY", "") or getattr(settings, "groq_api_key", "")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    custom_base_url = getattr(settings, "openai_base_url", "") or os.getenv("OPENAI_BASE_URL", "")

    # Auto-detect or explicit Groq
    if provider == "groq" or (provider == "auto" and groq_key):
        if not groq_key:
            raise ValueError("GROQ_API_KEY is required to use Groq.")
        return OpenAICompatibleClient(api_key=groq_key, base_url="https://api.groq.com/openai/v1")

    # DeepSeek
    if provider == "deepseek" or (provider == "auto" and deepseek_key):
        return OpenAICompatibleClient(api_key=deepseek_key, base_url="https://api.deepseek.com/v1")

    # OpenRouter
    if provider == "openrouter" or (provider == "auto" and openrouter_key):
        return OpenAICompatibleClient(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")

    # Custom OpenAI base URL
    if custom_base_url:
        api_key = os.getenv("OPENAI_API_KEY", "") or groq_key or "sk-dummy"
        return OpenAICompatibleClient(api_key=api_key, base_url=custom_base_url)

    # Fallback to Anthropic
    import anthropic
    return anthropic.Anthropic()
