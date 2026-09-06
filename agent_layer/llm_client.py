"""
Provider-agnostic LLM client.

All four call sites in this repo (claude_agent.py, autonomous_agent.py,
rules_review_agent.py, performance_reflection.py) were built against the
anthropic SDK's `.messages.create(...)` surface, including native
Anthropic tool-calling (tools=[...], tool_use / tool_result blocks).

get_client() returns either the real anthropic.Anthropic client, or a
FeatherlessClient that exposes the identical `.messages.create()` shape
but talks to Featherless's OpenAI-compatible endpoint underneath —
translating tool schemas and tool-call/tool-result messages back and
forth. Existing call sites need zero logic changes; only the
`self._client = ...` line changes.

Switch providers via LLM_PROVIDER env var: "anthropic" (default) or
"featherless".
"""
import json
import os
from types import SimpleNamespace

import anthropic


def get_client(config):
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "featherless":
        return FeatherlessClient()
    return anthropic.Anthropic(api_key=config.claude.api_key)


class _MessagesShim:
    def __init__(self, parent):
        self._parent = parent

    def create(self, model, max_tokens, messages, system=None, tools=None):
        return self._parent._create(model, max_tokens, messages, system, tools)


class FeatherlessClient:
    """
    Wraps an OpenAI-compatible client (Featherless) so it can be dropped
    in wherever anthropic.Anthropic() was used, with the same
    `.messages.create()` call signature and an Anthropic-shaped response
    (`.content` = list of blocks with `.type` of "text" or "tool_use").
    """

    def __init__(self):
        from openai import OpenAI  # local import: only needed on this path

        api_key = os.getenv("FEATHERLESS_API_KEY", "")
        if not api_key:
            raise RuntimeError("FEATHERLESS_API_KEY not set but LLM_PROVIDER=featherless")

        self._oa = OpenAI(
            api_key=api_key,
            base_url=os.getenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1"),
        )
        self.messages = _MessagesShim(self)

    def _create(self, model, max_tokens, messages, system=None, tools=None):
        oa_messages = []
        if system:
            oa_messages.append({"role": "system", "content": system})
        oa_messages.extend(_to_openai_messages(messages))

        kwargs = dict(model=model, max_tokens=max_tokens, messages=oa_messages)
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)

        resp = self._oa.chat.completions.create(**kwargs)
        return _to_anthropic_like_response(resp)


def _to_openai_tools(tools):
    """Anthropic tool schema -> OpenAI function-calling schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


def _to_openai_messages(messages):
    """
    Anthropic-shaped message history -> OpenAI chat format.

    Anthropic: assistant content is a list of blocks (text / tool_use);
    user content for tool results is a list of {"type": "tool_result",
    "tool_use_id", "content"} dicts.

    OpenAI: assistant messages carry a `tool_calls` list; each result is
    its own message with role="tool" and a matching `tool_call_id`.
    """
    out = []
    for m in messages:
        role, content = m["role"], m["content"]

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if role == "assistant":
            text_parts = [b.text for b in content if getattr(b, "type", None) == "text"]
            tool_calls = [
                {
                    "id": b.id,
                    "type": "function",
                    "function": {"name": b.name, "arguments": json.dumps(b.input)},
                }
                for b in content
                if getattr(b, "type", None) == "tool_use"
            ]
            entry = {"role": "assistant", "content": "\n".join(text_parts) or None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)

        elif role == "user":
            # list of tool_result dicts from the previous round
            for block in content:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": block["content"],
                    }
                )
    return out


def _to_anthropic_like_response(resp):
    """OpenAI ChatCompletion -> object shaped like an Anthropic Message."""
    choice = resp.choices[0].message
    blocks = []
    if choice.content:
        blocks.append(SimpleNamespace(type="text", text=choice.content))
    for tc in (choice.tool_calls or []):
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError):
            args = {}
        blocks.append(SimpleNamespace(type="tool_use", id=tc.id, name=tc.function.name, input=args))
    return SimpleNamespace(content=blocks)
