"""Anthropic 格式转换：块往返、thinking 签名规则、缓存断点位置。"""

from types import SimpleNamespace as NS

import pytest

from emiya_agent.providers.anthropic_provider import (
    AnthropicProvider,
    OFFICIAL_ANTHROPIC_SOURCE,
    anthropic_endpoint_identity,
    from_anthropic_content,
    to_anthropic_messages,
    to_anthropic_tools,
)
from emiya_agent.providers.base import ChatRequest, ToolSpec
from emiya_agent.types import (
    FallbackBlock,
    Message,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def test_basic_text_roundtrip():
    msgs = [Message(role="user", content=[TextBlock(text="你好")])]
    out = to_anthropic_messages(msgs)
    assert out == [{"role": "user", "content": [{"type": "text", "text": "你好"}]}]


def test_thinking_with_signature_passes_without_drops():
    msgs = [
        Message(
            role="assistant",
            content=[
                ThinkingBlock(
                    thinking="",
                    signature="sig123",
                    source=OFFICIAL_ANTHROPIC_SOURCE,
                ),  # fable-5 空文本思考块
                ThinkingBlock(thinking="orphan"),  # 无签名 → 丢弃
                TextBlock(text="answer"),
            ],
        )
    ]
    (out,) = to_anthropic_messages(msgs)
    types = [b["type"] for b in out["content"]]
    assert types == ["thinking", "text"]
    assert out["content"][0]["signature"] == "sig123"
    assert out["content"][0]["thinking"] == ""


def test_redacted_thinking_roundtrips_unchanged_for_tool_continuation():
    source = [
        NS(type="thinking", thinking="summary", signature="sig"),
        NS(type="redacted_thinking", data="opaque-encrypted-blob"),
        NS(type="tool_use", id="t1", name="read", input={"path": "a.py"}),
    ]

    blocks = from_anthropic_content(source)
    assert [block.type for block in blocks] == [
        "thinking",
        "redacted_thinking",
        "tool_use",
    ]
    assert isinstance(blocks[1], RedactedThinkingBlock)

    (out,) = to_anthropic_messages([Message(role="assistant", content=blocks)])
    assert out["content"] == [
        {"type": "thinking", "thinking": "summary", "signature": "sig"},
        {"type": "redacted_thinking", "data": "opaque-encrypted-blob"},
        {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "a.py"}},
    ]


def test_final_fallback_boundary_is_normalized_and_roundtrips_opaquely():
    fallback = NS(
        type="fallback",
        from_=NS(model="claude-fable-5"),
        to=NS(model="claude-opus-4-8"),
        trigger=NS(type="refusal", category="cyber"),
    )
    source = [
        NS(type="thinking", thinking="declined", signature="old-sig"),
        NS(type="redacted_thinking", data="old-redacted"),
        NS(type="tool_use", id="old", name="unsafe", input={}),
        NS(type="text", text="safe partial text"),
        fallback,
        NS(type="thinking", thinking="continued", signature="new-sig"),
        NS(type="redacted_thinking", data="new-redacted"),
        NS(type="tool_use", id="new", name="read", input={"path": "a.py"}),
        NS(type="text", text="fallback answer"),
    ]

    blocks = from_anthropic_content(source)

    assert [block.type for block in blocks] == [
        "text",
        "fallback",
        "thinking",
        "redacted_thinking",
        "tool_use",
        "text",
    ]
    boundary = blocks[1]
    assert isinstance(boundary, FallbackBlock)
    assert boundary.source == OFFICIAL_ANTHROPIC_SOURCE
    assert boundary.data == {
        "from": {"model": "claude-fable-5"},
        "to": {"model": "claude-opus-4-8"},
        "trigger": {"type": "refusal", "category": "cyber"},
    }

    (out,) = to_anthropic_messages([Message(role="assistant", content=blocks)])
    assert out["content"] == [
        {"type": "text", "text": "safe partial text"},
        {
            "from": {"model": "claude-fable-5"},
            "to": {"model": "claude-opus-4-8"},
            "trigger": {"type": "refusal", "category": "cyber"},
            "type": "fallback",
        },
        {"type": "thinking", "thinking": "continued", "signature": "new-sig"},
        {"type": "redacted_thinking", "data": "new-redacted"},
        {"type": "tool_use", "id": "new", "name": "read", "input": {"path": "a.py"}},
        {"type": "text", "text": "fallback answer"},
    ]


def test_official_opaque_blocks_never_cross_to_anthropic_compatible_endpoint():
    third_party = anthropic_endpoint_identity(
        "deepseek", "https://api.deepseek.com/anthropic"
    )
    assistant = Message(
        role="assistant",
        content=[
            ThinkingBlock(
                thinking="official-thinking",
                signature="official-signature",
                source=OFFICIAL_ANTHROPIC_SOURCE,
            ),
            RedactedThinkingBlock(
                data="official-redacted", source=OFFICIAL_ANTHROPIC_SOURCE
            ),
            FallbackBlock(
                data={
                    "from": {"model": "claude-fable-5"},
                    "to": {"model": "claude-opus-4-8"},
                    "trigger": {"type": "refusal"},
                },
                source=OFFICIAL_ANTHROPIC_SOURCE,
            ),
            ToolUseBlock(id="t1", name="read", input={"path": "a.py"}),
            TextBlock(text="visible"),
        ],
    )

    (out,) = to_anthropic_messages(
        [assistant], endpoint_identity=third_party
    )

    assert out["content"] == [
        {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "a.py"}},
        {"type": "text", "text": "visible"},
    ]
    assert "official-thinking" not in repr(out)
    assert "official-signature" not in repr(out)
    assert "official-redacted" not in repr(out)


def test_signed_thinking_only_replays_to_its_source_endpoint():
    third_party = anthropic_endpoint_identity(
        "deepseek", "https://api.deepseek.com/anthropic"
    )
    blocks = from_anthropic_content(
        [
            NS(type="thinking", thinking="deepseek thinking", signature="ds-sig"),
            NS(type="tool_use", id="t1", name="read", input={}),
        ],
        source=third_party,
    )
    message = Message(role="assistant", content=blocks)

    (same_endpoint,) = to_anthropic_messages(
        [message], endpoint_identity=third_party
    )
    (official_endpoint,) = to_anthropic_messages([message])

    assert same_endpoint["content"][0] == {
        "type": "thinking",
        "thinking": "deepseek thinking",
        "signature": "ds-sig",
    }
    assert [block["type"] for block in official_endpoint["content"]] == [
        "tool_use"
    ]


def test_opaque_blocks_without_provenance_are_never_replayed():
    message = Message(
        role="assistant",
        content=[
            ThinkingBlock(thinking="legacy", signature="unknown-origin"),
            RedactedThinkingBlock(data="legacy-redacted"),
            FallbackBlock(data={"to": {"model": "legacy-fallback"}}),
            TextBlock(text="safe"),
        ],
    )

    (out,) = to_anthropic_messages([message])

    assert out["content"] == [{"type": "text", "text": "safe"}]


def _third_party_provider() -> AnthropicProvider:
    provider = object.__new__(AnthropicProvider)
    provider._endpoint_identity = anthropic_endpoint_identity(
        "proxy", "https://anthropic-proxy.example/v1"
    )
    provider._official_endpoint = False
    return provider


def test_third_party_claude_keeps_claude_effort_and_thinking_rules():
    kwargs = _third_party_provider()._build_kwargs(
        ChatRequest(
            model="claude-opus-4-8",
            messages=[],
            effort="high",
        )
    )

    assert kwargs["output_config"] == {"effort": "high"}
    assert kwargs["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
    }


def test_third_party_non_claude_never_gets_effort_or_thinking():
    kwargs = _third_party_provider()._build_kwargs(
        ChatRequest(
            model="deepseek-v4-pro",
            messages=[],
            effort="high",
        )
    )

    assert "output_config" not in kwargs
    assert "thinking" not in kwargs


@pytest.mark.asyncio
async def test_provider_identity_uses_effective_anthropic_environment_url(
    monkeypatch,
):
    monkeypatch.setenv(
        "ANTHROPIC_BASE_URL", "https://untrusted-proxy.example/anthropic"
    )
    provider = AnthropicProvider(api_key="test-key", base_url=None)
    try:
        effective = str(provider._client.base_url)
        assert "untrusted-proxy.example" in effective
        assert provider._endpoint_identity == anthropic_endpoint_identity(
            "anthropic", effective
        )
        assert provider._official_endpoint is False
    finally:
        await provider._client.close()


def test_only_canonical_https_anthropic_origin_is_official():
    assert (
        anthropic_endpoint_identity(
            "anthropic", "https://api.anthropic.com/"
        )
        == OFFICIAL_ANTHROPIC_SOURCE
    )
    assert (
        anthropic_endpoint_identity(
            "anthropic", "http://api.anthropic.com/"
        )
        != OFFICIAL_ANTHROPIC_SOURCE
    )
    assert (
        anthropic_endpoint_identity(
            "anthropic", "https://api.anthropic.com.evil/"
        )
        != OFFICIAL_ANTHROPIC_SOURCE
    )


def test_fable_never_gets_explicit_thinking():
    kwargs = _third_party_provider()._build_kwargs(
        ChatRequest(
            model="claude-fable-5",
            messages=[],
            effort="max",
        )
    )

    assert kwargs["output_config"] == {"effort": "max"}
    assert "thinking" not in kwargs


def test_tool_use_and_result_mapping():
    msgs = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="t1", name="read", input={"path": "a.py"})],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="t1", content="data", is_error=True)],
        ),
    ]
    out = to_anthropic_messages(msgs)
    assert out[0]["content"][0] == {
        "type": "tool_use",
        "id": "t1",
        "name": "read",
        "input": {"path": "a.py"},
    }
    assert out[1]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": "data",
        "is_error": True,
    }


def test_cache_breakpoint_on_last_cacheable_block():
    msgs = [
        Message(role="user", content=[TextBlock(text="q1")]),
        Message(role="user", content=[TextBlock(text="q2")]),
    ]
    out = to_anthropic_messages(msgs, cache=True)
    assert "cache_control" not in out[0]["content"][0]
    assert out[1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_tools_sorted_by_name_for_cache_stability():
    tools = [
        ToolSpec(name="zeta", description="z", input_schema={}),
        ToolSpec(name="alpha", description="a", input_schema={}),
    ]
    assert [t["name"] for t in to_anthropic_tools(tools)] == ["alpha", "zeta"]


def test_from_anthropic_content_maps_blocks():
    content = [
        NS(type="thinking", thinking="deep", signature="s1"),
        NS(type="text", text="hello"),
        NS(type="tool_use", id="t1", name="grep", input={"pattern": "x"}),
        NS(type="server_tool_use", id="srv-1"),  # 未建模块跳过
        NS(type="redacted_thinking", data="blob"),
    ]
    blocks = from_anthropic_content(content)
    assert [b.type for b in blocks] == [
        "thinking",
        "text",
        "tool_use",
        "redacted_thinking",
    ]
    assert blocks[0].signature == "s1"
    assert blocks[2].input == {"pattern": "x"}
    assert blocks[3].data == "blob"


def test_from_anthropic_content_empty_thinking_kept():
    # fable-5 display=omitted：thinking 为空串但签名必须保留以便回传
    content = [NS(type="thinking", thinking="", signature="sig")]
    (b,) = from_anthropic_content(content)
    assert b.thinking == ""
    assert b.signature == "sig"
