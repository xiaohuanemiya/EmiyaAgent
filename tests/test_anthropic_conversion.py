"""Anthropic 格式转换：块往返、thinking 签名规则、缓存断点位置。"""

from types import SimpleNamespace as NS

from emiya_agent.providers.anthropic_provider import (
    from_anthropic_content,
    to_anthropic_messages,
    to_anthropic_tools,
)
from emiya_agent.providers.base import ToolSpec
from emiya_agent.types import (
    Message,
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
                ThinkingBlock(thinking="", signature="sig123"),  # fable-5 空文本思考块
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
        NS(type="fallback", to=NS(model="claude-opus-4-8")),  # 跳过
        NS(type="redacted_thinking", data="blob"),  # 跳过
    ]
    blocks = from_anthropic_content(content)
    assert [b.type for b in blocks] == ["thinking", "text", "tool_use"]
    assert blocks[0].signature == "s1"
    assert blocks[2].input == {"pattern": "x"}


def test_from_anthropic_content_empty_thinking_kept():
    # fable-5 display=omitted：thinking 为空串但签名必须保留以便回传
    content = [NS(type="thinking", thinking="", signature="sig")]
    (b,) = from_anthropic_content(content)
    assert b.thinking == ""
    assert b.signature == "sig"
