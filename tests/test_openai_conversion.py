"""OpenAI/DeepSeek 格式双向转换 —— 最高风险层的 golden 测试。"""

import json
from types import SimpleNamespace as NS

import pytest

from emiya_agent.providers.base import ChatRequest, MessageDone, ToolSpec
from emiya_agent.providers.openai_compat import (
    ChunkAggregator,
    OpenAICompatProvider,
    openai_endpoint_identity,
    to_openai_messages,
    to_openai_tools,
)
from emiya_agent.types import (
    FallbackBlock,
    Message,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def _req(messages, system="You are helpful."):
    return ChatRequest(model="deepseek-chat", system=system, messages=messages)


# ------------------------------------------------------------------ 出站


def test_system_becomes_first_message():
    msgs = to_openai_messages(_req([Message(role="user", content=[TextBlock(text="hi")])]))
    assert msgs[0] == {"role": "system", "content": "You are helpful."}
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_assistant_tool_use_becomes_tool_calls_with_json_string_args():
    assistant = Message(
        role="assistant",
        content=[
            TextBlock(text="I'll check the weather."),
            ToolUseBlock(id="toolu_1", name="get_weather", input={"city": "东京"}),
        ],
    )
    msgs = to_openai_messages(_req([assistant], system=""))
    (msg,) = msgs
    assert msg["role"] == "assistant"
    assert msg["content"] == "I'll check the weather."
    (tc,) = msg["tool_calls"]
    assert tc["id"] == "toolu_1"
    assert tc["function"]["name"] == "get_weather"
    # arguments 必须是 JSON 字符串且保留非 ASCII
    assert isinstance(tc["function"]["arguments"], str)
    assert json.loads(tc["function"]["arguments"]) == {"city": "东京"}
    assert "东京" in tc["function"]["arguments"]


def test_tool_results_become_tool_role_messages_before_user_text():
    user = Message(
        role="user",
        content=[
            ToolResultBlock(tool_use_id="toolu_1", content="72F sunny"),
            ToolResultBlock(tool_use_id="toolu_2", content="boom", is_error=True),
            TextBlock(text="continue"),
        ],
    )
    msgs = to_openai_messages(_req([user], system=""))
    assert [m["role"] for m in msgs] == ["tool", "tool", "user"]
    assert msgs[0] == {"role": "tool", "tool_call_id": "toolu_1", "content": "72F sunny"}
    assert msgs[1]["content"].startswith("Error")
    assert msgs[2] == {"role": "user", "content": "continue"}


def test_thinking_blocks_dropped_on_send():
    assistant = Message(
        role="assistant",
        content=[
            ThinkingBlock(thinking="hmm", signature="sig"),
            RedactedThinkingBlock(data="opaque"),
            FallbackBlock(data={"to": {"model": "fallback"}}),
            TextBlock(text="ok"),
        ],
    )
    (msg,) = to_openai_messages(_req([assistant], system=""))
    assert msg == {"role": "assistant", "content": "ok"}


def test_deepseek_tool_call_replays_unsigned_reasoning_content_only():
    assistant = Message(
        role="assistant",
        content=[
            ThinkingBlock(
                thinking="deepseek reasoning",
                source=openai_endpoint_identity(
                    "deepseek", "https://api.deepseek.com"
                ),
            ),
            ThinkingBlock(thinking="anthropic reasoning", signature="sig"),
            ThinkingBlock(
                thinking="other provider reasoning",
                source=openai_endpoint_identity("glm", "https://glm.example.test"),
            ),
            ToolUseBlock(id="call-1", name="read", input={"path": "a.py"}),
        ],
    )
    result = Message(
        role="user",
        content=[ToolResultBlock(tool_use_id="call-1", content="file contents")],
    )

    msg, tool_result = to_openai_messages(
        _req([assistant, result], system=""), replay_tool_reasoning=True
    )

    assert msg["reasoning_content"] == "deepseek reasoning"
    assert msg["content"] == ""
    assert msg["tool_calls"][0]["id"] == "call-1"
    assert tool_result == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "file contents",
    }


def test_deepseek_does_not_replay_reasoning_without_tool_call():
    assistant = Message(
        role="assistant",
        content=[ThinkingBlock(thinking="past reasoning"), TextBlock(text="answer")],
    )

    (msg,) = to_openai_messages(
        _req([assistant], system=""), replay_tool_reasoning=True
    )

    assert msg == {"role": "assistant", "content": "answer"}


def test_deepseek_does_not_replay_reasoning_without_provenance():
    assistant = Message(
        role="assistant",
        content=[
            ThinkingBlock(thinking="unknown origin"),
            ToolUseBlock(id="call-1", name="read", input={}),
        ],
    )

    (msg,) = to_openai_messages(
        _req([assistant], system=""), replay_tool_reasoning=True
    )

    assert msg["content"] == ""
    assert "reasoning_content" not in msg


def test_deepseek_reasoning_never_crosses_base_url_change():
    old_source = openai_endpoint_identity(
        "deepseek", "https://old.deepseek.example/v1"
    )
    new_source = openai_endpoint_identity(
        "deepseek", "https://new.deepseek.example/v1"
    )
    assistant = Message(
        role="assistant",
        content=[
            ThinkingBlock(thinking="endpoint-private", source=old_source),
            ToolUseBlock(id="call-1", name="read", input={}),
        ],
    )

    (msg,) = to_openai_messages(
        _req([assistant], system=""),
        replay_tool_reasoning=True,
        reasoning_source=new_source,
    )

    assert msg["content"] == ""
    assert "reasoning_content" not in msg


@pytest.mark.asyncio
async def test_provider_identity_uses_effective_openai_environment_url(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/openai/v1")
    provider = OpenAICompatProvider(name="deepseek", api_key="test-key")
    try:
        effective = str(provider._client.base_url)
        assert "proxy.example" in effective
        assert provider._endpoint_identity == openai_endpoint_identity(
            "deepseek", effective
        )
    finally:
        await provider._client.close()


def test_reasoning_effort_mapping():
    from emiya_agent.providers.openai_compat import reasoning_effort_for

    # OpenAI 推理模型：映射并压级
    assert reasoning_effort_for("openai", "gpt-5.6", "high") == "high"
    assert reasoning_effort_for("openai", "gpt-5.6", "xhigh") == "high"
    assert reasoning_effort_for("openai", "gpt-5.6", "max") == "high"
    assert reasoning_effort_for("openai", "o4-mini", "low") == "low"
    # 非推理模型 / 其他供应商 / 无 effort：不发参数
    assert reasoning_effort_for("openai", "gpt-4o", "high") is None
    assert reasoning_effort_for("deepseek", "deepseek-chat", "high") is None
    assert reasoning_effort_for("openai", "gpt-5.6", None) is None


def test_tools_conversion_sorted():
    tools = [
        ToolSpec(name="b_tool", description="B", input_schema={"type": "object"}),
        ToolSpec(name="a_tool", description="A", input_schema={"type": "object"}),
    ]
    out = to_openai_tools(tools)
    assert [t["function"]["name"] for t in out] == ["a_tool", "b_tool"]
    assert out[0]["type"] == "function"
    assert out[0]["function"]["parameters"] == {"type": "object"}


# ------------------------------------------------------------------ 入站（流式聚合）


def _chunk(delta=None, finish_reason=None, usage=None):
    choices = []
    if delta is not None or finish_reason is not None:
        choices = [NS(delta=delta, finish_reason=finish_reason)]
    return NS(choices=choices, usage=usage)


def _text_delta(text):
    return NS(content=text, tool_calls=None, reasoning_content=None)


def _reasoning_delta(text):
    return NS(content=None, tool_calls=None, reasoning_content=text)


def _tool_delta(index, id=None, name=None, arguments=None):
    return NS(
        content=None,
        reasoning_content=None,
        tool_calls=[NS(index=index, id=id, function=NS(name=name, arguments=arguments))],
    )


@pytest.mark.asyncio
async def test_deepseek_provider_replays_complete_tool_assistant_kwargs(monkeypatch):
    provider = OpenAICompatProvider(
        name="deepseek", api_key="test-key", base_url="https://api.deepseek.test"
    )
    captured: dict = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)

        async def chunks():
            yield _chunk(finish_reason="stop")

        return chunks()

    monkeypatch.setattr(provider._client.chat.completions, "create", fake_create)
    assistant = Message(
        role="assistant",
        content=[
            ThinkingBlock(
                thinking="continued reasoning",
                source=provider._endpoint_identity,
            ),
            ToolUseBlock(id="call-1", name="read", input={"path": "a.py"}),
        ],
    )
    result = Message(
        role="user",
        content=[ToolResultBlock(tool_use_id="call-1", content="file contents")],
    )

    try:
        events = [
            event
            async for event in provider.stream(
                ChatRequest(
                    model="deepseek-v4-pro",
                    system="",
                    messages=[assistant, result],
                )
            )
        ]
    finally:
        await provider._client.close()

    assert isinstance(events[-1], MessageDone)
    replayed = captured["messages"][0]
    assert replayed["content"] == ""
    assert replayed["reasoning_content"] == "continued reasoning"
    assert replayed["tool_calls"][0]["id"] == "call-1"


def test_aggregate_text_stream():
    agg = ChunkAggregator()
    events = []
    for c in [_chunk(_text_delta("Hel")), _chunk(_text_delta("lo")), _chunk(finish_reason="stop")]:
        events.extend(agg.feed(c))
    done = agg.finalize()
    assert isinstance(done, MessageDone)
    assert done.message.text() == "Hello"
    assert done.stop_reason == "end_turn"


def test_aggregate_tool_call_fragments():
    agg = ChunkAggregator()
    chunks = [
        _chunk(_tool_delta(0, id="call_abc", name="get_weather", arguments="")),
        _chunk(_tool_delta(0, arguments='{"ci')),
        _chunk(_tool_delta(0, arguments='ty": "Par')),
        _chunk(_tool_delta(0, arguments='is"}')),
        _chunk(finish_reason="tool_calls"),
        _chunk(usage=NS(prompt_tokens=100, completion_tokens=20, prompt_tokens_details=None)),
    ]
    for c in chunks:
        agg.feed(c)
    done = agg.finalize()
    assert done.stop_reason == "tool_use"
    (tu,) = done.message.tool_uses()
    assert tu.id == "call_abc"
    assert tu.name == "get_weather"
    assert tu.input == {"city": "Paris"}  # 已解析为 dict，非字符串
    assert done.usage.input_tokens == 100
    assert done.usage.output_tokens == 20


def test_aggregate_parallel_tool_calls():
    agg = ChunkAggregator()
    for c in [
        _chunk(_tool_delta(0, id="c0", name="a", arguments='{"x":1}')),
        _chunk(_tool_delta(1, id="c1", name="b", arguments='{"y":2}')),
        _chunk(finish_reason="tool_calls"),
    ]:
        agg.feed(c)
    done = agg.finalize()
    tus = done.message.tool_uses()
    assert [(t.id, t.name, t.input) for t in tus] == [
        ("c0", "a", {"x": 1}),
        ("c1", "b", {"y": 2}),
    ]


def test_aggregate_deepseek_reasoning_and_cache():
    agg = ChunkAggregator()
    for c in [
        _chunk(_reasoning_delta("thinking...")),
        _chunk(_text_delta("answer")),
        _chunk(finish_reason="stop"),
        _chunk(
            usage=NS(
                prompt_tokens=1000,
                completion_tokens=50,
                prompt_tokens_details=None,
                prompt_cache_hit_tokens=600,
            )
        ),
    ]:
        agg.feed(c)
    done = agg.finalize()
    blocks = done.message.content
    assert blocks[0].type == "thinking"
    assert blocks[0].thinking == "thinking..."
    assert blocks[1].type == "text"
    # 缓存语义对齐 Anthropic：input 为未缓存部分
    assert done.usage.cache_read_input_tokens == 600
    assert done.usage.input_tokens == 400


def test_aggregate_length_finish_maps_to_max_tokens():
    agg = ChunkAggregator()
    agg.feed(_chunk(_text_delta("partial")))
    agg.feed(_chunk(finish_reason="length"))
    assert agg.finalize().stop_reason == "max_tokens"


def test_aggregate_malformed_tool_args_do_not_crash():
    agg = ChunkAggregator()
    agg.feed(_chunk(_tool_delta(0, id="c0", name="a", arguments="{not json")))
    agg.feed(_chunk(finish_reason="tool_calls"))
    (tu,) = agg.finalize().message.tool_uses()
    assert tu.input == {"_raw": "{not json"}


def test_aggregate_rejects_ambiguous_or_nonstandard_tool_json():
    for arguments in ('{"path":"safe","path":"unsafe"}', '{"value":NaN}'):
        agg = ChunkAggregator()
        agg.feed(_chunk(_tool_delta(0, id="c0", name="a", arguments=arguments)))
        agg.feed(_chunk(finish_reason="tool_calls"))

        (tool_use,) = agg.finalize().message.tool_uses()
        assert tool_use.input == {"_raw": arguments}
