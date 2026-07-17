"""压缩：触发阈值、切分边界（不拆 tool 配对）、摘要与分块兜底。"""

import pytest

from emiya_agent.compaction.manager import (
    COMPACT_MARKER,
    COMPACT_SYSTEM,
    CompactionManager,
    serialize_messages,
)
from emiya_agent.types import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    user_message,
)


def turn(n: int) -> list[Message]:
    """一个完整回合：用户 → 助手(tool_use) → 工具结果 → 助手总结。"""
    return [
        user_message(f"请求 {n}"),
        Message(
            role="assistant",
            content=[ToolUseBlock(id=f"t{n}", name="read_file", input={"path": f"f{n}.py"})],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id=f"t{n}", content=f"content {n}")],
        ),
        Message(role="assistant", content=[TextBlock(text=f"done {n}")]),
    ]


def build_history(turns: int) -> list[Message]:
    messages: list[Message] = []
    for i in range(turns):
        messages.extend(turn(i))
    return messages


async def fake_complete(system: str, text: str) -> str:
    return f"SUMMARY({len(text)} chars)"


def make_manager(**kw) -> CompactionManager:
    return CompactionManager(fake_complete, **kw)


# ------------------------------------------------------------------ 触发


def test_should_compact_threshold():
    m = make_manager(threshold=0.7)
    window = 1_000_000
    assert not m.should_compact(Usage(input_tokens=600_000), window)
    assert m.should_compact(Usage(input_tokens=710_000), window)
    # 缓存读也计入提示词规模
    assert m.should_compact(
        Usage(input_tokens=100_000, cache_read_input_tokens=650_000), window
    )


def test_should_compact_zero_window_safe():
    assert not make_manager().should_compact(Usage(input_tokens=999), 0)


# ------------------------------------------------------------------ 切分


def test_split_keeps_recent_turns():
    m = make_manager(keep_recent_turns=2)
    messages = build_history(5)
    head, tail = m.split(messages)
    assert len(head) + len(tail) == len(messages)
    # tail 从真实用户消息开始
    assert tail[0].role == "user"
    assert tail[0].text() == "请求 3"
    # 保留最近 2 个回合
    assert [m2.text() for m2 in tail if m2.text().startswith("请求")] == [
        "请求 3",
        "请求 4",
    ]


def test_split_never_breaks_tool_pairs():
    m = make_manager(keep_recent_turns=2)
    head, tail = m.split(build_history(6))
    for part in (head, tail):
        use_ids = {
            b.id for msg in part for b in msg.content if isinstance(b, ToolUseBlock)
        }
        result_ids = {
            b.tool_use_id
            for msg in part
            for b in msg.content
            if isinstance(b, ToolResultBlock)
        }
        assert use_ids == result_ids  # 每段内部自洽，配对不被切断


def test_split_short_history_untouched():
    m = make_manager(keep_recent_turns=6)
    messages = build_history(3)
    head, tail = m.split(messages)
    assert head == []
    assert tail == messages


# ------------------------------------------------------------------ 压缩


@pytest.mark.asyncio
async def test_compact_replaces_head_with_summary():
    m = make_manager(keep_recent_turns=2)
    messages = build_history(5)
    compacted = await m.compact(messages)

    assert len(compacted) < len(messages)
    assert compacted[0].role == "user"
    assert COMPACT_MARKER in compacted[0].text()
    assert "SUMMARY" in compacted[0].text()
    # 尾部保留原样
    assert compacted[-1].text() == "done 4"


@pytest.mark.asyncio
async def test_compact_noop_on_short_history():
    m = make_manager(keep_recent_turns=6)
    messages = build_history(2)
    assert await m.compact(messages) == messages


@pytest.mark.asyncio
async def test_chunked_summarize_fallback():
    calls: list[str] = []

    async def recording_complete(system: str, text: str) -> str:
        calls.append(text[:20])
        return "PART"

    m = CompactionManager(
        recording_complete, keep_recent_turns=1, summarizer_window_chars=200
    )
    long_text = "x" * 1000
    result = await m._summarize(long_text)
    # 5 个分块 + 1 次合并
    assert len(calls) == 6
    assert result == "PART"


# ------------------------------------------------------------------ 序列化


def test_serialize_includes_tools_excludes_thinking():
    from emiya_agent.types import ThinkingBlock

    messages = [
        Message(
            role="assistant",
            content=[
                ThinkingBlock(thinking="secret reasoning", signature="s"),
                TextBlock(text="visible"),
                ToolUseBlock(id="t1", name="grep", input={"pattern": "foo"}),
            ],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="t1", content="match", is_error=True)],
        ),
    ]
    text = serialize_messages(messages)
    assert "visible" in text
    assert "grep" in text
    assert "[tool_error] match" in text
    assert "secret reasoning" not in text
