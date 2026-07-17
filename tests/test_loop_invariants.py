"""AgentLoop 不变量：tool_use/tool_result 配对、并行结果合并、中断补偿、迭代上限。"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import pytest

from emiya_agent.agent.loop import INTERRUPTED_NOTE, AgentLoop
from emiya_agent.providers.base import MessageDone
from emiya_agent.tools.base import Tool, ToolContext, ToolResult
from emiya_agent.tools.registry import ToolRegistry
from emiya_agent.types import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    user_message,
)
from emiya_agent.ui.render import StreamRenderer

from tests.conftest import FakeProvider, FakeUI, assistant_turn


class EchoTool(Tool):
    name: ClassVar[str] = "echo"
    description: ClassVar[str] = "echo"
    input_schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}
    read_only: ClassVar[bool] = True

    async def run(self, args, ctx):
        return ToolResult(content=f"echo: {args.get('text', '')}")


class BoomTool(Tool):
    name: ClassVar[str] = "boom"
    description: ClassVar[str] = "always raises"
    input_schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}
    read_only: ClassVar[bool] = True

    async def run(self, args, ctx):
        raise RuntimeError("kaboom")


class CancelTool(Tool):
    """执行时设置 cancel（模拟用户中断）。串行工具。"""

    name: ClassVar[str] = "cancel_self"
    description: ClassVar[str] = "sets cancel"
    input_schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}
    read_only: ClassVar[bool] = False

    async def run(self, args, ctx):
        ctx.cancel.set()
        return ToolResult(content="done, then cancelled")


class WriteMarkTool(Tool):
    name: ClassVar[str] = "mark"
    description: ClassVar[str] = "serial write"
    input_schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}
    read_only: ClassVar[bool] = False

    async def run(self, args, ctx):
        return ToolResult(content="marked")


def make_registry(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def make_loop(provider, registry, quiet_console, **kw) -> AgentLoop:
    renderer = StreamRenderer(quiet_console)
    return AgentLoop(
        provider, "fake-model", "fake/fake-model", registry, renderer.render, **kw
    )


def make_ctx(tmp_path, config) -> ToolContext:
    return ToolContext(cwd=tmp_path, config=config, ui=FakeUI())


def tool_use(id: str, name: str, **input: Any) -> ToolUseBlock:
    return ToolUseBlock(id=id, name=name, input=input)


# ------------------------------------------------------------------ 基本回合


@pytest.mark.asyncio
async def test_plain_answer_no_tools(tmp_path, config, quiet_console):
    provider = FakeProvider([assistant_turn([TextBlock(text="hi there")])])
    loop = make_loop(provider, make_registry(), quiet_console)
    messages = [user_message("hello")]
    result = await loop.run(messages, system="s", ctx=make_ctx(tmp_path, config))

    assert result.stop_reason == "end_turn"
    assert result.final_text == "hi there"
    assert len(messages) == 2  # user + assistant


@pytest.mark.asyncio
async def test_tool_roundtrip_pairing(tmp_path, config, quiet_console):
    provider = FakeProvider(
        [
            assistant_turn(
                [tool_use("t1", "echo", text="a")], stop_reason="tool_use"
            ),
            assistant_turn([TextBlock(text="final")]),
        ]
    )
    loop = make_loop(provider, make_registry(EchoTool()), quiet_console)
    messages = [user_message("go")]
    result = await loop.run(messages, system="s", ctx=make_ctx(tmp_path, config))

    assert result.final_text == "final"
    # user / assistant(tool_use) / user(tool_result) / assistant(final)
    assert [m.role for m in messages] == ["user", "assistant", "user", "assistant"]
    (tr,) = [b for b in messages[2].content if isinstance(b, ToolResultBlock)]
    assert tr.tool_use_id == "t1"
    assert tr.content == "echo: a"
    # 第二次请求应包含工具结果历史
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_parallel_calls_single_result_message(tmp_path, config, quiet_console):
    provider = FakeProvider(
        [
            assistant_turn(
                [
                    tool_use("t1", "echo", text="one"),
                    tool_use("t2", "echo", text="two"),
                    tool_use("t3", "echo", text="three"),
                ],
                stop_reason="tool_use",
            ),
            assistant_turn([TextBlock(text="done")]),
        ]
    )
    loop = make_loop(provider, make_registry(EchoTool()), quiet_console)
    messages = [user_message("go")]
    await loop.run(messages, system="s", ctx=make_ctx(tmp_path, config))

    result_msgs = [m for m in messages if any(
        isinstance(b, ToolResultBlock) for b in m.content
    )]
    assert len(result_msgs) == 1  # 不变量 2：全部结果在同一条 user 消息
    results = [b for b in result_msgs[0].content if isinstance(b, ToolResultBlock)]
    assert [r.tool_use_id for r in results] == ["t1", "t2", "t3"]  # 顺序与调用一致


@pytest.mark.asyncio
async def test_tool_exception_becomes_error_result(tmp_path, config, quiet_console):
    provider = FakeProvider(
        [
            assistant_turn([tool_use("t1", "boom")], stop_reason="tool_use"),
            assistant_turn([TextBlock(text="recovered")]),
        ]
    )
    loop = make_loop(provider, make_registry(BoomTool()), quiet_console)
    messages = [user_message("go")]
    result = await loop.run(messages, system="s", ctx=make_ctx(tmp_path, config))

    assert result.final_text == "recovered"  # 循环继续而非崩溃
    (tr,) = [b for b in messages[2].content if isinstance(b, ToolResultBlock)]
    assert tr.is_error
    assert "kaboom" in tr.content


@pytest.mark.asyncio
async def test_unknown_tool_is_error_result(tmp_path, config, quiet_console):
    provider = FakeProvider(
        [
            assistant_turn([tool_use("t1", "no_such_tool")], stop_reason="tool_use"),
            assistant_turn([TextBlock(text="ok")]),
        ]
    )
    loop = make_loop(provider, make_registry(), quiet_console)
    messages = [user_message("go")]
    await loop.run(messages, system="s", ctx=make_ctx(tmp_path, config))
    (tr,) = [b for b in messages[2].content if isinstance(b, ToolResultBlock)]
    assert tr.is_error and "unknown tool" in tr.content


# ------------------------------------------------------------------ 中断与上限


@pytest.mark.asyncio
async def test_cancel_mid_batch_synthesizes_remaining(tmp_path, config, quiet_console):
    """串行批次中第一个工具触发 cancel → 第二个工具补合成 is_error 结果。"""
    provider = FakeProvider(
        [
            assistant_turn(
                [tool_use("t1", "cancel_self"), tool_use("t2", "mark")],
                stop_reason="tool_use",
            ),
        ]
    )
    loop = make_loop(
        provider, make_registry(CancelTool(), WriteMarkTool()), quiet_console
    )
    messages = [user_message("go")]
    result = await loop.run(messages, system="s", ctx=make_ctx(tmp_path, config))

    assert result.stop_reason == "interrupted"
    results = [b for b in messages[-1].content if isinstance(b, ToolResultBlock)]
    assert len(results) == 2  # 不变量 1：每个 tool_use 都有配对结果
    by_id = {r.tool_use_id: r for r in results}
    assert by_id["t1"].content == "done, then cancelled"
    assert by_id["t2"].is_error and INTERRUPTED_NOTE in by_id["t2"].content


@pytest.mark.asyncio
async def test_max_iterations_guard(tmp_path, config, quiet_console):
    endless = [
        assistant_turn([tool_use(f"t{i}", "echo", text="x")], stop_reason="tool_use")
        for i in range(10)
    ]
    provider = FakeProvider(endless)
    loop = make_loop(
        provider, make_registry(EchoTool()), quiet_console, max_iterations=3
    )
    messages = [user_message("go")]
    result = await loop.run(messages, system="s", ctx=make_ctx(tmp_path, config))

    assert result.stop_reason == "max_iterations"
    assert len(provider.requests) == 3
    # 历史合法：最后一条是包含 tool_result 的 user 消息
    assert messages[-1].role == "user"
    assert all(
        isinstance(b, ToolResultBlock) for b in messages[-1].content
    )


@pytest.mark.asyncio
async def test_refusal_not_appended(tmp_path, config, quiet_console):
    provider = FakeProvider(
        [
            MessageDone(
                message=Message(role="assistant", content=[]),
                usage=__import__("emiya_agent.types", fromlist=["Usage"]).Usage(),
                stop_reason="refusal",
                refusal="cyber: blocked",
            )
        ]
    )
    loop = make_loop(provider, make_registry(), quiet_console)
    messages = [user_message("go")]
    result = await loop.run(messages, system="s", ctx=make_ctx(tmp_path, config))

    assert result.stop_reason == "refusal"
    assert result.refusal == "cyber: blocked"
    assert len(messages) == 1  # 拒绝回合不入历史


# ------------------------------------------------------------------ 回调


@pytest.mark.asyncio
async def test_callbacks_fire(tmp_path, config, quiet_console):
    provider = FakeProvider(
        [
            assistant_turn([tool_use("t1", "echo", text="a")], stop_reason="tool_use"),
            assistant_turn([TextBlock(text="fin")]),
        ]
    )
    persisted: list[Message] = []
    usages: list[str] = []
    tool_results: list[str] = []
    loop = make_loop(
        provider,
        make_registry(EchoTool()),
        quiet_console,
        on_message=persisted.append,
        on_usage=lambda route, u: usages.append(route),
        on_tool_result=lambda c, r: tool_results.append(c.name),
    )
    messages = [user_message("go")]
    await loop.run(messages, system="s", ctx=make_ctx(tmp_path, config))

    assert [m.role for m in persisted] == ["assistant", "user", "assistant"]
    assert usages == ["fake/fake-model", "fake/fake-model"]
    assert tool_results == ["echo"]
