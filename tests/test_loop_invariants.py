"""AgentLoop 不变量：tool_use/tool_result 配对、并行结果合并、中断补偿、迭代上限。"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import pytest

from emiya_agent.agent.control import RunController, RunLimits
from emiya_agent.agent.loop import (
    INTERRUPTED_NOTE,
    INVALID_STOP_NOTE,
    POLICY_DENIED_NOTE,
    AgentLoop,
    _args_brief,
)
from emiya_agent.config import ModelInfo
from emiya_agent.memory.manager import MemoryManager
from emiya_agent.providers.base import MessageDone
from emiya_agent.tools.base import Tool, ToolContext, ToolResult
from emiya_agent.tools.memory_tool import MemoryTool
from emiya_agent.tools.registry import ToolRegistry
from emiya_agent.types import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
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


class RecordingTool(Tool):
    """记录是否真正 dispatch，用于验证异常结束原因不会触发副作用。"""

    name: ClassVar[str] = "record"
    description: ClassVar[str] = "record dispatch"
    input_schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}
    read_only: ClassVar[bool] = False

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, args, ctx):
        self.calls += 1
        return ToolResult(content="dispatched")


class PolicyDeniedTool(RecordingTool):
    name: ClassVar[str] = "policy_denied"

    def permission_override(self, args, ctx):
        return "deny"


class ApprovalPreviewBoomTool(RecordingTool):
    name: ClassVar[str] = "approval_preview_boom"

    async def approval_preview(self, args, ctx):
        raise RuntimeError("preview exploded")


class ApprovalCleanupTool(RecordingTool):
    name: ClassVar[str] = "approval_cleanup"

    def __init__(self) -> None:
        super().__init__()
        self.cancelled = 0

    async def approval_cancelled(self, args, ctx):
        self.cancelled += 1


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


def tool_use(id: str, tool_name: str, **input: Any) -> ToolUseBlock:
    return ToolUseBlock(id=id, name=tool_name, input=input)


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
async def test_external_task_cancel_during_stream_propagates(
    tmp_path, config, quiet_console
):
    stream_started = asyncio.Event()
    stream_release = asyncio.Event()
    stream_finished = asyncio.Event()

    class BlockingProvider(FakeProvider):
        async def stream(self, req):
            self.requests.append(req)
            stream_started.set()
            try:
                await stream_release.wait()
            finally:
                stream_finished.set()
            yield assistant_turn([TextBlock(text="too late")])

    loop = make_loop(BlockingProvider([]), make_registry(), quiet_console)
    loop_task = asyncio.create_task(
        loop.run(
            [user_message("hello")],
            system="s",
            ctx=make_ctx(tmp_path, config),
        )
    )

    try:
        await asyncio.wait_for(stream_started.wait(), timeout=1)
        loop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(loop_task, timeout=1)
        assert stream_finished.is_set()
    finally:
        stream_release.set()
        if not loop_task.done():
            loop_task.cancel()
        await asyncio.gather(loop_task, return_exceptions=True)


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
async def test_parallel_self_cancel_settles_sibling_before_pairing(
    tmp_path, config, quiet_console
):
    sibling_started = asyncio.Event()
    sibling_release = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    sibling_finished = asyncio.Event()

    class SelfCancellingReadTool(Tool):
        name: ClassVar[str] = "self_cancel_read"
        description: ClassVar[str] = "cancels itself after its sibling starts"
        input_schema: ClassVar[dict[str, Any]] = {
            "type": "object",
            "properties": {},
        }
        read_only: ClassVar[bool] = True

        async def run(self, args, ctx):
            await sibling_started.wait()
            raise asyncio.CancelledError

    class BlockingReadTool(Tool):
        name: ClassVar[str] = "blocking_read"
        description: ClassVar[str] = "blocks until cancelled"
        input_schema: ClassVar[dict[str, Any]] = {
            "type": "object",
            "properties": {},
        }
        read_only: ClassVar[bool] = True

        async def run(self, args, ctx):
            sibling_started.set()
            try:
                await sibling_release.wait()
                return ToolResult(content="released without cancellation")
            finally:
                cleanup_started.set()
                await cleanup_release.wait()
                sibling_finished.set()

    provider = FakeProvider(
        [
            assistant_turn(
                [
                    tool_use("t1", "self_cancel_read"),
                    tool_use("t2", "blocking_read"),
                ],
                stop_reason="tool_use",
            )
        ]
    )
    loop = make_loop(
        provider,
        make_registry(SelfCancellingReadTool(), BlockingReadTool()),
        quiet_console,
    )
    messages = [user_message("go")]
    loop_task = asyncio.create_task(
        loop.run(messages, system="s", ctx=make_ctx(tmp_path, config))
    )
    cleanup_wait = asyncio.create_task(cleanup_started.wait())

    try:
        await asyncio.wait_for(sibling_started.wait(), timeout=1)
        done, _ = await asyncio.wait_for(
            asyncio.wait(
                {loop_task, cleanup_wait},
                return_when=asyncio.FIRST_COMPLETED,
            ),
            timeout=1,
        )
        assert cleanup_wait in done
        assert not loop_task.done(), (
            "AgentLoop returned before the cancelled sibling finished cleanup"
        )

        cleanup_release.set()
        result = await asyncio.wait_for(loop_task, timeout=1)

        assert result.stop_reason == "interrupted"
        assert sibling_finished.is_set()
        results = [
            block
            for block in messages[-1].content
            if isinstance(block, ToolResultBlock)
        ]
        assert [block.tool_use_id for block in results] == ["t1", "t2"]
        assert all(
            block.is_error and block.content == INTERRUPTED_NOTE for block in results
        )
    finally:
        sibling_release.set()
        cleanup_release.set()
        await asyncio.wait_for(sibling_finished.wait(), timeout=1)
        if not cleanup_wait.done():
            cleanup_wait.cancel()
        await asyncio.gather(cleanup_wait, return_exceptions=True)
        if not loop_task.done():
            await asyncio.wait_for(loop_task, timeout=1)


@pytest.mark.asyncio
async def test_external_task_cancel_pairs_results_after_parallel_cleanup(
    tmp_path, config, quiet_console
):
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    dispatch_release = asyncio.Event()
    first_cleanup_started = asyncio.Event()
    second_cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    first_finished = asyncio.Event()
    second_finished = asyncio.Event()

    async def block_until_cancelled(started, cleanup_started, finished):
        started.set()
        try:
            await dispatch_release.wait()
            return ToolResult(content="released without cancellation")
        finally:
            cleanup_started.set()
            await cleanup_release.wait()
            finished.set()

    class FirstBlockingReadTool(Tool):
        name: ClassVar[str] = "first_blocking_read"
        description: ClassVar[str] = "first blocking read"
        input_schema: ClassVar[dict[str, Any]] = {
            "type": "object",
            "properties": {},
        }
        read_only: ClassVar[bool] = True

        async def run(self, args, ctx):
            return await block_until_cancelled(
                first_started, first_cleanup_started, first_finished
            )

    class SecondBlockingReadTool(Tool):
        name: ClassVar[str] = "second_blocking_read"
        description: ClassVar[str] = "second blocking read"
        input_schema: ClassVar[dict[str, Any]] = {
            "type": "object",
            "properties": {},
        }
        read_only: ClassVar[bool] = True

        async def run(self, args, ctx):
            return await block_until_cancelled(
                second_started, second_cleanup_started, second_finished
            )

    provider = FakeProvider(
        [
            assistant_turn(
                [
                    tool_use("t1", "first_blocking_read"),
                    tool_use("t2", "second_blocking_read"),
                ],
                stop_reason="tool_use",
            )
        ]
    )
    loop = make_loop(
        provider,
        make_registry(FirstBlockingReadTool(), SecondBlockingReadTool()),
        quiet_console,
    )
    messages = [user_message("go")]
    loop_task = asyncio.create_task(
        loop.run(messages, system="s", ctx=make_ctx(tmp_path, config))
    )

    try:
        await asyncio.wait_for(
            asyncio.gather(first_started.wait(), second_started.wait()),
            timeout=1,
        )
        loop_task.cancel()
        await asyncio.wait_for(
            asyncio.gather(
                first_cleanup_started.wait(), second_cleanup_started.wait()
            ),
            timeout=1,
        )
        assert not loop_task.done(), (
            "AgentLoop exposed cancellation before parallel cleanup completed"
        )

        cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(loop_task, timeout=1)

        assert first_finished.is_set() and second_finished.is_set()
        results = [
            block
            for block in messages[-1].content
            if isinstance(block, ToolResultBlock)
        ]
        assert [block.tool_use_id for block in results] == ["t1", "t2"]
        assert all(
            block.is_error and block.content == INTERRUPTED_NOTE for block in results
        )
    finally:
        dispatch_release.set()
        cleanup_release.set()
        if not loop_task.done():
            loop_task.cancel()
        await asyncio.gather(loop_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_parallel_cleanup_escalates_after_first_cancel_is_swallowed(
    tmp_path, config, quiet_console, monkeypatch
):
    import emiya_agent.agent.loop as loop_module

    first_cancel_seen = asyncio.Event()
    finished = asyncio.Event()

    class CancellationResistantReadTool(Tool):
        name: ClassVar[str] = "cancellation_resistant_read"
        description: ClassVar[str] = "requires a second cancellation"
        input_schema: ClassVar[dict[str, Any]] = {
            "type": "object",
            "properties": {},
        }
        read_only: ClassVar[bool] = True

        async def run(self, args, ctx):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                first_cancel_seen.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    finished.set()

    class SelfCancellingReadTool(Tool):
        name: ClassVar[str] = "trigger_parallel_cancel"
        description: ClassVar[str] = "cancels its own child task"
        input_schema: ClassVar[dict[str, Any]] = {
            "type": "object",
            "properties": {},
        }
        read_only: ClassVar[bool] = True

        async def run(self, args, ctx):
            await asyncio.sleep(0)
            raise asyncio.CancelledError

    monkeypatch.setattr(loop_module, "PARALLEL_CANCEL_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(loop_module, "PARALLEL_CANCEL_FORCE_SECONDS", 0.2)
    provider = FakeProvider(
        [
            assistant_turn(
                [
                    tool_use("t1", "cancellation_resistant_read"),
                    tool_use("t2", "trigger_parallel_cancel"),
                ],
                stop_reason="tool_use",
            )
        ]
    )
    messages = [user_message("go")]
    loop = make_loop(
        provider,
        make_registry(CancellationResistantReadTool(), SelfCancellingReadTool()),
        quiet_console,
    )

    result = await asyncio.wait_for(
        loop.run(messages, system="s", ctx=make_ctx(tmp_path, config)),
        timeout=1,
    )

    assert result.stop_reason == "interrupted"
    assert first_cancel_seen.is_set()
    assert finished.is_set()
    results = [
        block for block in messages[-1].content if isinstance(block, ToolResultBlock)
    ]
    assert [block.tool_use_id for block in results] == ["t1", "t2"]
    assert all(block.is_error for block in results)


@pytest.mark.asyncio
async def test_parallel_cleanup_deadline_still_pairs_before_detached_task_exits(
    tmp_path, config, quiet_console, monkeypatch
):
    import emiya_agent.agent.loop as loop_module

    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release = asyncio.Event()
    first_finished = asyncio.Event()
    second_finished = asyncio.Event()

    async def stubborn(started, finished):
        started.set()
        try:
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue
            return ToolResult(content="released")
        finally:
            finished.set()

    class FirstStubbornReadTool(Tool):
        name: ClassVar[str] = "first_stubborn_read"
        description: ClassVar[str] = "first cancellation-resistant read"
        input_schema: ClassVar[dict[str, Any]] = {
            "type": "object",
            "properties": {},
        }
        read_only: ClassVar[bool] = True

        async def run(self, args, ctx):
            return await stubborn(first_started, first_finished)

    class SecondStubbornReadTool(Tool):
        name: ClassVar[str] = "second_stubborn_read"
        description: ClassVar[str] = "second cancellation-resistant read"
        input_schema: ClassVar[dict[str, Any]] = {
            "type": "object",
            "properties": {},
        }
        read_only: ClassVar[bool] = True

        async def run(self, args, ctx):
            return await stubborn(second_started, second_finished)

    monkeypatch.setattr(loop_module, "PARALLEL_CANCEL_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(loop_module, "PARALLEL_CANCEL_FORCE_SECONDS", 0.01)
    provider = FakeProvider(
        [
            assistant_turn(
                [
                    tool_use("t1", "first_stubborn_read"),
                    tool_use("t2", "second_stubborn_read"),
                ],
                stop_reason="tool_use",
            )
        ]
    )
    messages = [user_message("go")]
    loop = make_loop(
        provider,
        make_registry(FirstStubbornReadTool(), SecondStubbornReadTool()),
        quiet_console,
    )
    loop_task = asyncio.create_task(
        loop.run(messages, system="s", ctx=make_ctx(tmp_path, config))
    )

    try:
        await asyncio.wait_for(
            asyncio.gather(first_started.wait(), second_started.wait()),
            timeout=1,
        )
        loop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(loop_task, timeout=1)

        assert not first_finished.is_set()
        assert not second_finished.is_set()
        results = [
            block
            for block in messages[-1].content
            if isinstance(block, ToolResultBlock)
        ]
        assert [block.tool_use_id for block in results] == ["t1", "t2"]
        assert all(block.is_error for block in results)
    finally:
        release.set()
        await asyncio.wait_for(
            asyncio.gather(first_finished.wait(), second_finished.wait()),
            timeout=1,
        )
        if not loop_task.done():
            loop_task.cancel()
        await asyncio.gather(loop_task, return_exceptions=True)


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


@pytest.mark.parametrize("stop_reason", ["max_tokens", "end_turn", "error"])
@pytest.mark.asyncio
async def test_non_tool_stop_never_dispatches_embedded_tool_call(
    stop_reason, tmp_path, config, quiet_console
):
    """截断或异常结束中的 tool_use 只补配对错误结果，绝不执行。"""
    tool = RecordingTool()
    provider = FakeProvider(
        [
            assistant_turn(
                [tool_use("t1", "record", value="possibly truncated")],
                stop_reason=stop_reason,
            )
        ]
    )
    loop = make_loop(provider, make_registry(tool), quiet_console)
    messages = [user_message("go")]

    result = await loop.run(messages, system="s", ctx=make_ctx(tmp_path, config))

    assert result.stop_reason == stop_reason
    assert tool.calls == 0
    assert [m.role for m in messages] == ["user", "assistant", "user"]
    (synthetic,) = [
        b for b in messages[-1].content if isinstance(b, ToolResultBlock)
    ]
    assert synthetic.tool_use_id == "t1"
    assert synthetic.is_error
    assert INVALID_STOP_NOTE.format(stop_reason=stop_reason) == synthetic.content


@pytest.mark.parametrize(
    "calls",
    [
        [tool_use("t1", "record", _raw="{broken")],
        [
            tool_use("same", "record", value="first"),
            tool_use("same", "record", value="second"),
        ],
        [tool_use("", "record")],
        [tool_use("t1", "")],
    ],
)
@pytest.mark.asyncio
async def test_invalid_tool_batch_never_dispatches(
    calls, tmp_path, config, quiet_console
):
    tool = RecordingTool()
    provider = FakeProvider([assistant_turn(calls, stop_reason="tool_use")])
    persisted: list[Message] = []
    loop = make_loop(
        provider,
        make_registry(tool),
        quiet_console,
        on_message=persisted.append,
    )
    messages = [user_message("go")]

    result = await loop.run(messages, system="s", ctx=make_ctx(tmp_path, config))

    assert result.stop_reason == "error"
    assert tool.calls == 0
    assert messages == [user_message("go")]
    assert persisted == []


@pytest.mark.parametrize("stop_reason", ["tool_use", "max_tokens", "end_turn"])
@pytest.mark.asyncio
async def test_duplicate_tool_ids_do_not_poison_next_request(
    stop_reason, tmp_path, config, quiet_console
):
    calls = [tool_use("same", "record"), tool_use("same", "record")]
    messages = [user_message("first")]
    first_provider = FakeProvider([assistant_turn(calls, stop_reason=stop_reason)])
    first_loop = make_loop(
        first_provider, make_registry(RecordingTool()), quiet_console
    )

    result = await first_loop.run(
        messages, system="s", ctx=make_ctx(tmp_path, config)
    )
    assert result.stop_reason == "error"
    assert messages == [user_message("first")]

    messages.append(user_message("retry"))
    second_provider = FakeProvider(
        [assistant_turn([TextBlock(text="recovered")], stop_reason="end_turn")]
    )
    second_loop = make_loop(second_provider, make_registry(), quiet_console)
    second = await second_loop.run(
        messages, system="s", ctx=make_ctx(tmp_path, config)
    )

    assert second.final_text == "recovered"
    assert [m.role for m in second_provider.requests[0].messages] == ["user", "user"]
    assert all(not message.tool_uses() for message in second_provider.requests[0].messages)


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


# ------------------------------------------------------------------ 审批门控


@pytest.mark.asyncio
async def test_approval_deny_blocks_tool(tmp_path, config, quiet_console):
    """ask 模式下拒绝 → 工具不执行，返回 denied 错误结果，循环继续。"""
    config.tool_approval = "ask"
    provider = FakeProvider(
        [
            assistant_turn(
                [tool_use("t1", "approval_cleanup")], stop_reason="tool_use"
            ),
            assistant_turn([TextBlock(text="adjusted")]),
        ]
    )
    tool = ApprovalCleanupTool()
    loop = make_loop(provider, make_registry(tool), quiet_console)
    ui = FakeUI(approvals=[False])
    ctx = ToolContext(cwd=tmp_path, config=config, ui=ui)
    messages = [user_message("go")]
    result = await loop.run(messages, system="s", ctx=ctx)

    assert result.final_text == "adjusted"
    assert ui.approval_requests == ["approval_cleanup"]
    assert tool.cancelled == 1
    (tr,) = [b for b in messages[2].content if isinstance(b, ToolResultBlock)]
    assert tr.is_error
    from emiya_agent.agent.loop import DENIED_NOTE

    assert DENIED_NOTE in tr.content


@pytest.mark.asyncio
async def test_approval_allow_executes(tmp_path, config, quiet_console):
    config.tool_approval = "ask"
    provider = FakeProvider(
        [
            assistant_turn([tool_use("t1", "mark")], stop_reason="tool_use"),
            assistant_turn([TextBlock(text="done")]),
        ]
    )
    loop = make_loop(provider, make_registry(WriteMarkTool()), quiet_console)
    ui = FakeUI(approvals=[True])
    ctx = ToolContext(cwd=tmp_path, config=config, ui=ui)
    messages = [user_message("go")]
    await loop.run(messages, system="s", ctx=ctx)
    (tr,) = [b for b in messages[2].content if isinstance(b, ToolResultBlock)]
    assert not tr.is_error and tr.content == "marked"


@pytest.mark.asyncio
async def test_approval_preview_error_fails_closed_with_paired_result(
    tmp_path, config, quiet_console
):
    config.tool_approval = "ask"
    tool = ApprovalPreviewBoomTool()
    provider = FakeProvider(
        [
            assistant_turn(
                [tool_use("t1", "approval_preview_boom")], stop_reason="tool_use"
            ),
            assistant_turn([TextBlock(text="recovered")]),
        ]
    )
    loop = make_loop(provider, make_registry(tool), quiet_console)
    ui = FakeUI(approvals=[True])
    messages = [user_message("go")]

    result = await loop.run(
        messages,
        system="s",
        ctx=ToolContext(cwd=tmp_path, config=config, ui=ui),
    )

    assert result.final_text == "recovered"
    assert tool.calls == 0
    assert ui.approval_requests == []
    (tool_result,) = [
        block for block in messages[2].content if isinstance(block, ToolResultBlock)
    ]
    assert tool_result.is_error is True
    assert "Tool approval preview error" in tool_result.content


@pytest.mark.asyncio
async def test_approval_ui_error_fails_closed_with_paired_result(
    tmp_path, config, quiet_console
):
    class BrokenApprovalUI(FakeUI):
        async def confirm_tool(self, name, args_brief):
            raise RuntimeError("approval channel failed")

    config.tool_approval = "ask"
    provider = FakeProvider(
        [
            assistant_turn([tool_use("t1", "mark")], stop_reason="tool_use"),
            assistant_turn([TextBlock(text="recovered")]),
        ]
    )
    loop = make_loop(provider, make_registry(WriteMarkTool()), quiet_console)
    messages = [user_message("go")]

    result = await loop.run(
        messages,
        system="s",
        ctx=ToolContext(cwd=tmp_path, config=config, ui=BrokenApprovalUI()),
    )

    assert result.final_text == "recovered"
    (tool_result,) = [
        block for block in messages[2].content if isinstance(block, ToolResultBlock)
    ]
    assert tool_result.is_error is True
    assert "Tool approval UI error" in tool_result.content


@pytest.mark.asyncio
async def test_auto_mode_skips_approval(tmp_path, config, quiet_console):
    config.tool_approval = "auto"
    provider = FakeProvider(
        [
            assistant_turn([tool_use("t1", "mark")], stop_reason="tool_use"),
            assistant_turn([TextBlock(text="done")]),
        ]
    )
    loop = make_loop(provider, make_registry(WriteMarkTool()), quiet_console)
    ui = FakeUI()
    ctx = ToolContext(cwd=tmp_path, config=config, ui=ui)
    await loop.run([user_message("go")], system="s", ctx=ctx)
    assert ui.approval_requests == []  # 未询问


@pytest.mark.asyncio
async def test_permission_policy_denies_even_in_auto_mode(
    tmp_path, config, quiet_console
):
    config.tool_approval = "auto"
    tool = PolicyDeniedTool()
    provider = FakeProvider(
        [
            assistant_turn(
                [tool_use("t1", "policy_denied")], stop_reason="tool_use"
            ),
            assistant_turn([TextBlock(text="blocked")]),
        ]
    )
    loop = make_loop(provider, make_registry(tool), quiet_console)
    messages = [user_message("go")]

    await loop.run(messages, system="s", ctx=make_ctx(tmp_path, config))

    assert tool.calls == 0
    (result,) = [
        block
        for block in messages[2].content
        if isinstance(block, ToolResultBlock)
    ]
    assert result.is_error and POLICY_DENIED_NOTE in result.content


@pytest.mark.asyncio
async def test_default_approval_blocks_memory_write(
    tmp_path, config, quiet_console
):
    memory_dir = tmp_path / ".emiyacode" / "memory"
    provider = FakeProvider(
        [
            assistant_turn(
                [
                    tool_use(
                        "t1",
                        "memory",
                        action="write",
                        name="fact.md",
                        content="persistent side effect",
                    )
                ],
                stop_reason="tool_use",
            ),
            assistant_turn([TextBlock(text="noted")]),
        ]
    )
    loop = make_loop(
        provider,
        make_registry(MemoryTool(MemoryManager(memory_dir))),
        quiet_console,
    )
    ui = FakeUI(approvals=[False])
    ctx = ToolContext(cwd=tmp_path, config=config, ui=ui)

    await loop.run([user_message("remember")], system="s", ctx=ctx)

    assert ui.approval_requests == ["memory"]
    assert not (memory_dir / "fact.md").exists()


@pytest.mark.asyncio
async def test_read_only_tools_never_ask(tmp_path, config, quiet_console):
    config.tool_approval = "ask"
    provider = FakeProvider(
        [
            assistant_turn([tool_use("t1", "echo", text="x")], stop_reason="tool_use"),
            assistant_turn([TextBlock(text="done")]),
        ]
    )
    loop = make_loop(provider, make_registry(EchoTool()), quiet_console)
    ui = FakeUI()
    ctx = ToolContext(cwd=tmp_path, config=config, ui=ui)
    await loop.run([user_message("go")], system="s", ctx=ctx)
    assert ui.approval_requests == []


def test_exempt_tools_do_not_require_approval():
    from emiya_agent.tools import build_default_registry

    reg = build_default_registry()
    assert reg.needs_approval("shell")
    assert reg.needs_approval("write_file")
    assert reg.needs_approval("edit_file")
    assert not reg.needs_approval("read_file")
    assert not reg.needs_approval("todo_write")  # 显式豁免
    assert not reg.needs_approval("grep")


def test_approval_preview_keeps_dangerous_command_tail():
    command = "safe-prefix " + ("x" * 500) + " ; dangerous-tail"
    preview = _args_brief(tool_use("t1", "shell", command=command), limit=100)

    assert preview.startswith("command=safe-prefix")
    assert "omitted" in preview
    assert preview.endswith("dangerous-tail")


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


@pytest.mark.asyncio
async def test_before_request_runs_for_each_internal_model_turn(
    tmp_path, config, quiet_console
):
    provider = FakeProvider(
        [
            assistant_turn([tool_use("t1", "echo", text="a")], stop_reason="tool_use"),
            assistant_turn([TextBlock(text="fin")]),
        ]
    )
    observed_lengths: list[int] = []

    async def before_request(messages: list[Message]) -> None:
        observed_lengths.append(len(messages))

    loop = make_loop(
        provider,
        make_registry(EchoTool()),
        quiet_console,
        before_request=before_request,
    )

    await loop.run(
        [user_message("go")], system="s", ctx=make_ctx(tmp_path, config)
    )

    assert observed_lengths == [1, 3]


@pytest.mark.asyncio
async def test_global_turn_limit_stops_before_another_provider_request(
    tmp_path, config, quiet_console
):
    provider = FakeProvider(
        [
            assistant_turn(
                [tool_use("t1", "echo", text="a")], stop_reason="tool_use"
            ),
            assistant_turn([TextBlock(text="must not be requested")]),
        ]
    )
    controller = RunController(config, RunLimits(max_turns=1))
    loop = make_loop(
        provider,
        make_registry(EchoTool()),
        quiet_console,
        controller=controller,
    )
    messages = [user_message("go")]

    result = await loop.run(
        messages, system="s", ctx=make_ctx(tmp_path, config)
    )

    assert result.stop_reason == "max_turns"
    assert len(provider.requests) == 1
    assert controller.turns == 1
    assert messages[-1].content[0].tool_use_id == "t1"


@pytest.mark.asyncio
async def test_budget_stop_pairs_but_never_dispatches_tool_calls(
    tmp_path, config, quiet_console
):
    route = "fake/fake-model"
    config.models[route] = ModelInfo(input_price=1.0, output_price=1.0)
    provider = FakeProvider(
        [
            assistant_turn(
                [tool_use("t1", "record")],
                stop_reason="tool_use",
                usage=Usage(output_tokens=1),
            )
        ]
    )
    tool = RecordingTool()
    controller = RunController(
        config, RunLimits(max_budget_usd="0.000001")
    )
    loop = make_loop(
        provider,
        make_registry(tool),
        quiet_console,
        controller=controller,
    )
    messages = [user_message("go")]

    result = await loop.run(
        messages, system="s", ctx=make_ctx(tmp_path, config)
    )

    assert result.stop_reason == "budget_exceeded"
    assert tool.calls == 0
    assert len(messages) == 3
    synthetic = messages[-1].content[0]
    assert isinstance(synthetic, ToolResultBlock)
    assert synthetic.tool_use_id == "t1"
    assert synthetic.is_error is True
    assert "budget_exceeded" in synthetic.content
