"""Agent 工具循环。

不变量（单测覆盖，绝不破坏）：
1. 每个 tool_use 必有且仅有一个配对 tool_result —— 中断/异常时补 is_error 合成结果，
   否则下一次请求会 400。
2. 同一助手回合的所有 tool_result 合入同一条 user 消息。
3. 并行安全工具（只读且非交互）并发执行；其余按顺序串行。
4. 助手消息（含 thinking 块）原样追加进历史。
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from emiya_agent.agent.control import RunController, RunLimitExceeded
from emiya_agent.hooks import HookDecision, HookEvent, HookRunner
from emiya_agent.providers.base import ChatRequest, MessageDone, Provider
from emiya_agent.tools.base import ToolContext, truncate_output
from emiya_agent.tools.registry import ToolRegistry
from emiya_agent.types import Message, ToolResultBlock, ToolUseBlock, Usage

INTERRUPTED_NOTE = "Interrupted by user"
DENIED_NOTE = "User denied this tool call"
POLICY_DENIED_NOTE = "Blocked by the user's permission policy"
LIMIT_STOP_NOTE = "Tool call not executed because the run stopped: {reason}"
INVALID_STOP_NOTE = (
    "Tool call not executed because the model response ended with "
    "stop_reason={stop_reason!r}, not 'tool_use'"
)
APPROVAL_PREVIEW_CHARS = 8_000
PARALLEL_CANCEL_GRACE_SECONDS = 2.0
PARALLEL_CANCEL_FORCE_SECONDS = 0.5


def _args_brief(
    call: ToolUseBlock, limit: int = APPROVAL_PREVIEW_CHARS
) -> str:
    """审批摘要保留危险命令的首尾，并转义终端控制字符。"""
    parts: list[str] = []
    for k, v in call.input.items():
        raw = str(v)
        safe = "".join(ch if ch.isprintable() else repr(ch)[1:-1] for ch in raw)
        parts.append(f"{k}={safe}")
    text = ", ".join(parts)
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - (half * 2)
    return (
        f"{text[:half]}\n... [approval preview omitted {omitted} chars] ...\n"
        f"{text[-half:]}"
    )


def _invalid_call_batch(calls: list[ToolUseBlock]) -> str | None:
    if any(not call.id.strip() for call in calls):
        return "empty tool_use id"
    ids = [call.id for call in calls]
    if len(ids) != len(set(ids)):
        return "duplicate tool_use ids"
    if any(not call.name.strip() for call in calls):
        return "empty tool name"
    if any(
        not _is_strict_json(call.id) or not _is_strict_json(call.name)
        for call in calls
    ):
        return "at least one tool_use id or name is not valid UTF-8 JSON text"
    if any("_raw" in call.input for call in calls):
        return "at least one tool input contains malformed raw JSON"
    try:
        strict_inputs = all(_is_strict_json(call.input) for call in calls)
    except RecursionError:
        strict_inputs = False
    if not strict_inputs:
        return "at least one tool input is not strict JSON"
    return None


def _is_strict_json(value: object, ancestors: set[int] | None = None) -> bool:
    """Accept only finite, UTF-8 representable values from the JSON data model."""

    if value is None or type(value) in {bool, int}:
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is str:
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            return False
        return True
    if type(value) not in {list, dict}:
        return False

    seen = ancestors if ancestors is not None else set()
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    try:
        if type(value) is list:
            return all(_is_strict_json(item, seen) for item in value)
        return all(
            type(key) is str
            and _is_strict_json(key, seen)
            and _is_strict_json(item, seen)
            for key, item in value.items()
        )
    finally:
        seen.remove(identity)


def _current_task_is_cancelling() -> bool:
    """Whether this coroutine's owner received a real ``Task.cancel()``."""

    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _consume_future_result(future: asyncio.Future[object]) -> None:
    """Retrieve a detached cleanup result so asyncio does not log it later."""

    try:
        future.result()
    except BaseException:
        pass


async def _cancel_and_settle_parallel_tasks(
    tasks: list[asyncio.Task[ToolResultBlock]],
) -> bool:
    """Cancel an owned batch and reap it within two bounded grace periods.

    A child task that raises ``CancelledError`` does not make ``gather`` cancel
    its siblings.  Returning before those siblings finish cancellation would
    let tool or PostToolUse work continue after paired interruption results have
    already entered history.  A second cancellation is used after the normal
    cleanup grace period so a task stuck in an async ``finally`` cannot make the
    owning AgentLoop uninterruptible forever.
    """

    def cancel_pending() -> None:
        for task in tasks:
            if not task.done():
                task.cancel()

    async def wait_until(
        settlement: asyncio.Future[list[ToolResultBlock | BaseException]],
        deadline: float,
    ) -> bool:
        loop = asyncio.get_running_loop()
        while not settlement.done():
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            try:
                await asyncio.wait_for(
                    asyncio.shield(settlement), timeout=remaining
                )
            except (asyncio.TimeoutError, TimeoutError):
                return settlement.done()
            except (KeyboardInterrupt, asyncio.CancelledError):
                # Repeated user cancellation escalates to every child while
                # preserving the fixed deadline for this cleanup phase.
                cancel_pending()
        return True

    cancel_pending()
    settlement = asyncio.gather(*tasks, return_exceptions=True)
    loop = asyncio.get_running_loop()
    settled = await wait_until(
        settlement, loop.time() + PARALLEL_CANCEL_GRACE_SECONDS
    )
    if not settled:
        cancel_pending()
        settled = await wait_until(
            settlement, loop.time() + PARALLEL_CANCEL_FORCE_SECONDS
        )

    if settled:
        _consume_future_result(settlement)
        return True

    # A coroutine can deliberately catch every CancelledError; asyncio has no
    # safe hard-kill primitive.  Do not make the whole AgentLoop immortal.  A
    # final cancellation plus result consumer prevents unobserved exceptions if
    # the read-only task eventually exits.
    cancel_pending()
    settlement.add_done_callback(_consume_future_result)
    return False


# 渲染回调：消费一次流式请求，返回 MessageDone（REPL 注入 StreamRenderer.render）
RenderFn = Callable[..., Awaitable[MessageDone]]
# 每产出一条消息的回调（会话持久化用）
OnMessage = Callable[[Message], None]
# 每次模型请求前的上下文维护回调（主循环用于轮内压缩）
BeforeRequest = Callable[[list[Message]], Awaitable[None]]


@dataclass
class LoopResult:
    final_text: str
    stop_reason: str  # provider stop, interruption, protocol error, or run limit
    served_by: str | None = None
    refusal: str | None = None


class AgentLoop:
    def __init__(
        self,
        provider: Provider,
        model: str,
        route: str,
        tools: ToolRegistry,
        render: RenderFn,
        *,
        max_tokens: int = 16000,
        effort: str | None = None,
        max_iterations: int = 40,
        on_message: OnMessage | None = None,
        on_usage: Callable[[str, Usage], None] | None = None,
        on_tool_result: Callable[[ToolUseBlock, ToolResultBlock], None] | None = None,
        before_request: BeforeRequest | None = None,
        controller: RunController | None = None,
        hooks: HookRunner | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.route = route
        self.tools = tools
        self.render = render
        self.max_tokens = max_tokens
        self.effort = effort
        self.max_iterations = max_iterations
        self.on_message = on_message or (lambda m: None)
        self.on_usage = on_usage or (lambda r, u: None)
        self.on_tool_result = on_tool_result or (lambda c, r: None)
        self.before_request = before_request
        self.controller = controller
        self.hooks = hooks

    def _observe_tool_result(
        self, call: ToolUseBlock, result: ToolResultBlock
    ) -> None:
        """Notify optional render/event observers without changing protocol state."""

        try:
            self.on_tool_result(call, result)
        except Exception:
            # The result still has to reach history/provider when a terminal
            # consumer closes early or a UI renderer fails.
            pass

    @staticmethod
    def _notify(ctx: ToolContext, text: str) -> None:
        """Hook diagnostics are best-effort and never gate tool-result pairing."""

        try:
            ctx.ui.notify(text)
        except Exception:
            pass

    async def run(
        self, messages: list[Message], *, system: str, ctx: ToolContext
    ) -> LoopResult:
        """messages 原地增长（调用方持有同一列表实现历史延续）。"""
        last = LoopResult(final_text="", stop_reason="max_iterations")
        for _ in range(self.max_iterations):
            try:
                if self.before_request is not None:
                    await self.before_request(messages)
                if self.controller is not None:
                    await self.controller.before_request(self.route)
            except RunLimitExceeded as exc:
                return LoopResult(
                    final_text=last.final_text,
                    stop_reason=exc.reason,
                    served_by=last.served_by,
                )
            req = ChatRequest(
                model=self.model,
                system=system,
                messages=messages,
                tools=self.tools.specs(),
                max_tokens=self.max_tokens,
                effort=self.effort,
            )
            try:
                done = await self.render(self.provider.stream(req))
            except asyncio.CancelledError:
                if _current_task_is_cancelling():
                    raise
                # A provider/renderer may use CancelledError as an internal
                # interruption signal without cancelling this AgentLoop task.
                return LoopResult(final_text="", stop_reason="interrupted")
            except KeyboardInterrupt:
                # 流式中被打断：assistant 消息未提交，历史保持合法
                return LoopResult(final_text="", stop_reason="interrupted")
            if _current_task_is_cancelling():
                # A provider is allowed to clean up, but must not erase its
                # caller's cancellation request by suppressing CancelledError.
                raise asyncio.CancelledError

            self.on_usage(self.route, done.usage)
            if self.controller is not None:
                self.controller.record_usage(self.route, done.usage)

            if done.stop_reason == "refusal":
                # 拒绝回合不入历史（内容为空或不完整）
                return LoopResult(
                    final_text="",
                    stop_reason="refusal",
                    refusal=done.refusal,
                    served_by=done.served_by,
                )

            last = LoopResult(
                final_text=done.message.text(),
                stop_reason=done.stop_reason,
                served_by=done.served_by,
            )

            calls = done.message.tool_uses()
            invalid_reason = _invalid_call_batch(calls)
            if invalid_reason is not None:
                # 重复/空 ID 无法与 tool_result 唯一配对；损坏的 JSON 也不能
                # 安全回放。整条 assistant 消息不进入活跃历史，避免下一轮或
                # 恢复会话时把非法协议状态再次发给 provider。
                return LoopResult(
                    final_text=last.final_text,
                    stop_reason="error",
                    served_by=last.served_by,
                )

            messages.append(done.message)
            self.on_message(done.message)
            limit_reason = (
                self.controller.stop_reason
                if self.controller is not None
                else None
            )
            if limit_reason is not None:
                if calls:
                    skipped = [
                        ToolResultBlock(
                            tool_use_id=call.id,
                            content=LIMIT_STOP_NOTE.format(reason=limit_reason),
                            is_error=True,
                        )
                        for call in calls
                    ]
                    for call, result in zip(calls, skipped):
                        self._observe_tool_result(call, result)
                    user_msg = Message(role="user", content=list(skipped))
                    messages.append(user_msg)
                    self.on_message(user_msg)
                return LoopResult(
                    final_text=last.final_text,
                    stop_reason=limit_reason,
                    served_by=last.served_by,
                )
            if done.stop_reason != "tool_use":
                if calls:
                    # 非 tool_use 结束（尤其 max_tokens）中的调用参数可能被截断。
                    # 绝不 dispatch；补配对错误结果以保持后续恢复/切模时历史合法。
                    skipped = [
                        ToolResultBlock(
                            tool_use_id=call.id,
                            content=INVALID_STOP_NOTE.format(
                                stop_reason=done.stop_reason
                            ),
                            is_error=True,
                        )
                        for call in calls
                    ]
                    for call, result in zip(calls, skipped):
                        self._observe_tool_result(call, result)
                    user_msg = Message(role="user", content=list(skipped))
                    messages.append(user_msg)
                    self.on_message(user_msg)
                return last
            if not calls:  # 声称 tool_use 却没有调用块：按结束处理
                return last

            results, interrupted, externally_cancelled = await self._execute_calls(
                calls, ctx
            )
            user_msg = Message(role="user", content=list(results))
            messages.append(user_msg)
            try:
                self.on_message(user_msg)
            finally:
                if externally_cancelled:
                    # The paired synthetic results above must enter history
                    # before exposing cancellation to the owning lifecycle.
                    raise asyncio.CancelledError
            if interrupted or ctx.cancel.is_set():
                return LoopResult(final_text=last.final_text, stop_reason="interrupted")

        return LoopResult(final_text=last.final_text, stop_reason="max_iterations")

    # ------------------------------------------------------------- 工具执行

    async def _cancel_approval_state(
        self, call: ToolUseBlock, ctx: ToolContext
    ) -> None:
        try:
            await self.tools.approval_cancelled(call.name, call.input, ctx)
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception:
            # Cleanup cannot turn a denied approval into an unpaired tool call.
            pass

    async def _prepare_call(
        self, call: ToolUseBlock, ctx: ToolContext
    ) -> tuple[str, str, bool]:
        try:
            base = await self.tools.permission_decision_for(
                call.name, call.input, ctx
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception as exc:
            return (
                "deny",
                f"Tool approval policy error: {type(exc).__name__}: {exc}",
                False,
            )
        base_reason = POLICY_DENIED_NOTE if base == "deny" else ""
        if self.hooks is None:
            return base, base_reason, False
        try:
            outcome = await self.hooks.run(
                HookEvent.PRE_TOOL_USE,
                call.name,
                {
                    "tool_use_id": call.id,
                    "tool_input": call.input,
                    "cwd": str(ctx.cwd),
                    "route": self.route,
                },
                initial_decision=HookDecision(base),
                initial_reason=base_reason,
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception as exc:
            return "deny", f"Hook infrastructure error: {type(exc).__name__}: {exc}", True
        for diagnostic in outcome.diagnostics:
            if diagnostic.status != "ok":
                self._notify(
                    ctx,
                    f"Hook {diagnostic.hook_id} ({diagnostic.status}): "
                    f"{diagnostic.reason}",
                )
        hook_denied = outcome.decision == HookDecision.DENY and base != "deny"
        return outcome.decision.value, outcome.reason, hook_denied

    async def _post_hook(
        self,
        call: ToolUseBlock,
        result: ToolResultBlock,
        ctx: ToolContext,
        *,
        duration_ms: int,
    ) -> ToolResultBlock:
        if self.hooks is None:
            return result
        try:
            outcome = await self.hooks.run(
                HookEvent.POST_TOOL_USE,
                call.name,
                {
                    "tool_use_id": call.id,
                    "tool_input": call.input,
                    "tool_response": {
                        "content": result.content,
                        "is_error": result.is_error,
                        "executed": True,
                    },
                    "duration_ms": duration_ms,
                    "cwd": str(ctx.cwd),
                    "route": self.route,
                },
                initial_decision=HookDecision.ALLOW,
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception as exc:
            failure = (
                "PostToolUse hook infrastructure error: "
                f"{type(exc).__name__}: {exc}"
            )
            outcome = self.hooks.failure_result(
                HookEvent.POST_TOOL_USE,
                call.name,
                failure,
                initial_decision=HookDecision.ALLOW,
            )
        for diagnostic in outcome.diagnostics:
            if diagnostic.status != "ok":
                self._notify(
                    ctx,
                    f"Hook {diagnostic.hook_id} ({diagnostic.status}): "
                    f"{diagnostic.reason}",
                )
        if outcome.decision != HookDecision.DENY:
            return result
        reason = outcome.reason or "PostToolUse hook rejected the result"
        content = truncate_output(
            f"{result.content}\n\n"
            f"Error: tool already executed, but a PostToolUse hook reported: {reason}"
        )
        return ToolResultBlock(
            tool_use_id=call.id,
            content=content,
            is_error=True,
        )

    async def _dispatch_with_hooks(
        self,
        call: ToolUseBlock,
        ctx: ToolContext,
        *,
        cancellation_barrier: asyncio.Event | None = None,
    ) -> ToolResultBlock:
        started = time.monotonic()
        result = await self.tools.dispatch(call, ctx)
        if cancellation_barrier is not None and cancellation_barrier.is_set():
            # A cancellation-resistant read may return after its batch already
            # timed out.  Never run PostToolUse hooks in that detached state.
            raise asyncio.CancelledError
        duration_ms = max(0, round((time.monotonic() - started) * 1000))
        return await self._post_hook(
            call, result, ctx, duration_ms=duration_ms
        )

    async def _execute_calls(
        self, calls: list[ToolUseBlock], ctx: ToolContext
    ) -> tuple[list[ToolResultBlock], bool, bool]:
        """执行一批调用。

        返回 (与 calls 一一对应的结果, 是否被中断,
        是否由当前 asyncio Task 的外部取消触发)。
        """
        results: dict[str, ToolResultBlock] = {}
        externally_cancelled = _current_task_is_cancelling()
        interrupted = externally_cancelled

        # Read-only/non-interactive calls may still be preflighted together and
        # dispatched concurrently. Stateful serial calls are deliberately not
        # prepared here: each PreToolUse policy must observe the effects of all
        # earlier serial calls in this batch.
        prepared: dict[str, tuple[str, str, bool]] = {}
        # Do not move a read ahead of an earlier write: in a mixed batch its
        # policy and result may depend on that write. Only an entirely safe
        # batch may be preflighted and dispatched as one parallel group.
        parallel_candidates = (
            list(calls)
            if all(self.tools.is_parallel_safe(call.name) for call in calls)
            else []
        )
        for call in parallel_candidates:
            if _current_task_is_cancelling():
                externally_cancelled = True
                interrupted = True
                break
            if ctx.cancel.is_set():
                interrupted = True
                break
            try:
                prepared[call.id] = await self._prepare_call(call, ctx)
                if _current_task_is_cancelling():
                    externally_cancelled = True
                    interrupted = True
                    break
            except (KeyboardInterrupt, asyncio.CancelledError):
                interrupted = True
                externally_cancelled = _current_task_is_cancelling()
                break

        parallel = [
            c
            for c in parallel_candidates
            if prepared.get(c.id, ("deny", "", False))[0] == "allow"
        ]
        serial = [c for c in calls if c not in parallel]

        if parallel and not interrupted and not ctx.cancel.is_set():
            parallel_cancelled = asyncio.Event()
            parallel_tasks = [
                asyncio.create_task(
                    self._dispatch_with_hooks(
                        call,
                        ctx,
                        cancellation_barrier=parallel_cancelled,
                    )
                )
                for call in parallel
            ]
            parallel_batch = asyncio.gather(*parallel_tasks)
            try:
                # Keep ownership of child cancellation here. If this task is
                # cancelled, gather must not detach or repeatedly cancel tools
                # while their finally/PostToolUse cleanup is still running.
                done_list = await asyncio.shield(parallel_batch)
                for call, res in zip(parallel, done_list):
                    results[call.id] = res
                    self._observe_tool_result(call, res)
            except (KeyboardInterrupt, asyncio.CancelledError):
                interrupted = True
                externally_cancelled = _current_task_is_cancelling()
                parallel_cancelled.set()
                settled = await _cancel_and_settle_parallel_tasks(parallel_tasks)
                if not settled:
                    self._notify(
                        ctx,
                        "Parallel tool cleanup exceeded its cancellation deadline; "
                        "the batch was interrupted and paired with error results.",
                    )
                externally_cancelled = (
                    externally_cancelled or _current_task_is_cancelling()
                )
                # ``parallel_batch`` may have completed with a child's
                # CancelledError while the settlement gather reaped siblings.
                if parallel_batch.done():
                    _consume_future_result(parallel_batch)
                else:
                    parallel_batch.add_done_callback(_consume_future_result)

        for call in serial:
            if _current_task_is_cancelling():
                externally_cancelled = True
                interrupted = True
                break
            if interrupted or ctx.cancel.is_set():
                break
            try:
                decision_info = prepared.get(call.id)
                if decision_info is None:
                    decision_info = await self._prepare_call(call, ctx)
                    if _current_task_is_cancelling():
                        externally_cancelled = True
                        interrupted = True
                        break
                decision, reason, hook_denied = decision_info
                if decision == "deny":
                    content = (
                        f"Blocked by PreToolUse hook: {reason}"
                        if hook_denied
                        else reason or POLICY_DENIED_NOTE
                    )
                    res = ToolResultBlock(
                        tool_use_id=call.id,
                        content=truncate_output(content),
                        is_error=True,
                    )
                    results[call.id] = res
                    self._observe_tool_result(call, res)
                    continue
                if decision == "ask":
                    try:
                        preview = await self.tools.approval_preview(
                            call.name, call.input, ctx
                        )
                        if preview is not None and not isinstance(preview, str):
                            raise TypeError("approval preview must be a string or None")
                    except (KeyboardInterrupt, asyncio.CancelledError):
                        await self._cancel_approval_state(call, ctx)
                        raise
                    except Exception as exc:
                        await self._cancel_approval_state(call, ctx)
                        res = ToolResultBlock(
                            tool_use_id=call.id,
                            content=truncate_output(
                                "Tool approval preview error: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                            is_error=True,
                        )
                        results[call.id] = res
                        self._observe_tool_result(call, res)
                        continue
                    if _current_task_is_cancelling():
                        await self._cancel_approval_state(call, ctx)
                        externally_cancelled = True
                        interrupted = True
                        break
                    try:
                        approved = await ctx.ui.confirm_tool(
                            call.name,
                            truncate_output(preview, limit=4_000)
                            if preview
                            else _args_brief(call),
                        )
                    except (KeyboardInterrupt, asyncio.CancelledError):
                        await self._cancel_approval_state(call, ctx)
                        raise
                    except Exception as exc:
                        await self._cancel_approval_state(call, ctx)
                        res = ToolResultBlock(
                            tool_use_id=call.id,
                            content=truncate_output(
                                "Tool approval UI error: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                            is_error=True,
                        )
                        results[call.id] = res
                        self._observe_tool_result(call, res)
                        continue
                    if _current_task_is_cancelling():
                        await self._cancel_approval_state(call, ctx)
                        externally_cancelled = True
                        interrupted = True
                        break
                    if approved is not True:
                        await self._cancel_approval_state(call, ctx)
                        res = ToolResultBlock(
                            tool_use_id=call.id, content=DENIED_NOTE, is_error=True
                        )
                        results[call.id] = res
                        self._observe_tool_result(call, res)
                        continue
                res = await self._dispatch_with_hooks(call, ctx)
                results[call.id] = res
                self._observe_tool_result(call, res)
                if _current_task_is_cancelling():
                    externally_cancelled = True
                    interrupted = True
                    break
            except (KeyboardInterrupt, asyncio.CancelledError):
                interrupted = True
                externally_cancelled = _current_task_is_cancelling()
                break

        # 不变量 1：为所有未执行的调用补合成结果
        out: list[ToolResultBlock] = []
        for call in calls:
            res = results.get(call.id)
            if res is None:
                interrupted = True
                res = ToolResultBlock(
                    tool_use_id=call.id, content=INTERRUPTED_NOTE, is_error=True
                )
            out.append(res)
        return (
            out,
            interrupted or ctx.cancel.is_set(),
            externally_cancelled or _current_task_is_cancelling(),
        )
