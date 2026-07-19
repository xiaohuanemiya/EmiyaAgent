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
from dataclasses import dataclass
from typing import Awaitable, Callable

from emiya_agent.providers.base import ChatRequest, MessageDone, Provider
from emiya_agent.tools.base import ToolContext
from emiya_agent.tools.registry import ToolRegistry
from emiya_agent.types import Message, ToolResultBlock, ToolUseBlock, Usage

INTERRUPTED_NOTE = "Interrupted by user"
DENIED_NOTE = "User denied this tool call"


def _args_brief(call: ToolUseBlock, limit: int = 200) -> str:
    parts = []
    for k, v in call.input.items():
        s = str(v).replace("\n", "\\n")
        parts.append(f"{k}={s[:limit]}")
    return ", ".join(parts)

# 渲染回调：消费一次流式请求，返回 MessageDone（REPL 注入 StreamRenderer.render）
RenderFn = Callable[..., Awaitable[MessageDone]]
# 每产出一条消息的回调（会话持久化用）
OnMessage = Callable[[Message], None]


@dataclass
class LoopResult:
    final_text: str
    stop_reason: str  # end_turn | interrupted | max_iterations | refusal | max_tokens | error
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

    async def run(
        self, messages: list[Message], *, system: str, ctx: ToolContext
    ) -> LoopResult:
        """messages 原地增长（调用方持有同一列表实现历史延续）。"""
        last = LoopResult(final_text="", stop_reason="max_iterations")
        for _ in range(self.max_iterations):
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
            except (KeyboardInterrupt, asyncio.CancelledError):
                # 流式中被打断：assistant 消息未提交，历史保持合法
                return LoopResult(final_text="", stop_reason="interrupted")

            if done.stop_reason == "refusal":
                # 拒绝回合不入历史（内容为空或不完整）
                return LoopResult(
                    final_text="",
                    stop_reason="refusal",
                    refusal=done.refusal,
                    served_by=done.served_by,
                )

            messages.append(done.message)
            self.on_message(done.message)
            self.on_usage(self.route, done.usage)
            last = LoopResult(
                final_text=done.message.text(),
                stop_reason=done.stop_reason,
                served_by=done.served_by,
            )

            calls = done.message.tool_uses()
            if done.stop_reason != "tool_use" and not calls:
                return last
            if not calls:  # 声称 tool_use 却没有调用块：按结束处理
                return last

            results, interrupted = await self._execute_calls(calls, ctx)
            user_msg = Message(role="user", content=list(results))
            messages.append(user_msg)
            self.on_message(user_msg)

            if interrupted or ctx.cancel.is_set():
                return LoopResult(final_text=last.final_text, stop_reason="interrupted")

        return LoopResult(final_text=last.final_text, stop_reason="max_iterations")

    # ------------------------------------------------------------- 工具执行

    async def _execute_calls(
        self, calls: list[ToolUseBlock], ctx: ToolContext
    ) -> tuple[list[ToolResultBlock], bool]:
        """执行一批调用。返回 (结果列表——与 calls 一一对应, 是否被中断)。"""
        results: dict[str, ToolResultBlock] = {}
        interrupted = False

        parallel = [c for c in calls if self.tools.is_parallel_safe(c.name)]
        serial = [c for c in calls if not self.tools.is_parallel_safe(c.name)]

        if parallel and not ctx.cancel.is_set():
            try:
                done_list = await asyncio.gather(
                    *(self.tools.dispatch(c, ctx) for c in parallel)
                )
                for call, res in zip(parallel, done_list):
                    results[call.id] = res
                    self.on_tool_result(call, res)
            except (KeyboardInterrupt, asyncio.CancelledError):
                interrupted = True

        for call in serial:
            if interrupted or ctx.cancel.is_set():
                break
            try:
                # 审批门控：写操作/shell/MCP 在 ask 模式下须用户确认
                if (
                    ctx.config.tool_approval == "ask"
                    and self.tools.needs_approval(call.name)
                ):
                    approved = await ctx.ui.confirm_tool(
                        call.name, _args_brief(call)
                    )
                    if not approved:
                        res = ToolResultBlock(
                            tool_use_id=call.id, content=DENIED_NOTE, is_error=True
                        )
                        results[call.id] = res
                        self.on_tool_result(call, res)
                        continue
                res = await self.tools.dispatch(call, ctx)
                results[call.id] = res
                self.on_tool_result(call, res)
            except (KeyboardInterrupt, asyncio.CancelledError):
                interrupted = True
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
        return out, interrupted or ctx.cancel.is_set()
