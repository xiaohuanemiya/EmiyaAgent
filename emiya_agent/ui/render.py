"""流式渲染：单一 rich Console，思考灰显、正文直出、工具调用标头。

规则：绝不在 prompt_toolkit 提示符激活时渲染（调用方保证时序）。
"""

from __future__ import annotations

from typing import AsyncIterator

from rich.console import Console
from rich.text import Text

from emiya_agent.providers.base import (
    BlockStop,
    MessageDone,
    StreamEvent,
    StreamStart,
    TextDelta,
    ThinkingDelta,
)
from emiya_agent.ui.panels import safe_terminal_text


class StreamRenderer:
    def __init__(self, console: Console) -> None:
        self.console = console

    async def render(self, events: AsyncIterator[StreamEvent]) -> MessageDone:
        """消费事件流并实时打印，返回最终 MessageDone。"""
        done: MessageDone | None = None
        in_thinking = False
        printed_any = False

        async for ev in events:
            if isinstance(ev, StreamStart):
                if ev.block_type == "thinking":
                    self.console.print("[dim]· 思考中…[/dim]")
                    in_thinking = True
                elif ev.block_type == "tool_use":
                    self.console.print(
                        Text(
                            f"⚙ {safe_terminal_text(ev.tool_name or 'tool')}",
                            style="bold cyan",
                        )
                    )
                elif ev.block_type == "fallback":
                    self.console.print("[yellow]⤷ 已切换到备援模型[/yellow]")
            elif isinstance(ev, ThinkingDelta):
                self.console.print(
                    Text(safe_terminal_text(ev.thinking), style="dim"),
                    end="",
                    highlight=False,
                )
                printed_any = True
            elif isinstance(ev, TextDelta):
                if in_thinking:
                    self.console.print()  # 思考与正文之间换行
                    in_thinking = False
                self.console.print(
                    Text(safe_terminal_text(ev.text)),
                    end="",
                    highlight=False,
                )
                printed_any = True
            elif isinstance(ev, BlockStop):
                if in_thinking:
                    in_thinking = False
                    self.console.print()
            elif isinstance(ev, MessageDone):
                done = ev

        if printed_any:
            self.console.print()  # 收尾换行
        assert done is not None, "事件流未以 MessageDone 结束"
        return done


class QuietRenderer:
    """后台任务使用：完整消费 provider 流，但不触碰共享终端。"""

    async def render(self, events: AsyncIterator[StreamEvent]) -> MessageDone:
        done: MessageDone | None = None
        async for event in events:
            if isinstance(event, MessageDone):
                done = event
        assert done is not None, "事件流未以 MessageDone 结束"
        return done
