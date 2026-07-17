"""spec 确认界面：中英对照展示 + approve / edit / revise / reject 选择。"""

from __future__ import annotations

from typing import Awaitable, Callable

from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from emiya_agent.pipeline.translation import ConfirmResult, Decision, SpecDraft

# 由 REPL 注入的输入函数
ReadLine = Callable[[str], Awaitable[str]]
ReadMultiline = Callable[[str, str], Awaitable[str]]  # (提示, 初始文本)


class ConfirmUI:
    def __init__(
        self, console: Console, read_line: ReadLine, read_multiline: ReadMultiline
    ) -> None:
        self.console = console
        self.read_line = read_line
        self.read_multiline = read_multiline

    async def __call__(self, draft: SpecDraft) -> ConfirmResult:
        self._show(draft)
        edit_label = "编辑英文spec" if draft.mirror is not None else "编辑spec"
        while True:
            answer = (
                await self.read_line(
                    f"[a]批准执行  [e]{edit_label}  [r]中文反馈重写  [q]放弃 > "
                )
            ).strip().lower()
            if answer in ("a", "approve", "y", "yes", "批准"):
                return ConfirmResult(decision=Decision.APPROVE)
            if answer in ("q", "reject", "n", "no", "放弃"):
                return ConfirmResult(decision=Decision.REJECT)
            if answer in ("e", "edit", "编辑"):
                edited = await self.read_multiline(
                    "编辑 spec（Esc+Enter 或按提示提交）:", draft.spec
                )
                if edited.strip():
                    return ConfirmResult(decision=Decision.EDIT_EN, text=edited)
                self.console.print("[yellow]编辑为空，已忽略。[/yellow]")
                continue
            if answer in ("r", "revise", "反馈"):
                feedback = (await self.read_line("请输入中文修改意见 > ")).strip()
                if feedback:
                    return ConfirmResult(decision=Decision.REVISE_ZH, text=feedback)
                self.console.print("[yellow]反馈为空，已忽略。[/yellow]")
                continue
            self.console.print("[yellow]请输入 a / e / r / q[/yellow]")

    def _show(self, draft: SpecDraft) -> None:
        if draft.mirror is None:
            # native 模式：同语言规格，单栏确认
            self.console.print(
                Panel(
                    Markdown(draft.spec),
                    title="需求规格（确认后执行，免翻译）",
                    border_style="cyan",
                )
            )
            return
        width = self.console.width
        zh_panel = Panel(
            Markdown(draft.mirror),
            title="中文对照",
            border_style="green",
        )
        en_panel = Panel(
            Markdown(draft.spec),
            title="English Spec（实际执行内容）",
            border_style="cyan",
        )
        if width >= 140:
            self.console.print(
                Columns([zh_panel, en_panel], equal=True, expand=True)
            )
        else:  # 窄终端上下排列
            self.console.print(zh_panel)
            self.console.print(en_panel)
