"""工具调用结果 / todo 的面板渲染。"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

from emiya_agent.types import ToolResultBlock, ToolUseBlock

PREVIEW_LINES = 6
PREVIEW_CHARS = 500


def _args_summary(call: ToolUseBlock) -> str:
    parts = []
    for k, v in call.input.items():
        s = str(v).replace("\n", "\\n")
        if len(s) > 80:
            s = s[:77] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def print_tool_call(console: Console, call: ToolUseBlock) -> None:
    console.print(
        f"[bold cyan]⚙ {call.name}[/bold cyan][dim]({_args_summary(call)})[/dim]",
        highlight=False,
    )


def print_tool_result(
    console: Console, call: ToolUseBlock, result: ToolResultBlock
) -> None:
    lines = result.content.splitlines()
    preview = "\n".join(lines[:PREVIEW_LINES])[:PREVIEW_CHARS]
    if len(lines) > PREVIEW_LINES or len(result.content) > PREVIEW_CHARS:
        preview += f"\n[dim]… 共 {len(lines)} 行[/dim]"
    style = "red" if result.is_error else "dim"
    console.print(
        Panel(preview or "(no output)", border_style=style, padding=(0, 1)),
        highlight=False,
    )


def print_todos(console: Console, rendered: str) -> None:
    console.print(Panel(rendered, title="任务进度", border_style="blue", padding=(0, 1)))
