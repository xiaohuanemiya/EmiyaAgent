"""工具调用结果 / todo 的面板渲染。"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from emiya_agent.types import ToolResultBlock, ToolUseBlock

PREVIEW_LINES = 6
PREVIEW_CHARS = 500


def safe_terminal_text(value: object) -> str:
    """Render untrusted text without Rich markup or terminal control codes."""
    text = str(value)
    return "".join(
        character
        if character in {"\n", "\t"} or character.isprintable()
        else f"\\x{ord(character):02x}"
        for character in text
    )


def _args_summary(call: ToolUseBlock) -> str:
    parts = []
    for k, v in call.input.items():
        s = safe_terminal_text(v).replace("\n", "\\n")
        if len(s) > 80:
            s = s[:77] + "..."
        parts.append(f"{safe_terminal_text(k)}={s}")
    return ", ".join(parts)


def print_tool_call(console: Console, call: ToolUseBlock) -> None:
    line = Text(f"⚙ {safe_terminal_text(call.name)}", style="bold cyan")
    line.append(f"({_args_summary(call)})", style="dim")
    console.print(line, highlight=False)


def print_tool_result(
    console: Console, call: ToolUseBlock, result: ToolResultBlock
) -> None:
    lines = result.content.splitlines()
    preview = safe_terminal_text(
        "\n".join(lines[:PREVIEW_LINES])[:PREVIEW_CHARS]
    )
    rendered = Text(preview or "(no output)")
    if len(lines) > PREVIEW_LINES or len(result.content) > PREVIEW_CHARS:
        rendered.append(f"\n… 共 {len(lines)} 行", style="dim")
    style = "red" if result.is_error else "dim"
    console.print(
        Panel(rendered, border_style=style, padding=(0, 1)),
        highlight=False,
    )


def print_todos(console: Console, rendered: str) -> None:
    console.print(
        Panel(
            Text(safe_terminal_text(rendered)),
            title="任务进度",
            border_style="blue",
            padding=(0, 1),
        )
    )
