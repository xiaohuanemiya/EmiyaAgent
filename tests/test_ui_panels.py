from __future__ import annotations

import io

from rich.console import Console

from emiya_agent.types import ToolResultBlock, ToolUseBlock
from emiya_agent.ui import panels


def _console() -> tuple[Console, io.StringIO]:
    output = io.StringIO()
    return Console(file=output, color_system=None, width=100), output


def test_tool_panels_render_untrusted_markup_and_controls_as_plain_text() -> None:
    console, output = _console()
    call = ToolUseBlock(
        id="call-1",
        name="lsp[red]spoof[/red]\x1b",
        input={"path": "[link=https://example.invalid]file.py[/link]"},
    )
    result = ToolResultBlock(
        tool_use_id="call-1",
        content="[bold red]not markup[/bold red]\x1b]0;fake-title\x07",
    )

    panels.print_tool_call(console, call)
    panels.print_tool_result(console, call, result)

    rendered = output.getvalue()
    assert "[red]spoof[/red]" in rendered
    assert "[bold red]not markup[/bold red]" in rendered
    assert "\\x1b" in rendered
    assert "\\x07" in rendered
    assert "\x1b" not in rendered


def test_todo_panel_does_not_interpret_model_supplied_markup() -> None:
    console, output = _console()

    panels.print_todos(console, "[green]fake success[/green]\x00")

    rendered = output.getvalue()
    assert "[green]fake success[/green]" in rendered
    assert "\\x00" in rendered

