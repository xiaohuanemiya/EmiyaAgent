"""Single read-only tool exposing bounded Language Server Protocol queries."""

from __future__ import annotations

import json
from typing import Any, ClassVar

from emiya_agent.lsp.client import LSPClientError
from emiya_agent.lsp.manager import LSPManager, LSPManagerError
from emiya_agent.lsp.protocol import LSPProtocolError
from emiya_agent.tools.base import Tool, ToolContext, ToolResult, truncate_output

_MANAGER_STATE_KEY = "lsp_manager"
_PENDING_TICKET_STATE_KEY = "_lsp_pending_launch_ticket"
_LAUNCH_REVIEW_REQUIRED_STATE_KEY = "_lsp_launch_review_required"
_POSITION_ACTIONS = frozenset({"definition", "references", "hover"})


class LSPTool(Tool):
    name: ClassVar[str] = "lsp"
    description: ClassVar[str] = (
        "Query a configured local language server for diagnostics, definitions, "
        "references, hover information, or document/workspace symbols. Paths, "
        "lines, and columns are workspace-relative and one-based."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "diagnostics",
                    "definition",
                    "references",
                    "hover",
                    "document_symbols",
                    "workspace_symbols",
                ],
            },
            "path": {
                "type": "string",
                "description": (
                    "Workspace file used as the document and language-server selector"
                ),
            },
            "line": {
                "type": "integer",
                "minimum": 1,
                "description": "One-based line for definition/references/hover",
            },
            "column": {
                "type": "integer",
                "minimum": 1,
                "description": "One-based Unicode-codepoint column",
            },
            "query": {
                "type": "string",
                "description": "Query for workspace_symbols",
            },
            "include_declaration": {
                "type": "boolean",
                "description": "Include the declaration in references (default: true)",
            },
        },
        "required": ["action", "path"],
        "additionalProperties": False,
    }
    read_only: ClassVar[bool] = True
    # Starting a configured server executes local code.  Until the registry can
    # ask the manager whether a connection is already warm, conservatively gate
    # every invocation.
    requires_approval: ClassVar[bool | None] = True

    async def approval_preview(self, args: dict[str, Any], ctx: ToolContext) -> str:
        """Describe the exact prospective launch without starting a process."""

        manager = self._manager(ctx)
        self._discard_pending_ticket(manager, ctx)
        ctx.state[_LAUNCH_REVIEW_REQUIRED_STATE_KEY] = True
        try:
            preview = await manager.connection_preview(
                args.get("path", ""), issue_launch_ticket=True
            )
        except Exception:
            # Dynamic approval must not disrupt the tool loop. This branch is a
            # fail-closed fallback for unexpected preview failures.
            return "LSP launch details unavailable; approval is still required"
        ticket = preview.get("launch_ticket")
        if isinstance(ticket, str):
            ctx.state[_PENDING_TICKET_STATE_KEY] = ticket
        state = "will start" if preview["requires_start"] else "already running"
        command = truncate_output(
            " ".join(repr(part) for part in preview["argv"]), limit=3_000
        )
        return (
            f"LSP server {preview['server']!r} ({state})\n"
            f"root: {preview['root']!r}\ncommand: {command}\n"
            "warning: a language server may execute project plugins, build tools, "
            "or workspace configuration"
        )

    async def approval_required_for(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> bool:
        """Return whether this invocation would launch local code."""

        manager = self._manager(ctx)
        self._discard_pending_ticket(manager, ctx)
        try:
            preview = await manager.connection_preview(args.get("path", ""))
        except LSPManagerError:
            # Disabled/unconfigured/missing/invalid invocations cannot launch;
            # let run() return their bounded actionable error without an ask.
            return False
        except Exception:
            # An unknown preview failure must not accidentally bypass execution
            # approval, but it also must not bubble out of the Agent loop.
            return True
        return bool(preview["requires_start"])

    async def approval_cancelled(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> None:
        self._discard_pending_ticket(self._manager(ctx), ctx)

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        manager = self._manager(ctx)
        ticket = ctx.state.pop(_PENDING_TICKET_STATE_KEY, None)
        review_required = (
            ctx.state.pop(_LAUNCH_REVIEW_REQUIRED_STATE_KEY, False) is True
        )
        if not isinstance(ticket, str):
            ticket = None

        def error(message: str) -> ToolResult:
            manager.discard_launch_ticket(ticket)
            return ToolResult(content=message, is_error=True)

        action = args.get("action")
        path = args.get("path")
        if (
            not isinstance(action, str)
            or action not in self.input_schema["properties"]["action"]["enum"]
        ):
            return error(f"Error: unknown LSP action {action!r}")
        if not isinstance(path, str) or not path.strip():
            return error("Error: LSP path is required")
        if ctx.cancel.is_set():
            return error("Error: LSP request was cancelled")

        if action in _POSITION_ACTIONS:
            line = args.get("line")
            column = args.get("column")
            if (
                not isinstance(line, int)
                or isinstance(line, bool)
                or line < 1
                or not isinstance(column, int)
                or isinstance(column, bool)
                or column < 1
            ):
                return error(
                    (
                        f"Error: LSP action {action!r} requires positive integer "
                        "line and column values"
                    )
                )

        if action == "references" and "include_declaration" in args:
            if not isinstance(args["include_declaration"], bool):
                return error("Error: include_declaration must be a boolean")

        launch_options = {
            "launch_ticket": ticket,
            "require_launch_ticket": review_required
            or getattr(ctx.config, "tool_approval", "auto") == "ask",
        }

        try:
            if action == "diagnostics":
                result: Any = await manager.diagnostics(path, **launch_options)
            elif action == "definition":
                result = await manager.definition(
                    path, args["line"], args["column"], **launch_options
                )
            elif action == "references":
                result = await manager.references(
                    path,
                    args["line"],
                    args["column"],
                    include_declaration=args.get("include_declaration", True),
                    **launch_options,
                )
            elif action == "hover":
                result = await manager.hover(
                    path, args["line"], args["column"], **launch_options
                )
            elif action == "document_symbols":
                result = await manager.document_symbols(path, **launch_options)
            else:
                query = args.get("query")
                if not isinstance(query, str):
                    return error(
                        "Error: workspace_symbols requires a string query"
                    )
                result = await manager.workspace_symbols(
                    path, query, **launch_options
                )
        except (LSPManagerError, LSPClientError, LSPProtocolError) as exc:
            return ToolResult(content=f"Error: {exc}", is_error=True)
        finally:
            manager.discard_launch_ticket(ticket)
        return ToolResult(
            content=json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
        )

    @staticmethod
    def _manager(ctx: ToolContext) -> LSPManager:
        manager = ctx.state.get(_MANAGER_STATE_KEY)
        if not isinstance(manager, LSPManager):
            manager = LSPManager(ctx.config, ctx.cwd)
            ctx.state[_MANAGER_STATE_KEY] = manager
        return manager

    @staticmethod
    def _discard_pending_ticket(manager: LSPManager, ctx: ToolContext) -> None:
        pending = ctx.state.pop(_PENDING_TICKET_STATE_KEY, None)
        ctx.state.pop(_LAUNCH_REVIEW_REQUIRED_STATE_KEY, None)
        manager.discard_launch_ticket(pending if isinstance(pending, str) else None)


__all__ = ["LSPTool"]
