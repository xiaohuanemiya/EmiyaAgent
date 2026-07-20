"""spawn_subagent 工具：把 SubagentSpawner 暴露给模型。"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from emiya_agent.agent.task_runtime import TaskStatus
from emiya_agent.tools.base import Tool, ToolContext, ToolResult


class SpawnSubagentTool(Tool):
    name: ClassVar[str] = "spawn_subagent"
    description: ClassVar[str] = (
        "Optionally delegate a self-contained task to a subagent with an isolated "
        "context when doing so materially improves quality or useful concurrency. "
        "Handle greetings, small talk, simple questions, and clear one-step or "
        "local tasks directly; the request language is never a reason to delegate. "
        "There is no fixed explore -> plan -> general chain: select only the role "
        "that is useful and avoid duplicate delegation. "
        "For a precision-sensitive cross-language handoff, use translate_for_model "
        "for the selected target role first; translation is not a mandatory stage. "
        "Roles: 'explore' (read-only codebase search, fast/cheap — use for broad "
        "fan-out searches where you only need conclusions), 'plan' (read-only "
        "design for genuinely complex, high-risk, or cross-module work), 'general' "
        "(full tools for a bounded independent implementation). The prompt must be "
        "self-contained — the subagent cannot see this conversation. You only "
        "receive its final report."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "role": {"type": "string", "enum": ["explore", "plan", "general"]},
            "prompt": {
                "type": "string",
                "description": "Self-contained task description for the subagent",
            },
            "background": {
                "type": "boolean",
                "default": False,
                "description": "Return a task ID immediately instead of waiting",
            },
            "isolation": {
                "type": "string",
                "enum": ["shared", "worktree"],
                "default": "shared",
                "description": (
                    "shared is only safe for background explore/plan; "
                    "background general requires worktree"
                ),
            },
        },
        "required": ["role", "prompt"],
    }
    read_only: ClassVar[bool] = False  # 输出流占用终端，串行执行
    interactive: ClassVar[bool] = True
    # 子agent 本体豁免审批；其内部的写工具在子循环中各自走门控
    requires_approval: ClassVar[bool | None] = False

    def __init__(self, spawner, task_service=None) -> None:
        self._spawner = spawner
        self._tasks = task_service

    def approval_required(self, args: dict[str, Any]) -> bool:
        return args.get("isolation", "shared") == "worktree"

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        background_arg = args.get("background", False)
        if type(background_arg) is not bool:
            return ToolResult(
                content="Error: background must be a boolean",
                is_error=True,
            )
        background = background_arg
        isolation = args.get("isolation", "shared")
        if isolation not in {"shared", "worktree"}:
            return ToolResult(
                content="Error: isolation must be 'shared' or 'worktree'",
                is_error=True,
            )
        if not background and isolation == "shared":
            report = await self._spawner.spawn(args["role"], args["prompt"], ctx)
            is_error = report.startswith("Error:") or report.startswith(
                "Subagent refused"
            )
            return ToolResult(content=report, is_error=is_error)
        if self._tasks is None:
            return ToolResult(
                content="Error: background task runtime is unavailable",
                is_error=True,
            )
        try:
            handle = await self._tasks.start(
                args["role"], args["prompt"], isolation, ctx
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            return ToolResult(content=f"Error: {exc}", is_error=True)
        if background:
            return ToolResult(
                content=(
                    f"Background subagent started: task_id={handle.id}, "
                    f"role={handle.spec.role}, isolation={handle.spec.isolation}. "
                    "Use subagent_task to inspect or wait for it."
                )
            )

        try:
            record = await handle.wait()
        except asyncio.CancelledError:
            await handle.cancel()
            raise
        report = record.report or record.error or f"Task ended: {record.status.value}"
        is_error = record.status != TaskStatus.SUCCEEDED or report.startswith(
            ("Error:", "Subagent refused")
        )
        return ToolResult(content=report, is_error=is_error)


class SubagentTaskTool(Tool):
    name: ClassVar[str] = "subagent_task"
    description: ClassVar[str] = (
        "Inspect, wait for, cancel, or clean up background subagent tasks. "
        "Actions: list, get, wait, cancel, cleanup."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "get", "wait", "cancel", "cleanup"],
            },
            "task_id": {"type": "string"},
            "timeout_seconds": {
                "type": "number",
                "minimum": 0,
                "maximum": 600,
                "default": 30,
            },
            "force": {"type": "boolean", "default": False},
        },
        "required": ["action"],
    }
    read_only: ClassVar[bool] = False
    requires_approval: ClassVar[bool | None] = False

    def __init__(self, task_service) -> None:
        self._tasks = task_service

    def approval_required(self, args: dict[str, Any]) -> bool:
        return args.get("action") == "cleanup"

    @staticmethod
    def _summary(record) -> str:
        prompt = record.spec.prompt.strip().splitlines()[0][:80]
        return (
            f"{record.id}  {record.status.value:11s}  {record.spec.role:7s}  "
            f"{record.spec.isolation:8s}  {prompt}"
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        action = args["action"]
        task_id = str(args.get("task_id") or "")
        if action == "list":
            records = self._tasks.list()
            return ToolResult(
                content="\n".join(self._summary(record) for record in records)
                or "(no subagent tasks)"
            )
        if not task_id:
            return ToolResult(content="Error: task_id is required", is_error=True)

        if action == "get":
            record = self._tasks.get(task_id)
            if record is None:
                return ToolResult(
                    content=f"Error: unknown task {task_id}", is_error=True
                )
            worktree = self._tasks.worktree(task_id)
            detail = self._summary(record)
            if worktree is not None:
                detail += f"\nworktree: {worktree.path}\nbranch: {worktree.branch or '(detached)'}"
            if record.report:
                detail += f"\n\n{record.report}"
            if record.error:
                detail += f"\nerror: {record.error}"
            return ToolResult(content=detail)

        if action == "wait":
            timeout = args.get("timeout_seconds", 30)
            try:
                record = await self._tasks.wait(
                    task_id,
                    timeout=float(timeout) if timeout is not None else None,
                )
            except TimeoutError:
                record = self._tasks.get(task_id)
                return ToolResult(
                    content=(
                        f"Task {task_id} is still "
                        f"{record.status.value if record else 'unknown'}"
                    )
                )
            except KeyError:
                return ToolResult(
                    content=f"Error: unknown task {task_id}", is_error=True
                )
            content = record.report or record.error or self._summary(record)
            return ToolResult(
                content=content,
                is_error=record.status not in {TaskStatus.SUCCEEDED},
            )

        if action == "cancel":
            try:
                changed = await self._tasks.cancel(task_id)
            except KeyError:
                return ToolResult(
                    content=f"Error: unknown task {task_id}", is_error=True
                )
            return ToolResult(
                content=(
                    f"Cancellation requested for {task_id}"
                    if changed
                    else f"Task {task_id} is already terminal"
                )
            )

        if action == "cleanup":
            force_arg = args.get("force", False)
            if type(force_arg) is not bool:
                return ToolResult(
                    content="Error: force must be a boolean",
                    is_error=True,
                )
            try:
                await self._tasks.cleanup(task_id, force=force_arg)
            except (KeyError, RuntimeError, ValueError) as exc:
                return ToolResult(content=f"Error: {exc}", is_error=True)
            return ToolResult(content=f"Cleaned up worktree for {task_id}")

        return ToolResult(content=f"Error: unknown action {action!r}", is_error=True)
