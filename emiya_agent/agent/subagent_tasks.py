"""后台子 Agent 编排：任务生命周期、静默执行与可选 Git worktree。"""

from __future__ import annotations

import hashlib

from emiya_agent.agent.git_worktree import (
    GitWorktreeError,
    WorktreeInfo,
    cleanup_worktree,
    create_worktree,
)
from emiya_agent.agent.prompts import load_agent_md
from emiya_agent.agent.subagent import SubagentSpawner
from emiya_agent.agent.task_runtime import (
    TERMINAL_STATUSES,
    TaskHandle,
    TaskManager,
    TaskRecord,
    TaskSpec,
)
from emiya_agent.config import Config
from emiya_agent.skills.loader import (
    default_skill_dirs,
    discover_skills,
    skills_index_text,
)
from emiya_agent.tools import build_default_registry
from emiya_agent.tools.base import ToolContext
from emiya_agent.tools.skill_tool import SkillTool


class BackgroundUI:
    """后台任务永不读取终端；任何未预先允许的交互都安全拒绝。"""

    async def ask(self, question: str, options: list[str] | None = None) -> str:
        return "Background task cannot ask the user"

    def notify(self, text: str) -> None:
        return None

    async def confirm_tool(self, name: str, args_brief: str) -> bool:
        return False


class SubagentTaskService:
    def __init__(
        self,
        spawner: SubagentSpawner,
        config: Config,
        *,
        max_concurrency: int = 2,
        max_queued: int = 16,
    ) -> None:
        self.spawner = spawner
        self.config = config
        self.manager = TaskManager(
            max_concurrency=max_concurrency, max_queued=max_queued
        )
        self._worktrees: dict[str, WorktreeInfo] = {}
        project_key = hashlib.sha256(
            str(config.cwd.resolve()).casefold().encode("utf-8")
        ).hexdigest()[:16]
        self._worktree_base = Config.user_dir() / "worktrees" / project_key
        self._background_ui = BackgroundUI()

    async def start(
        self,
        role: str,
        prompt: str,
        isolation: str,
        parent_ctx: ToolContext,
    ) -> TaskHandle:
        if isolation == "shared" and role == "general":
            raise ValueError(
                "background general tasks require isolation='worktree'"
            )
        if isolation == "worktree" and role == "general":
            if self.config.tool_approval != "auto":
                raise ValueError(
                    "background general worktree tasks require tool_approval='auto'; "
                    "background tasks never prompt on stdin"
                )

        spec = TaskSpec(
            role=role,
            prompt=prompt,
            background=True,
            isolation=isolation,
        )
        route = self.config.route(f"subagent_{role}")

        async def runner(handle: TaskHandle) -> str:
            handle.set_route(route)
            effective_config = self.config
            effective_tools = self.spawner.tools
            shared_context: str | None = None

            if isolation == "worktree":
                info = await create_worktree(
                    self.config.cwd,
                    self._worktree_base,
                    handle.id,
                    detached=role != "general",
                )
                self._worktrees[handle.id] = info
                try:
                    relative_cwd = self.config.cwd.resolve().relative_to(
                        info.repo_root
                    )
                    task_cwd = (info.path / relative_cwd).resolve()
                    task_cwd.relative_to(info.path)
                except ValueError as exc:
                    raise GitWorktreeError(
                        "task working directory escapes its managed worktree"
                    ) from exc
                task_cwd.mkdir(parents=True, exist_ok=True)
                effective_config = self.config.model_copy(
                    update={
                        "cwd": task_cwd,
                        "allow_external_paths": False,
                    }
                )
                skills = discover_skills(default_skill_dirs(effective_config))
                effective_tools = build_default_registry()
                effective_tools.register(SkillTool(skills))
                sections = [
                    f"# Environment\n- Working directory: {task_cwd}"
                ]
                instructions = load_agent_md(task_cwd)
                if instructions:
                    sections.append(f"# Project instructions\n{instructions}")
                skill_index = skills_index_text(skills)
                if skill_index:
                    sections.append(f"# Skills\n{skill_index}")
                shared_context = "\n\n".join(sections)

            report = await self.spawner.spawn(
                role,
                prompt,
                parent_ctx,
                cancel=handle.cancel_event,
                quiet=True,
                usage_scope=handle.id,
                config=effective_config,
                tools=effective_tools,
                shared_context=shared_context,
                ui=self._background_ui,
            )
            if report.startswith(("Error:", "Subagent refused")):
                raise RuntimeError(report)
            return report

        return await self.manager.start(spec, runner, route=route)

    def list(self) -> list[TaskRecord]:
        return self.manager.list()

    def get(self, task_id: str) -> TaskRecord | None:
        return self.manager.get(task_id)

    async def wait(
        self, task_id: str, timeout: float | None = None
    ) -> TaskRecord:
        return await self.manager.wait(task_id, timeout=timeout)

    async def cancel(self, task_id: str) -> bool:
        return await self.manager.cancel(task_id)

    def worktree(self, task_id: str) -> WorktreeInfo | None:
        return self._worktrees.get(task_id)

    async def cleanup(self, task_id: str, *, force: bool = False) -> None:
        record = self.manager.get(task_id)
        if record is None:
            raise KeyError(task_id)
        if record.status not in TERMINAL_STATUSES:
            raise GitWorktreeError("cannot clean up a running task worktree")
        info = self._worktrees.get(task_id)
        if info is None:
            raise GitWorktreeError("task has no managed worktree")
        await cleanup_worktree(info, force=force)
        del self._worktrees[task_id]

    async def shutdown(self) -> None:
        await self.manager.shutdown()
