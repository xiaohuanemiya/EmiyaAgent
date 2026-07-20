from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from emiya_agent.agent.git_worktree import WorktreeInfo
from emiya_agent.agent.subagent_tasks import SubagentTaskService
from emiya_agent.agent.task_runtime import TaskStatus
from emiya_agent.config import Config
from emiya_agent.tools.base import ToolContext
from emiya_agent.tools.task import SpawnSubagentTool, SubagentTaskTool

from tests.conftest import FakeUI


class FakeSpawner:
    def __init__(self) -> None:
        self.tools = object()
        self.calls: list[dict[str, Any]] = []
        self.started = asyncio.Event()

    async def spawn(
        self,
        role: str,
        prompt: str,
        parent_ctx: ToolContext,
        **kwargs: Any,
    ) -> str:
        self.calls.append(
            {
                "role": role,
                "prompt": prompt,
                "parent_ctx": parent_ctx,
                "kwargs": kwargs,
            }
        )
        self.started.set()
        if prompt == "block until cancelled":
            await asyncio.Event().wait()
        return f"report: {prompt}"


@pytest.fixture
def task_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    user_dir = tmp_path / "isolated-user-dir"
    monkeypatch.setattr(Config, "user_dir", staticmethod(lambda: user_dir))
    return Config(
        cwd=tmp_path,
        routes={
            "subagent_explore": "fake/explore",
            "subagent_plan": "fake/plan",
            "subagent_general": "fake/general",
        },
    )


@pytest.fixture
def tool_ctx(task_config: Config) -> ToolContext:
    return ToolContext(
        cwd=task_config.cwd,
        config=task_config,
        ui=FakeUI(),
    )


async def test_shared_explore_background_succeeds(
    task_config: Config, tool_ctx: ToolContext
) -> None:
    spawner = FakeSpawner()
    service = SubagentTaskService(spawner, task_config)  # type: ignore[arg-type]
    tool = SpawnSubagentTool(spawner, service)

    try:
        started = await tool.run(
            {
                "role": "explore",
                "prompt": "inspect imports",
                "background": True,
                "isolation": "shared",
            },
            tool_ctx,
        )

        assert started.is_error is False
        assert "Background subagent started" in started.content
        record = await service.wait(service.list()[0].id)
        assert record.status == TaskStatus.SUCCEEDED
        assert record.report == "report: inspect imports"
        assert record.route == "fake/explore"
        assert spawner.calls[0]["kwargs"]["quiet"] is True
        assert spawner.calls[0]["kwargs"]["usage_scope"] == record.id
        assert spawner.calls[0]["kwargs"]["cancel"] is not tool_ctx.cancel
    finally:
        await service.shutdown()


@pytest.mark.parametrize(
    ("isolation", "approval", "expected"),
    [
        ("shared", "auto", "background general tasks require isolation='worktree'"),
        (
            "worktree",
            "ask",
            "background general worktree tasks require tool_approval='auto'",
        ),
    ],
)
async def test_background_general_rejects_unsafe_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolation: str,
    approval: str,
    expected: str,
) -> None:
    user_dir = tmp_path / "isolated-user-dir"
    monkeypatch.setattr(Config, "user_dir", staticmethod(lambda: user_dir))
    config = Config(
        cwd=tmp_path,
        tool_approval=approval,
        routes={"subagent_general": "fake/general"},
    )
    ctx = ToolContext(cwd=tmp_path, config=config, ui=FakeUI())
    spawner = FakeSpawner()
    service = SubagentTaskService(spawner, config)  # type: ignore[arg-type]

    try:
        result = await SpawnSubagentTool(spawner, service).run(
            {
                "role": "general",
                "prompt": "change a file",
                "background": True,
                "isolation": isolation,
            },
            ctx,
        )

        assert result.is_error is True
        assert expected in result.content
        assert service.list() == []
        assert spawner.calls == []
    finally:
        await service.shutdown()


async def test_task_tool_lists_gets_and_waits_for_completed_task(
    task_config: Config, tool_ctx: ToolContext
) -> None:
    spawner = FakeSpawner()
    service = SubagentTaskService(spawner, task_config)  # type: ignore[arg-type]
    handle = await service.start("plan", "design the change", "shared", tool_ctx)
    record = await handle.wait()
    tool = SubagentTaskTool(service)

    try:
        listed = await tool.run({"action": "list"}, tool_ctx)
        assert handle.id in listed.content
        assert "succeeded" in listed.content
        assert "design the change" in listed.content

        detail = await tool.run({"action": "get", "task_id": handle.id}, tool_ctx)
        assert detail.is_error is False
        assert record.report in detail.content

        waited = await tool.run({"action": "wait", "task_id": handle.id}, tool_ctx)
        assert waited.is_error is False
        assert waited.content == "report: design the change"
    finally:
        await service.shutdown()


async def test_task_tool_wait_timeout_and_cancel_are_non_blocking(
    task_config: Config, tool_ctx: ToolContext
) -> None:
    spawner = FakeSpawner()
    service = SubagentTaskService(spawner, task_config)  # type: ignore[arg-type]
    handle = await service.start("explore", "block until cancelled", "shared", tool_ctx)
    tool = SubagentTaskTool(service)

    try:
        await spawner.started.wait()
        waiting = await tool.run(
            {
                "action": "wait",
                "task_id": handle.id,
                "timeout_seconds": 0,
            },
            tool_ctx,
        )
        assert waiting.is_error is False
        assert f"Task {handle.id} is still running" == waiting.content
        assert service.get(handle.id).status == TaskStatus.RUNNING  # type: ignore[union-attr]

        cancelled = await tool.run({"action": "cancel", "task_id": handle.id}, tool_ctx)
        assert cancelled.is_error is False
        assert cancelled.content == f"Cancellation requested for {handle.id}"
        assert (await handle.wait()).status == TaskStatus.CANCELLED

        repeated = await tool.run({"action": "cancel", "task_id": handle.id}, tool_ctx)
        assert repeated.content == f"Task {handle.id} is already terminal"
    finally:
        await service.shutdown()


@pytest.mark.parametrize("role", ["explore", "plan", "general"])
async def test_spawn_tool_runs_only_the_selected_role(
    task_config: Config, tool_ctx: ToolContext, role: str
) -> None:
    spawner = FakeSpawner()
    service = SubagentTaskService(spawner, task_config)  # type: ignore[arg-type]
    tool = SpawnSubagentTool(spawner, service)

    try:
        result = await tool.run(
            {
                "role": role,
                "prompt": "answer now",
                "background": False,
                "isolation": "shared",
            },
            tool_ctx,
        )

        assert result.is_error is False
        assert result.content == "report: answer now"
        assert len(spawner.calls) == 1
        assert spawner.calls[0]["role"] == role
        assert spawner.calls[0]["kwargs"] == {}
        assert service.list() == []
    finally:
        await service.shutdown()


def test_spawn_tool_description_makes_delegation_optional() -> None:
    description = SpawnSubagentTool.description.lower()

    assert "optionally delegate" in description
    assert "greetings" in description
    assert "one-step" in description
    assert "language is never a reason" in description
    assert "no fixed" in description
    assert "translate_for_model" in description
    assert "not a mandatory stage" in description


async def test_worktree_task_preserves_nested_repository_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    source_cwd = repo / "packages" / "agent"
    source_cwd.mkdir(parents=True)
    worktree = tmp_path / "managed-worktree"
    (worktree / "packages" / "agent").mkdir(parents=True)
    monkeypatch.setattr(Config, "user_dir", staticmethod(lambda: tmp_path / "user"))
    config = Config(
        cwd=source_cwd,
        routes={"subagent_explore": "fake/explore"},
    )
    ctx = ToolContext(cwd=source_cwd, config=config, ui=FakeUI())
    spawner = FakeSpawner()
    service = SubagentTaskService(spawner, config)  # type: ignore[arg-type]

    async def fake_create(*args: Any, **kwargs: Any) -> WorktreeInfo:
        return WorktreeInfo(
            repo_root=repo.resolve(),
            path=worktree.resolve(),
            task_id=str(args[2]),
            head="a" * 40,
            branch=None,
            detached=True,
        )

    monkeypatch.setattr("emiya_agent.agent.subagent_tasks.create_worktree", fake_create)
    try:
        handle = await service.start("explore", "inspect nested", "worktree", ctx)
        assert (await handle.wait()).status == TaskStatus.SUCCEEDED
        effective = spawner.calls[0]["kwargs"]["config"]
        assert effective.cwd == (worktree / "packages" / "agent").resolve()
    finally:
        await service.shutdown()


async def test_worktree_spawn_requires_approval_and_force_is_strict_boolean(
    task_config: Config, tool_ctx: ToolContext
) -> None:
    spawner = FakeSpawner()
    service = SubagentTaskService(spawner, task_config)  # type: ignore[arg-type]
    spawn_tool = SpawnSubagentTool(spawner, service)
    task_tool = SubagentTaskTool(service)

    try:
        assert spawn_tool.approval_required({"isolation": "worktree"}) is True
        assert spawn_tool.approval_required({"isolation": "shared"}) is False
        result = await task_tool.run(
            {"action": "cleanup", "task_id": "missing", "force": "false"},
            tool_ctx,
        )
        assert result.is_error is True
        assert result.content == "Error: force must be a boolean"
    finally:
        await service.shutdown()
