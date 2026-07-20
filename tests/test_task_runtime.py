from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from emiya_agent.agent.task_runtime import (
    TaskManager,
    TaskNotFoundError,
    TaskQueueFullError,
    TaskSpec,
    TaskStatus,
)


def spec(prompt: str = "inspect the project") -> TaskSpec:
    return TaskSpec(role="explore", prompt=prompt)


async def test_successful_task_retains_report_and_metadata() -> None:
    manager = TaskManager(max_concurrency=1)

    async def runner(handle) -> str:
        handle.set_route("subagent_explore")
        handle.set_stop_reason("end_turn")
        return "inspection complete"

    handle = await manager.start(spec(), runner)
    record = await handle.wait()

    assert record.id == handle.id == handle.task_id
    assert record.task_id == handle.id
    assert record.status == TaskStatus.SUCCEEDED
    assert record.report == "inspection complete"
    assert record.route == "subagent_explore"
    assert record.stop_reason == "end_turn"
    assert record.started_at is not None
    assert record.finished_at is not None
    assert manager.list() == [record]


async def test_concurrency_and_queue_capacity_are_bounded() -> None:
    manager = TaskManager(max_concurrency=1, max_queued=1)
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release_first = asyncio.Event()
    release_second = asyncio.Event()

    async def runner(handle) -> str:
        if handle.spec.prompt == "first":
            first_started.set()
            await release_first.wait()
        else:
            second_started.set()
            await release_second.wait()
        return handle.spec.prompt

    first = await manager.start(spec("first"), runner)
    await first_started.wait()
    second = await manager.start(spec("second"), runner)

    assert second.record().status == TaskStatus.QUEUED
    with pytest.raises(TaskQueueFullError):
        await manager.start(spec("overflow"), runner)

    release_first.set()
    assert (await first.wait()).status == TaskStatus.SUCCEEDED
    await second_started.wait()
    release_second.set()
    assert (await second.wait()).status == TaskStatus.SUCCEEDED


async def test_cancel_has_independent_event_and_cancelling_transition() -> None:
    manager = TaskManager(max_concurrency=2)
    started = asyncio.Event()

    async def runner(handle) -> str:
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    first = await manager.start(spec("first"), runner)
    second = await manager.start(spec("second"), runner)
    await started.wait()

    assert first.cancel_event is not second.cancel_event
    assert await first.cancel() is True
    assert first.cancel_requested is True
    assert second.cancel_requested is False
    assert first.record().status == TaskStatus.CANCELLING
    assert (await first.wait()).status == TaskStatus.CANCELLED
    assert await first.cancel() is False
    await manager.cancel(second.id)


async def test_wait_timeout_does_not_cancel_task() -> None:
    manager = TaskManager(max_concurrency=1)
    started = asyncio.Event()
    release = asyncio.Event()

    async def runner(handle) -> str:
        started.set()
        await release.wait()
        return "eventually done"

    handle = await manager.start(spec(), runner)
    await started.wait()

    with pytest.raises(TimeoutError):
        await manager.wait(handle.id, timeout=0.001)
    assert handle.record().status == TaskStatus.RUNNING

    release.set()
    record = await handle.wait()
    assert record.status == TaskStatus.SUCCEEDED
    assert record.report == "eventually done"


async def test_runner_failure_is_recorded() -> None:
    manager = TaskManager()

    async def runner(handle) -> str:
        raise ValueError("bad delegation")

    record = await (await manager.start(spec(), runner)).wait()
    assert record.status == TaskStatus.FAILED
    assert record.stop_reason == "error"
    assert record.error == "ValueError: bad delegation"
    assert record.report is None


async def test_shutdown_interrupts_work_and_is_idempotent() -> None:
    manager = TaskManager(max_concurrency=1, max_queued=1)
    running = asyncio.Event()

    async def runner(handle) -> str:
        running.set()
        await asyncio.Event().wait()
        return "unreachable"

    first = await manager.start(spec("running"), runner)
    await running.wait()
    second = await manager.start(spec("queued"), runner)

    await manager.shutdown()
    assert (await first.wait()).status == TaskStatus.INTERRUPTED
    assert (await second.wait()).status == TaskStatus.INTERRUPTED
    assert first.cancel_requested and second.cancel_requested
    await manager.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        await manager.start(spec("late"), runner)


async def test_shutdown_deadline_does_not_wait_forever_for_stubborn_runner() -> None:
    manager = TaskManager(
        shutdown_timeout=0.01,
        force_cancel_timeout=0.01,
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def runner(handle) -> str:
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue
        return "released"

    handle = await manager.start(spec("stubborn"), runner)
    await started.wait()
    started_at = asyncio.get_running_loop().time()
    await manager.shutdown()
    elapsed = asyncio.get_running_loop().time() - started_at

    assert elapsed < 0.5
    stopping = handle.record()
    assert stopping.status == TaskStatus.CANCELLING
    assert stopping.error == "shutdown deadline exceeded; task is still stopping"

    release.set()
    await asyncio.wait_for(manager._tasks[handle.id], timeout=1)
    assert (await handle.wait()).status == TaskStatus.INTERRUPTED


async def test_unknown_task_operations_raise_but_get_returns_none() -> None:
    manager = TaskManager()
    assert manager.get("missing") is None
    with pytest.raises(TaskNotFoundError):
        await manager.wait("missing")
    with pytest.raises(TaskNotFoundError):
        await manager.cancel("missing")


def test_spec_validation_and_manager_limits() -> None:
    with pytest.raises(ValidationError):
        TaskSpec(role="explore", prompt="   ")
    with pytest.raises(ValueError):
        TaskManager(max_concurrency=0)
    with pytest.raises(ValueError):
        TaskManager(max_queued=-1)
    with pytest.raises(ValueError):
        TaskManager(shutdown_timeout=0)
