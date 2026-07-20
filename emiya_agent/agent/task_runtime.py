"""In-process runtime for bounded background tasks.

The runtime deliberately contains no agent, worktree, or persistence policy.  A
caller supplies a coroutine for each task and may use the handle's cancellation
event for cooperative cleanup.  ``asyncio.Task.cancel`` is also used so a
blocked runner cannot keep shutdown from completing indefinitely.
"""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


TERMINAL_STATUSES = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.INTERRUPTED,
    }
)


class TaskSpec(BaseModel):
    """Serializable input needed to launch a delegated agent task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["explore", "plan", "general"]
    prompt: str = Field(min_length=1)
    background: bool = True
    isolation: Literal["shared", "worktree"] = "shared"

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class TaskRecord(BaseModel):
    """A point-in-time, externally safe snapshot of a task."""

    model_config = ConfigDict(extra="forbid")

    id: str
    spec: TaskSpec
    status: TaskStatus = TaskStatus.QUEUED
    route: str | None = None
    report: str | None = None
    error: str | None = None
    stop_reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def task_id(self) -> str:
        """Compatibility alias for callers that prefer an explicit name."""

        return self.id


class TaskQueueFullError(RuntimeError):
    """Raised when both execution and waiting capacity are occupied."""


class TaskNotFoundError(KeyError):
    """Raised by task operations that require an existing task."""


@dataclass(frozen=True, slots=True)
class TaskHandle:
    """Runner-facing task identity and cooperative cancellation signal."""

    id: str
    spec: TaskSpec
    cancel_event: asyncio.Event
    _manager: TaskManager = field(repr=False, compare=False)

    @property
    def task_id(self) -> str:
        return self.id

    @property
    def cancel_requested(self) -> bool:
        return self.cancel_event.is_set()

    def record(self) -> TaskRecord:
        record = self._manager.get(self.id)
        if record is None:  # Defensive: handles are only made for stored tasks.
            raise TaskNotFoundError(self.id)
        return record

    def set_route(self, route: str | None) -> None:
        self._manager._set_runtime_metadata(self.id, route=route)

    def set_stop_reason(self, stop_reason: str | None) -> None:
        self._manager._set_runtime_metadata(self.id, stop_reason=stop_reason)

    async def wait(self, timeout: float | None = None) -> TaskRecord:
        return await self._manager.wait(self.id, timeout=timeout)

    async def cancel(self) -> bool:
        return await self._manager.cancel(self.id)


TaskRunner: TypeAlias = Callable[[TaskHandle], Awaitable[str]]


class TaskManager:
    """Run a bounded number of tasks and retain their terminal snapshots."""

    def __init__(
        self,
        *,
        max_concurrency: int = 2,
        max_queued: int = 32,
        shutdown_timeout: float = 20.0,
        force_cancel_timeout: float = 5.0,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if max_queued < 0:
            raise ValueError("max_queued must not be negative")
        if shutdown_timeout <= 0 or force_cancel_timeout <= 0:
            raise ValueError("shutdown timeouts must be positive")

        self.max_concurrency = max_concurrency
        self.max_queued = max_queued
        self.shutdown_timeout = shutdown_timeout
        self.force_cancel_timeout = force_cancel_timeout
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._lock = asyncio.Lock()
        self._records: dict[str, TaskRecord] = {}
        self._handles: dict[str, TaskHandle] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._done: dict[str, asyncio.Event] = {}
        self._interrupted: set[str] = set()
        self._closed = False

    async def start(
        self,
        spec: TaskSpec,
        runner: TaskRunner,
        *,
        route: str | None = None,
    ) -> TaskHandle:
        """Admit a task and return immediately with its handle.

        Capacity includes tasks already running plus ``max_queued`` waiters, so
        admission remains deterministic even before newly created asyncio tasks
        have had a chance to acquire the semaphore.
        """

        async with self._lock:
            if self._closed:
                raise RuntimeError("task manager is shut down")

            active = sum(
                record.status not in TERMINAL_STATUSES
                for record in self._records.values()
            )
            if active >= self.max_concurrency + self.max_queued:
                raise TaskQueueFullError(
                    "background task capacity is full "
                    f"({self.max_concurrency} running + {self.max_queued} queued)"
                )

            task_id = self._new_id()
            cancel_event = asyncio.Event()
            handle = TaskHandle(task_id, spec, cancel_event, self)
            self._records[task_id] = TaskRecord(id=task_id, spec=spec, route=route)
            self._handles[task_id] = handle
            self._done[task_id] = asyncio.Event()
            self._tasks[task_id] = asyncio.create_task(
                self._execute(handle, runner), name=f"emiya-task-{task_id}"
            )
            return handle

    def list(self, status: TaskStatus | None = None) -> list[TaskRecord]:
        """Return creation-ordered snapshots, optionally filtered by status."""

        return [
            record.model_copy(deep=True)
            for record in self._records.values()
            if status is None or record.status == status
        ]

    def get(self, task_id: str) -> TaskRecord | None:
        record = self._records.get(task_id)
        return record.model_copy(deep=True) if record is not None else None

    async def wait(self, task_id: str, timeout: float | None = None) -> TaskRecord:
        """Wait for a terminal state without cancelling the underlying task."""

        done = self._done.get(task_id)
        if done is None:
            raise TaskNotFoundError(task_id)

        waiter = asyncio.shield(done.wait())
        if timeout is None:
            await waiter
        else:
            await asyncio.wait_for(waiter, timeout)

        record = self.get(task_id)
        if record is None:  # The manager never evicts records, kept defensive.
            raise TaskNotFoundError(task_id)
        return record

    async def cancel(self, task_id: str) -> bool:
        """Request cancellation; return false when the task is already terminal."""

        async with self._lock:
            record = self._records.get(task_id)
            if record is None:
                raise TaskNotFoundError(task_id)
            if record.status in TERMINAL_STATUSES:
                return False

            self._handles[task_id].cancel_event.set()
            task = self._tasks[task_id]
            if record.status == TaskStatus.QUEUED:
                self._finish(record, TaskStatus.CANCELLED, "cancelled")
                self._done[task_id].set()
            else:
                record.status = TaskStatus.CANCELLING
            task.cancel()
            return True

    async def shutdown(self) -> None:
        """Stop accepting work and interrupt every unfinished task."""

        async with self._lock:
            if self._closed and all(task.done() for task in self._tasks.values()):
                return
            self._closed = True
            for task_id, record in self._records.items():
                if record.status in TERMINAL_STATUSES:
                    continue
                self._interrupted.add(task_id)
                self._handles[task_id].cancel_event.set()
                task = self._tasks[task_id]
                if record.status == TaskStatus.QUEUED:
                    self._finish(record, TaskStatus.INTERRUPTED, "interrupted")
                    self._done[task_id].set()
                else:
                    record.status = TaskStatus.CANCELLING
                task.cancel()
            active_tasks = [task for task in self._tasks.values() if not task.done()]

        pending: set[asyncio.Task[None]] = set(active_tasks)
        if pending:
            _, pending = await asyncio.wait(
                pending, timeout=self.shutdown_timeout
            )
        if pending:
            for task in pending:
                task.cancel()
            _, pending = await asyncio.wait(
                pending, timeout=self.force_cancel_timeout
            )

        # A task cancelled before its coroutine first ran cannot execute finally.
        for task_id in self._interrupted:
            record = self._records[task_id]
            if self._tasks[task_id] in pending:
                record.error = "shutdown deadline exceeded; task is still stopping"
                continue
            if record.status not in TERMINAL_STATUSES:
                self._finish(record, TaskStatus.INTERRUPTED, "interrupted")
            self._done[task_id].set()
        for task in pending:
            task.add_done_callback(self._consume_task_result)

    async def _execute(self, handle: TaskHandle, runner: TaskRunner) -> None:
        record = self._records[handle.id]
        try:
            async with self._semaphore:
                if record.status in TERMINAL_STATUSES:
                    return
                record.status = TaskStatus.RUNNING
                record.started_at = datetime.now(UTC)
                report = await runner(handle)
                if not isinstance(report, str):
                    raise TypeError("task runner must return str")

                if handle.id in self._interrupted:
                    self._finish(record, TaskStatus.INTERRUPTED, "interrupted")
                elif handle.cancel_requested:
                    self._finish(record, TaskStatus.CANCELLED, "cancelled")
                else:
                    record.report = report
                    self._finish(
                        record,
                        TaskStatus.SUCCEEDED,
                        record.stop_reason or "completed",
                    )
        except asyncio.CancelledError:
            if handle.id in self._interrupted:
                self._finish(record, TaskStatus.INTERRUPTED, "interrupted")
            elif record.status not in TERMINAL_STATUSES:
                self._finish(record, TaskStatus.CANCELLED, "cancelled")
        except Exception as exc:
            if record.status not in TERMINAL_STATUSES:
                record.error = f"{type(exc).__name__}: {exc}"
                if handle.id in self._interrupted:
                    self._finish(record, TaskStatus.INTERRUPTED, "interrupted")
                else:
                    self._finish(record, TaskStatus.FAILED, "error")
        finally:
            self._done[handle.id].set()

    def _set_runtime_metadata(
        self,
        task_id: str,
        *,
        route: str | None | Any = ...,
        stop_reason: str | None | Any = ...,
    ) -> None:
        record = self._records.get(task_id)
        if record is None:
            raise TaskNotFoundError(task_id)
        if route is not ...:
            record.route = route
        if stop_reason is not ...:
            record.stop_reason = stop_reason

    @staticmethod
    def _finish(record: TaskRecord, status: TaskStatus, stop_reason: str) -> None:
        record.status = status
        record.stop_reason = stop_reason
        record.finished_at = datetime.now(UTC)

    def _new_id(self) -> str:
        while True:
            task_id = secrets.token_urlsafe(12)
            if task_id not in self._records:
                return task_id

    @staticmethod
    def _consume_task_result(task: asyncio.Task[None]) -> None:
        try:
            task.exception()
        except asyncio.CancelledError:
            pass
