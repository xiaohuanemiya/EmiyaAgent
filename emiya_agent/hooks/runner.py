"""Bounded asynchronous runner for command hook protocol v1."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from emiya_agent.hooks.models import (
    HookCommandConfig,
    HookDecision,
    HookDiagnostic,
    HookEvent,
    HookRunResult,
)
from emiya_agent.tools.shell import (
    _WindowsJob,
    _resume_windows_process,
    _terminate_process_tree,
)

HOOK_PROTOCOL_VERSION = 1
MAX_HOOK_OUTPUT_BYTES = 64 * 1024


def _environment_value(env: Mapping[str, str], name: str) -> str | None:
    if os.name != "nt":
        return env.get(name)
    folded = name.casefold()
    return next(
        (value for key, value in env.items() if key.casefold() == folded),
        None,
    )


def _resolve_hook_argv(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str]
) -> tuple[str, ...]:
    """Resolve a hook executable without Windows' cwd-first search order."""

    executable = command[0]
    command_path = Path(executable)
    explicit = command_path.is_absolute() or any(
        separator in executable for separator in ("/", "\\")
    )
    candidates: list[Path] = []
    if explicit:
        candidates.append(command_path if command_path.is_absolute() else cwd / command_path)
    else:
        raw_path = _environment_value(env, "PATH") or ""
        directories: list[Path] = []
        for entry in raw_path.split(os.pathsep):
            entry = entry.strip().strip('"')
            if not entry:
                continue
            directory = Path(entry)
            # Empty/relative PATH entries reintroduce repository-controlled
            # executable lookup and are intentionally ignored.
            if directory.is_absolute():
                directories.append(directory)

        names = [executable]
        if os.name == "nt" and not command_path.suffix:
            raw_extensions = _environment_value(env, "PATHEXT") or (
                ".COM;.EXE;.BAT;.CMD"
            )
            names = [
                executable + extension
                for extension in raw_extensions.split(";")
                if extension.startswith(".")
                and not any(char in extension for char in ("/", "\\", "\x00"))
            ]
        candidates.extend(
            directory / name for directory in directories for name in names
        )

    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if not resolved.is_file():
            continue
        if os.name != "nt" and not os.access(resolved, os.X_OK):
            continue
        return (str(resolved), *command[1:])
    raise FileNotFoundError(
        "hook executable is unavailable on the absolute allowed PATH: "
        f"{executable}"
    )


@dataclass(frozen=True, slots=True)
class _CapturedStream:
    data: bytes
    omitted: int


@dataclass(frozen=True, slots=True)
class _ProcessResult:
    exit_code: int
    stdout: _CapturedStream
    stderr: _CapturedStream


@dataclass(frozen=True, slots=True)
class _HookOutcome:
    decision: HookDecision
    diagnostic: HookDiagnostic


async def _read_bounded(
    stream: asyncio.StreamReader | None,
) -> _CapturedStream:
    if stream is None:
        return _CapturedStream(b"", 0)
    captured = bytearray()
    omitted = 0
    while True:
        chunk = await stream.read(16 * 1024)
        if not chunk:
            break
        remaining = max(0, MAX_HOOK_OUTPUT_BYTES - len(captured))
        retained = min(remaining, len(chunk))
        captured.extend(chunk[:retained])
        omitted += len(chunk) - retained
    return _CapturedStream(bytes(captured), omitted)


async def _write_stdin(
    stream: asyncio.StreamWriter | None, payload: bytes
) -> None:
    if stream is None:
        return
    try:
        stream.write(payload)
        await stream.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        stream.close()
        try:
            await stream.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            pass


async def _exchange(
    process: asyncio.subprocess.Process, payload: bytes
) -> _ProcessResult:
    stdout_task = asyncio.create_task(_read_bounded(process.stdout))
    stderr_task = asyncio.create_task(_read_bounded(process.stderr))
    stdin_task = asyncio.create_task(_write_stdin(process.stdin, payload))
    tasks = (stdout_task, stderr_task, stdin_task)
    try:
        await stdin_task
        await process.wait()
        stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert process.returncode is not None
    return _ProcessResult(process.returncode, stdout, stderr)


async def _finish_termination(
    process: asyncio.subprocess.Process, job: _WindowsJob | None
) -> None:
    """Finish tree cleanup even if the owning task receives another cancel."""

    cleanup = asyncio.create_task(_terminate_process_tree(process, job))
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            continue
    await cleanup


async def _settle_spawn_after_cancellation(
    task: asyncio.Task[asyncio.subprocess.Process],
) -> asyncio.subprocess.Process | None:
    """Recover a spawned process handle before propagating cancellation."""

    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except BaseException:
            break
    try:
        return task.result()
    except BaseException:
        return None


def _decode(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def _containment_failure_outcome(
    config: HookCommandConfig, event: HookEvent, reason: str
) -> _HookOutcome:
    """Process-containment setup is mandatory regardless of on_error."""

    return _HookOutcome(
        HookDecision.DENY,
        HookDiagnostic(
            hook_id=config.id,
            event=event,
            status="error",
            decision=HookDecision.DENY,
            reason=reason,
        ),
    )


def _json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _parse_success_output(
    output: _CapturedStream,
) -> tuple[HookDecision, str]:
    if output.omitted:
        raise ValueError(
            f"hook stdout exceeded {MAX_HOOK_OUTPUT_BYTES} bytes"
        )
    try:
        text = output.data.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("hook stdout is not valid UTF-8") from exc
    if not text:
        return HookDecision.ALLOW, ""
    try:
        document = json.loads(text, parse_constant=_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("hook stdout is not strict JSON") from exc
    if not isinstance(document, dict):
        raise ValueError("hook JSON response must be an object")
    unknown = set(document) - {"decision", "reason"}
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ValueError(f"unknown hook JSON field(s): {names}")
    if "decision" not in document:
        raise ValueError("hook JSON response requires decision")
    try:
        decision = HookDecision(document["decision"])
    except (TypeError, ValueError) as exc:
        raise ValueError("hook decision must be allow, ask, or deny") from exc
    reason = document.get("reason", "")
    if not isinstance(reason, str):
        raise ValueError("hook reason must be a string")
    return decision, reason


def _failure_outcome(
    config: HookCommandConfig,
    *,
    reason: str,
    event: HookEvent,
    status: str,
    exit_code: int | None = None,
    stdout: _CapturedStream | None = None,
    stderr: _CapturedStream | None = None,
) -> _HookOutcome:
    blocked = config.on_error == "block"
    decision = HookDecision.DENY if blocked else HookDecision.ALLOW
    diagnostic = HookDiagnostic(
        hook_id=config.id,
        event=event,
        status=("error" if blocked and status == "warning" else status),
        decision=decision,
        reason=reason,
        exit_code=exit_code,
        stdout=_decode(stdout.data) if stdout else "",
        stderr=_decode(stderr.data) if stderr else "",
        stdout_omitted_bytes=stdout.omitted if stdout else 0,
        stderr_omitted_bytes=stderr.omitted if stderr else 0,
    )
    return _HookOutcome(decision, diagnostic)


class HookRunner:
    """Run matching hooks concurrently without invoking a command shell.

    ``env`` is used verbatim: the runner never adds values inherited from the
    agent process. The caller therefore owns the environment allowlist policy.
    """

    def __init__(
        self,
        configs: Sequence[HookCommandConfig],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> None:
        identifiers = [config.id for config in configs]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("hook ids must be unique")
        resolved_cwd = Path(cwd).resolve(strict=True)
        if not resolved_cwd.is_dir():
            raise ValueError("hook cwd must be a directory")
        copied_env: dict[str, str] = {}
        for key, value in env.items():
            if not key or "=" in key or "\x00" in key or "\x00" in value:
                raise ValueError("hook environment contains an invalid entry")
            copied_env[str(key)] = str(value)
        self._configs = tuple(configs)
        self._cwd = resolved_cwd
        self._env = copied_env

    def for_context(
        self, *, cwd: Path, env: Mapping[str, str]
    ) -> "HookRunner":
        """Reuse the injected policy in another execution context.

        Foreground subagents keep the exact injected runner. Isolated
        worktrees receive a runner with the same audited hook definitions but
        a worktree-local cwd and freshly selected environment.
        """

        resolved_cwd = Path(cwd).resolve(strict=True)
        if resolved_cwd == self._cwd and dict(env) == self._env:
            return self
        return HookRunner(self._configs, cwd=resolved_cwd, env=env)

    def has_hooks_for(self, event: HookEvent) -> bool:
        """Return whether any configured hook listens to this event."""

        return any(config.event == event for config in self._configs)

    def _matching(
        self, event: HookEvent, tool_name: str
    ) -> tuple[HookCommandConfig, ...]:
        # ``tool_name`` is the event's matcher subject: the tool name for
        # tool events, and the source/reason/trigger for lifecycle events.
        return tuple(
            config
            for config in self._configs
            if config.event == event and config.matches(tool_name)
        )

    @staticmethod
    def _aggregate(
        outcomes: Sequence[_HookOutcome],
        *,
        initial_decision: HookDecision,
        initial_reason: str,
    ) -> HookRunResult:
        rank = {
            HookDecision.ALLOW: 0,
            HookDecision.ASK: 1,
            HookDecision.DENY: 2,
        }
        final_decision = initial_decision
        for outcome in outcomes:
            if rank[outcome.decision] > rank[final_decision]:
                final_decision = outcome.decision

        reasons = [
            outcome.diagnostic.reason
            for outcome in outcomes
            if outcome.decision == final_decision
            and outcome.diagnostic.reason
        ]
        if rank[initial_decision] == rank[final_decision] and initial_reason:
            reasons.insert(0, initial_reason)
        return HookRunResult(
            decision=final_decision,
            reason="; ".join(reasons),
            diagnostics=tuple(outcome.diagnostic for outcome in outcomes),
        )

    def failure_result(
        self,
        event: HookEvent,
        tool_name: str,
        reason: str,
        *,
        initial_decision: HookDecision = HookDecision.ASK,
        initial_reason: str = "",
    ) -> HookRunResult:
        """Apply each matching hook's ``on_error`` policy to runner failures."""

        event = HookEvent(event)
        initial_decision = HookDecision(initial_decision)
        outcomes = [
            _failure_outcome(
                config,
                event=event,
                status="warning",
                reason=reason,
            )
            for config in self._matching(event, tool_name)
        ]
        return self._aggregate(
            outcomes,
            initial_decision=initial_decision,
            initial_reason=initial_reason,
        )

    async def run(
        self,
        event: HookEvent,
        tool_name: str,
        payload: Mapping[str, Any] | None = None,
        *,
        initial_decision: HookDecision = HookDecision.ASK,
        initial_reason: str = "",
    ) -> HookRunResult:
        """Run all matches and aggregate with ``deny > ask > allow``.

        An ``allow`` response can never lower ``initial_decision``. This keeps
        hooks from bypassing the normal tool approval policy. For lifecycle
        events ``tool_name`` carries the event subject (start source, stop
        reason, compaction trigger, …) both for matching and on stdin; whether
        the aggregate decision gates anything is the caller's contract (see
        ``DECISION_EVENTS`` in :mod:`emiya_agent.hooks.models`).
        """

        event = HookEvent(event)
        initial_decision = HookDecision(initial_decision)
        matching = self._matching(event, tool_name)
        if not matching:
            return HookRunResult(
                decision=initial_decision,
                reason=initial_reason,
            )

        try:
            stdin = json.dumps(
                {
                    "protocol_version": HOOK_PROTOCOL_VERSION,
                    "event": event.value,
                    "tool_name": tool_name,
                    "payload": dict(payload or {}),
                },
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8") + b"\n"
        except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
            return self.failure_result(
                event,
                tool_name,
                f"hook input is not strict JSON: {type(exc).__name__}: {exc}",
                initial_decision=initial_decision,
                initial_reason=initial_reason,
            )

        raw_outcomes = await asyncio.gather(
            *(self._run_one(config, event, stdin) for config in matching),
            return_exceptions=True,
        )
        outcomes: list[_HookOutcome] = []
        for config, outcome in zip(matching, raw_outcomes):
            if isinstance(outcome, (KeyboardInterrupt, asyncio.CancelledError)):
                raise outcome
            if isinstance(outcome, BaseException):
                outcomes.append(
                    _failure_outcome(
                        config,
                        event=event,
                        status="warning",
                        reason=(
                            "hook infrastructure error: "
                            f"{type(outcome).__name__}: {outcome}"
                        ),
                    )
                )
            else:
                outcomes.append(outcome)
        return self._aggregate(
            outcomes,
            initial_decision=initial_decision,
            initial_reason=initial_reason,
        )

    async def _run_one(
        self,
        config: HookCommandConfig,
        event: HookEvent,
        stdin: bytes,
    ) -> _HookOutcome:
        spawn_options: dict[str, Any] = {}
        if sys.platform == "win32":
            spawn_options["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                # CPython exposes CREATE_NEW_PROCESS_GROUP but not the
                # documented Win32 CREATE_SUSPENDED flag.
                | 0x00000004
            )
        else:
            spawn_options["start_new_session"] = True

        try:
            argv = _resolve_hook_argv(
                config.command,
                cwd=self._cwd,
                env=self._env,
            )
        except OSError as exc:
            return _failure_outcome(
                config,
                event=event,
                status="warning",
                reason=f"hook process could not start: {exc}",
            )
        spawn_task = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._cwd),
                env=self._env,
                **spawn_options,
            )
        )
        try:
            process = await asyncio.shield(spawn_task)
        except (KeyboardInterrupt, asyncio.CancelledError):
            process = await _settle_spawn_after_cancellation(spawn_task)
            if process is not None:
                await _finish_termination(process, None)
            raise
        except OSError as exc:
            return _failure_outcome(
                config,
                event=event,
                status="warning",
                reason=f"hook process could not start: {exc}",
            )

        job: _WindowsJob | None = None
        try:
            if sys.platform == "win32":
                try:
                    job = _WindowsJob(process.pid)
                except Exception as exc:
                    return _containment_failure_outcome(
                        config,
                        event,
                        "hook process containment failed: "
                        f"{type(exc).__name__}: {exc}",
                    )
                if job.handle is None:
                    return _containment_failure_outcome(
                        config,
                        event,
                        "hook process containment failed: "
                        "could not assign the suspended process to a Job Object",
                    )
                try:
                    _resume_windows_process(process.pid)
                except Exception as exc:
                    return _containment_failure_outcome(
                        config,
                        event,
                        "hook process could not resume after containment: "
                        f"{type(exc).__name__}: {exc}",
                    )
            try:
                result = await asyncio.wait_for(
                    _exchange(process, stdin), timeout=config.timeout
                )
            except (asyncio.TimeoutError, TimeoutError):
                return _failure_outcome(
                    config,
                    event=event,
                    status="timeout",
                    reason=f"hook timed out after {config.timeout:g} seconds",
                )

            if result.exit_code == 2:
                reason = _decode(result.stderr.data).strip()
                if not reason:
                    reason = f"hook {config.id} blocked the tool"
                return _HookOutcome(
                    HookDecision.DENY,
                    HookDiagnostic(
                        hook_id=config.id,
                        event=event,
                        status="blocked",
                        decision=HookDecision.DENY,
                        reason=reason,
                        exit_code=result.exit_code,
                        stdout=_decode(result.stdout.data),
                        stderr=_decode(result.stderr.data),
                        stdout_omitted_bytes=result.stdout.omitted,
                        stderr_omitted_bytes=result.stderr.omitted,
                    ),
                )
            if result.exit_code != 0:
                return _failure_outcome(
                    config,
                    event=event,
                    status="warning",
                    reason=(
                        f"hook exited with code {result.exit_code}"
                    ),
                    exit_code=result.exit_code,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )

            try:
                decision, reason = _parse_success_output(result.stdout)
            except ValueError as exc:
                return _failure_outcome(
                    config,
                    event=event,
                    status="warning",
                    reason=str(exc),
                    exit_code=result.exit_code,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            return _HookOutcome(
                decision,
                HookDiagnostic(
                    hook_id=config.id,
                    event=event,
                    status="blocked"
                    if decision == HookDecision.DENY
                    else "ok",
                    decision=decision,
                    reason=reason,
                    exit_code=result.exit_code,
                    stdout=_decode(result.stdout.data),
                    stderr=_decode(result.stderr.data),
                    stdout_omitted_bytes=result.stdout.omitted,
                    stderr_omitted_bytes=result.stderr.omitted,
                ),
            )
        finally:
            # Hooks are bounded subprocesses, not service launchers. Always
            # close the Job Object / terminate the POSIX process group even
            # after a clean parent exit so redirected background children do
            # not outlive the hook invocation.
            await _finish_termination(process, job)
