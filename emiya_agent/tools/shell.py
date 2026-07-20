"""Shell 执行工具：Windows 用 PowerShell，POSIX 用 bash。

超时默认 120s / 上限 600s；finally 中 terminate+kill 防僵尸进程（Proactor 循环）。
"""

from __future__ import annotations

import asyncio
import ctypes
import os
import signal
import sys
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from emiya_agent.config import select_environment
from emiya_agent.tools.base import (
    PermissionAction,
    Tool,
    ToolContext,
    ToolResult,
)

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600
MAX_CAPTURE_BYTES = 2_000_000
_CREATE_SUSPENDED = 0x00000004
_CREATE_NEW_PROCESS_GROUP = 0x00000200

_T = TypeVar("_T")


class _ContainmentError(RuntimeError):
    """Raised when a process cannot be started inside its OS containment."""


def _decode(data: bytes) -> str:
    """Windows 控制台输出可能是 GBK；先试 UTF-8，失败回退本地编码。"""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        import locale

        return data.decode(locale.getpreferredencoding(False), errors="replace")


class _WindowsJob:
    """Kill-on-close Job Object used as a mandatory Windows boundary."""

    def __init__(self, pid: int) -> None:
        self.handle: int | None = None
        if sys.platform != "win32":
            return

        from ctypes import wintypes

        class BasicLimit(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class ExtendedLimit(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimit),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
        ]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            return
        limits = ExtendedLimit()
        limits.BasicLimitInformation.LimitFlags = 0x2000
        configured = kernel32.SetInformationJobObject(
            handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        )
        process = kernel32.OpenProcess(0x0100 | 0x0001, False, pid)
        assigned = bool(
            process and kernel32.AssignProcessToJobObject(handle, process)
        )
        if process:
            kernel32.CloseHandle(process)
        if not configured or not assigned:
            kernel32.CloseHandle(handle)
            return
        self.handle = int(handle)

    def close(self) -> bool:
        if self.handle is None:
            return True
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        closed = bool(kernel32.CloseHandle(wintypes.HANDLE(self.handle)))
        if closed:
            self.handle = None
        return closed


def _resume_windows_process(pid: int) -> None:
    """Resume every thread of a process created with CREATE_SUSPENDED."""

    from ctypes import wintypes

    class ThreadEntry32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ThreadID", wintypes.DWORD),
            ("th32OwnerProcessID", wintypes.DWORD),
            ("tpBasePri", wintypes.LONG),
            ("tpDeltaPri", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Thread32First.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ThreadEntry32),
    ]
    kernel32.Thread32First.restype = wintypes.BOOL
    kernel32.Thread32Next.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ThreadEntry32),
    ]
    kernel32.Thread32Next.restype = wintypes.BOOL
    kernel32.OpenThread.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel32.OpenThread.restype = wintypes.HANDLE
    kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel32.ResumeThread.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000004, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if not snapshot or int(snapshot) == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    resumed = 0
    try:
        entry = ThreadEntry32()
        entry.dwSize = ctypes.sizeof(entry)
        present = bool(kernel32.Thread32First(snapshot, ctypes.byref(entry)))
        while present:
            if entry.th32OwnerProcessID == pid:
                thread = kernel32.OpenThread(0x0002, False, entry.th32ThreadID)
                if not thread:
                    raise ctypes.WinError(ctypes.get_last_error())
                try:
                    previous = kernel32.ResumeThread(thread)
                    if previous == 0xFFFFFFFF:
                        raise ctypes.WinError(ctypes.get_last_error())
                    resumed += 1
                finally:
                    kernel32.CloseHandle(thread)
            present = bool(kernel32.Thread32Next(snapshot, ctypes.byref(entry)))
    finally:
        kernel32.CloseHandle(snapshot)
    if not resumed:
        raise OSError("could not find the suspended process thread")


async def _capture_limited(
    proc: asyncio.subprocess.Process,
) -> tuple[bytes, int]:
    assert proc.stdout is not None
    captured = bytearray()
    omitted = 0
    while True:
        chunk = await proc.stdout.read(64 * 1024)
        if not chunk:
            break
        remaining = MAX_CAPTURE_BYTES - len(captured)
        if remaining > 0:
            captured.extend(chunk[:remaining])
        omitted += max(0, len(chunk) - max(remaining, 0))
    return bytes(captured), omitted


async def _read_limited(
    proc: asyncio.subprocess.Process,
) -> tuple[bytes, int]:
    """Compatibility helper for bounded internal subprocesses."""

    captured = await _capture_limited(proc)
    await proc.wait()
    return captured


def _windows_system_executable(name: str) -> str:
    """从 Win32 系统目录解析内置程序，不走仓库可控 PATH/cwd。"""
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetSystemDirectoryW.argtypes = [wintypes.LPWSTR, wintypes.UINT]
    kernel32.GetSystemDirectoryW.restype = wintypes.UINT
    buffer = ctypes.create_unicode_buffer(32768)
    length = kernel32.GetSystemDirectoryW(buffer, len(buffer))
    if not length or length >= len(buffer):
        raise OSError("无法解析 Windows System32 目录")
    path = Path(buffer.value) / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def _posix_bash() -> str:
    for candidate in (Path("/bin/bash"), Path("/usr/bin/bash")):
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("未找到可信的 /bin/bash 或 /usr/bin/bash")


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _await_uninterruptibly(
    task: asyncio.Task[_T],
) -> tuple[_T, bool]:
    """Await an owned lifecycle task to completion and report cancellation.

    Cancellation must not strand a just-created suspended process or interrupt
    process-tree cleanup.  The caller re-raises ``CancelledError`` after the
    lifecycle operation has reached a terminal state.
    """

    interrupted = False
    while True:
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            interrupted = True
            if task.done():
                break
        except BaseException:
            if interrupted:
                raise asyncio.CancelledError from None
            raise
        else:
            return result, interrupted
    try:
        result = task.result()
    except BaseException:
        if interrupted:
            raise asyncio.CancelledError from None
        raise
    return result, interrupted


async def _taskkill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Use the trusted System32 taskkill fallback for an unassigned process."""

    try:
        cmd = _windows_system_executable("cmd.exe")
        killer = await asyncio.create_subprocess_exec(
            _windows_system_executable("taskkill.exe"),
            "/PID",
            str(proc.pid),
            "/T",
            "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            stdin=asyncio.subprocess.DEVNULL,
            env={"SYSTEMROOT": str(Path(cmd).parent.parent)},
        )
    except OSError:
        return
    try:
        await asyncio.wait_for(killer.wait(), timeout=5)
    except (asyncio.TimeoutError, TimeoutError):
        killer.kill()
        try:
            await asyncio.wait_for(killer.wait(), timeout=3)
        except (asyncio.TimeoutError, TimeoutError):
            pass


async def _terminate_process_tree(
    proc: asyncio.subprocess.Process, job: _WindowsJob | None
) -> None:
    if sys.platform == "win32":
        job_closed = False
        if job is not None and job.handle is not None:
            job_closed = job.close()
        if not job_closed:
            await _taskkill_process_tree(proc)
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        deadline = asyncio.get_running_loop().time() + 3
        while _process_group_exists(proc.pid):
            if asyncio.get_running_loop().time() >= deadline:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            await asyncio.sleep(0.05)

    if proc.returncode is None:
        try:
            await asyncio.wait_for(proc.wait(), timeout=3)
        except (asyncio.TimeoutError, TimeoutError):
            if sys.platform != "win32":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=3)
            except (asyncio.TimeoutError, TimeoutError):
                pass


class _ProcessTree:
    """One process-tree identity with an idempotent cleanup operation.

    Retaining the cleanup task prevents timeout, cancellation, and ``finally``
    paths from signalling a recycled numeric PID/PGID more than once.
    """

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        job: _WindowsJob | None,
    ) -> None:
        self.process = process
        self.job = job
        self._cleanup_task: asyncio.Task[None] | None = None

    async def terminate(self) -> None:
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(
                _terminate_process_tree(self.process, self.job)
            )
        _, interrupted = await _await_uninterruptibly(self._cleanup_task)
        if interrupted:
            raise asyncio.CancelledError


async def _read_contained_process(
    tree: _ProcessTree,
) -> tuple[bytes, int]:
    """Capture output while treating leader exit as the cleanup boundary.

    Waiting for pipe EOF first is unsafe: a background child may inherit the
    write handle and keep it open after the shell leader exits.  Monitor the
    leader concurrently, close containment immediately when it exits, then
    finish draining the now-closed pipe.
    """

    capture_task = asyncio.create_task(_capture_limited(tree.process))
    try:
        # asyncio's Process.wait() may wait for inherited stdout pipe handles
        # as well as the leader.  Poll the documented returncode instead so a
        # descendant cannot postpone Job/group cleanup merely by holding the
        # pipe open.
        while tree.process.returncode is None:
            await asyncio.sleep(0.01)
        await tree.terminate()
        return await capture_task
    finally:
        try:
            await tree.terminate()
        finally:
            if not capture_task.done():
                capture_task.cancel()
            await asyncio.gather(capture_task, return_exceptions=True)


async def _kill_suspended_process(
    process: asyncio.subprocess.Process,
) -> bool:
    """Terminate a Windows process that has never been allowed to execute."""

    async def kill_and_wait() -> None:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=3)
            except (asyncio.TimeoutError, TimeoutError):
                pass

    task = asyncio.create_task(kill_and_wait())
    _, interrupted = await _await_uninterruptibly(task)
    return interrupted


async def _spawn_process_tree(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> _ProcessTree:
    """Start a shell inside mandatory process-tree containment.

    Windows starts the real PowerShell suspended.  It is assigned to a
    kill-on-close Job Object before its initial thread is resumed.  The spawn
    coroutine itself is shielded so cancellation cannot lose ownership of a
    process created just before ``create_subprocess_exec`` returns.
    """

    spawn_options: dict[str, Any]
    if sys.platform == "win32":
        spawn_options = {
            "creationflags": _CREATE_NEW_PROCESS_GROUP | _CREATE_SUSPENDED
        }
    else:
        spawn_options = {"start_new_session": True}

    spawn_task = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(cwd),
            env=env,
            **spawn_options,
        )
    )
    process, spawn_cancelled = await _await_uninterruptibly(spawn_task)

    if sys.platform != "win32":
        tree = _ProcessTree(process, None)
        if spawn_cancelled:
            await tree.terminate()
            raise asyncio.CancelledError
        return tree

    if spawn_cancelled:
        await _kill_suspended_process(process)
        raise asyncio.CancelledError

    try:
        job = _WindowsJob(process.pid)
    except Exception as exc:
        interrupted = await _kill_suspended_process(process)
        if interrupted:
            raise asyncio.CancelledError from None
        raise _ContainmentError(
            f"could not create the shell Job Object: {type(exc).__name__}: {exc}"
        ) from None
    if job.handle is None:
        interrupted = await _kill_suspended_process(process)
        if interrupted:
            raise asyncio.CancelledError from None
        raise _ContainmentError(
            "could not assign the suspended shell to a Job Object"
        )

    tree = _ProcessTree(process, job)
    try:
        _resume_windows_process(process.pid)
    except Exception as exc:
        await tree.terminate()
        raise _ContainmentError(
            f"could not resume the contained shell: {type(exc).__name__}: {exc}"
        ) from None
    return tree


class ShellTool(Tool):
    name: ClassVar[str] = "shell"
    description: ClassVar[str] = (
        "Execute a shell command and return combined stdout/stderr. "
        "On Windows this runs PowerShell; on POSIX it runs bash. "
        "The working directory persists as the project directory for every call. "
        "Stdin is closed and only user-allowlisted environment variables are inherited. "
        "Use for running programs, tests, git, package managers, etc."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to execute"},
            "timeout_seconds": {
                "type": "integer",
                "description": f"Timeout in seconds (default {DEFAULT_TIMEOUT}, max {MAX_TIMEOUT})",
            },
        },
        "required": ["command"],
    }
    read_only: ClassVar[bool] = False

    def permission_override(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> PermissionAction | None:
        return ctx.config.shell_permission(str(args.get("command", "")))

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = args.get("command", "")
        if not isinstance(command, str) or not command.strip():
            return ToolResult(content="Error: command 为空", is_error=True)
        try:
            timeout = max(
                1,
                min(
                    int(args.get("timeout_seconds") or DEFAULT_TIMEOUT),
                    MAX_TIMEOUT,
                ),
            )
        except (TypeError, ValueError):
            return ToolResult(content="Error: timeout_seconds 必须为整数", is_error=True)

        try:
            if sys.platform == "win32":
                argv = [
                    _windows_system_executable(
                        "WindowsPowerShell\\v1.0\\powershell.exe"
                    ),
                    "-NoProfile",
                    "-Command",
                    command,
                ]
            else:
                argv = [_posix_bash(), "-c", command]
            tree = await _spawn_process_tree(
                argv,
                cwd=ctx.cwd,
                env=select_environment(ctx.config.shell_inherit_env),
            )
        except _ContainmentError as exc:
            return ToolResult(
                content=f"Error: 命令未执行：进程收容失败（{exc}）",
                is_error=True,
            )
        except OSError as exc:
            return ToolResult(
                content=f"Error: 命令无法启动（{type(exc).__name__}: {exc}）",
                is_error=True,
            )

        proc = tree.process
        try:
            stdout, omitted = await asyncio.wait_for(
                _read_contained_process(tree), timeout=timeout
            )
        except (asyncio.TimeoutError, TimeoutError):
            return ToolResult(
                content=f"Error: 命令超时（{timeout}s）已被终止: {command}",
                is_error=True,
            )
        finally:
            # Always close containment, including after a clean shell exit.
            # A background child can keep running after its shell leader exits.
            await tree.terminate()

        output = _decode(stdout or b"").strip()
        if omitted:
            output += f"\n[output truncated: {omitted} bytes omitted]"
        if proc.returncode != 0:
            output = f"{output}\n[exit code: {proc.returncode}]".strip()
            return ToolResult(content=output or f"[exit code: {proc.returncode}]", is_error=True)
        return ToolResult(content=output or "(no output)")
