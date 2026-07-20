"""Asynchronous, bounded stdio client for the Language Server Protocol."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import ctypes
import math
import os
import secrets
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from emiya_agent.lsp.protocol import (
    MAX_FRAME_BYTES,
    MAX_HEADER_BYTES,
    LSPConnectionClosed,
    LSPProtocolError,
    encoded_json_size,
    read_message_with_size,
    write_message,
)

DEFAULT_REQUEST_TIMEOUT = 30.0
DEFAULT_SHUTDOWN_TIMEOUT = 3.0
MAX_STDERR_BYTES = 64 * 1024
MAX_NOTIFICATIONS = 256
MAX_NOTIFICATION_BYTES = MAX_FRAME_BYTES
MAX_PENDING_REQUESTS = 256
MAX_PENDING_RESPONSE_BYTES = 2 * MAX_FRAME_BYTES
MAX_SERVER_REQUESTS = 32
MAX_SERVER_REQUEST_BYTES = MAX_FRAME_BYTES
MAX_CONFIGURATION_ITEMS = 128
MAX_CONFIGURATION_RESPONSE_BYTES = 1024 * 1024
MAX_CONFIGURATION_SECTION_LENGTH = 1024
MAX_CONFIGURATION_COMPONENTS = 64
MAX_TIMEOUT_SECONDS = 600.0
_CANCEL_SEND_TIMEOUT = 1.0
_STDIN_CLOSE_TIMEOUT = 1.0
_PROCESS_CLEANUP_TIMEOUT = 12.0
_MISSING = object()

_WINDOWS_BROKER = r"""
import ctypes
import subprocess
import sys
from ctypes import wintypes

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
kernel32.OpenEventW.restype = wintypes.HANDLE
kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel32.WaitForSingleObject.restype = wintypes.DWORD
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL
handle = kernel32.OpenEventW(0x00100000, False, sys.argv[1])
if not handle:
    raise SystemExit(125)
try:
    if kernel32.WaitForSingleObject(handle, 15000) != 0:
        raise SystemExit(125)
finally:
    kernel32.CloseHandle(handle)
try:
    child = subprocess.Popen(
        sys.argv[2:],
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
        close_fds=False,
    )
except Exception:
    raise SystemExit(126) from None
raise SystemExit(child.wait())
"""


class _NotificationsClosed:
    __slots__ = ("message",)

    def __init__(self, message: str) -> None:
        self.message = message


class _WindowsStartGate:
    """Hold a trusted broker until it has been assigned to the Job Object."""

    def __init__(self) -> None:
        self.handle: int | None = None
        self.name = (
            f"Local\\EmiyaAgent-LSP-{os.getpid()}-{secrets.token_hex(16)}"
        )
        if sys.platform != "win32":
            return
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateEventW.argtypes = [
            ctypes.c_void_p,
            wintypes.BOOL,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        ]
        kernel32.CreateEventW.restype = wintypes.HANDLE
        handle = kernel32.CreateEventW(None, True, False, self.name)
        if not handle:
            raise OSError("cannot create the language server start gate")
        self.handle = int(handle)

    def release(self) -> None:
        if self.handle is None:
            raise OSError("language server start gate is unavailable")
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.SetEvent.argtypes = [wintypes.HANDLE]
        kernel32.SetEvent.restype = wintypes.BOOL
        if not kernel32.SetEvent(wintypes.HANDLE(self.handle)):
            raise OSError("cannot release the language server start gate")

    def close(self) -> None:
        if self.handle is None:
            return
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle(wintypes.HANDLE(self.handle))
        self.handle = None


def _bounded_timeout(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite positive number") from None
    if not math.isfinite(result) or result <= 0 or result > MAX_TIMEOUT_SECONDS:
        raise ValueError(
            f"{name} must be in the range (0, {MAX_TIMEOUT_SECONDS:g}]"
        )
    return result


class _WindowsJob:
    """Best-effort kill-on-close Job Object for a language server tree."""

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

    def close(self) -> None:
        if self.handle is None:
            return
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle(wintypes.HANDLE(self.handle))
        self.handle = None


def _windows_system_executable(name: str) -> str:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetSystemDirectoryW.argtypes = [wintypes.LPWSTR, wintypes.UINT]
    kernel32.GetSystemDirectoryW.restype = wintypes.UINT
    buffer = ctypes.create_unicode_buffer(32768)
    length = kernel32.GetSystemDirectoryW(buffer, len(buffer))
    if not length or length >= len(buffer):
        raise OSError("cannot resolve the Windows system directory")
    path = Path(buffer.value) / name
    if not path.is_file():
        raise FileNotFoundError("required Windows system executable is unavailable")
    return str(path)


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _terminate_process_tree(
    process: asyncio.subprocess.Process, job: _WindowsJob | None
) -> None:
    if job is not None and job.handle is not None:
        job.close()
    elif sys.platform == "win32":
        taskkill = await asyncio.create_subprocess_exec(
            _windows_system_executable("taskkill.exe"),
            "/PID",
            str(process.pid),
            "/T",
            "/F",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env={
                "SYSTEMROOT": str(
                    Path(_windows_system_executable("cmd.exe")).parent.parent
                )
            },
        )
        try:
            await asyncio.wait_for(taskkill.wait(), timeout=5)
        except (asyncio.TimeoutError, TimeoutError):
            taskkill.kill()
            await taskkill.wait()
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        deadline = asyncio.get_running_loop().time() + 3
        while _process_group_exists(process.pid):
            if asyncio.get_running_loop().time() >= deadline:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                break
            await asyncio.sleep(0.05)

    if process.returncode is None:
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except (asyncio.TimeoutError, TimeoutError):
            if sys.platform == "win32":
                process.kill()
            else:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            await process.wait()


class LSPClientError(RuntimeError):
    """Base exception for LSP client lifecycle and response failures."""


class LSPRequestTimeout(LSPClientError):
    """Raised after a request deadline expires and cancellation is sent."""


class LSPResponseError(LSPClientError):
    """A structured JSON-RPC error returned by the language server."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"language server error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


def _valid_id(value: Any) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(
        value, str
    )


class LSPClient:
    """One LSP server process and its concurrent JSON-RPC request channel.

    ``start`` launches the argv directly, performs ``initialize``, and sends the
    matching ``initialized`` notification. The environment is retained only for
    process creation and is deliberately omitted from representations/errors.
    """

    def __init__(
        self,
        argv: Sequence[str | os.PathLike[str]],
        *,
        cwd: str | os.PathLike[str],
        env: Mapping[str, str],
        settings: Mapping[str, Any] | None = None,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT,
    ) -> None:
        if isinstance(argv, (str, bytes)) or not argv:
            raise ValueError("argv must be a non-empty sequence")
        converted = tuple(os.fspath(value) for value in argv)
        if any(not value or "\x00" in value for value in converted):
            raise ValueError("argv entries must be non-empty and contain no NUL")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in env.items()
        ):
            raise ValueError("LSP environment keys and values must be strings")

        self._argv = converted
        self._cwd = Path(cwd).resolve()
        self._env = dict(env)
        self._settings = copy.deepcopy(dict(settings or {}))
        self._request_timeout = _bounded_timeout(
            request_timeout, name="request_timeout"
        )
        self._shutdown_timeout = _bounded_timeout(
            shutdown_timeout, name="shutdown_timeout"
        )

        self._process: asyncio.subprocess.Process | None = None
        self._job: _WindowsJob | None = None
        self._start_gate: _WindowsStartGate | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._server_request_tasks: dict[asyncio.Task[None], int] = {}
        self._server_request_bytes = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._pending_response_sizes: dict[asyncio.Future[Any], int] = {}
        self._pending_response_bytes = 0
        self._next_id = 1
        self._write_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._terminate_lock = asyncio.Lock()
        self._notifications: asyncio.Queue[
            tuple[dict[str, Any], int] | _NotificationsClosed
        ] = asyncio.Queue(maxsize=MAX_NOTIFICATIONS)
        self._queued_notification_bytes = 0
        self._notification_closed: _NotificationsClosed | None = None
        self._stderr = bytearray()
        self._stderr_omitted = 0
        self._start_attempted = False
        self._started = False
        self._initialized = False
        self._closing = False
        self._closed = False
        self._connection_error: BaseException | None = None
        self._containment_closed = False
        self.server_capabilities: dict[str, Any] = {}

    def __repr__(self) -> str:
        state = "closed" if self._closed else "running" if self._started else "new"
        return f"<{type(self).__name__} state={state}>"

    @property
    def process(self) -> asyncio.subprocess.Process | None:
        return self._process

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def stderr_text(self) -> str:
        text = bytes(self._stderr).decode("utf-8", errors="replace")
        if self._stderr_omitted:
            text += f"\n[stderr truncated: {self._stderr_omitted} bytes omitted]"
        return text

    async def __aenter__(self) -> LSPClient:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def start(
        self, initialize_params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Launch the server, initialize it, and return InitializeResult."""

        async with self._start_lock:
            if self._start_attempted or self._started:
                raise LSPClientError(
                    "language server client has already been started"
                )
            if self._closed:
                raise LSPClientError("language server client is closed")
            self._start_attempted = True

            spawn_options: dict[str, Any] = {"limit": MAX_HEADER_BYTES}
            gate: _WindowsStartGate | None = None
            spawn_argv = self._argv
            if sys.platform == "win32":
                spawn_options["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP
                )
                try:
                    gate = _WindowsStartGate()
                except OSError:
                    self._closed = True
                    self._env.clear()
                    self._signal_notifications_closed(
                        "failed to prepare language server containment"
                    )
                    raise LSPClientError(
                        "failed to prepare language server containment"
                    ) from None
                spawn_argv = (
                    sys.executable,
                    "-I",
                    "-c",
                    _WINDOWS_BROKER,
                    gate.name,
                    *self._argv,
                )
            else:
                spawn_options["start_new_session"] = True
            try:
                process = await asyncio.create_subprocess_exec(
                    *spawn_argv,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self._cwd),
                    env=self._env,
                    **spawn_options,
                )
            except asyncio.CancelledError:
                if gate is not None:
                    gate.close()
                self._closed = True
                self._signal_notifications_closed(
                    "language server start was cancelled"
                )
                raise
            except Exception:
                if gate is not None:
                    gate.close()
                self._closed = True
                self._signal_notifications_closed(
                    "failed to start language server"
                )
                raise LSPClientError("failed to start language server") from None
            finally:
                self._env.clear()

            self._process = process
            self._job = (
                _WindowsJob(process.pid) if sys.platform == "win32" else None
            )
            if (
                sys.platform == "win32"
                and (self._job is None or self._job.handle is None)
            ):
                assert gate is not None
                gate.close()
                await self._terminate_bounded()
                self._closed = True
                self._signal_notifications_closed(
                    "failed to contain language server process tree"
                )
                raise LSPClientError(
                    "failed to contain language server process tree"
                )
            if gate is not None:
                try:
                    gate.release()
                except OSError:
                    gate.close()
                    await self._terminate_bounded()
                    self._closed = True
                    self._signal_notifications_closed(
                        "failed to release language server start gate"
                    )
                    raise LSPClientError(
                        "failed to release language server start gate"
                    ) from None
                # Keep the named event alive until initialize proves that the
                # broker opened it and launched the real server.
                self._start_gate = gate
            self._started = True
            self._reader_task = asyncio.create_task(
                self._reader_loop(), name=f"lsp-reader-{process.pid}"
            )
            self._stderr_task = asyncio.create_task(
                self._drain_stderr(), name=f"lsp-stderr-{process.pid}"
            )

        params: dict[str, Any] = {
            "processId": os.getpid(),
            "clientInfo": {"name": "EmiyaAgent"},
            "rootUri": self._cwd.as_uri(),
            "capabilities": {},
        }
        if initialize_params is not None:
            if not isinstance(initialize_params, Mapping):
                await self.close(graceful=False)
                raise ValueError("initialize_params must be a mapping")
            params.update(copy.deepcopy(dict(initialize_params)))

        try:
            result = await self.request("initialize", params)
            if not isinstance(result, dict):
                raise LSPProtocolError("initialize result must be a JSON object")
            capabilities = result.get("capabilities", {})
            if not isinstance(capabilities, dict):
                raise LSPProtocolError(
                    "initialize result capabilities must be a JSON object"
                )
            self.server_capabilities = copy.deepcopy(capabilities)
            await self.notify("initialized", {})
            self._initialized = True
            if self._start_gate is not None:
                self._start_gate.close()
                self._start_gate = None
            return result
        except BaseException:
            await self.close(graceful=False)
            raise

    async def request(
        self,
        method: str,
        params: Any = _MISSING,
        *,
        timeout: float | None = None,
    ) -> Any:
        """Send a request; timeout sends ``$/cancelRequest`` first."""

        return await self._request(method, params, timeout=timeout)

    async def _request(
        self,
        method: str,
        params: Any,
        *,
        timeout: float | None,
        allow_closing: bool = False,
    ) -> Any:
        self._ensure_usable(allow_closing=allow_closing)
        if not isinstance(method, str) or not method:
            raise ValueError("LSP method must be a non-empty string")
        deadline = (
            self._request_timeout
            if timeout is None
            else _bounded_timeout(timeout, name="request timeout")
        )
        if len(self._pending) >= MAX_PENDING_REQUESTS:
            raise LSPClientError(
                f"language server has {MAX_PENDING_REQUESTS} pending requests"
            )

        request_id = self._next_id
        self._next_id += 1
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not _MISSING:
            message["params"] = params
        try:
            async with asyncio.timeout(deadline):
                await self._send(message, allow_closing=allow_closing)
                return await asyncio.shield(future)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            self._remove_pending(request_id, future)
            await self._send_cancel(request_id, allow_closing=allow_closing)
            raise LSPRequestTimeout(
                f"language server request timed out: {method}"
            ) from exc
        except asyncio.CancelledError:
            self._remove_pending(request_id, future)
            await self._send_cancel(request_id, allow_closing=allow_closing)
            raise
        except BaseException:
            self._remove_pending(request_id, future)
            raise
        finally:
            response_bytes = self._pending_response_sizes.pop(future, 0)
            self._pending_response_bytes = max(
                0, self._pending_response_bytes - response_bytes
            )
            if future.done() and future.cancelled():
                self._pending.pop(request_id, None)

    async def notify(self, method: str, params: Any = _MISSING) -> None:
        """Send a JSON-RPC notification."""

        await self._notify(method, params)

    async def _notify(
        self,
        method: str,
        params: Any = _MISSING,
        *,
        allow_closing: bool = False,
    ) -> None:
        self._ensure_usable(allow_closing=allow_closing)
        if not isinstance(method, str) or not method:
            raise ValueError("LSP method must be a non-empty string")
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not _MISSING:
            message["params"] = params
        await self._send(message, allow_closing=allow_closing)

    async def next_notification(
        self, *, timeout: float | None = None
    ) -> dict[str, Any]:
        """Return the next server notification from the bounded queue."""

        if timeout is None:
            item = await self._notifications.get()
        else:
            deadline = _bounded_timeout(timeout, name="notification timeout")
            item = await asyncio.wait_for(
                self._notifications.get(), timeout=deadline
            )
        if isinstance(item, _NotificationsClosed):
            # Keep one terminal marker available so every current or future
            # waiter observes closure instead of hanging behind the first one.
            with contextlib.suppress(asyncio.QueueFull):
                self._notifications.put_nowait(item)
            raise LSPConnectionClosed(item.message)
        message, frame_bytes = item
        self._queued_notification_bytes = max(
            0, self._queued_notification_bytes - frame_bytes
        )
        return message

    async def close(self, *, graceful: bool = True) -> None:
        """Send shutdown/exit when possible, then boundedly reap the process tree."""

        cancelled = False
        async with self._close_lock:
            if self._closed:
                return
            task = asyncio.create_task(self._close_impl(graceful=graceful))
            while True:
                try:
                    await asyncio.shield(task)
                    break
                except asyncio.CancelledError:
                    if task.done():
                        break
                    current = asyncio.current_task()
                    if current is not None:
                        current.uncancel()
                    cancelled = True
            if task.cancelled() and not self._closed:
                # An independently cancelled cleanup task must not turn the
                # shield loop into a busy-spin or leave a process tree behind.
                await self._finalize_close()
            elif not task.cancelled():
                task.result()
            if cancelled:
                raise asyncio.CancelledError

    async def _close_impl(self, *, graceful: bool) -> None:
        # If close races the subprocess spawn, wait until start either publishes
        # the process or marks the one-shot client closed.
        async with self._start_lock:
            pass
        self._closing = True
        process = self._process
        try:
            if (
                process is not None
                and graceful
                and self._initialized
                and self._connection_error is None
            ):
                with contextlib.suppress(Exception):
                    await self._request(
                        "shutdown",
                        None,
                        timeout=self._shutdown_timeout,
                        allow_closing=True,
                    )
                with contextlib.suppress(Exception):
                    await self._notify("exit", allow_closing=True)
        finally:
            await self._finalize_close()

    async def _finalize_close(self) -> None:
        process = self._process
        try:
            if process is not None and process.stdin is not None:
                try:
                    process.stdin.close()
                    await asyncio.wait_for(
                        process.stdin.wait_closed(),
                        timeout=_STDIN_CLOSE_TIMEOUT,
                    )
                except (asyncio.TimeoutError, TimeoutError, OSError):
                    with contextlib.suppress(Exception):
                        process.stdin.transport.abort()

            if process is not None and process.returncode is None:
                with contextlib.suppress(asyncio.TimeoutError, TimeoutError):
                    await asyncio.wait_for(
                        process.wait(), timeout=self._shutdown_timeout
                    )

            # Always clean the containment unit.  The server leader may already
            # have exited while descendants remain in its process group/job.
            await self._terminate_bounded()

            if self._reader_task is not None and not self._reader_task.done():
                self._reader_task.cancel()
            await self._await_task(self._reader_task, timeout=1.0)

            server_tasks = tuple(self._server_request_tasks)
            for task in server_tasks:
                task.cancel()
            if server_tasks:
                await asyncio.gather(*server_tasks, return_exceptions=True)
            await self._await_task(self._stderr_task, timeout=1.0)
        finally:
            error = LSPConnectionClosed("language server client closed")
            self._fail_pending(error)
            self._signal_notifications_closed(str(error))
            if self._job is not None:
                self._job.close()
            if self._start_gate is not None:
                self._start_gate.close()
                self._start_gate = None
            self._closed = True

    def _ensure_usable(self, *, allow_closing: bool) -> None:
        if not self._started:
            raise LSPClientError("language server client has not been started")
        if self._closed or (self._closing and not allow_closing):
            raise LSPConnectionClosed("language server client is closed")
        if self._connection_error is not None:
            raise LSPConnectionClosed("language server connection failed") from (
                self._connection_error
            )

    async def _send(
        self, message: Mapping[str, Any], *, allow_closing: bool
    ) -> None:
        self._ensure_usable(allow_closing=allow_closing)
        process = self._process
        if process is None or process.stdin is None:
            raise LSPConnectionClosed("language server stdin is unavailable")
        async with self._write_lock:
            # close() may have begun while this sender waited for another
            # writer.  Re-check under the serialization lock before writing.
            self._ensure_usable(allow_closing=allow_closing)
            await write_message(process.stdin, message)

    def _remove_pending(
        self, request_id: int, future: asyncio.Future[Any]
    ) -> None:
        if self._pending.get(request_id) is future:
            self._pending.pop(request_id, None)
        if not future.done():
            future.cancel()
        elif not future.cancelled():
            # A connection failure can complete the future while this request
            # is still blocked in drain().  Retrieve that discarded exception
            # so the event loop does not report an orphaned Future error.
            with contextlib.suppress(Exception):
                future.exception()

    async def _send_cancel(
        self, request_id: int, *, allow_closing: bool
    ) -> None:
        if self._closed or self._connection_error is not None:
            return
        task = asyncio.create_task(
            self._notify(
                "$/cancelRequest",
                {"id": request_id},
                allow_closing=allow_closing,
            )
        )
        with contextlib.suppress(Exception):
            await asyncio.wait_for(task, timeout=_CANCEL_SEND_TIMEOUT)

    async def _reader_loop(self) -> None:
        process = self._process
        assert process is not None and process.stdout is not None
        error: BaseException | None = None
        try:
            while True:
                message, frame_bytes = await read_message_with_size(
                    process.stdout
                )
                self._dispatch_message(message, frame_bytes=frame_bytes)
        except asyncio.CancelledError:
            if not self._closing:
                error = LSPConnectionClosed(
                    "language server reader was cancelled"
                )
                self._connection_error = error
            raise
        except BaseException as exc:
            error = exc
            if not self._closing:
                self._connection_error = exc
        finally:
            if error is not None:
                self._fail_pending(error)
                self._signal_notifications_closed(str(error))
            if not self._closing:
                await self._terminate_bounded()

    def _dispatch_message(
        self, message: dict[str, Any], *, frame_bytes: int
    ) -> None:
        if message.get("jsonrpc") != "2.0":
            raise LSPProtocolError("LSP message has an invalid jsonrpc version")
        has_id = "id" in message
        has_method = "method" in message

        if has_method:
            method = message["method"]
            if not isinstance(method, str) or not method:
                raise LSPProtocolError("LSP method must be a non-empty string")
            if has_id:
                if not _valid_id(message["id"]):
                    raise LSPProtocolError("LSP request has an invalid id")
                if (
                    len(self._server_request_tasks) >= MAX_SERVER_REQUESTS
                    or self._server_request_bytes + frame_bytes
                    > MAX_SERVER_REQUEST_BYTES
                ):
                    raise LSPProtocolError(
                        "language server sent too many concurrent requests"
                    )
                task = asyncio.create_task(self._handle_server_request(message))
                self._server_request_tasks[task] = frame_bytes
                self._server_request_bytes += frame_bytes
                task.add_done_callback(self._server_request_done)
            else:
                self._queue_notification(message, frame_bytes=frame_bytes)
            return

        if not has_id or not _valid_id(message["id"]):
            raise LSPProtocolError("LSP response has an invalid id")
        has_result = "result" in message
        has_error = "error" in message
        if has_result == has_error:
            raise LSPProtocolError(
                "LSP response must contain exactly one of result or error"
            )
        if has_result:
            response_id = message["id"]
            future = self._take_response_future(
                response_id, frame_bytes=frame_bytes
            )
            if future is None:
                return
            future.set_result(message["result"])
            return
        error = message["error"]
        if not isinstance(error, dict):
            raise LSPProtocolError("LSP response error must be an object")
        code = error.get("code")
        error_message = error.get("message")
        if (
            not isinstance(code, int)
            or isinstance(code, bool)
            or not isinstance(error_message, str)
        ):
            raise LSPProtocolError("LSP response error is malformed")
        response_id = message["id"]
        future = self._take_response_future(
            response_id, frame_bytes=frame_bytes
        )
        if future is None:
            return
        future.set_exception(
            LSPResponseError(code, error_message, error.get("data"))
        )

    def _take_response_future(
        self, response_id: int | str, *, frame_bytes: int
    ) -> asyncio.Future[Any] | None:
        future = self._pending.get(response_id)
        if future is None or future.done():
            return None
        if (
            self._pending_response_bytes + frame_bytes
            > MAX_PENDING_RESPONSE_BYTES
        ):
            raise LSPProtocolError(
                "language server responses exceed the pending byte budget"
            )
        self._pending.pop(response_id, None)
        self._pending_response_sizes[future] = frame_bytes
        self._pending_response_bytes += frame_bytes
        return future

    def _server_request_done(self, task: asyncio.Task[None]) -> None:
        frame_bytes = self._server_request_tasks.pop(task, 0)
        self._server_request_bytes = max(
            0, self._server_request_bytes - frame_bytes
        )
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.exception()

    def _pop_notification(self) -> None:
        item = self._notifications.get_nowait()
        if isinstance(item, tuple):
            self._queued_notification_bytes = max(
                0, self._queued_notification_bytes - item[1]
            )

    def _queue_notification(
        self, message: dict[str, Any], *, frame_bytes: int
    ) -> None:
        if self._notification_closed is not None:
            return
        while (
            self._notifications.full()
            or self._queued_notification_bytes + frame_bytes
            > MAX_NOTIFICATION_BYTES
        ):
            with contextlib.suppress(asyncio.QueueEmpty):
                self._pop_notification()
                continue
            break
        if frame_bytes > MAX_NOTIFICATION_BYTES:
            return
        self._notifications.put_nowait((message, frame_bytes))
        self._queued_notification_bytes += frame_bytes

    def _signal_notifications_closed(self, message: str) -> None:
        if self._notification_closed is not None:
            return
        terminal = _NotificationsClosed(message)
        self._notification_closed = terminal
        if self._notifications.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._pop_notification()
        self._notifications.put_nowait(terminal)

    async def _handle_server_request(self, message: dict[str, Any]) -> None:
        request_id = message["id"]
        method = message["method"]
        try:
            if method == "workspace/applyEdit":
                result: Any = {
                    "applied": False,
                    "failureReason": "EmiyaAgent does not apply server workspace edits",
                }
                await self._send_response(request_id, result=result)
                return
            if method == "workspace/configuration":
                try:
                    result = self._configuration_result(message.get("params"))
                except ValueError:
                    await self._send_response(
                        request_id,
                        error={"code": -32602, "message": "Invalid params"},
                    )
                else:
                    await self._send_response(request_id, result=result)
                return
            await self._send_response(
                request_id,
                error={"code": -32601, "message": "Method not found"},
            )
        except (LSPClientError, LSPProtocolError):
            return

    def _configuration_result(self, params: Any) -> list[Any]:
        if not isinstance(params, dict) or not isinstance(params.get("items"), list):
            raise ValueError("invalid workspace/configuration params")
        items = params["items"]
        if len(items) > MAX_CONFIGURATION_ITEMS:
            raise ValueError("too many workspace/configuration items")
        values: list[Any] = []
        encoded_size = 2  # JSON array brackets
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("invalid workspace/configuration item")
            section = item.get("section")
            if section is None:
                value: Any = self._settings
            else:
                if not isinstance(section, str):
                    raise ValueError("invalid workspace/configuration section")
                value = self._setting_for_section(section)
            remaining = MAX_CONFIGURATION_RESPONSE_BYTES - encoded_size
            try:
                value_size = encoded_json_size(value, max_bytes=remaining)
            except LSPProtocolError as exc:
                raise ValueError(
                    "workspace/configuration response is too large or invalid"
                ) from exc
            separator_size = 1 if values else 0
            if encoded_size + separator_size + value_size > (
                MAX_CONFIGURATION_RESPONSE_BYTES
            ):
                raise ValueError(
                    "workspace/configuration response is too large"
                )
            encoded_size += separator_size + value_size
            values.append(copy.deepcopy(value))
        return values

    def _setting_for_section(self, section: str) -> Any:
        if len(section) > MAX_CONFIGURATION_SECTION_LENGTH:
            raise ValueError("workspace/configuration section is too long")
        component_count = 0 if not section else section.count(".") + 1
        if component_count > MAX_CONFIGURATION_COMPONENTS:
            raise ValueError(
                "workspace/configuration section has too many components"
            )
        if section in self._settings:
            return self._settings[section]
        value: Any = self._settings
        for component in section.split(".") if section else ():
            if not isinstance(value, Mapping) or component not in value:
                return None
            value = value[component]
        return value

    async def _send_response(
        self,
        request_id: int | str,
        *,
        result: Any = _MISSING,
        error: Mapping[str, Any] | None = None,
    ) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
        if error is not None:
            message["error"] = dict(error)
        elif result is not _MISSING:
            message["result"] = result
        else:
            message["result"] = None
        await self._send(message, allow_closing=False)

    async def _drain_stderr(self) -> None:
        process = self._process
        assert process is not None and process.stderr is not None
        while True:
            chunk = await process.stderr.read(64 * 1024)
            if not chunk:
                return
            remaining = MAX_STDERR_BYTES - len(self._stderr)
            if remaining > 0:
                self._stderr.extend(chunk[:remaining])
            self._stderr_omitted += max(0, len(chunk) - max(remaining, 0))

    async def _terminate_bounded(self) -> None:
        async with self._terminate_lock:
            if self._containment_closed:
                return
            process = self._process
            if process is None:
                if self._job is not None:
                    self._job.close()
                self._containment_closed = True
                return
            try:
                await asyncio.wait_for(
                    _terminate_process_tree(process, self._job),
                    timeout=_PROCESS_CLEANUP_TIMEOUT,
                )
            except (asyncio.TimeoutError, TimeoutError, OSError):
                if self._job is not None:
                    self._job.close()
                if sys.platform != "win32":
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                elif process.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                if process.returncode is None:
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(process.wait(), timeout=1.0)
            finally:
                # Never signal this numeric process-group id again after it can
                # be recycled for an unrelated process.
                self._containment_closed = True

    @staticmethod
    async def _await_task(
        task: asyncio.Task[None] | None, *, timeout: float
    ) -> None:
        if task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        except asyncio.CancelledError:
            if not task.cancelled():
                raise
        except Exception:
            return

    def _fail_pending(self, error: BaseException) -> None:
        pending = tuple(self._pending.values())
        self._pending.clear()
        for future in pending:
            if not future.done():
                future.set_exception(error)
