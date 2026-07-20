"""Workspace-confined orchestration for configured language servers.

The manager deliberately does not discover or install servers.  It selects a
user-configured server, resolves its executable before spawning, lazily caches
one client per ``(server, project root)``, and normalizes untrusted LSP results
before they can enter the model context.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import math
import os
import secrets
import stat
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit
from urllib.request import url2pathname

from emiya_agent.config import Config, select_environment
from emiya_agent.lsp.client import LSPClient, LSPResponseError
from emiya_agent.lsp.protocol import (
    MAX_FRAME_BYTES,
    LSPProtocolError,
    encoded_json_size,
)
from emiya_agent.safe_paths import UnsafePathError, confined_path, open_confined

MAX_RESULTS = 100
MAX_NOTIFICATION_SCAN = 64
MAX_SYMBOL_DEPTH = 32
MAX_SCANNED_SYMBOLS = 800
MAX_TEXT_CHARS = 12_000
MAX_RESULT_JSON_BYTES = 20_000
MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
MAX_RESULT_FILES = 8
MAX_OPEN_DOCUMENTS_PER_CONNECTION = 8
MAX_CONNECTIONS = 8
MAX_PENDING_LAUNCH_TICKETS = 32
MAX_EXECUTABLE_FINGERPRINT_BYTES = 512 * 1024 * 1024
MAX_HOVER_DEPTH = 16
MAX_HOVER_NODES = 256
MAX_PATH_SEARCH_ENTRIES = 256
PUBLISH_DIAGNOSTICS_TIMEOUT = 2.0
METHOD_NOT_FOUND = -32601


class LSPManagerError(RuntimeError):
    """A safe, user-facing LSP configuration or orchestration error."""


class LSPDisabledError(LSPManagerError):
    """Raised when the LSP feature is disabled."""


class LSPConfigurationError(LSPManagerError):
    """Raised when no valid configured server can handle a request."""


class LSPServerNotInstalledError(LSPManagerError):
    """Raised when a configured executable cannot be resolved locally."""


class LSPPathError(LSPManagerError):
    """Raised when a path or server-returned URI escapes the workspace."""


@dataclass(frozen=True)
class _ServerSelection:
    name: str
    spec: Any
    extension: str
    language: str


@dataclass
class _DocumentState:
    text: str
    version: int = 1
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    diagnostic_result_id: str | None = None


@dataclass
class _Connection:
    selection: _ServerSelection
    root: Path
    client: Any
    argv: tuple[str, ...]
    launch_signature: str
    executable_fingerprint: tuple[int, int, int, int, int, str]
    documents: OrderedDict[Path, _DocumentState] = field(default_factory=OrderedDict)
    next_document_version: int = 1
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(frozen=True)
class _LaunchSnapshot:
    path: Path
    server: str
    root: Path
    signature: str
    executable_fingerprint: tuple[int, int, int, int, int, str]


def _bounded_text(value: Any, limit: int = MAX_TEXT_CHARS) -> str | None:
    if not isinstance(value, str):
        return None
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n... [truncated {len(value) - limit} chars]"


def _integer(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _extension(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        return ""
    return normalized if normalized.startswith(".") else f".{normalized}"


def _environment_value(environment: Mapping[str, str], name: str) -> str | None:
    if os.name != "nt":
        return environment.get(name)
    folded = name.casefold()
    for key, value in environment.items():
        if key.casefold() == folded:
            return value
    return None


def _absolute_path_executable(
    command: str, environment: Mapping[str, str]
) -> str | None:
    """Resolve a bare command using absolute PATH entries only.

    In particular, this avoids Windows ``NeedCurrentDirectoryForExePath``
    behavior, empty PATH components, and relative entries silently searching
    the untrusted workspace.
    """

    raw_path = _environment_value(environment, "PATH")
    if not raw_path:
        return None

    suffixes = ("",)
    if os.name == "nt":
        raw_pathext = _environment_value(environment, "PATHEXT") or ".COM;.EXE"
        native_extensions: list[str] = []
        for raw_extension in raw_pathext.split(os.pathsep)[:32]:
            extension = raw_extension.strip().upper()
            if extension in {".COM", ".EXE"} and extension not in native_extensions:
                native_extensions.append(extension)
        command_suffix = Path(command).suffix.upper()
        suffixes = (
            ("",)
            if command_suffix in native_extensions
            else tuple(native_extensions or [".COM", ".EXE"])
        )

    for raw_entry in raw_path.split(os.pathsep)[:MAX_PATH_SEARCH_ENTRIES]:
        entry_text = raw_entry.strip('"') if os.name == "nt" else raw_entry
        if not entry_text:
            continue
        entry = Path(entry_text)
        if not entry.is_absolute():
            continue
        for suffix in suffixes:
            try:
                candidate = (entry / f"{command}{suffix}").resolve(strict=True)
            except OSError:
                continue
            if not candidate.is_file():
                continue
            if os.name != "nt" and not os.access(candidate, os.X_OK):
                continue
            return str(candidate)
    return None


def _utf16_offset(line: str, column: int) -> int:
    """Convert a one-based Python-codepoint column to a zero-based UTF-16 offset."""

    if column < 1 or column > len(line) + 1:
        raise LSPManagerError(
            f"column {column} is outside the line (valid range: 1..{len(line) + 1})"
        )
    return len(line[: column - 1].encode("utf-16-le")) // 2


def _codepoint_column(line: str, utf16_offset: int) -> int | None:
    """Convert a UTF-16 offset to a one-based Python-codepoint column."""

    if utf16_offset < 0:
        return None
    units = 0
    for index, character in enumerate(line):
        if units == utf16_offset:
            return index + 1
        width = 2 if ord(character) > 0xFFFF else 1
        if units + width > utf16_offset:
            return None  # the peer pointed into the middle of a surrogate pair
        units += width
    return len(line) + 1 if units == utf16_offset else None


def _line_at(text: str, line: int) -> str | None:
    lines = text.split("\n")
    if line < 0 or line >= len(lines):
        return None
    return lines[line]


def _request_position(text: str, line: int, column: int) -> dict[str, int]:
    if line < 1:
        raise LSPManagerError("line must be a positive, one-based integer")
    line_text = _line_at(text, line - 1)
    if line_text is None:
        raise LSPManagerError(
            f"line {line} is outside the document (valid range: 1..{len(text.split(chr(10)))})"
        )
    return {"line": line - 1, "character": _utf16_offset(line_text, column)}


def _range_for_lines(value: Any, lines: Sequence[str]) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    start = value.get("start")
    end = value.get("end")
    if not isinstance(start, Mapping) or not isinstance(end, Mapping):
        return None
    start_line = _integer(start.get("line"))
    start_character = _integer(start.get("character"))
    end_line = _integer(end.get("line"))
    end_character = _integer(end.get("character"))
    if None in {start_line, start_character, end_line, end_character}:
        return None
    assert start_line is not None and start_character is not None
    assert end_line is not None and end_character is not None
    if start_line < 0 or end_line < 0:
        return None
    if start_line >= len(lines) or end_line >= len(lines):
        return None
    start_text = lines[start_line]
    end_text = lines[end_line]
    start_column = _codepoint_column(start_text, start_character)
    end_column = _codepoint_column(end_text, end_character)
    if start_column is None or end_column is None:
        return None
    if (end_line, end_column) < (start_line, start_column):
        return None
    return {
        "line": start_line + 1,
        "column": start_column,
        "end_line": end_line + 1,
        "end_column": end_column,
    }


def _dedupe_sorted(
    values: Sequence[dict[str, Any]],
    *,
    sort_key: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> list[dict[str, Any]]:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for value in values:
        key = tuple((name, repr(item)) for name, item in sorted(value.items()))
        unique.setdefault(key, value)
    ordered = sorted(unique.values(), key=sort_key)
    bounded: list[dict[str, Any]] = []
    encoded_bytes = 2  # surrounding JSON list brackets
    for value in ordered[:MAX_RESULTS]:
        item_bytes = len(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        separator_bytes = 1 if bounded else 0
        if encoded_bytes + separator_bytes + item_bytes > MAX_RESULT_JSON_BYTES:
            break
        bounded.append(value)
        encoded_bytes += separator_bytes + item_bytes
    return bounded


def _executable_fingerprint(path: Path) -> tuple[int, int, int, int, int, str]:
    """Fingerprint an approved executable and detect mutation while reading it."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LSPServerNotInstalledError(
            f"language server executable is unavailable: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise LSPServerNotInstalledError(
                f"language server executable is not a regular file: {path}"
            )
        if before.st_size > MAX_EXECUTABLE_FINGERPRINT_BYTES:
            raise LSPConfigurationError(
                "language server executable is too large to bind safely to an "
                f"approval ({MAX_EXECUTABLE_FINGERPRINT_BYTES} byte limit): {path}"
            )
        digest = hashlib.sha256()
        hashed_bytes = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            hashed_bytes += len(chunk)
            if hashed_bytes > MAX_EXECUTABLE_FINGERPRINT_BYTES:
                raise LSPConfigurationError(
                    "language server executable grew beyond the approval "
                    f"fingerprint limit while being inspected: {path}"
                )
            digest.update(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise LSPServerNotInstalledError(
            f"cannot inspect language server executable: {path}"
        ) from exc
    finally:
        os.close(descriptor)

    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise LSPManagerError(
            "language server executable changed while it was being inspected"
        )
    return (*after_identity, digest.hexdigest())


def _read_utf8_document(path: Path, workspace: Path) -> str:
    """Read one bounded UTF-8 document and normalize newline representation."""

    try:
        relative = path.relative_to(workspace)
        with open_confined(
            workspace,
            relative,
            "rb",
            label="LSP document",
        ) as stream:
            raw = stream.read(MAX_DOCUMENT_BYTES + 1)
    except (OSError, UnsafePathError, ValueError) as exc:
        raise LSPPathError(f"cannot read LSP document: {path}") from exc
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise LSPPathError(
            f"LSP document exceeds the {MAX_DOCUMENT_BYTES}-byte limit: {path}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LSPPathError(f"LSP requires a UTF-8 text file: {path}") from exc
    return text.replace("\r\n", "\n").replace("\r", "\n")


class LSPManager:
    """Lazy LSP connection manager confined to one workspace."""

    def __init__(
        self,
        config: Config,
        cwd: str | os.PathLike[str] | None = None,
        *,
        client_factory: Callable[..., Any] = LSPClient,
    ) -> None:
        self.config = config
        self.cwd = Path(cwd if cwd is not None else config.cwd).absolute()
        self._resolved_workspace: Path | None = None
        self._client_factory = client_factory
        self._connections: dict[tuple[str, Path], _Connection] = {}
        self._launch_tickets: OrderedDict[str, _LaunchSnapshot] = OrderedDict()
        self._connection_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False

    @property
    def connected_count(self) -> int:
        return len(self._connections)

    async def connection_preview(
        self,
        path: str | os.PathLike[str],
        *,
        issue_launch_ticket: bool = False,
    ) -> dict[str, Any]:
        """Resolve a prospective launch without starting a server process."""

        if self._closed:
            raise LSPManagerError("LSP manager is closed")
        self._ensure_enabled()
        resolved = await asyncio.to_thread(self._resolve_file, path)
        async with self._connection_lock:
            if self._closed:
                raise LSPManagerError("LSP manager is closed")
            selection, root, (argv, env) = await self._select_available(resolved)
            requires_start = (selection.name, root) not in self._connections
            ticket: str | None = None
            if issue_launch_ticket and requires_start:
                signature = self._launch_signature(selection, root, argv, env)
                fingerprint = await asyncio.to_thread(
                    _executable_fingerprint, Path(argv[0])
                )
                ticket = secrets.token_urlsafe(24)
                self._launch_tickets[ticket] = _LaunchSnapshot(
                    path=resolved,
                    server=selection.name,
                    root=root,
                    signature=signature,
                    executable_fingerprint=fingerprint,
                )
                while len(self._launch_tickets) > MAX_PENDING_LAUNCH_TICKETS:
                    self._launch_tickets.popitem(last=False)
            return {
                "server": selection.name,
                "root": str(root),
                "argv": list(argv),
                "requires_start": requires_start,
                "launch_ticket": ticket,
            }

    def discard_launch_ticket(self, ticket: str | None) -> None:
        """Invalidate an unused approval snapshot without exposing its contents."""

        if isinstance(ticket, str):
            self._launch_tickets.pop(ticket, None)

    async def diagnostics(
        self,
        path: str | os.PathLike[str],
        *,
        launch_ticket: str | None = None,
        require_launch_ticket: bool = False,
    ) -> list[dict[str, Any]]:
        resolved, selection, connection = await self._prepare(
            path,
            launch_ticket=launch_ticket,
            require_launch_ticket=require_launch_ticket,
        )
        async with connection.lock:
            text, document = await self._sync_document(connection, resolved, selection)
            uri = resolved.as_uri()
            params: dict[str, Any] = {"textDocument": {"uri": uri}}
            if document.diagnostic_result_id is not None:
                params["previousResultId"] = document.diagnostic_result_id
            try:
                result = await self._request(
                    connection, "textDocument/diagnostic", params
                )
            except LSPResponseError as exc:
                if exc.code != METHOD_NOT_FOUND:
                    raise
                return await self._published_diagnostics(
                    connection, resolved, text, document
                )

            if isinstance(result, Mapping) and result.get("kind") == "unchanged":
                result_id = result.get("resultId")
                if isinstance(result_id, str):
                    document.diagnostic_result_id = result_id
                return copy.deepcopy(document.diagnostics)
            items: Any
            if isinstance(result, Mapping):
                items = result.get("items", [])
                result_id = result.get("resultId")
                document.diagnostic_result_id = (
                    result_id if isinstance(result_id, str) else None
                )
            else:
                items = result
                document.diagnostic_result_id = None
            normalized = self._normalize_diagnostics(items, resolved, text)
            document.diagnostics = normalized
            return copy.deepcopy(normalized)

    async def definition(
        self,
        path: str | os.PathLike[str],
        line: int,
        column: int,
        *,
        launch_ticket: str | None = None,
        require_launch_ticket: bool = False,
    ) -> list[dict[str, Any]]:
        resolved, selection, connection = await self._prepare(
            path,
            launch_ticket=launch_ticket,
            require_launch_ticket=require_launch_ticket,
        )
        async with connection.lock:
            text, _ = await self._sync_document(connection, resolved, selection)
            result = await self._request(
                connection,
                "textDocument/definition",
                self._text_position_params(resolved, text, line, column),
            )
            return await self._normalize_locations(result)

    async def references(
        self,
        path: str | os.PathLike[str],
        line: int,
        column: int,
        *,
        include_declaration: bool = True,
        launch_ticket: str | None = None,
        require_launch_ticket: bool = False,
    ) -> list[dict[str, Any]]:
        resolved, selection, connection = await self._prepare(
            path,
            launch_ticket=launch_ticket,
            require_launch_ticket=require_launch_ticket,
        )
        async with connection.lock:
            text, _ = await self._sync_document(connection, resolved, selection)
            params = self._text_position_params(resolved, text, line, column)
            params["context"] = {"includeDeclaration": bool(include_declaration)}
            result = await self._request(connection, "textDocument/references", params)
            return await self._normalize_locations(result)

    async def hover(
        self,
        path: str | os.PathLike[str],
        line: int,
        column: int,
        *,
        launch_ticket: str | None = None,
        require_launch_ticket: bool = False,
    ) -> dict[str, Any] | None:
        resolved, selection, connection = await self._prepare(
            path,
            launch_ticket=launch_ticket,
            require_launch_ticket=require_launch_ticket,
        )
        async with connection.lock:
            text, _ = await self._sync_document(connection, resolved, selection)
            result = await self._request(
                connection,
                "textDocument/hover",
                self._text_position_params(resolved, text, line, column),
            )
            if not isinstance(result, Mapping):
                return None
            contents = self._hover_text(result.get("contents"))
            if not contents:
                return None
            normalized: dict[str, Any] = {"contents": contents}
            hover_range = _range_for_lines(result.get("range"), text.split("\n"))
            if hover_range is not None:
                normalized["range"] = hover_range
            return normalized

    async def document_symbols(
        self,
        path: str | os.PathLike[str],
        *,
        launch_ticket: str | None = None,
        require_launch_ticket: bool = False,
    ) -> list[dict[str, Any]]:
        resolved, selection, connection = await self._prepare(
            path,
            launch_ticket=launch_ticket,
            require_launch_ticket=require_launch_ticket,
        )
        async with connection.lock:
            text, _ = await self._sync_document(connection, resolved, selection)
            result = await self._request(
                connection,
                "textDocument/documentSymbol",
                {"textDocument": {"uri": resolved.as_uri()}},
            )
            return await self._normalize_symbols(
                result, default_path=resolved, default_text=text
            )

    async def workspace_symbols(
        self,
        path: str | os.PathLike[str],
        query: str,
        *,
        launch_ticket: str | None = None,
        require_launch_ticket: bool = False,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str):
            raise LSPManagerError("query must be a string")
        query = query[:1024]
        resolved, selection, connection = await self._prepare(
            path,
            launch_ticket=launch_ticket,
            require_launch_ticket=require_launch_ticket,
        )
        async with connection.lock:
            text, _ = await self._sync_document(connection, resolved, selection)
            result = await self._request(
                connection, "workspace/symbol", {"query": query}
            )
            return await self._normalize_symbols(
                result, default_path=resolved, default_text=text
            )

    async def close_all(self) -> None:
        """Close every cached server; one failure never prevents other cleanup."""

        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            async with self._connection_lock:
                connections = tuple(self._connections.values())
                self._connections.clear()
                self._launch_tickets.clear()
            if connections:
                await asyncio.gather(
                    *(connection.client.close() for connection in connections),
                    return_exceptions=True,
                )

    async def _prepare(
        self,
        path: str | os.PathLike[str],
        *,
        launch_ticket: str | None = None,
        require_launch_ticket: bool = False,
    ) -> tuple[Path, _ServerSelection, _Connection]:
        if self._closed:
            raise LSPManagerError("LSP manager is closed")
        self._ensure_enabled()
        resolved = await asyncio.to_thread(self._resolve_file, path)
        if launch_ticket is not None:
            connection = await self._get_connection(
                None,
                None,
                spawn_configuration=None,
                launch_ticket=launch_ticket,
                resolved_path=resolved,
                require_launch_ticket=True,
            )
            return resolved, connection.selection, connection

        selection, root, spawn_configuration = await self._select_available(resolved)
        connection = await self._get_connection(
            selection,
            root,
            spawn_configuration=spawn_configuration,
            require_launch_ticket=require_launch_ticket,
        )
        return resolved, selection, connection

    def _ensure_enabled(self) -> None:
        if not bool(getattr(self.config, "lsp_enabled", False)):
            raise LSPDisabledError(
                "LSP is disabled; set lsp_enabled = true in user configuration"
            )
        servers = getattr(self.config, "lsp_servers", None)
        if not isinstance(servers, Mapping) or not servers:
            raise LSPConfigurationError("no LSP servers are configured")

    def _workspace(self) -> Path:
        if self._resolved_workspace is not None:
            return self._resolved_workspace
        try:
            resolved = self.cwd.resolve(strict=True)
        except OSError as exc:
            raise LSPPathError(f"workspace is unavailable: {self.cwd}") from exc
        self._resolved_workspace = resolved
        return resolved

    def _resolve_file(self, path: str | os.PathLike[str]) -> Path:
        if not os.fspath(path).strip():
            raise LSPPathError("path must not be empty")
        candidate = Path(path)
        lexical = candidate if candidate.is_absolute() else self.cwd / candidate
        try:
            resolved = lexical.resolve(strict=True)
            workspace = self._workspace()
            resolved.relative_to(workspace)
        except (OSError, ValueError) as exc:
            raise LSPPathError(
                f"LSP path must be an existing file inside the workspace: {lexical}"
            ) from exc
        if not resolved.is_file():
            raise LSPPathError(f"LSP path is not a file: {resolved}")
        try:
            size = resolved.stat().st_size
        except OSError as exc:
            raise LSPPathError(f"cannot inspect LSP document: {resolved}") from exc
        if size > MAX_DOCUMENT_BYTES:
            raise LSPPathError(
                f"LSP document exceeds the {MAX_DOCUMENT_BYTES}-byte limit: {resolved}"
            )
        return resolved

    def _server_candidates(self, path: Path) -> list[_ServerSelection]:
        servers = getattr(self.config, "lsp_servers", {})
        filename = path.name.lower()
        candidates: list[tuple[int, int, str, _ServerSelection]] = []
        for name, spec in servers.items():
            if bool(getattr(spec, "disabled", False)):
                continue
            extensions = getattr(spec, "extensions", None)
            if not isinstance(name, str) or not isinstance(extensions, Mapping):
                continue
            matches: list[tuple[str, str]] = []
            for raw_extension, raw_language in extensions.items():
                if not isinstance(raw_extension, str) or not isinstance(
                    raw_language, str
                ):
                    continue
                normalized = _extension(raw_extension)
                if normalized and filename.endswith(normalized):
                    matches.append((normalized, raw_language))
            if not matches:
                continue
            extension, language = max(matches, key=lambda item: len(item[0]))
            raw_priority = getattr(spec, "priority", 0)
            priority = raw_priority if _integer(raw_priority) is not None else 0
            selection = _ServerSelection(name, spec, extension, language)
            candidates.append((-priority, -len(extension), name, selection))
        if not candidates:
            suffix = path.suffix.lower() or "(none)"
            raise LSPConfigurationError(
                f"no configured LSP server handles file extension {suffix}"
            )
        candidates.sort(key=lambda item: item[:3])
        return [candidate[3] for candidate in candidates]

    async def _select_available(
        self, path: Path
    ) -> tuple[_ServerSelection, Path, tuple[tuple[str, ...], dict[str, str]]]:
        """Choose the highest-priority safely resolvable local server."""

        attempts: list[tuple[str, str]] = []
        for selection in self._server_candidates(path):
            root = await asyncio.to_thread(self._find_root, path, selection.spec)
            cached = self._connections.get((selection.name, root))
            if cached is not None:
                # A running process remains usable if its executable later leaves
                # PATH. Keep its original absolute argv for approval previews.
                return selection, root, (cached.argv, {})
            try:
                spawn_configuration = await asyncio.to_thread(
                    self._spawn_configuration, selection, root
                )
            except LSPServerNotInstalledError:
                command = getattr(selection.spec, "command", "")
                attempts.append((selection.name, str(command)))
                continue
            if len(self._connections) >= MAX_CONNECTIONS:
                raise LSPManagerError(
                    f"LSP connection limit reached ({MAX_CONNECTIONS}); "
                    "close existing language servers before starting another root"
                )
            return selection, root, spawn_configuration

        rendered = ", ".join(
            f"{name[:80]!r} ({command[:160]!r})" for name, command in attempts[:16]
        )
        if len(attempts) > 16:
            rendered += f", ... ({len(attempts) - 16} more)"
        raise LSPServerNotInstalledError(
            "matching LSP servers are not installed; tried: " + (rendered or "(none)")
        )

    def _find_root(self, path: Path, spec: Any) -> Path:
        workspace = self._workspace()
        raw_markers = getattr(spec, "root_markers", ()) or ()
        markers: list[Path] = []
        for marker in raw_markers:
            marker_path = Path(marker) if isinstance(marker, str) else None
            if (
                marker_path is None
                or not marker
                or "\x00" in marker
                or marker_path.is_absolute()
                or ".." in marker_path.parts
            ):
                raise LSPConfigurationError(
                    "LSP root markers must be safe relative paths"
                )
            markers.append(marker_path)

        current = path.parent
        while True:
            if markers and any((current / marker).exists() for marker in markers):
                return current
            if current == workspace:
                return workspace
            try:
                current.relative_to(workspace)
            except ValueError:
                return workspace
            parent = current.parent
            if parent == current:
                return workspace
            current = parent

    async def _get_connection(
        self,
        selection: _ServerSelection | None,
        root: Path | None,
        *,
        spawn_configuration: tuple[tuple[str, ...], dict[str, str]] | None,
        launch_ticket: str | None = None,
        resolved_path: Path | None = None,
        require_launch_ticket: bool = False,
    ) -> _Connection:
        async with self._connection_lock:
            if launch_ticket is not None:
                snapshot = self._launch_tickets.pop(launch_ticket, None)
                if snapshot is None:
                    raise LSPManagerError(
                        "LSP launch approval is missing, expired, or already used; retry the request"
                    )
                if resolved_path != snapshot.path:
                    raise LSPManagerError(
                        "LSP document changed since launch approval; retry the request"
                    )
                approved_connection = self._connections.get(
                    (snapshot.server, snapshot.root)
                )
                if approved_connection is not None:
                    if (
                        approved_connection.launch_signature != snapshot.signature
                        or approved_connection.executable_fingerprint
                        != snapshot.executable_fingerprint
                    ):
                        raise LSPManagerError(
                            "a different LSP launch became active after approval; retry the request"
                        )
                    return approved_connection

                assert resolved_path is not None
                selection, root, spawn_configuration = await self._select_available(
                    resolved_path
                )
                argv, env = spawn_configuration
                signature = self._launch_signature(selection, root, argv, env)
                fingerprint = await asyncio.to_thread(
                    _executable_fingerprint, Path(argv[0])
                )
                if (
                    selection.name != snapshot.server
                    or root != snapshot.root
                    or signature != snapshot.signature
                    or fingerprint != snapshot.executable_fingerprint
                ):
                    raise LSPManagerError(
                        "approved LSP launch changed before execution; retry to review the new server, root, and command"
                    )

            if selection is None or root is None or spawn_configuration is None:
                raise LSPManagerError("invalid LSP launch state")
            key = (selection.name, root)
            existing = self._connections.get(key)
            if existing is not None:
                return existing
            if self._closed:
                raise LSPManagerError("LSP manager is closed")
            if len(self._connections) >= MAX_CONNECTIONS:
                raise LSPManagerError(
                    f"LSP connection limit reached ({MAX_CONNECTIONS}); "
                    "close existing language servers before starting another root"
                )
            if require_launch_ticket and launch_ticket is None:
                raise LSPManagerError(
                    "starting this language server requires a fresh approval; retry the tool call"
                )

            argv, env = spawn_configuration
            request_timeout = self._positive_timeout(
                selection.spec, "request_timeout", 30.0
            )
            shutdown_timeout = self._positive_timeout(
                selection.spec, "shutdown_timeout", 3.0
            )
            startup_timeout = self._positive_timeout(
                selection.spec, "startup_timeout", 30.0
            )
            settings = getattr(selection.spec, "settings", {}) or {}
            self._validate_launch_root(root)
            final_signature = self._launch_signature(selection, root, argv, env)
            final_fingerprint = await asyncio.to_thread(
                _executable_fingerprint, Path(argv[0])
            )
            if launch_ticket is not None:
                if final_fingerprint != snapshot.executable_fingerprint:
                    raise LSPManagerError(
                        "approved LSP executable changed immediately before launch; retry the request"
                    )
                if final_signature != snapshot.signature:
                    raise LSPManagerError(
                        "approved LSP launch configuration changed immediately before launch; retry the request"
                    )
            client = self._client_factory(
                argv,
                cwd=root,
                env=env,
                settings=settings,
                request_timeout=request_timeout,
                shutdown_timeout=shutdown_timeout,
            )
            initialize_options = (
                getattr(selection.spec, "initialization_options", {}) or {}
            )
            initialize_params = {
                "rootUri": root.as_uri(),
                "rootPath": str(root),
                "initializationOptions": copy.deepcopy(initialize_options),
                "capabilities": {
                    "general": {"positionEncodings": ["utf-16"]},
                    "textDocument": {"diagnostic": {}},
                    "workspace": {"symbol": {}},
                },
            }
            try:
                async with asyncio.timeout(startup_timeout):
                    initialize_result = await client.start(initialize_params)
                    capabilities = (
                        initialize_result.get("capabilities")
                        if isinstance(initialize_result, Mapping)
                        else None
                    )
                    position_encoding = (
                        capabilities.get("positionEncoding")
                        if isinstance(capabilities, Mapping)
                        else None
                    )
                    if position_encoding is not None and position_encoding != "utf-16":
                        raise LSPConfigurationError(
                            f"language server {selection.name!r} selected unsupported "
                            f"positionEncoding {position_encoding!r}; only 'utf-16' was offered"
                        )
                    if settings:
                        await client.notify(
                            "workspace/didChangeConfiguration",
                            {"settings": copy.deepcopy(settings)},
                        )
            except (asyncio.TimeoutError, TimeoutError) as exc:
                await self._close_failed_client(client)
                raise LSPManagerError(
                    f"language server {selection.name!r} startup timed out"
                ) from exc
            except BaseException as exc:
                await self._close_failed_client(client)
                if isinstance(exc, asyncio.CancelledError):
                    raise
                if isinstance(exc, LSPManagerError):
                    raise
                raise LSPManagerError(
                    f"failed to start language server {selection.name!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            connection = _Connection(
                selection,
                root,
                client,
                argv,
                final_signature,
                final_fingerprint,
            )
            self._connections[key] = connection
            return connection

    async def _close_failed_client(self, client: Any) -> None:
        with contextlib.suppress(BaseException):
            async with asyncio.timeout(10.0):
                await client.close(graceful=False)

    def _launch_signature(
        self,
        selection: _ServerSelection,
        root: Path,
        argv: tuple[str, ...],
        env: Mapping[str, str],
    ) -> str:
        """Hash every launch-affecting value without retaining environment secrets."""

        spec = selection.spec
        payload = {
            "server": selection.name,
            "extension": selection.extension,
            "language": selection.language,
            "root": str(root),
            "argv": list(argv),
            "env": dict(env),
            "initialization_options": copy.deepcopy(
                getattr(spec, "initialization_options", {}) or {}
            ),
            "settings": copy.deepcopy(getattr(spec, "settings", {}) or {}),
            "startup_timeout": self._positive_timeout(
                spec, "startup_timeout", 30.0
            ),
            "request_timeout": self._positive_timeout(
                spec, "request_timeout", 30.0
            ),
            "shutdown_timeout": self._positive_timeout(
                spec, "shutdown_timeout", 3.0
            ),
        }
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise LSPConfigurationError(
                f"language server {selection.name!r} launch settings must be strict JSON"
            ) from exc
        if len(encoded) > MAX_FRAME_BYTES:
            raise LSPConfigurationError(
                f"language server {selection.name!r} launch settings exceed the frame limit"
            )
        return hashlib.sha256(encoded).hexdigest()

    def _spawn_configuration(
        self, selection: _ServerSelection, root: Path
    ) -> tuple[tuple[str, ...], dict[str, str]]:
        self._validate_launch_root(root)
        spec = selection.spec
        command = getattr(spec, "command", None)
        if not isinstance(command, str) or not command.strip() or "\x00" in command:
            raise LSPConfigurationError(
                f"language server {selection.name!r} has no valid command"
            )
        patterns = getattr(self.config, "lsp_inherit_env", ()) or ()
        overrides = getattr(spec, "env", {}) or {}
        if not isinstance(overrides, Mapping) or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not key
            or "=" in key
            or "\x00" in key
            or "\x00" in value
            for key, value in overrides.items()
        ):
            raise LSPConfigurationError(
                f"language server {selection.name!r} has an invalid environment"
            )
        env = select_environment(list(patterns), dict(overrides))
        if os.name == "nt":
            # Windows environment keys are case-insensitive. Collapse aliases
            # after overrides so PATH cannot retain an earlier mixed-case value.
            env = {key.upper(): value for key, value in env.items()}
        executable = self._resolve_executable(command, root, env, selection.name)

        raw_args = getattr(spec, "args", ()) or ()
        if isinstance(raw_args, (str, bytes)) or any(
            not isinstance(argument, str) or "\x00" in argument for argument in raw_args
        ):
            raise LSPConfigurationError(
                f"language server {selection.name!r} has invalid args"
            )
        return (executable, *tuple(raw_args)), env

    def _validate_launch_root(self, root: Path) -> None:
        workspace = self._workspace()
        try:
            relative = root.relative_to(workspace)
            confined = confined_path(workspace, relative, label="LSP project root")
            if not confined.is_dir() or confined.resolve(strict=True) != root:
                raise LSPPathError("LSP project root changed before launch")
        except (OSError, UnsafePathError, ValueError) as exc:
            raise LSPPathError(
                "LSP project root must remain a real directory inside the workspace"
            ) from exc

    def _resolve_executable(
        self, command: str, root: Path, env: Mapping[str, str], server_name: str
    ) -> str:
        candidate_path = Path(command)
        if candidate_path.is_absolute():
            candidate: str | None = str(candidate_path)
        elif any(separator in command for separator in ("/", "\\")):
            candidate = str(root / candidate_path)
        else:
            candidate = _absolute_path_executable(command, env)
        if candidate is None:
            raise LSPServerNotInstalledError(
                f"language server executable is not installed or not on the allowed PATH: {command}"
            )
        try:
            resolved = Path(candidate).resolve(strict=True)
        except OSError as exc:
            raise LSPServerNotInstalledError(
                f"language server executable is unavailable: {command}"
            ) from exc
        if not resolved.is_file():
            raise LSPServerNotInstalledError(
                f"language server executable is not a file: {command}"
            )
        if sys.platform == "win32" and resolved.suffix.lower() not in {".exe", ".com"}:
            raise LSPConfigurationError(
                f"language server {server_name!r} does not resolve to a native "
                ".exe/.com executable; configure a directly executable binary instead"
            )
        if os.name != "nt" and not os.access(resolved, os.X_OK):
            raise LSPServerNotInstalledError(
                f"language server command is not executable: {command}"
            )
        return str(resolved)

    @staticmethod
    def _positive_timeout(spec: Any, name: str, default: float) -> float:
        raw = getattr(spec, name, default)
        if isinstance(raw, bool):
            raise LSPConfigurationError(f"{name} must be a positive number")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise LSPConfigurationError(f"{name} must be a positive number") from None
        if not math.isfinite(value) or value <= 0 or value > 600:
            raise LSPConfigurationError(f"{name} must be in the range (0, 600]")
        return value

    async def _sync_document(
        self, connection: _Connection, path: Path, selection: _ServerSelection
    ) -> tuple[str, _DocumentState]:
        text = await asyncio.to_thread(self._read_document, path)
        uri = path.as_uri()
        document = connection.documents.get(path)
        if document is None:
            if len(connection.documents) >= MAX_OPEN_DOCUMENTS_PER_CONNECTION:
                evicted_path = next(iter(connection.documents))
                await self._notify(
                    connection,
                    "textDocument/didClose",
                    {"textDocument": {"uri": evicted_path.as_uri()}},
                )
                connection.documents.pop(evicted_path, None)
            version = connection.next_document_version
            await self._notify(
                connection,
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": selection.language,
                        "version": version,
                        "text": text,
                    }
                },
            )
            connection.next_document_version += 1
            document = _DocumentState(text=text, version=version)
            connection.documents[path] = document
        elif document.text != text:
            next_version = connection.next_document_version
            await self._notify(
                connection,
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": next_version},
                    "contentChanges": [{"text": text}],
                },
            )
            document.text = text
            document.version = next_version
            connection.next_document_version += 1
            document.diagnostics = []
            document.diagnostic_result_id = None
            connection.documents.move_to_end(path)
        else:
            connection.documents.move_to_end(path)
        return text, document

    async def _request(self, connection: _Connection, method: str, params: Any) -> Any:
        timeout = self._positive_timeout(
            connection.selection.spec, "request_timeout", 30.0
        )
        return await connection.client.request(method, params, timeout=timeout)

    async def _notify(self, connection: _Connection, method: str, params: Any) -> None:
        timeout = self._positive_timeout(
            connection.selection.spec, "request_timeout", 30.0
        )
        try:
            await asyncio.to_thread(
                encoded_json_size,
                {"jsonrpc": "2.0", "method": method, "params": params},
                max_bytes=MAX_FRAME_BYTES,
            )
            async with asyncio.timeout(timeout):
                await connection.client.notify(method, params)
        except LSPProtocolError as exc:
            raise LSPManagerError(
                f"language server notification exceeds the frame limit: {method}"
            ) from exc
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise LSPManagerError(
                f"language server notification timed out: {method}"
            ) from exc

    @staticmethod
    def _text_position_params(
        path: Path, text: str, line: int, column: int
    ) -> dict[str, Any]:
        if _integer(line) is None or _integer(column) is None:
            raise LSPManagerError("line and column must be integers")
        return {
            "textDocument": {"uri": path.as_uri()},
            "position": _request_position(text, line, column),
        }

    async def _published_diagnostics(
        self,
        connection: _Connection,
        path: Path,
        text: str,
        document: _DocumentState,
    ) -> list[dict[str, Any]]:
        uri = path.as_uri()
        deadline = asyncio.get_running_loop().time() + min(
            PUBLISH_DIAGNOSTICS_TIMEOUT,
            self._positive_timeout(connection.selection.spec, "request_timeout", 30.0),
        )
        message: dict[str, Any] | None = None
        scanned = 0
        while message is None and scanned < MAX_NOTIFICATION_SCAN:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                notification = await connection.client.next_notification(
                    timeout=remaining
                )
            except (asyncio.TimeoutError, TimeoutError):
                break
            scanned += 1
            if notification.get("method") != "textDocument/publishDiagnostics":
                continue
            params = notification.get("params")
            if not isinstance(params, dict) or not isinstance(params.get("uri"), str):
                continue
            target_uri = params["uri"]
            if target_uri == uri:
                message = params
            # Unmatched notifications are deliberately discarded. Retaining raw
            # server params here would turn the client's bounded queue into an
            # unbounded secondary cache.

        if message is None:
            return copy.deepcopy(document.diagnostics)
        version = _integer(message.get("version"))
        if version is not None and version != document.version:
            return copy.deepcopy(document.diagnostics)
        normalized = self._normalize_diagnostics(
            message.get("diagnostics", []), path, text
        )
        document.diagnostics = normalized
        return copy.deepcopy(normalized)

    def _normalize_diagnostics(
        self, raw: Any, path: Path, text: str
    ) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        output: list[dict[str, Any]] = []
        lines = text.split("\n")
        for item in raw[: MAX_RESULTS * 4]:
            if not isinstance(item, Mapping):
                continue
            normalized_range = _range_for_lines(item.get("range"), lines)
            message = _bounded_text(item.get("message"), 4000)
            if normalized_range is None or message is None:
                continue
            diagnostic: dict[str, Any] = {
                "path": self._display_path(path),
                **normalized_range,
                "message": message,
            }
            severity = _integer(item.get("severity"))
            if severity is not None and 1 <= severity <= 4:
                diagnostic["severity"] = severity
            code = item.get("code")
            if isinstance(code, (str, int)) and not isinstance(code, bool):
                diagnostic["code"] = str(code)[:256]
            source = _bounded_text(item.get("source"), 256)
            if source:
                diagnostic["source"] = source
            output.append(diagnostic)
        return _dedupe_sorted(
            output,
            sort_key=lambda item: (
                item["path"],
                item["line"],
                item["column"],
                item.get("severity", 0),
                item["message"],
            ),
        )

    async def _normalize_locations(self, raw: Any) -> list[dict[str, Any]]:
        if raw is None:
            return []
        values = raw if isinstance(raw, list) else [raw]
        output: list[dict[str, Any]] = []
        line_cache: dict[Path, list[str] | None] = {}
        for value in values[: MAX_RESULTS * 4]:
            if not isinstance(value, Mapping):
                continue
            if "targetUri" in value:
                uri = value.get("targetUri")
                raw_range = value.get("targetSelectionRange", value.get("targetRange"))
            else:
                uri = value.get("uri")
                raw_range = value.get("range")
            if not isinstance(uri, str):
                continue
            path = await self._path_from_uri(uri)
            if path is None:
                continue
            if path not in line_cache:
                if len(line_cache) >= MAX_RESULT_FILES:
                    continue
                text = await self._read_result_text(path)
                line_cache[path] = text.split("\n") if text is not None else None
            lines = line_cache[path]
            if lines is None:
                continue
            normalized_range = _range_for_lines(raw_range, lines)
            if normalized_range is None:
                continue
            output.append({"path": self._display_path(path), **normalized_range})
        return _dedupe_sorted(
            output,
            sort_key=lambda item: (
                item["path"],
                item["line"],
                item["column"],
                item["end_line"],
                item["end_column"],
            ),
        )

    async def _normalize_symbols(
        self, raw: Any, *, default_path: Path, default_text: str
    ) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        output: list[dict[str, Any]] = []
        line_cache: dict[Path, list[str] | None] = {
            default_path: default_text.split("\n")
        }
        scanned = 0

        async def visit(item: Any, *, container: str | None, depth: int) -> None:
            nonlocal scanned
            if (
                scanned >= MAX_SCANNED_SYMBOLS
                or len(output) >= MAX_RESULTS * 4
                or depth > MAX_SYMBOL_DEPTH
                or not isinstance(item, Mapping)
            ):
                return
            scanned += 1
            name = _bounded_text(item.get("name"), 512)
            kind = _integer(item.get("kind"))
            if not name or kind is None:
                return
            symbol_path = default_path
            raw_range = item.get("selectionRange", item.get("range"))
            location = item.get("location")
            if isinstance(location, Mapping):
                uri = location.get("uri")
                if not isinstance(uri, str):
                    return
                maybe_path = await self._path_from_uri(uri)
                if maybe_path is None:
                    return
                symbol_path = maybe_path
                raw_range = location.get("range")
            if symbol_path not in line_cache:
                if len(line_cache) >= MAX_RESULT_FILES:
                    return
                symbol_text = await self._read_result_text(symbol_path)
                line_cache[symbol_path] = (
                    symbol_text.split("\n") if symbol_text is not None else None
                )
            symbol_lines = line_cache[symbol_path]
            if symbol_lines is None:
                return
            normalized_range = _range_for_lines(raw_range, symbol_lines)
            if normalized_range is None:
                return
            explicit_container = _bounded_text(item.get("containerName"), 512)
            symbol: dict[str, Any] = {
                "name": name,
                "kind": kind,
                "path": self._display_path(symbol_path),
                **normalized_range,
            }
            effective_container = explicit_container or container
            if effective_container:
                symbol["container"] = effective_container
            detail = _bounded_text(item.get("detail"), 1000)
            if detail:
                symbol["detail"] = detail
            output.append(symbol)
            children = item.get("children")
            if isinstance(children, list):
                for child in children[: MAX_RESULTS * 4]:
                    if scanned >= MAX_SCANNED_SYMBOLS:
                        break
                    await visit(child, container=name, depth=depth + 1)

        for value in raw[: MAX_RESULTS * 4]:
            if scanned >= MAX_SCANNED_SYMBOLS:
                break
            await visit(value, container=None, depth=0)
        return _dedupe_sorted(
            output,
            sort_key=lambda item: (
                item["name"].casefold(),
                item["path"],
                item["line"],
                item["column"],
                item["kind"],
                item.get("container", ""),
            ),
        )

    @staticmethod
    def _hover_text(contents: Any) -> str:
        parts: list[str] = []
        used_chars = 0
        visited = 0
        stack: list[tuple[Any, int]] = [(contents, 0)]

        def append_bounded(text: str) -> None:
            nonlocal used_chars
            if not text or used_chars >= MAX_TEXT_CHARS:
                return
            separator = 2 if parts else 0
            remaining = MAX_TEXT_CHARS - used_chars - separator
            if remaining <= 0:
                return
            bounded = text[:remaining]
            if bounded:
                parts.append(bounded)
                used_chars += separator + len(bounded)

        while stack and visited < MAX_HOVER_NODES and used_chars < MAX_TEXT_CHARS:
            value, depth = stack.pop()
            visited += 1
            if depth > MAX_HOVER_DEPTH:
                continue
            if isinstance(value, str):
                append_bounded(value)
                continue
            if isinstance(value, Mapping):
                text = value.get("value")
                if isinstance(text, str) and text:
                    language = value.get("language")
                    if isinstance(language, str) and language:
                        append_bounded(f"```{language[:64]}\n{text}\n```")
                    else:
                        append_bounded(text)
                continue
            if isinstance(value, list) and depth < MAX_HOVER_DEPTH:
                # Reverse-push preserves the peer's order without recursive calls.
                for child in reversed(value[:MAX_HOVER_NODES]):
                    stack.append((child, depth + 1))

        return "\n\n".join(parts)

    async def _path_from_uri(self, uri: str) -> Path | None:
        return await asyncio.to_thread(self._path_from_uri_sync, uri)

    def _path_from_uri_sync(self, uri: str) -> Path | None:
        try:
            parsed = urlsplit(uri)
            if parsed.scheme.lower() != "file" or parsed.query or parsed.fragment:
                return None
            if parsed.netloc and parsed.netloc.lower() != "localhost":
                if os.name != "nt":
                    return None
                raw_path = f"//{parsed.netloc}{url2pathname(parsed.path)}"
            else:
                raw_path = url2pathname(parsed.path)
            candidate = Path(raw_path).resolve(strict=False)
            candidate.relative_to(self._workspace())
            return candidate
        except (OSError, ValueError):
            return None

    async def _read_result_text(self, path: Path) -> str | None:
        try:
            return await asyncio.to_thread(self._read_document, path)
        except LSPPathError:
            return None

    def _read_document(self, path: Path) -> str:
        return _read_utf8_document(path, self._workspace())

    def _display_path(self, path: Path) -> str:
        try:
            return path.relative_to(self._workspace()).as_posix()
        except ValueError as exc:  # defense in depth: never expose outside locations
            raise LSPPathError(
                "language server returned a path outside the workspace"
            ) from exc


__all__ = [
    "LSPConfigurationError",
    "LSPDisabledError",
    "LSPManager",
    "LSPManagerError",
    "LSPPathError",
    "LSPServerNotInstalledError",
]
