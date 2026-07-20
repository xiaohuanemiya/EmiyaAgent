from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

import emiya_agent.lsp.manager as manager_module
from emiya_agent.lsp.client import LSPResponseError
from emiya_agent.lsp.manager import (
    LSPConfigurationError,
    LSPDisabledError,
    LSPManager,
    LSPPathError,
    LSPServerNotInstalledError,
)
from emiya_agent.tools.base import ToolContext
from emiya_agent.tools.lsp import LSPTool


@dataclass
class ServerSpec:
    command: str = sys.executable
    args: list[str] = field(default_factory=lambda: ["--fake-lsp"])
    extensions: dict[str, str] = field(default_factory=lambda: {".py": "python"})
    root_markers: list[str] = field(default_factory=lambda: ["pyproject.toml"])
    env: dict[str, str] = field(default_factory=dict)
    initialization_options: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    startup_timeout: float = 1.0
    request_timeout: float = 1.0
    shutdown_timeout: float = 1.0
    priority: int = 0
    disabled: bool = False


@dataclass
class ManagerConfig:
    cwd: Path
    lsp_enabled: bool = True
    tool_approval: str = "auto"
    lsp_inherit_env: list[str] = field(default_factory=list)
    lsp_servers: dict[str, ServerSpec] = field(default_factory=dict)


class FakeClient:
    def __init__(
        self,
        factory: FakeFactory,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: dict[str, str],
        settings: dict[str, Any],
        request_timeout: float,
        shutdown_timeout: float,
    ) -> None:
        self.factory = factory
        self.argv = tuple(argv)
        self.cwd = Path(cwd)
        self.env = dict(env)
        self.settings = settings
        self.request_timeout = request_timeout
        self.shutdown_timeout = shutdown_timeout
        self.initialize_params: dict[str, Any] | None = None
        self.notifications: list[tuple[str, Any]] = []
        self.requests: list[tuple[str, Any, float | None]] = []
        self.incoming_notifications: list[dict[str, Any]] = []
        self.closed = False
        self.close_graceful: bool | None = None

    async def start(self, initialize_params: dict[str, Any]) -> dict[str, Any]:
        self.initialize_params = initialize_params
        return self.factory.start_result

    async def notify(self, method: str, params: Any = None) -> None:
        self.notifications.append((method, params))

    async def request(
        self, method: str, params: Any, *, timeout: float | None = None
    ) -> Any:
        self.requests.append((method, params, timeout))
        return self.factory.handler(method, params, self)

    async def next_notification(
        self, *, timeout: float | None = None
    ) -> dict[str, Any]:
        if self.incoming_notifications:
            return self.incoming_notifications.pop(0)
        await asyncio.sleep(timeout or 0)
        raise TimeoutError

    async def close(self, *, graceful: bool = True) -> None:
        self.closed = True
        self.close_graceful = graceful


class FakeFactory:
    def __init__(
        self,
        handler: Callable[[str, Any, FakeClient], Any] | None = None,
        *,
        start_result: dict[str, Any] | None = None,
    ) -> None:
        self.instances: list[FakeClient] = []
        self.handler = handler or (lambda method, params, client: None)
        self.start_result = start_result or {"capabilities": {}}

    def __call__(self, argv: tuple[str, ...], **kwargs: Any) -> FakeClient:
        client = FakeClient(self, tuple(argv), **kwargs)
        self.instances.append(client)
        return client


class FakeUI:
    async def ask(self, question: str, options: list[str] | None = None) -> str:
        return "ok"

    def notify(self, text: str) -> None:
        return None

    async def confirm_tool(self, name: str, args_brief: str) -> bool:
        return True


def _config(tmp_path: Path, **servers: ServerSpec) -> ManagerConfig:
    return ManagerConfig(cwd=tmp_path, lsp_servers=servers or {"python": ServerSpec()})


@pytest.mark.asyncio
async def test_lazy_selection_root_environment_and_document_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "pkg"
    source_dir = package / "src"
    source_dir.mkdir(parents=True)
    (package / "pyproject.toml").write_text("", encoding="utf-8")
    source = source_dir / "main.py"
    source.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setenv("EMIYA_LSP_ALLOWED", "inherited")
    monkeypatch.setenv("EMIYA_LSP_SECRET", "excluded")

    low = ServerSpec(priority=1)
    high = ServerSpec(
        priority=10,
        env={"EXPLICIT": "configured"},
        initialization_options={"feature": True},
        settings={"python": {"analysis": "strict"}},
    )
    config = _config(tmp_path, low=low, high=high)
    config.lsp_inherit_env = ["EMIYA_LSP_ALLOWED"]
    factory = FakeFactory(
        lambda method, params, client: (
            {"contents": "hover"} if method == "textDocument/hover" else None
        )
    )
    manager = LSPManager(config, client_factory=factory)  # type: ignore[arg-type]

    assert manager.connected_count == 0
    assert await manager.hover(source, 1, 1) == {"contents": "hover"}
    source.write_text("value = 2\n", encoding="utf-8")
    assert await manager.hover(source, 1, 1) == {"contents": "hover"}

    assert len(factory.instances) == 1
    client = factory.instances[0]
    assert Path(client.argv[0]) == Path(sys.executable).resolve()
    assert client.cwd == package
    assert client.env == {
        "EMIYA_LSP_ALLOWED": "inherited",
        "EXPLICIT": "configured",
    }
    assert client.initialize_params is not None
    assert client.initialize_params["initializationOptions"] == {"feature": True}
    assert client.initialize_params["capabilities"]["general"] == {
        "positionEncodings": ["utf-16"]
    }
    assert [method for method, _ in client.notifications] == [
        "workspace/didChangeConfiguration",
        "textDocument/didOpen",
        "textDocument/didChange",
    ]
    assert client.notifications[1][1]["textDocument"]["version"] == 1
    assert client.notifications[2][1]["textDocument"]["version"] == 2

    await manager.close_all()
    assert manager.connected_count == 0
    assert client.closed is True


@pytest.mark.asyncio
async def test_definition_converts_utf16_deduplicates_and_drops_external_uri(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    target = tmp_path / "target file.py"
    external = tmp_path.parent / "external.py"
    source.write_text("a😀b\n", encoding="utf-8")
    target.write_text("x😀y\n", encoding="utf-8")
    external.write_text("outside\n", encoding="utf-8")

    def handler(method: str, params: Any, client: FakeClient) -> Any:
        assert method == "textDocument/definition"
        assert params["position"] == {"line": 0, "character": 3}
        location = {
            "targetUri": target.as_uri(),
            "targetSelectionRange": {
                "start": {"line": 0, "character": 3},
                "end": {"line": 0, "character": 4},
            },
        }
        return [
            location,
            dict(location),
            {
                "uri": external.as_uri(),
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 1},
                },
            },
        ]

    factory = FakeFactory(handler)
    manager = LSPManager(
        _config(tmp_path),
        client_factory=factory,  # type: ignore[arg-type]
    )
    result = await manager.definition(source, 1, 3)

    assert result == [
        {
            "path": "target file.py",
            "line": 1,
            "column": 3,
            "end_line": 1,
            "end_column": 4,
        }
    ]


@pytest.mark.asyncio
async def test_selection_skips_disabled_and_missing_servers_by_priority(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("x\n", encoding="utf-8")
    config = _config(
        tmp_path,
        disabled=ServerSpec(priority=30, disabled=True),
        missing=ServerSpec(command="missing-emiya-lsp", priority=20),
        available=ServerSpec(priority=10),
    )
    factory = FakeFactory(lambda method, params, client: {"contents": "selected"})
    manager = LSPManager(config, client_factory=factory)  # type: ignore[arg-type]

    assert await manager.hover(source, 1, 1) == {"contents": "selected"}
    assert len(factory.instances) == 1
    assert Path(factory.instances[0].argv[0]) == Path(sys.executable).resolve()


def test_bare_command_search_ignores_cwd_empty_and_relative_path_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = "emiya-path-test"
    suffix = ".exe" if sys.platform == "win32" else ""
    hijack = tmp_path / f"{command}{suffix}"
    hijack.write_bytes(b"workspace hijack")
    safe_dir = tmp_path / "installed"
    safe_dir.mkdir()
    safe = safe_dir / f"{command}{suffix}"
    safe.write_bytes(b"installed executable")
    if sys.platform != "win32":
        hijack.chmod(0o755)
        safe.chmod(0o755)
    monkeypatch.chdir(tmp_path)

    relative_only = os.pathsep.join(("", ".", "installed"))
    assert (
        manager_module._absolute_path_executable(
            command,
            {
                "Path" if sys.platform == "win32" else "PATH": relative_only,
                "PATHEXT": ".EXE;.CMD",
            },
        )
        is None
    )

    safe_path = os.pathsep.join(("", ".", str(safe_dir.resolve())))
    resolved = manager_module._absolute_path_executable(
        command,
        {
            "Path" if sys.platform == "win32" else "PATH": safe_path,
            "PathExt" if sys.platform == "win32" else "PATHEXT": ".CMD;.EXE",
        },
    )
    assert resolved == str(safe.resolve())


@pytest.mark.asyncio
async def test_document_size_frame_and_result_file_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("x\n", encoding="utf-8")
    oversized = tmp_path / "oversized.py"
    oversized.write_bytes(b"123456789")
    target_a = tmp_path / "a.py"
    target_b = tmp_path / "b.py"
    target_a.write_text("a\n", encoding="utf-8")
    target_b.write_text("b\n", encoding="utf-8")
    location_range = {
        "start": {"line": 0, "character": 0},
        "end": {"line": 0, "character": 1},
    }

    def handler(method: str, params: Any, client: FakeClient) -> Any:
        return [
            {"uri": target_a.as_uri(), "range": location_range},
            {"uri": target_b.as_uri(), "range": location_range},
            {"uri": oversized.as_uri(), "range": location_range},
        ]

    monkeypatch.setattr(manager_module, "MAX_DOCUMENT_BYTES", 8)
    monkeypatch.setattr(manager_module, "MAX_RESULT_FILES", 2)
    factory = FakeFactory(handler)
    manager = LSPManager(
        _config(tmp_path),
        client_factory=factory,  # type: ignore[arg-type]
    )

    with pytest.raises(LSPPathError, match="byte limit"):
        await manager.connection_preview(oversized)
    assert factory.instances == []
    assert [item["path"] for item in await manager.definition(source, 1, 1)] == [
        "a.py",
        "b.py",
    ]

    # Even a file below the raw byte cap can expand beyond a JSON frame because
    # control characters must be escaped. Preflight rejects it before notify().
    escaped = tmp_path / "escaped.py"
    escaped.write_bytes(b'"' * 8)
    monkeypatch.setattr(manager_module, "MAX_FRAME_BYTES", 100)
    with pytest.raises(manager_module.LSPManagerError, match="frame limit"):
        await manager.hover(escaped, 1, 1)


@pytest.mark.asyncio
async def test_document_lru_sends_did_close_and_connection_count_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(manager_module, "MAX_OPEN_DOCUMENTS_PER_CONNECTION", 2)
    monkeypatch.setattr(manager_module, "MAX_CONNECTIONS", 1)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "pyproject.toml").write_text("", encoding="utf-8")
    (second_root / "pyproject.toml").write_text("", encoding="utf-8")
    first_files = [first_root / f"file{index}.py" for index in range(3)]
    for path in first_files:
        path.write_text("x\n", encoding="utf-8")
    second_file = second_root / "other.py"
    second_file.write_text("x\n", encoding="utf-8")
    factory = FakeFactory(lambda method, params, client: {"contents": "ok"})
    manager = LSPManager(
        _config(tmp_path),
        client_factory=factory,  # type: ignore[arg-type]
    )

    for path in first_files:
        assert await manager.hover(path, 1, 1) == {"contents": "ok"}
    assert len(factory.instances) == 1
    client = factory.instances[0]
    methods = [method for method, _ in client.notifications]
    assert methods == [
        "textDocument/didOpen",
        "textDocument/didOpen",
        "textDocument/didClose",
        "textDocument/didOpen",
    ]
    assert client.notifications[2][1] == {
        "textDocument": {"uri": first_files[0].as_uri()}
    }
    assert await manager.hover(first_files[0], 1, 1) == {"contents": "ok"}
    open_versions = [
        params["textDocument"]["version"]
        for method, params in client.notifications
        if method == "textDocument/didOpen"
    ]
    assert open_versions == [1, 2, 3, 4]

    with pytest.raises(manager_module.LSPManagerError, match="connection limit"):
        await manager.hover(second_file, 1, 1)
    assert len(factory.instances) == 1


@pytest.mark.asyncio
async def test_diagnostics_falls_back_to_bounded_publish_notification(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    other = tmp_path / "other.py"
    source.write_text("a😀b\n", encoding="utf-8")
    other.write_text("other\n", encoding="utf-8")

    def handler(method: str, params: Any, client: FakeClient) -> Any:
        if method == "textDocument/diagnostic":
            raise LSPResponseError(-32601, "Method not found")
        return None

    factory = FakeFactory(handler)
    manager = LSPManager(
        _config(tmp_path),
        client_factory=factory,  # type: ignore[arg-type]
    )

    task = asyncio.create_task(manager.diagnostics(source))
    while not factory.instances:
        await asyncio.sleep(0)
    client = factory.instances[0]
    diagnostic = {
        "range": {
            "start": {"line": 0, "character": 3},
            "end": {"line": 0, "character": 4},
        },
        "severity": 2,
        "code": "E1",
        "source": "fake",
        "message": "problem",
    }
    client.incoming_notifications.extend(
        [
            {
                "method": "textDocument/publishDiagnostics",
                "params": {"uri": other.as_uri(), "diagnostics": []},
            },
            {
                "method": "textDocument/publishDiagnostics",
                "params": {
                    "uri": source.as_uri(),
                    "version": 1,
                    "diagnostics": [diagnostic, dict(diagnostic)],
                },
            },
        ]
    )

    assert await task == [
        {
            "path": "source.py",
            "line": 1,
            "column": 3,
            "end_line": 1,
            "end_column": 4,
            "severity": 2,
            "code": "E1",
            "source": "fake",
            "message": "problem",
        }
    ]
    assert client.requests[0][0] == "textDocument/diagnostic"


@pytest.mark.asyncio
async def test_document_change_clears_cached_diagnostics_and_rejects_old_version(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("x\n", encoding="utf-8")
    diagnostic = {
        "range": {
            "start": {"line": 0, "character": 0},
            "end": {"line": 0, "character": 1},
        },
        "message": "old diagnostic",
    }
    pulls = 0

    def handler(method: str, params: Any, client: FakeClient) -> Any:
        nonlocal pulls
        if method != "textDocument/diagnostic":
            return None
        pulls += 1
        if pulls == 1:
            return {"kind": "full", "resultId": "old", "items": [diagnostic]}
        raise LSPResponseError(-32601, "Method not found")

    factory = FakeFactory(handler)
    manager = LSPManager(
        _config(tmp_path),
        client_factory=factory,  # type: ignore[arg-type]
    )
    assert len(await manager.diagnostics(source)) == 1
    source.write_text("changed\n", encoding="utf-8")
    factory.instances[0].incoming_notifications.append(
        {
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": source.as_uri(),
                "version": 1,
                "diagnostics": [diagnostic],
            },
        }
    )

    assert await manager.diagnostics(source) == []


@pytest.mark.asyncio
async def test_references_hover_and_symbols_are_normalized_and_sorted(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    target = tmp_path / "target.py"
    source.write_text("root\n  child\n", encoding="utf-8")
    target.write_text("target\n", encoding="utf-8")

    def handler(method: str, params: Any, client: FakeClient) -> Any:
        zero_range = {
            "start": {"line": 0, "character": 0},
            "end": {"line": 0, "character": 1},
        }
        if method == "textDocument/references":
            assert params["context"] == {"includeDeclaration": False}
            return [{"uri": target.as_uri(), "range": zero_range}]
        if method == "textDocument/hover":
            return {
                "contents": [
                    {"language": "python", "value": "def root(): ..."},
                    "documentation",
                ],
                "range": zero_range,
            }
        if method == "textDocument/documentSymbol":
            return [
                {
                    "name": "Root",
                    "kind": 12,
                    "range": zero_range,
                    "selectionRange": zero_range,
                    "children": [
                        {
                            "name": "Child",
                            "kind": 13,
                            "range": {
                                "start": {"line": 1, "character": 2},
                                "end": {"line": 1, "character": 7},
                            },
                            "selectionRange": {
                                "start": {"line": 1, "character": 2},
                                "end": {"line": 1, "character": 7},
                            },
                        }
                    ],
                }
            ]
        if method == "workspace/symbol":
            assert params == {"query": "tar"}
            return [
                {
                    "name": "Target",
                    "kind": 12,
                    "containerName": "module",
                    "location": {"uri": target.as_uri(), "range": zero_range},
                }
            ]
        raise AssertionError(method)

    factory = FakeFactory(handler)
    manager = LSPManager(
        _config(tmp_path),
        client_factory=factory,  # type: ignore[arg-type]
    )

    assert await manager.references(source, 1, 1, include_declaration=False) == [
        {
            "path": "target.py",
            "line": 1,
            "column": 1,
            "end_line": 1,
            "end_column": 2,
        }
    ]
    hover = await manager.hover(source, 1, 1)
    assert hover == {
        "contents": "```python\ndef root(): ...\n```\n\ndocumentation",
        "range": {
            "line": 1,
            "column": 1,
            "end_line": 1,
            "end_column": 2,
        },
    }
    symbols = await manager.document_symbols(source)
    assert [symbol["name"] for symbol in symbols] == ["Child", "Root"]
    assert symbols[0]["container"] == "Root"
    assert await manager.workspace_symbols(source, "tar") == [
        {
            "name": "Target",
            "kind": 12,
            "path": "target.py",
            "line": 1,
            "column": 1,
            "end_line": 1,
            "end_column": 2,
            "container": "module",
        }
    ]


@pytest.mark.asyncio
async def test_symbol_scan_has_independent_bound_for_invalid_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("root\n", encoding="utf-8")
    symbol_range = {
        "start": {"line": 0, "character": 0},
        "end": {"line": 0, "character": 1},
    }

    class MustNotVisit(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise AssertionError("symbol scan exceeded its independent node budget")

    root_symbol = {
        "name": "Root",
        "kind": 12,
        "range": symbol_range,
        "selectionRange": symbol_range,
        "children": [{}, {}, MustNotVisit()],
    }
    monkeypatch.setattr(manager_module, "MAX_SCANNED_SYMBOLS", 3)
    factory = FakeFactory(
        lambda method, params, client: (
            [root_symbol] if method == "textDocument/documentSymbol" else None
        )
    )
    manager = LSPManager(
        _config(tmp_path),
        client_factory=factory,  # type: ignore[arg-type]
    )

    symbols = await manager.document_symbols(source)
    assert [symbol["name"] for symbol in symbols] == ["Root"]


@pytest.mark.asyncio
async def test_clear_disabled_missing_unconfigured_and_path_errors(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("x\n", encoding="utf-8")

    disabled = _config(tmp_path)
    disabled.lsp_enabled = False
    with pytest.raises(LSPDisabledError, match="disabled"):
        await LSPManager(disabled).diagnostics(source)  # type: ignore[arg-type]

    unconfigured = ManagerConfig(cwd=tmp_path)
    with pytest.raises(LSPConfigurationError, match="no LSP servers"):
        await LSPManager(unconfigured).diagnostics(source)  # type: ignore[arg-type]

    wrong_extension = _config(tmp_path, python=ServerSpec(extensions={".ts": "ts"}))
    with pytest.raises(LSPConfigurationError, match="extension"):
        await LSPManager(wrong_extension).diagnostics(source)  # type: ignore[arg-type]

    missing = _config(
        tmp_path, python=ServerSpec(command="certainly-not-installed-emiya")
    )
    with pytest.raises(LSPServerNotInstalledError, match="not installed"):
        await LSPManager(missing).diagnostics(source)  # type: ignore[arg-type]

    external = tmp_path.parent / "external-lsp.py"
    external.write_text("x\n", encoding="utf-8")
    with pytest.raises(LSPPathError, match="inside the workspace"):
        await LSPManager(_config(tmp_path)).diagnostics(external)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_lsp_tool_is_read_only_but_conservatively_requires_approval(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("x\n", encoding="utf-8")
    config = _config(tmp_path)
    config.lsp_enabled = False
    tool = LSPTool()
    ctx = ToolContext(
        cwd=tmp_path,
        config=config,  # type: ignore[arg-type]
        ui=FakeUI(),
    )

    assert tool.read_only is True
    assert tool.approval_required({"action": "diagnostics"}) is True
    result = await tool.run({"action": "diagnostics", "path": str(source)}, ctx)
    assert result.is_error is True
    assert "LSP is disabled" in result.content
    assert (
        await tool.approval_required_for(
            {"action": "diagnostics", "path": str(source)}, ctx
        )
        is False
    )

    missing_config = _config(
        tmp_path, python=ServerSpec(command="missing-approval-lsp")
    )
    missing_ctx = ToolContext(
        cwd=tmp_path,
        config=missing_config,  # type: ignore[arg-type]
        ui=FakeUI(),
    )
    assert (
        await tool.approval_required_for(
            {"action": "diagnostics", "path": str(source)}, missing_ctx
        )
        is False
    )


@pytest.mark.asyncio
async def test_lsp_tool_dynamic_approval_previews_launch_then_skips_warm_connection(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("x\n", encoding="utf-8")
    config = _config(tmp_path)
    factory = FakeFactory(lambda method, params, client: {"kind": "full", "items": []})
    manager = LSPManager(config, client_factory=factory)  # type: ignore[arg-type]
    ctx = ToolContext(
        cwd=tmp_path,
        config=config,  # type: ignore[arg-type]
        ui=FakeUI(),
        state={"lsp_manager": manager},
    )
    tool = LSPTool()
    args = {"action": "diagnostics", "path": str(source)}

    assert await tool.approval_required_for(args, ctx) is True
    assert factory.instances == []
    preview = await tool.approval_preview(args, ctx)
    assert "will start" in preview
    assert repr(str(Path(sys.executable).resolve())) in preview
    assert repr(str(tmp_path.resolve())) in preview

    result = await tool.run(args, ctx)
    assert result.is_error is False
    assert result.content == "[]"
    assert len(factory.instances) == 1
    assert await tool.approval_required_for(args, ctx) is False


@pytest.mark.asyncio
async def test_launch_ticket_rejects_root_and_argv_changes_and_is_single_use(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    source = package / "source.py"
    source.write_text("x\n", encoding="utf-8")
    spec = ServerSpec()
    manager = LSPManager(
        _config(tmp_path, python=spec),
        client_factory=FakeFactory(),  # type: ignore[arg-type]
    )

    root_preview = await manager.connection_preview(
        source, issue_launch_ticket=True
    )
    root_ticket = root_preview["launch_ticket"]
    assert isinstance(root_ticket, str)
    assert Path(root_preview["root"]) == tmp_path.resolve()
    (package / "pyproject.toml").write_text("", encoding="utf-8")
    with pytest.raises(manager_module.LSPManagerError, match="changed"):
        await manager.hover(source, 1, 1, launch_ticket=root_ticket)
    with pytest.raises(manager_module.LSPManagerError, match="already used"):
        await manager.hover(source, 1, 1, launch_ticket=root_ticket)

    argv_preview = await manager.connection_preview(
        source, issue_launch_ticket=True
    )
    argv_ticket = argv_preview["launch_ticket"]
    assert isinstance(argv_ticket, str)
    spec.args.append("--changed-after-approval")
    with pytest.raises(manager_module.LSPManagerError, match="changed"):
        await manager.hover(source, 1, 1, launch_ticket=argv_ticket)
    assert manager.connected_count == 0


@pytest.mark.asyncio
async def test_launch_ticket_rejects_executable_replacement(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("x\n", encoding="utf-8")
    executable = tmp_path / ("server.exe" if sys.platform == "win32" else "server")
    executable.write_bytes(b"approved executable\n")
    if sys.platform != "win32":
        executable.chmod(0o755)
    spec = ServerSpec(command=str(executable), args=[])
    factory = FakeFactory()
    manager = LSPManager(
        _config(tmp_path, python=spec),
        client_factory=factory,  # type: ignore[arg-type]
    )

    preview = await manager.connection_preview(source, issue_launch_ticket=True)
    ticket = preview["launch_ticket"]
    assert isinstance(ticket, str)
    executable.write_bytes(b"replaced executable\n")
    if sys.platform != "win32":
        executable.chmod(0o755)

    with pytest.raises(manager_module.LSPManagerError, match="changed"):
        await manager.hover(source, 1, 1, launch_ticket=ticket)
    assert factory.instances == []


@pytest.mark.asyncio
async def test_launch_ticket_matches_only_the_same_concurrent_start(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("x\n", encoding="utf-8")
    factory = FakeFactory(lambda method, params, client: {"contents": "ok"})
    manager = LSPManager(
        _config(tmp_path),
        client_factory=factory,  # type: ignore[arg-type]
    )
    preview = await manager.connection_preview(source, issue_launch_ticket=True)
    ticket = preview["launch_ticket"]
    assert isinstance(ticket, str)

    assert await manager.hover(source, 1, 1) == {"contents": "ok"}
    assert await manager.hover(source, 1, 1, launch_ticket=ticket) == {
        "contents": "ok"
    }
    assert len(factory.instances) == 1

    other_spec = ServerSpec()
    other_factory = FakeFactory(lambda method, params, client: {"contents": "ok"})
    other = LSPManager(
        _config(tmp_path, python=other_spec),
        client_factory=other_factory,  # type: ignore[arg-type]
    )
    other_preview = await other.connection_preview(
        source, issue_launch_ticket=True
    )
    other_ticket = other_preview["launch_ticket"]
    assert isinstance(other_ticket, str)
    other_spec.args.append("--different-concurrent-launch")
    assert await other.hover(source, 1, 1) == {"contents": "ok"}
    with pytest.raises(manager_module.LSPManagerError, match="different LSP launch"):
        await other.hover(source, 1, 1, launch_ticket=other_ticket)
    assert len(other_factory.instances) == 1


@pytest.mark.asyncio
async def test_non_utf16_position_encoding_is_rejected_and_client_closed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("x\n", encoding="utf-8")
    factory = FakeFactory(
        start_result={"capabilities": {"positionEncoding": "utf-8"}}
    )
    manager = LSPManager(
        _config(tmp_path),
        client_factory=factory,  # type: ignore[arg-type]
    )

    with pytest.raises(LSPConfigurationError, match="positionEncoding"):
        await manager.hover(source, 1, 1)
    assert len(factory.instances) == 1
    assert factory.instances[0].closed is True
    assert factory.instances[0].close_graceful is False
    assert manager.connected_count == 0


@pytest.mark.asyncio
async def test_hover_has_independent_depth_node_and_character_budgets(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("x\n", encoding="utf-8")

    class MustNotVisit(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise AssertionError("hover traversal exceeded its node budget")

    too_deep: Any = "must not be reached"
    for _ in range(manager_module.MAX_HOVER_DEPTH + 10):
        too_deep = [too_deep]
    node_contents: list[Any] = [
        {} for _ in range(manager_module.MAX_HOVER_NODES - 1)
    ]
    node_contents.append(MustNotVisit())
    responses = iter(
        [
            {"contents": too_deep},
            {"contents": node_contents},
            {"contents": "x" * (manager_module.MAX_TEXT_CHARS * 2)},
        ]
    )
    factory = FakeFactory(
        lambda method, params, client: next(responses)
    )
    manager = LSPManager(
        _config(tmp_path),
        client_factory=factory,  # type: ignore[arg-type]
    )

    assert await manager.hover(source, 1, 1) is None
    assert await manager.hover(source, 1, 1) is None
    hover = await manager.hover(source, 1, 1)
    assert hover is not None
    assert len(hover["contents"]) <= manager_module.MAX_TEXT_CHARS


@pytest.mark.asyncio
async def test_lsp_tool_requires_ticket_for_cold_start_in_ask_mode(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("x\n", encoding="utf-8")
    config = _config(tmp_path)
    config.tool_approval = "ask"
    factory = FakeFactory(lambda method, params, client: {"kind": "full", "items": []})
    manager = LSPManager(config, client_factory=factory)  # type: ignore[arg-type]
    ctx = ToolContext(
        cwd=tmp_path,
        config=config,  # type: ignore[arg-type]
        ui=FakeUI(),
        state={"lsp_manager": manager},
    )
    tool = LSPTool()
    args = {"action": "diagnostics", "path": str(source)}

    bypass = await tool.run(args, ctx)
    assert bypass.is_error is True
    assert "fresh approval" in bypass.content
    assert factory.instances == []

    assert await tool.approval_required_for(args, ctx) is True
    preview = await tool.approval_preview(args, ctx)
    assert "will start" in preview
    approved = await tool.run(args, ctx)
    assert approved.is_error is False
    assert len(factory.instances) == 1


@pytest.mark.asyncio
async def test_hook_elevated_review_fails_closed_when_preview_breaks_in_auto_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("x\n", encoding="utf-8")
    config = _config(tmp_path)
    config.tool_approval = "auto"
    factory = FakeFactory(lambda method, params, client: {"kind": "full", "items": []})
    manager = LSPManager(config, client_factory=factory)  # type: ignore[arg-type]
    ctx = ToolContext(
        cwd=tmp_path,
        config=config,  # type: ignore[arg-type]
        ui=FakeUI(),
        state={"lsp_manager": manager},
    )
    tool = LSPTool()
    args = {"action": "diagnostics", "path": str(source)}

    async def broken_preview(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("preview channel broke")

    monkeypatch.setattr(manager, "connection_preview", broken_preview)
    assert "details unavailable" in await tool.approval_preview(args, ctx)

    result = await tool.run(args, ctx)
    assert result.is_error is True
    assert "fresh approval" in result.content
    assert factory.instances == []


@pytest.mark.asyncio
async def test_cancelled_launch_review_discards_ticket_and_review_flag(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("x\n", encoding="utf-8")
    config = _config(tmp_path)
    config.tool_approval = "auto"
    factory = FakeFactory(lambda method, params, client: {"kind": "full", "items": []})
    manager = LSPManager(config, client_factory=factory)  # type: ignore[arg-type]
    ctx = ToolContext(
        cwd=tmp_path,
        config=config,  # type: ignore[arg-type]
        ui=FakeUI(),
        state={"lsp_manager": manager},
    )
    tool = LSPTool()
    args = {"action": "diagnostics", "path": str(source)}

    assert "will start" in await tool.approval_preview(args, ctx)
    await tool.approval_cancelled(args, ctx)
    assert set(ctx.state) == {"lsp_manager"}

    # A later auto-mode invocation is a fresh policy decision, not a reuse of
    # the denied review or its launch snapshot.
    result = await tool.run(args, ctx)
    assert result.is_error is False
    assert len(factory.instances) == 1
