"""MCP 桥接：命名空间、schema 透传、错误映射、Windows 垫片、内容解析。"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from emiya_agent.config import MCPServerConfig
from emiya_agent.mcp_client.adapter import MCPTool, build_mcp_tools
from emiya_agent.mcp_client.manager import (
    MCPServerConn,
    _resolve_stdio_executable,
    build_stdio_params,
    content_to_text,
)
from emiya_agent.safe_paths import UnsafePathError
from emiya_agent.tools.base import ToolContext

from tests.conftest import FakeUI


class FakeConn:
    def __init__(self, name="srv", connected=True, tools=(), reply=("ok", False)):
        self.name = name
        self.connected = connected
        self.tools = list(tools)
        self.reply = reply
        self.calls: list[tuple[str, dict]] = []

    async def call(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return self.reply


@pytest.fixture
def ctx(tmp_path, config):
    return ToolContext(cwd=tmp_path, config=config, ui=FakeUI())


# ------------------------------------------------------------------ 命名与 schema


def test_namespacing_and_schema():
    conn = FakeConn(name="linear")
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    tool = MCPTool(conn, "search_issues", "Search Linear issues", schema)
    assert tool.name == "mcp__linear__search_issues"
    assert "MCP:linear" in tool.description
    assert tool.input_schema == schema
    assert tool.read_only is False


def test_empty_schema_gets_default():
    tool = MCPTool(FakeConn(), "t", "", {})
    assert tool.input_schema == {"type": "object", "properties": {}}


@pytest.mark.asyncio
async def test_run_forwards_bare_tool_name(ctx):
    conn = FakeConn(reply=("result text", False))
    tool = MCPTool(conn, "do_thing", "", {})
    result = await tool.run({"x": 1}, ctx)
    assert conn.calls == [("do_thing", {"x": 1})]  # 不带命名空间前缀
    assert result.content == "result text"
    assert not result.is_error


@pytest.mark.asyncio
async def test_run_maps_error(ctx):
    conn = FakeConn(reply=("boom", True))
    result = await MCPTool(conn, "t", "", {}).run({}, ctx)
    assert result.is_error


def test_build_mcp_tools_skips_disconnected():
    manager = NS(
        conns={
            "up": FakeConn(
                name="up",
                tools=[NS(name="a", description="A", inputSchema={"type": "object"})],
            ),
            "down": FakeConn(name="down", connected=False, tools=[NS(name="b")]),
        }
    )
    tools = build_mcp_tools(manager)
    assert [t.name for t in tools] == ["mcp__up__a"]


# ------------------------------------------------------------------ stdio 垫片


def test_stdio_params_plain_exe():
    spec = MCPServerConfig(command=sys.executable, args=["-m", "server"])
    params = build_stdio_params(spec)
    assert Path(params.command).is_absolute()
    if sys.platform == "win32":
        assert Path(params.command).suffix.lower() == ".exe"
    assert "-m" in params.args


@pytest.mark.skipif(sys.platform != "win32", reason="Windows 专属垫片")
def test_stdio_params_cmd_shim(tmp_path, monkeypatch):
    npx = tmp_path / "npx.CMD"
    npx.write_text("@exit /b 0", encoding="utf-8")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("PATHEXT", ".EXE;.CMD")
    spec = MCPServerConfig(command="npx", args=["-y", "some-server"])
    params = build_stdio_params(spec)
    assert Path(params.command).name.lower() == "cmd.exe"
    assert params.args == [
        "/d",
        "/s",
        "/c",
        str(npx.resolve()),
        "-y",
        "some-server",
    ]


def test_stdio_params_env_is_allowlisted_and_explicit_values_override(monkeypatch):
    monkeypatch.setenv("BASE_VAR", "base")
    monkeypatch.setenv("SECRET_TOKEN", "secret")
    spec = MCPServerConfig(command=sys.executable, env={"EXTRA": "extra"})
    params = build_stdio_params(spec)
    assert "BASE_VAR" not in params.env
    assert "SECRET_TOKEN" not in params.env
    assert params.env["EXTRA"] == "extra"

    opted_in = build_stdio_params(spec, inherit_env=["BASE_VAR"])
    assert opted_in.env == {"BASE_VAR": "base", "EXTRA": "extra"}

    no_inheritance = build_stdio_params(spec, inherit_env=[])
    assert no_inheritance.env == {"EXTRA": "extra"}


def test_stdio_resolution_does_not_search_project_or_relative_path(tmp_path):
    project = tmp_path / "project"
    safe_bin = tmp_path / "safe-bin"
    project.mkdir()
    safe_bin.mkdir()
    filename = "server.EXE" if os.name == "nt" else "server"
    malicious = project / filename
    expected = safe_bin / filename
    malicious.write_bytes(b"")
    expected.write_bytes(b"")
    if os.name != "nt":
        malicious.chmod(0o700)
        expected.chmod(0o700)

    resolved = _resolve_stdio_executable(
        "server",
        cwd=project,
        env={"PATH": str(safe_bin), "PATHEXT": ".EXE"},
    )
    assert resolved == expected.resolve()

    with pytest.raises(FileNotFoundError, match="absolute allowed PATH"):
        _resolve_stdio_executable(
            "server",
            cwd=project,
            env={"PATH": ".", "PATHEXT": ".EXE"},
        )


def test_mcp_log_name_cannot_traverse_log_directory(tmp_path):
    conn = MCPServerConn(
        "../../outside", MCPServerConfig(command="server"), log_dir=tmp_path
    )

    log = conn._open_errlog()
    try:
        log_path = Path(log.name)
        assert log_path.parent == tmp_path
        assert ".." not in log_path.name
    finally:
        log.close()


def test_mcp_log_symlink_cannot_append_outside_directory(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    outside = tmp_path / "outside.log"
    outside.write_text("original", encoding="utf-8")
    try:
        (log_dir / "mcp-evil.log").symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    conn = MCPServerConn(
        "evil", MCPServerConfig(command="server"), log_dir=log_dir
    )
    with pytest.raises(UnsafePathError):
        conn._open_errlog()
    assert outside.read_text(encoding="utf-8") == "original"


# ------------------------------------------------------------------ 内容解析


def test_content_to_text_variants():
    result = NS(
        content=[
            NS(type="text", text="hello"),
            NS(type="image", data="..."),
            NS(type="resource", resource=NS(text="resource body")),
        ]
    )
    text = content_to_text(result)
    assert "hello" in text
    assert "[image content]" in text
    assert "resource body" in text


def test_content_to_text_empty():
    assert content_to_text(NS(content=[])) == "(empty result)"
