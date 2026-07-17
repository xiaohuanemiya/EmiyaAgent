"""MCP 桥接：命名空间、schema 透传、错误映射、Windows 垫片、内容解析。"""

import sys
from types import SimpleNamespace as NS

import pytest

from emiya_agent.config import MCPServerConfig
from emiya_agent.mcp_client.adapter import MCPTool, build_mcp_tools
from emiya_agent.mcp_client.manager import build_stdio_params, content_to_text
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
    spec = MCPServerConfig(command="python", args=["-m", "server"])
    params = build_stdio_params(spec)
    if sys.platform == "win32":
        # python 解析为 .exe → 不加垫片
        assert params.command == "python"
    assert "-m" in params.args


@pytest.mark.skipif(sys.platform != "win32", reason="Windows 专属垫片")
def test_stdio_params_cmd_shim(monkeypatch):
    monkeypatch.setattr(
        "shutil.which", lambda cmd: r"C:\nodejs\npx.cmd" if cmd == "npx" else None
    )
    spec = MCPServerConfig(command="npx", args=["-y", "some-server"])
    params = build_stdio_params(spec)
    assert params.command == "cmd"
    assert params.args == ["/c", "npx", "-y", "some-server"]


def test_stdio_params_env_merged(monkeypatch):
    monkeypatch.setenv("BASE_VAR", "base")
    spec = MCPServerConfig(command="python", env={"EXTRA": "extra"})
    params = build_stdio_params(spec)
    assert params.env["BASE_VAR"] == "base"
    assert params.env["EXTRA"] == "extra"


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
