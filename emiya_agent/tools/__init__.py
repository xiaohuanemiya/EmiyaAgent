"""内置工具集合。"""

from emiya_agent.tools.ask_user import AskUserTool
from emiya_agent.tools.edit import EditFileTool
from emiya_agent.tools.fs import ListDirTool, ReadFileTool, WriteFileTool
from emiya_agent.tools.registry import ToolRegistry
from emiya_agent.tools.search import GlobTool, GrepTool
from emiya_agent.tools.shell import ShellTool
from emiya_agent.tools.todo import TodoWriteTool


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool_cls in (
        ShellTool,
        ReadFileTool,
        WriteFileTool,
        ListDirTool,
        EditFileTool,
        GlobTool,
        GrepTool,
        TodoWriteTool,
        AskUserTool,
    ):
        registry.register(tool_cls())
    return registry
