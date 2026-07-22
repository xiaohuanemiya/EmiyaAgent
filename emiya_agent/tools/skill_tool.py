"""skill 工具：按需加载技能完整内容（渐进披露）。"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from emiya_agent.safe_paths import UnsafePathError
from emiya_agent.skills.loader import Skill
from emiya_agent.tools.base import Tool, ToolContext, ToolResult


class SkillTool(Tool):
    name: ClassVar[str] = "skill"
    description: ClassVar[str] = (
        "Load the full instructions of an available skill by name. "
        "Available skills and their descriptions are listed in the system prompt."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Skill name to load"},
        },
        "required": ["name"],
    }
    read_only: ClassVar[bool] = True

    def __init__(self, skills: dict[str, Skill]) -> None:
        self._skills = skills

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = args["name"]
        skill = self._skills.get(name)
        if skill is None:
            available = ", ".join(sorted(self._skills)) or "(none)"
            return ToolResult(
                content=f"Error: 未找到技能 {name!r}。可用技能: {available}",
                is_error=True,
            )
        try:
            # Skills may live on slow/network-backed user directories.  The
            # read-time confinement check and traversal must not block the loop.
            content, extras = await asyncio.to_thread(
                lambda: (skill.content(), skill.bundled_files())
            )
        except (OSError, UnicodeDecodeError, UnsafePathError) as exc:
            return ToolResult(
                content=(
                    "Error: 技能文件在发现后发生变化、越界或无法安全读取: "
                    f"{type(exc).__name__}"
                ),
                is_error=True,
            )
        if extras:
            listing = "\n".join(f"- {p}" for p in extras[:50])
            content += f"\n\n[Bundled files in this skill directory]\n{listing}"
        return ToolResult(content=content)
