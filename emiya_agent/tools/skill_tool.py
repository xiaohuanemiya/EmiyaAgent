"""skill 工具：按需加载技能完整内容（渐进披露）。"""

from __future__ import annotations

from typing import Any, ClassVar

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
        content = skill.content()
        extras = skill.bundled_files()
        if extras:
            listing = "\n".join(f"- {p}" for p in extras[:50])
            content += f"\n\n[Bundled files in this skill directory]\n{listing}"
        return ToolResult(content=content)
