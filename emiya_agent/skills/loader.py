"""Skills 加载：目录发现 + SKILL.md frontmatter 解析。

结构（Claude Code 风格）：
    <skills_dir>/<skill-name>/SKILL.md   # YAML frontmatter: name, description
    <skills_dir>/<skill-name>/...        # 技能附带文件

描述注入系统提示词；完整内容由 skill 工具按需加载（渐进披露）。
项目级覆盖用户级同名技能。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from emiya_agent.config import CONFIG_DIR_NAME, Config

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass
class Skill:
    name: str
    description: str
    path: Path  # SKILL.md 所在路径

    def content(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def bundled_files(self) -> list[Path]:
        """技能目录中除 SKILL.md 外的文件。"""
        return [
            p
            for p in sorted(self.path.parent.rglob("*"))
            if p.is_file() and p.name != "SKILL.md"
        ]


def parse_skill_md(path: Path) -> Skill | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = _FRONTMATTER_RE.match(text)
    meta: dict = {}
    if match:
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
    name = str(meta.get("name") or path.parent.name).strip()
    description = str(meta.get("description") or "").strip()
    if not description:
        # 无描述时取正文首个非空行，保证索引可用
        body = _FRONTMATTER_RE.sub("", text)
        for line in body.splitlines():
            if line.strip() and not line.startswith("#"):
                description = line.strip()[:200]
                break
    return Skill(name=name, description=description, path=path)


def discover_skills(dirs: list[Path]) -> dict[str, Skill]:
    """按顺序扫描目录，后面的目录（项目级）覆盖前面的同名技能。"""
    skills: dict[str, Skill] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for skill_md in sorted(d.glob("*/SKILL.md")):
            skill = parse_skill_md(skill_md)
            if skill is not None:
                skills[skill.name] = skill
    return skills


def default_skill_dirs(config: Config) -> list[Path]:
    return [
        Path.home() / CONFIG_DIR_NAME / "skills",  # 用户级
        config.cwd / CONFIG_DIR_NAME / "skills",  # 项目级（优先）
    ]


def skills_index_text(skills: dict[str, Skill]) -> str:
    """注入系统提示词的技能索引。"""
    if not skills:
        return ""
    lines = [
        "The following skills are available. When a task matches a skill's "
        "description, load its full instructions with the `skill` tool BEFORE "
        "attempting the task:"
    ]
    for skill in sorted(skills.values(), key=lambda s: s.name):
        lines.append(f"- {skill.name}: {skill.description}")
    return "\n".join(lines)
