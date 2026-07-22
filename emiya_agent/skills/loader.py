"""Skills 加载：目录发现 + SKILL.md frontmatter 解析。

结构（Claude Code 风格）：
    <skills_dir>/<skill-name>/SKILL.md   # YAML frontmatter: name, description
    <skills_dir>/<skill-name>/...        # 技能附带文件

描述注入系统提示词；完整内容由 skill 工具按需加载（渐进披露）。
项目级覆盖用户级同名技能。
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from emiya_agent.config import CONFIG_DIR_NAME, Config
from emiya_agent.safe_paths import UnsafePathError, confined_path, open_confined

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass
class Skill:
    name: str
    description: str
    path: Path  # SKILL.md 所在路径
    root: Path  # 发现时声明的技能根目录（保留词法路径以检测后续替换）
    resolved_root: Path  # 发现时的规范根目录身份
    root_identity: tuple[int, int]
    content_digest: bytes

    def _relative_path(self, path: Path) -> Path:
        """Revalidate the discovery root and return a confined relative path."""

        try:
            root_stat = self.root.stat()
            if (
                self.root.resolve(strict=True) != self.resolved_root
                or (root_stat.st_dev, root_stat.st_ino) != self.root_identity
            ):
                raise UnsafePathError("技能根目录在发现后发生变化")
            relative = path.absolute().relative_to(self.root.absolute())
            path.resolve(strict=True).relative_to(self.resolved_root)
        except (OSError, ValueError) as exc:
            raise UnsafePathError(f"技能路径在发现后越界或失效: {path}") from exc
        # confined_path rejects any link/reparse component, including a final
        # SKILL.md swapped after discovery. open_confined adds O_NOFOLLOW on POSIX.
        confined_path(self.root, relative, label="技能文件")
        return relative

    def content(self) -> str:
        relative = self._relative_path(self.path)
        with open_confined(
            self.root,
            relative,
            "r",
            label="技能文件",
            encoding="utf-8",
        ) as stream:
            content = stream.read()
        if hashlib.sha256(content.encode("utf-8")).digest() != self.content_digest:
            raise UnsafePathError("技能文件内容在发现后发生变化")
        return content

    def bundled_files(self) -> list[Path]:
        """技能目录中除 SKILL.md 外的文件。"""
        skill_dir = self.root / self._relative_path(self.path.parent)
        files: list[Path] = []
        for path in sorted(skill_dir.rglob("*")):
            if path.name == "SKILL.md" or not path.is_file():
                continue
            try:
                self._relative_path(path)
            except UnsafePathError:
                continue
            files.append(path)
        return files


def parse_skill_md(path: Path, *, root: Path | None = None) -> Skill | None:
    declared_root = (root or path.parent).absolute()
    try:
        resolved_root = declared_root.resolve(strict=True)
        root_stat = declared_root.stat()
        relative = path.absolute().relative_to(declared_root)
        candidate = confined_path(declared_root, relative, label="技能文件")
        candidate.resolve(strict=True).relative_to(resolved_root)

        # Parse only bytes read through the confined handle.  Checking the
        # opened file identity against the path after the read closes the
        # classic validate-then-open / swap-then-restore race as well.
        with open_confined(
            declared_root,
            relative,
            "r",
            label="技能文件",
            encoding="utf-8",
        ) as stream:
            opened_stat = os.fstat(stream.fileno())
            text = stream.read()
            # Keep the handle alive until the comparison finishes so a closed
            # file identifier cannot be recycled into a false match.
            current = confined_path(declared_root, relative, label="技能文件")
            current_stat = current.stat(follow_symlinks=False)
            current_root_stat = declared_root.stat()
            if (
                declared_root.resolve(strict=True) != resolved_root
                or (current_root_stat.st_dev, current_root_stat.st_ino)
                != (root_stat.st_dev, root_stat.st_ino)
                or (current_stat.st_dev, current_stat.st_ino)
                != (opened_stat.st_dev, opened_stat.st_ino)
            ):
                return None
    except (OSError, UnicodeDecodeError, UnsafePathError, ValueError):
        return None
    match = _FRONTMATTER_RE.match(text)
    meta: dict = {}
    if match:
        try:
            loaded = yaml.safe_load(match.group(1))
            meta = loaded if isinstance(loaded, dict) else {}
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
    return Skill(
        name=name,
        description=description,
        path=path.absolute(),
        root=declared_root,
        resolved_root=resolved_root,
        root_identity=(root_stat.st_dev, root_stat.st_ino),
        content_digest=hashlib.sha256(text.encode("utf-8")).digest(),
    )


def discover_skills(dirs: list[Path]) -> dict[str, Skill]:
    """按顺序扫描目录，后面的目录（项目级）覆盖前面的同名技能。"""
    skills: dict[str, Skill] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        try:
            resolved_root = d.resolve(strict=True)
        except OSError:
            continue
        for skill_md in sorted(d.glob("*/SKILL.md")):
            try:
                # SKILL.md 会被自动索引、按需完整发送给模型；拒绝单文件
                # symlink 逃出其声明的技能根目录。
                skill_md.resolve(strict=True).relative_to(resolved_root)
            except (OSError, ValueError):
                continue
            skill = parse_skill_md(skill_md, root=d)
            if skill is not None:
                skills[skill.name] = skill
    return skills


def default_skill_dirs(config: Config) -> list[Path]:
    """兼容主流技能目录，并保持项目级与 .emiyacode 的覆盖优先级。

    discover_skills 按顺序加载且后者覆盖前者，因此所有用户级目录先于项目级；
    每个作用域内再让 EmiyaAgent 原生目录最后加载。
    """
    user_root = Path.home()
    project_root = config.cwd
    relative_dirs = (
        Path(".agents") / "skills",
        Path(".claude") / "skills",
        Path(".codex") / "skills",
        Path(CONFIG_DIR_NAME) / "skills",
    )
    user_dirs = [user_root / relative for relative in relative_dirs]
    project_dirs: list[Path] = []
    resolved_project = project_root.resolve()
    for relative in relative_dirs:
        candidate = project_root / relative
        try:
            # 项目级技能根本身也不能通过父目录链接逃出工作区。
            candidate.resolve().relative_to(resolved_project)
        except (OSError, ValueError):
            continue
        project_dirs.append(candidate)
    return [*user_dirs, *project_dirs]


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
