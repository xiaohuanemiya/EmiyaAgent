"""skills 发现与 frontmatter 解析、项目级覆盖用户级、memory 管理器。"""

from pathlib import Path

import pytest

from emiya_agent.config import Config
from emiya_agent.memory.manager import MemoryManager
from emiya_agent.tools.memory_tool import MemoryTool
from emiya_agent.skills.loader import (
    default_skill_dirs,
    discover_skills,
    parse_skill_md,
    skills_index_text,
)


def make_skill(root, name, description="", body="# Body\ndo things"):
    d = root / name
    d.mkdir(parents=True)
    fm = f"---\nname: {name}\ndescription: {description}\n---\n" if description else ""
    (d / "SKILL.md").write_text(fm + body, encoding="utf-8")
    return d


def test_parse_frontmatter(tmp_path):
    d = make_skill(tmp_path, "deploy", "How to deploy this project")
    skill = parse_skill_md(d / "SKILL.md")
    assert skill.name == "deploy"
    assert skill.description == "How to deploy this project"
    assert "do things" in skill.content()


def test_parse_without_frontmatter_falls_back(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    (d / "SKILL.md").write_text("# Title\nFirst meaningful line.\n", encoding="utf-8")
    skill = parse_skill_md(d / "SKILL.md")
    assert skill.name == "raw"  # 目录名兜底
    assert skill.description == "First meaningful line."


def test_project_overrides_user(tmp_path):
    user_dir = tmp_path / "user_skills"
    project_dir = tmp_path / "project_skills"
    make_skill(user_dir, "deploy", "user version")
    make_skill(project_dir, "deploy", "project version")
    make_skill(user_dir, "review", "user only")

    skills = discover_skills([user_dir, project_dir])
    assert skills["deploy"].description == "project version"
    assert skills["review"].description == "user only"


def test_default_dirs_support_ecosystem_and_preserve_precedence(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    config = Config(cwd=project)

    dirs = default_skill_dirs(config)
    assert dirs == [
        home / ".agents" / "skills",
        home / ".claude" / "skills",
        home / ".codex" / "skills",
        home / ".emiyacode" / "skills",
        project / ".agents" / "skills",
        project / ".claude" / "skills",
        project / ".codex" / "skills",
        project / ".emiyacode" / "skills",
    ]

    make_skill(home / ".emiyacode" / "skills", "shared", "user native")
    make_skill(project / ".agents" / "skills", "shared", "project standard")
    make_skill(project / ".codex" / "skills", "native-wins", "project codex")
    make_skill(project / ".emiyacode" / "skills", "native-wins", "project native")

    skills = discover_skills(dirs)
    assert skills["shared"].description == "project standard"
    assert skills["native-wins"].description == "project native"


def test_missing_dirs_tolerated(tmp_path):
    assert discover_skills([tmp_path / "nope", tmp_path / "also_nope"]) == {}


def test_external_skill_file_symlink_is_ignored(tmp_path):
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "leak"
    skill_dir.mkdir(parents=True)
    secret = tmp_path / "secret.txt"
    secret.write_text(
        "---\nname: leak\ndescription: secret\n---\nprivate",
        encoding="utf-8",
    )
    try:
        (skill_dir / "SKILL.md").symlink_to(secret)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    assert "leak" not in discover_skills([skills_dir])


def test_index_text(tmp_path):
    d = tmp_path / "s"
    make_skill(d, "alpha", "does alpha")
    make_skill(d, "beta", "does beta")
    text = skills_index_text(discover_skills([d]))
    assert "- alpha: does alpha" in text
    assert "- beta: does beta" in text
    assert skills_index_text({}) == ""


def test_bundled_files(tmp_path):
    d = make_skill(tmp_path, "withfiles", "has extras")
    (d / "template.py").write_text("x", encoding="utf-8")
    skill = parse_skill_md(d / "SKILL.md")
    names = [p.name for p in skill.bundled_files()]
    assert names == ["template.py"]


# ------------------------------------------------------------------ memory


def test_memory_write_read_list_delete(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    assert mm.list_files() == []
    assert mm.index_text() == ""

    mm.write("MEMORY.md", "- [约定](conventions.md)")
    mm.write("conventions.md", "用 uv 管理环境")

    assert mm.index_text() == "- [约定](conventions.md)"
    assert mm.read("conventions.md") == "用 uv 管理环境"
    assert mm.list_files() == ["MEMORY.md", "conventions.md"]

    mm.delete("conventions.md")
    assert mm.list_files() == ["MEMORY.md"]


def test_memory_rejects_traversal(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    with pytest.raises(ValueError):
        mm.write("../escape.md", "bad")
    with pytest.raises(ValueError):
        mm.write("", "bad")


def test_memory_read_missing_raises(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    with pytest.raises(FileNotFoundError):
        mm.read("nope.md")


def test_memory_write_and_delete_require_approval(tmp_path):
    tool = MemoryTool(MemoryManager(tmp_path / "memory"))

    assert tool.approval_required({"action": "list"}) is False
    assert tool.approval_required({"action": "read"}) is False
    assert tool.approval_required({"action": "write"}) is True
    assert tool.approval_required({"action": "delete"}) is True


def test_memory_index_symlink_cannot_read_outside_directory(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("do not exfiltrate", encoding="utf-8")
    try:
        (memory_dir / "MEMORY.md").symlink_to(secret)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    manager = MemoryManager(memory_dir)
    assert manager.index_text() == ""
    with pytest.raises(ValueError, match="非法记忆路径"):
        manager.read("MEMORY.md")
