"""skills 发现与 frontmatter 解析、项目级覆盖用户级、memory 管理器。"""

import pytest

from emiya_agent.memory.manager import MemoryManager
from emiya_agent.skills.loader import (
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


def test_missing_dirs_tolerated(tmp_path):
    assert discover_skills([tmp_path / "nope", tmp_path / "also_nope"]) == {}


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
