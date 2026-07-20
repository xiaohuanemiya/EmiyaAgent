"""项目指令的层级发现、文件优先级与 system prompt 来源标注。"""

import pytest

from emiya_agent.agent.prompts import build_system_prompt, load_agent_md
from emiya_agent.config import load_config
from emiya_agent.ui.repl import Repl


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_loads_instructions_from_git_root_to_cwd_with_sources(tmp_path):
    root = tmp_path / "repo"
    cwd = root / "packages" / "api"
    (root / ".git").mkdir(parents=True)
    cwd.mkdir(parents=True)

    _write(root / "AGENTS.md", "root standard")
    _write(root / "AGENT.md", "ignored root legacy")
    _write(root / "packages" / "AGENT.md", "package legacy")
    _write(cwd / "AGENTS.md", "api standard")

    instructions = load_agent_md(cwd)
    assert instructions is not None
    assert "root standard" in instructions
    assert "package legacy" in instructions
    assert "api standard" in instructions
    assert "ignored root legacy" not in instructions
    assert instructions.index("root standard") < instructions.index("package legacy")
    assert instructions.index("package legacy") < instructions.index("api standard")
    assert f"## Source: {root / 'AGENTS.md'}" in instructions
    assert f"## Source: {root / 'packages' / 'AGENT.md'}" in instructions
    assert f"## Source: {cwd / 'AGENTS.md'}" in instructions

    prompt = build_system_prompt(cwd, instructions)
    assert "# Project instructions" in prompt
    assert f"## Source: {root / 'AGENTS.md'}" in prompt
    assert "# Project instructions (AGENT.md)" not in prompt


def test_override_wins_within_each_directory(tmp_path):
    root = tmp_path / "repo"
    cwd = root / "src"
    (root / ".git").mkdir(parents=True)
    cwd.mkdir(parents=True)

    _write(root / "AGENTS.md", "root rules")
    _write(cwd / "AGENT.md", "legacy rules")
    _write(cwd / "AGENTS.md", "standard rules")
    _write(cwd / "AGENTS.override.md", "override rules")

    instructions = load_agent_md(cwd)
    assert instructions is not None
    assert "root rules" in instructions
    assert "override rules" in instructions
    assert "standard rules" not in instructions
    assert "legacy rules" not in instructions
    assert f"## Source: {cwd / 'AGENTS.override.md'}" in instructions


def test_non_git_directory_does_not_scan_parents(tmp_path):
    cwd = tmp_path / "standalone"
    cwd.mkdir()
    _write(tmp_path / "AGENTS.md", "parent rules")
    _write(cwd / "AGENT.md", "local legacy")

    instructions = load_agent_md(cwd)
    assert instructions is not None
    assert "local legacy" in instructions
    assert "parent rules" not in instructions


def test_no_instruction_file_returns_none(tmp_path):
    assert load_agent_md(tmp_path) is None


def test_external_instruction_symlink_is_ignored(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("never send this secret", encoding="utf-8")
    try:
        (root / "AGENTS.md").symlink_to(secret)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    assert load_agent_md(root) is None


def test_subagents_receive_project_instructions_and_skill_index(tmp_path):
    _write(tmp_path / "AGENTS.md", "always run the focused tests")
    _write(
        tmp_path / ".agents" / "skills" / "audit" / "SKILL.md",
        "---\nname: audit\ndescription: inspect risky changes\n---\nsteps",
    )

    repl = Repl(load_config(tmp_path))

    assert "always run the focused tests" in repl.spawner.shared_context
    assert "audit: inspect risky changes" in repl.spawner.shared_context
    assert "translate_for_model" in repl.tools.names()


def test_main_agent_decides_when_delegation_is_useful(tmp_path):
    prompt = build_system_prompt(tmp_path).lower()

    assert "handle greetings, small talk, simple questions" in prompt
    assert "clear one-step or local tasks directly" in prompt
    assert "request is written in chinese or any other language" in prompt
    assert "delegation is optional and benefit-driven" in prompt
    assert "there is no fixed" in prompt
    assert "explore -> plan -> general chain" in prompt
    assert "use explore" in prompt
    assert "use plan only" in prompt
    assert "use general" in prompt
    assert "translate_for_model tool description" in prompt
    assert "translation is never a mandatory stage" in prompt
    assert "never guess or supply the target language" in prompt
