from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "emiya_agent"


def _production_trees():
    for path in PACKAGE_ROOT.rglob("*.py"):
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        yield relative, ast.parse(source, filename=str(path))


def _named_call_sites(name: str) -> set[str]:
    sites: set[str] = set()
    for relative, tree in _production_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == name:
                    sites.add(relative)
    return sites


def _method_calls(name: str) -> set[tuple[str, str | None]]:
    calls: set[tuple[str, str | None]] = set()
    for relative, tree in _production_trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != name:
                continue
            first_argument = node.args[0] if node.args else None
            literal = (
                first_argument.value
                if isinstance(first_argument, ast.Constant)
                and isinstance(first_argument.value, str)
                else None
            )
            calls.add((relative, literal))
    return calls


def test_chat_requests_are_created_only_inside_agent_loop() -> None:
    """Prevent auxiliary roles from bypassing lifecycle and usage accounting."""

    assert _named_call_sites("ChatRequest") == {"agent/loop.py"}
    non_provider_stream_calls = {
        relative
        for relative, _ in _method_calls("stream")
        if not relative.startswith("providers/")
    }
    assert non_provider_stream_calls == {"agent/loop.py"}
    assert _named_call_sites("complete_text") == set()
    assert not (PACKAGE_ROOT / "agent" / "completion.py").exists()


def test_agent_loop_has_only_root_and_subagent_owners() -> None:
    """The executor remains root; every other model role uses the spawner."""

    assert _named_call_sites("AgentLoop") == {
        "agent/subagent.py",
        "ui/repl.py",
    }


def test_internal_model_roles_use_only_the_internal_spawner_entrypoint() -> None:
    assert _method_calls("spawn_internal") == {
        ("tools/translate.py", "translator"),
        ("ui/repl.py", "compactor"),
    }
