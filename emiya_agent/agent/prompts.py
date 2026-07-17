"""系统提示词组装。

缓存稳定性规则：会话内字节级稳定 —— 不含时间戳、不含每轮变化的内容；
动态上下文（记忆更新等）注入到消息列表，绝不进 system。
"""

from __future__ import annotations

import platform
from pathlib import Path

CORE_SYSTEM = """\
You are EmiyaAgent, an AI coding agent running in a terminal.

You help the user with software engineering tasks: writing code, debugging, \
refactoring, running commands, and answering questions.

# Working style
- Be direct and concise. Lead with the outcome.
- Respond in the language the user writes in (Chinese input → Chinese response).
- When writing code, match the conventions of the surrounding project.
- If a request is ambiguous, state your assumption and proceed, or use \
ask_user for a single focused question — only when genuinely blocked.
- Format responses in Markdown; fenced code blocks with language tags.

# Tool use
- Read files before editing them. Prefer edit_file (exact string replace) for \
existing files; write_file only for new files or full rewrites.
- Use glob/grep to locate code instead of guessing paths.
- Use todo_write to track multi-step tasks so the user can follow progress; \
update it as you complete each step.
- Run shell commands to verify your changes (tests, builds) before declaring \
them done. Report failures honestly.
- Independent read-only lookups can be issued as parallel tool calls in one turn.
"""


def build_system_prompt(
    cwd: Path,
    agent_md: str | None = None,
    skills_index: str | None = None,
    memory_index: str | None = None,
) -> str:
    """静态核心 + 环境 + 项目指令 + 技能索引 + 记忆索引。

    所有部分都是会话开始时的静态快照（会话内字节级稳定，缓存安全）；
    会话中新写入的记忆下个会话生效。
    """
    prompt = CORE_SYSTEM
    prompt += (
        f"\n# Environment\n"
        f"- Working directory: {cwd}\n"
        f"- Platform: {platform.system()} {platform.release()}\n"
        f"- Shell tool: {'PowerShell' if platform.system() == 'Windows' else 'bash'}\n"
    )
    if agent_md and agent_md.strip():
        prompt += f"\n# Project instructions (AGENT.md)\n{agent_md.strip()}\n"
    if skills_index and skills_index.strip():
        prompt += f"\n# Skills\n{skills_index.strip()}\n"
    if memory_index and memory_index.strip():
        prompt += (
            f"\n# Project memory (MEMORY.md index)\n{memory_index.strip()}\n"
            "Use the `memory` tool to read a memory file when its pointer is "
            "relevant, and to save new durable facts you learn.\n"
        )
    return prompt


def load_agent_md(cwd: Path) -> str | None:
    """项目根的 AGENT.md（类 CLAUDE.md 项目指令）。"""
    path = cwd / "AGENT.md"
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
    return None
