"""系统提示词组装。

缓存稳定性规则：会话内字节级稳定 —— 不含时间戳、不含每轮变化的内容；
动态上下文（记忆更新等）注入到消息列表，绝不进 system。
"""

from __future__ import annotations

import platform
from pathlib import Path


_INSTRUCTION_FILES = ("AGENTS.override.md", "AGENTS.md", "AGENT.md")

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
- When the lsp tool is enabled, prefer it for semantic diagnostics, definitions, \
references, hover details, and symbols; still use grep and builds/tests for final verification.
- Use todo_write to track multi-step tasks so the user can follow progress; \
update it as you complete each step.
- Run shell commands to verify your changes (tests, builds) before declaring \
them done. Report failures honestly.
- Independent read-only lookups can be issued as parallel tool calls in one turn.

# Delegation
- Handle greetings, small talk, simple questions, and clear one-step or local \
tasks directly. Do not create a plan, todo list, or subagent merely because the \
request is written in Chinese or any other language.
- Delegation is optional and benefit-driven. There is no fixed \
explore -> plan -> general chain; choose only the role or roles that materially \
improve the result and avoid duplicating work.
- Use explore for broad read-only investigation where independent search or \
fan-out is useful.
- Use plan only for genuinely complex, ambiguous, high-risk, or cross-module \
design where a separate read-only plan is likely to prevent rework.
- Use general for a bounded, independent implementation task that benefits from \
isolated execution or concurrency.
- Before a cross-language handoff, read the current target model languages shown \
in the translate_for_model tool description. Call that translator Agent only \
when adaptation materially improves a complex or precision-sensitive prompt or \
report; same-language, short, simple, or code-heavy handoffs should remain direct. \
Translation is never a mandatory stage and never implies that plan is needed.
- When translation is useful, pass the returned translated_text to the intended \
downstream role. Never guess or supply the target language yourself.
- Integrate and verify delegated results yourself before answering the user.
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
        prompt += f"\n# Project instructions\n{agent_md.strip()}\n"
    if skills_index and skills_index.strip():
        prompt += f"\n# Skills\n{skills_index.strip()}\n"
    if memory_index and memory_index.strip():
        prompt += (
            f"\n# Project memory (MEMORY.md index)\n{memory_index.strip()}\n"
            "Use the `memory` tool to read a memory file when its pointer is "
            "relevant, and to save new durable facts you learn.\n"
        )
    return prompt


def _git_root(cwd: Path) -> Path:
    """返回包含 cwd 的最近 Git 工作区根；非 Git 目录退化为 cwd。"""
    cwd = cwd.resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / ".git").exists():
            return candidate
    return cwd


def _instruction_dirs(root: Path, cwd: Path) -> list[Path]:
    """Git 根到 cwd（含两端）的目录链，顺序从通用到具体。"""
    dirs = [root]
    current = root
    for part in cwd.relative_to(root).parts:
        current /= part
        dirs.append(current)
    return dirs


def load_agent_md(cwd: Path) -> str | None:
    """发现并合并项目指令，同时在文本中保留每段指令的实际来源。

    从最近的 Git 项目根逐层扫描到 cwd。每一层只加载一个文件，优先级为
    AGENTS.override.md > AGENTS.md > AGENT.md；最后一个是 EmiyaAgent 旧格式的
    兼容回退。更深目录的内容排在后面，因而可以补充或覆盖上层约定。
    """
    cwd = cwd.resolve()
    root = _git_root(cwd)
    sections: list[str] = []
    for directory in _instruction_dirs(root, cwd):
        for filename in _INSTRUCTION_FILES:
            path = directory / filename
            if not path.is_file():
                continue
            try:
                # 项目指令会自动发送给 provider；拒绝仓库用符号链接读取
                # Git 工作区外的凭据或其他用户文件。
                path.resolve(strict=True).relative_to(root.resolve(strict=True))
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError, ValueError):
                # 不可读的高优先级文件不应屏蔽同层兼容回退。
                continue
            if text.strip():
                sections.append(f"## Source: {path}\n{text.strip()}")
            break
    return "\n\n".join(sections) or None
