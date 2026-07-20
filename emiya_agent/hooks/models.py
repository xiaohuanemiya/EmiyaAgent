"""Public models for command-based lifecycle hooks."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HookEvent(StrEnum):
    """Lifecycle points supported by the first hook protocol version.

    ``matcher`` applies to a per-event subject string:

    - ``PreToolUse`` / ``PostToolUse``: the tool name.
    - ``UserPromptSubmit``: the raw prompt text.
    - ``SessionStart``: the start source (``startup`` / ``resume`` / ``one-shot``).
    - ``SessionEnd``: the end reason (``exit``).
    - ``Stop``: the turn's stop reason (``end_turn`` / ``max_tokens`` / …).
    - ``SubagentStop``: the subagent role name.
    - ``PreCompact`` / ``PostCompact``: the trigger (``auto`` / ``manual``).
    """

    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    SESSION_START = "SessionStart"
    SESSION_END = "SessionEnd"
    STOP = "Stop"
    SUBAGENT_STOP = "SubagentStop"
    PRE_COMPACT = "PreCompact"
    POST_COMPACT = "PostCompact"


class HookDecision(StrEnum):
    """A hook may preserve, raise, or deny the caller's approval level."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


# Events whose aggregate decision gates the surrounding action. All other
# events are observe-only: their decisions are recorded in diagnostics but
# never change control flow, so a misbehaving hook cannot force the agent to
# keep running or alter results retroactively.
DECISION_EVENTS = frozenset(
    {
        HookEvent.PRE_TOOL_USE,
        HookEvent.POST_TOOL_USE,
        HookEvent.USER_PROMPT_SUBMIT,
        HookEvent.PRE_COMPACT,
    }
)


class HookCommandConfig(BaseModel):
    """Validated configuration for one argv-based command hook."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=128)
    event: HookEvent
    matcher: str = Field(default=".*", max_length=1024)
    command: tuple[str, ...] = Field(min_length=1)
    timeout: float = Field(default=10.0, gt=0, le=120)
    on_error: Literal["warn", "block"] = "warn"

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if value != value.strip() or any(ord(char) < 32 for char in value):
            raise ValueError("hook id must not contain surrounding/control whitespace")
        return value

    @field_validator("matcher")
    @classmethod
    def _validate_matcher(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"invalid hook matcher regex: {exc}") from exc
        return value

    @field_validator("command")
    @classmethod
    def _validate_command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value[0]:
            raise ValueError("hook executable must not be empty")
        if any("\x00" in argument for argument in value):
            raise ValueError("hook command arguments must not contain NUL")
        return value

    def matches(self, tool_name: str) -> bool:
        """Return whether this hook applies to a tool name."""

        return re.search(self.matcher, tool_name) is not None


HookStatus = Literal["ok", "blocked", "warning", "error", "timeout"]


class HookDiagnostic(BaseModel):
    """Bounded process evidence for one matching hook."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hook_id: str
    event: HookEvent
    status: HookStatus
    decision: HookDecision | None = None
    reason: str = ""
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    stdout_omitted_bytes: int = 0
    stderr_omitted_bytes: int = 0


class HookRunResult(BaseModel):
    """Aggregate hook decision with diagnostics in configuration order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: HookDecision
    reason: str = ""
    diagnostics: tuple[HookDiagnostic, ...] = ()
