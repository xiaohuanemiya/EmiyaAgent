"""共享测试基建：FakeProvider（脚本化回合）、FakeUI、静默渲染。"""

from __future__ import annotations

import io
from typing import AsyncIterator

import pytest
from rich.console import Console

from emiya_agent.config import Config
from emiya_agent.providers.base import (
    ChatRequest,
    MessageDone,
    Provider,
    StreamEvent,
)
from emiya_agent.types import Message, Usage


class FakeProvider(Provider):
    """按脚本依次返回 assistant 回合；记录收到的每个请求供断言。"""

    name = "fake"

    def __init__(self, turns: list[MessageDone]) -> None:
        self.turns = list(turns)
        self.requests: list[ChatRequest] = []

    async def stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(req)
        if not self.turns:
            raise AssertionError("FakeProvider 脚本已耗尽")
        yield self.turns.pop(0)


def assistant_turn(
    content: list, stop_reason: str = "end_turn", usage: Usage | None = None
) -> MessageDone:
    return MessageDone(
        message=Message(role="assistant", content=content),
        usage=usage or Usage(input_tokens=10, output_tokens=5),
        stop_reason=stop_reason,
    )


class FakeUI:
    def __init__(self, answers: list[str] | None = None) -> None:
        self.answers = list(answers or [])
        self.questions: list[str] = []
        self.notifications: list[str] = []

    async def ask(self, question: str, options: list[str] | None = None) -> str:
        self.questions.append(question)
        return self.answers.pop(0) if self.answers else "ok"

    def notify(self, text: str) -> None:
        self.notifications.append(text)


@pytest.fixture
def quiet_console() -> Console:
    return Console(file=io.StringIO(), width=100)


@pytest.fixture
def config(tmp_path) -> Config:
    return Config(cwd=tmp_path)
