"""Provider 抽象层：统一请求结构 + 流式事件联合类型。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, ClassVar

from pydantic import BaseModel, Field

from emiya_agent.types import Message, Usage


class ToolSpec(BaseModel):
    """工具定义，input_schema 为 JSON Schema（Anthropic 原生；OpenAI 侧转换为 function）。"""

    name: str
    description: str
    input_schema: dict[str, Any]


class ChatRequest(BaseModel):
    model: str  # 裸模型 id，如 "claude-fable-5" / "deepseek-chat"
    system: str = ""
    messages: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list)
    max_tokens: int = 16000
    effort: str | None = None  # → Anthropic output_config.effort；OpenAI 侧忽略
    cache: bool = True  # → Anthropic cache_control 断点


# ---------------------------------------------------------------- 流式事件


@dataclass
class StreamStart:
    """一个内容块开始。block_type: text | thinking | tool_use | fallback ..."""

    index: int
    block_type: str
    tool_use_id: str | None = None
    tool_name: str | None = None


@dataclass
class TextDelta:
    index: int
    text: str


@dataclass
class ThinkingDelta:
    index: int
    thinking: str


@dataclass
class ToolInputDelta:
    index: int
    partial_json: str


@dataclass
class BlockStop:
    index: int


@dataclass
class MessageDone:
    """一轮流式请求结束。stop_reason 归一化为:
    end_turn | tool_use | max_tokens | refusal | 其余原样透传。"""

    message: Message
    usage: Usage
    stop_reason: str
    served_by: str | None = None  # fable-5 服务端 fallback 实际服务的模型
    refusal: str | None = None  # stop_reason==refusal 时的分类/解释


StreamEvent = (
    StreamStart | TextDelta | ThinkingDelta | ToolInputDelta | BlockStop | MessageDone
)


class Provider(ABC):
    name: ClassVar[str] = ""

    @abstractmethod
    def stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]:
        """流式对话。最后一个事件必须是 MessageDone。"""

    async def count_tokens(self, req: ChatRequest) -> int | None:
        """精确计数（仅 Anthropic 支持）；其余返回 None，调用方用启发式。"""
        return None
