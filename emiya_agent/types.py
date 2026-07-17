"""内部消息模型 —— 以 Anthropic content-block 为原生形态，全项目唯一数据真源。

OpenAI/DeepSeek 的格式转换在 providers/openai_compat.py 内完成，
其余模块一律只操作这里定义的类型。
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(BaseModel):
    """思考块。fable-5 默认 display=omitted 时 thinking 为空串；
    signature 是不透明签名，同模型续写时必须原样回传。"""

    type: Literal["thinking"] = "thinking"
    thinking: str = ""
    signature: str | None = None


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = Annotated[
    TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]


class Message(BaseModel):
    """system 提示词不是消息，是 ChatRequest 的独立字段（Anthropic 原生语义）。"""

    role: Literal["user", "assistant"]
    content: list[ContentBlock]

    def text(self) -> str:
        """拼接所有文本块。"""
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]


def user_message(text: str) -> Message:
    return Message(role="user", content=[TextBlock(text=text)])


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def total_prompt_tokens(self) -> int:
        """真实提示词规模 = 未缓存 + 缓存读 + 缓存写 三者之和。"""
        return (
            self.input_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens
            + other.cache_read_input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
        )
