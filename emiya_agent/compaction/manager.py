"""客户端上下文压缩（跨供应商通用）。

- 触发：上一请求 usage.total_prompt_tokens > threshold × 执行模型窗口
- 切分：只在"真实用户消息"边界切，永不拆开 tool_use/tool_result 配对
- 摘要：便宜模型（compactor 路由）生成结构化摘要；超出其窗口时分块 map-reduce 兜底
"""

from __future__ import annotations

from typing import Awaitable, Callable

from emiya_agent.agent.specialist_prompts import COMPACT_SYSTEM
from emiya_agent.types import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    user_message,
)

COMPACT_MARKER = "[COMPACTED HISTORY]"

# (system, text) -> 摘要文本；由调用方绑定 compactor 路由
CompleteFn = Callable[[str, str], Awaitable[str]]


def _is_user_boundary(msg: Message) -> bool:
    """真实用户消息（含正文文本、不含 tool_result）才是安全切分点。"""
    if msg.role != "user":
        return False
    has_text = any(isinstance(b, TextBlock) and b.text.strip() for b in msg.content)
    has_tool_result = any(isinstance(b, ToolResultBlock) for b in msg.content)
    return has_text and not has_tool_result


def serialize_messages(messages: list[Message]) -> str:
    """把消息序列化为供摘要模型阅读的纯文本。"""
    parts: list[str] = []
    for m in messages:
        for b in m.content:
            if isinstance(b, TextBlock):
                if b.text.strip():
                    parts.append(f"[{m.role}] {b.text.strip()}")
            elif isinstance(b, ToolUseBlock):
                args = ", ".join(f"{k}={str(v)[:120]}" for k, v in b.input.items())
                parts.append(f"[tool_call] {b.name}({args})")
            elif isinstance(b, ToolResultBlock):
                content = b.content[:800]
                prefix = "[tool_error]" if b.is_error else "[tool_result]"
                parts.append(f"{prefix} {content}")
            elif isinstance(b, ThinkingBlock):
                continue  # 思考不进摘要
    return "\n".join(parts)


class CompactionManager:
    def __init__(
        self,
        complete: CompleteFn,
        *,
        threshold: float = 0.70,
        keep_recent_turns: int = 6,
        summarizer_window_chars: int = 2_000_000,  # DeepSeek v4 1M tok ≈ 数百万字符，保守取 200 万
    ) -> None:
        self.complete = complete
        self.threshold = threshold
        self.keep_recent_turns = keep_recent_turns
        self.summarizer_window_chars = summarizer_window_chars

    # ------------------------------------------------------------- 触发判断

    def should_compact(self, last_usage: Usage, context_window: int) -> bool:
        if context_window <= 0:
            return False
        return last_usage.total_prompt_tokens > self.threshold * context_window

    # ------------------------------------------------------------- 切分

    def split(self, messages: list[Message]) -> tuple[list[Message], list[Message]]:
        """返回 (head 待摘要, tail 原样保留)。无可压缩内容时 head 为空。"""
        boundaries = [i for i, m in enumerate(messages) if _is_user_boundary(m)]
        if len(boundaries) <= self.keep_recent_turns:
            return [], list(messages)
        cut_at = boundaries[-self.keep_recent_turns]
        if cut_at == 0:
            return [], list(messages)
        return list(messages[:cut_at]), list(messages[cut_at:])

    # ------------------------------------------------------------- 压缩

    async def compact(self, messages: list[Message]) -> list[Message]:
        head, tail = self.split(messages)
        if not head:
            return list(messages)
        summary = await self._summarize(serialize_messages(head))
        summary_msg = user_message(
            f"{COMPACT_MARKER}\n以下是本会话早前内容的压缩摘要（原始记录已存档）：\n\n{summary}"
        )
        return [summary_msg, *tail]

    async def _summarize(self, text: str) -> str:
        """单次摘要；超出摘要模型窗口时分块 map-reduce 兜底。"""
        if len(text) <= self.summarizer_window_chars:
            return await self.complete(COMPACT_SYSTEM, text)
        chunks = [
            text[i : i + self.summarizer_window_chars]
            for i in range(0, len(text), self.summarizer_window_chars)
        ]
        partials = []
        for i, chunk in enumerate(chunks, 1):
            partials.append(
                await self.complete(
                    COMPACT_SYSTEM,
                    f"(part {i}/{len(chunks)} of a longer transcript)\n{chunk}",
                )
            )
        merged = "\n\n".join(partials)
        return await self.complete(
            COMPACT_SYSTEM,
            "Merge these partial summaries of one conversation into a single "
            f"coherent briefing:\n\n{merged}",
        )
