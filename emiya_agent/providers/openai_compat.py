"""OpenAI 兼容 Provider —— OpenAI (GPT-5.6) 与 DeepSeek 共用。

全部格式转换集中在本文件（内部 content-block ↔ chat.completions）：
- 内部 tool_use.input(dict) ↔ OpenAI tool_calls.function.arguments(JSON 字符串)
- 内部 user 消息内的 tool_result 块 ↔ 独立的 role:"tool" 消息（每结果一条）
- 内部 system 字段 ↔ 首条 role:"system" 消息
- DeepSeek 的 reasoning_content ↔ 内部 ThinkingBlock（无签名，不回传）
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import openai

from emiya_agent.errors import ProviderError
from emiya_agent.providers.base import (
    BlockStop,
    ChatRequest,
    MessageDone,
    Provider,
    StreamEvent,
    StreamStart,
    TextDelta,
    ThinkingDelta,
    ToolInputDelta,
    ToolSpec,
)
from emiya_agent.types import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_tool_arguments(value: str) -> Any:
    return json.loads(
        value,
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
    )

_FINISH_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}

# 伪块索引：OpenAI 流没有内容块编号，为渲染器约定固定布局
_THINKING_INDEX = 0
_TEXT_INDEX = 1
_TOOL_INDEX_BASE = 2

# 内部 effort → OpenAI reasoning_effort（仅 gpt-5/o 系推理模型支持）
_REASONING_EFFORT_MAP = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


def reasoning_effort_for(provider_name: str, model: str, effort: str | None) -> str | None:
    """OpenAI 官方推理模型才映射 effort；其余供应商/模型不发该参数（避免 400）。"""
    if provider_name != "openai" or not effort:
        return None
    if not model.startswith(("gpt-5", "o1", "o3", "o4")):
        return None
    return _REASONING_EFFORT_MAP.get(effort)


def to_openai_messages(req: ChatRequest) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    if req.system:
        msgs.append({"role": "system", "content": req.system})
    for m in req.messages:
        if m.role == "assistant":
            text = m.text()
            tool_calls = [
                {
                    "id": b.id,
                    "type": "function",
                    "function": {
                        "name": b.name,
                        "arguments": json.dumps(b.input, ensure_ascii=False),
                    },
                }
                for b in m.content
                if isinstance(b, ToolUseBlock)
            ]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            msgs.append(msg)
            # ThinkingBlock 出站丢弃（跨供应商不可回传）
        else:
            # tool_result 必须先于同轮的用户文本（紧跟 assistant 的 tool_calls）
            for b in m.content:
                if isinstance(b, ToolResultBlock):
                    content = b.content
                    if b.is_error and not content.startswith("Error"):
                        content = f"Error: {content}"
                    msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": b.tool_use_id,
                            "content": content,
                        }
                    )
            text = m.text()
            if text:
                msgs.append({"role": "user", "content": text})
    return msgs


def to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in sorted(tools, key=lambda t: t.name)
    ]


class ChunkAggregator:
    """聚合 chat.completions 流式分片 → 内部 StreamEvent + 最终 Message。

    与网络解耦，便于用 SimpleNamespace 伪分片做单测。
    """

    def __init__(self) -> None:
        self._text: list[str] = []
        self._thinking: list[str] = []
        self._tool_calls: dict[int, dict[str, Any]] = {}  # index → {id,name,args}
        self._finish_reason: str | None = None
        self._usage: Any = None
        self._text_started = False
        self._thinking_started = False

    def feed(self, chunk: Any) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        if getattr(chunk, "usage", None) is not None:
            self._usage = chunk.usage
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return events
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is not None:
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                if not self._thinking_started:
                    self._thinking_started = True
                    events.append(
                        StreamStart(index=_THINKING_INDEX, block_type="thinking")
                    )
                self._thinking.append(reasoning)
                events.append(
                    ThinkingDelta(index=_THINKING_INDEX, thinking=reasoning)
                )
            content = getattr(delta, "content", None)
            if content:
                if not self._text_started:
                    self._text_started = True
                    if self._thinking_started:
                        events.append(BlockStop(index=_THINKING_INDEX))
                    events.append(StreamStart(index=_TEXT_INDEX, block_type="text"))
                self._text.append(content)
                events.append(TextDelta(index=_TEXT_INDEX, text=content))
            for tc in getattr(delta, "tool_calls", None) or []:
                idx = tc.index
                slot = self._tool_calls.setdefault(
                    idx, {"id": None, "name": None, "args": []}
                )
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        if slot["name"] is None:
                            events.append(
                                StreamStart(
                                    index=_TOOL_INDEX_BASE + idx,
                                    block_type="tool_use",
                                    tool_use_id=slot["id"],
                                    tool_name=fn.name,
                                )
                            )
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["args"].append(fn.arguments)
                        events.append(
                            ToolInputDelta(
                                index=_TOOL_INDEX_BASE + idx,
                                partial_json=fn.arguments,
                            )
                        )
        if getattr(choice, "finish_reason", None):
            self._finish_reason = choice.finish_reason
        return events

    def finalize(self) -> MessageDone:
        blocks: list[Any] = []
        if self._thinking:
            blocks.append(ThinkingBlock(thinking="".join(self._thinking)))
        if self._text:
            blocks.append(TextBlock(text="".join(self._text)))
        for idx in sorted(self._tool_calls):
            slot = self._tool_calls[idx]
            args_str = "".join(slot["args"]).strip()
            try:
                args = _parse_tool_arguments(args_str) if args_str else {}
            except ValueError:
                args = {"_raw": args_str}
            blocks.append(
                ToolUseBlock(
                    id=slot["id"] or f"call_{idx}",
                    name=slot["name"] or "unknown",
                    input=args if isinstance(args, dict) else {"_raw": args_str},
                )
            )

        usage = Usage()
        if self._usage is not None:
            prompt = getattr(self._usage, "prompt_tokens", 0) or 0
            completion = getattr(self._usage, "completion_tokens", 0) or 0
            cached = 0
            details = getattr(self._usage, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", 0) or 0
            if not cached:  # DeepSeek 命名
                cached = getattr(self._usage, "prompt_cache_hit_tokens", 0) or 0
            usage = Usage(
                input_tokens=max(prompt - cached, 0),
                output_tokens=completion,
                cache_read_input_tokens=cached,
            )

        raw = self._finish_reason or "stop"
        return MessageDone(
            message=Message(role="assistant", content=blocks),
            usage=usage,
            stop_reason=_FINISH_MAP.get(raw, raw),
        )


class OpenAICompatProvider(Provider):
    def __init__(self, name: str, api_key: str, base_url: str | None = None) -> None:
        self.name = name  # "openai" | "deepseek"
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]:
        kwargs: dict[str, Any] = {
            "model": req.model,
            "messages": to_openai_messages(req),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if req.tools:
            kwargs["tools"] = to_openai_tools(req.tools)
        if self.name == "openai":
            kwargs["max_completion_tokens"] = req.max_tokens
        else:
            kwargs["max_tokens"] = req.max_tokens
        reasoning = reasoning_effort_for(self.name, req.model, req.effort)
        if reasoning:
            kwargs["reasoning_effort"] = reasoning

        agg = ChunkAggregator()
        try:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                for ev in agg.feed(chunk):
                    yield ev
        except openai.APIStatusError as exc:
            raise ProviderError(
                f"{self.name} API 错误 (HTTP {exc.status_code}): {exc.message}"
            ) from exc
        except openai.APIConnectionError as exc:
            raise ProviderError(f"{self.name} 连接失败: {exc}") from exc
        yield agg.finalize()
