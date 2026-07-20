"""不进入工具循环的单次模型补全。"""

from __future__ import annotations

from emiya_agent.agent.control import RunController
from emiya_agent.providers.base import ChatRequest, MessageDone, Provider
from emiya_agent.types import Usage, user_message


async def complete_text(
    provider: Provider,
    model: str,
    system: str,
    text: str,
    max_tokens: int = 16000,
    *,
    route: str | None = None,
    controller: RunController | None = None,
) -> tuple[str, Usage]:
    """消费一次模型流并返回完整文本与权威 usage。"""
    if controller is not None:
        if route is None:
            raise ValueError("route is required when a run controller is used")
        await controller.before_request(route)
    request = ChatRequest(
        model=model,
        system=system,
        messages=[user_message(text)],
        max_tokens=max_tokens,
    )
    done: MessageDone | None = None
    async for event in provider.stream(request):
        if isinstance(event, MessageDone):
            done = event
    if done is None:
        raise RuntimeError("provider stream ended without MessageDone")
    if controller is not None:
        assert route is not None
        controller.record_usage(route, done.usage)
    return done.message.text(), done.usage
