"""会话级 token / 费用统计（数据来源：provider 返回的 usage 字段，权威且免费）。"""

from __future__ import annotations

from emiya_agent.config import Config
from emiya_agent.types import Usage


class UsageTracker:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._per_route: dict[str, Usage] = {}
        self.last_route: str | None = None
        self.last_usage: Usage = Usage()

    def add(self, route: str, usage: Usage) -> None:
        self._per_route[route] = self._per_route.get(route, Usage()) + usage
        self.last_route = route
        self.last_usage = usage

    def total(self) -> Usage:
        total = Usage()
        for u in self._per_route.values():
            total = total + u
        return total

    def cost(self) -> float:
        """按配置单价估算累计费用（USD）。"""
        cost = 0.0
        for route, u in self._per_route.items():
            info = self._config.model_info(route)
            cost += u.input_tokens / 1e6 * info.input_price
            cost += u.output_tokens / 1e6 * info.output_price
            cost += u.cache_read_input_tokens / 1e6 * info.cache_read_price
            # 缓存写按输入价 1.25 倍近似
            cost += u.cache_creation_input_tokens / 1e6 * info.input_price * 1.25
        return cost

    def footer(self) -> str:
        u = self.last_usage
        parts = [
            f"↑{u.input_tokens}",
            f"↓{u.output_tokens}",
        ]
        if u.cache_read_input_tokens:
            parts.append(f"缓存读 {u.cache_read_input_tokens}")
        if u.cache_creation_input_tokens:
            parts.append(f"缓存写 {u.cache_creation_input_tokens}")
        total = self.total()
        parts.append(f"累计 {total.total_prompt_tokens + total.output_tokens} tok")
        cost = self.cost()
        if cost > 0:
            parts.append(f"${cost:.4f}")
        return " · ".join(parts)
