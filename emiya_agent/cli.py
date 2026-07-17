"""命令行入口：加载配置 → 启动 REPL。"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from emiya_agent import __version__
from emiya_agent.config import load_config
from emiya_agent.errors import EmiyaError


def _harden_stdio() -> None:
    """Windows 控制台编码防护：
    - 管道/重定向场景强制 UTF-8（否则默认 GBK）；
    - 任何场景下编码失败降级为替换符而非崩溃。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            if not stream.isatty():
                stream.reconfigure(encoding="utf-8", errors="replace")
            else:
                stream.reconfigure(errors="replace")
        except Exception:
            pass


def main() -> None:
    _harden_stdio()
    parser = argparse.ArgumentParser(
        prog="emiya", description="EmiyaAgent — 多智能体 AI 编程助手"
    )
    parser.add_argument("--version", action="version", version=f"emiya {__version__}")
    parser.add_argument(
        "--cwd", type=Path, default=None, help="工作目录（默认当前目录）"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.cwd)
    except EmiyaError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        raise SystemExit(1)

    # Windows: 保持默认 Proactor 事件循环（子进程/MCP stdio 依赖它），勿切换
    from emiya_agent.ui.repl import Repl

    try:
        asyncio.run(Repl(config).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
