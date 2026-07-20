"""命令行入口：交互 REPL 与可机器解析的一次性执行。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from emiya_agent import __version__
from emiya_agent.agent.control import (
    ExitCode,
    RunController,
    RunLimits,
    exit_code_for_reason,
)
from emiya_agent.config import load_config
from emiya_agent.errors import ConfigError, EmiyaError, ProviderError
from emiya_agent.ui.json_output import JsonEventSink, final_payload


def _harden_stdio() -> None:
    """管道统一 UTF-8；任何编码错误都不应击穿 CLI。"""

    for stream in (sys.stdout, sys.stderr):
        try:
            if not stream.isatty():
                stream.reconfigure(encoding="utf-8", errors="replace")
            else:
                stream.reconfigure(errors="replace")
        except Exception:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emiya",
        description="EmiyaAgent — 多智能体 AI 编程助手",
        epilog="子命令: emiya doctor [--json]  运行脱敏自检",
    )
    parser.add_argument("--version", action="version", version=f"emiya {__version__}")
    parser.add_argument(
        "--cwd", type=Path, default=None, help="工作目录（默认当前目录）"
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default=None,
        help="一次性执行模式：执行该提示词后退出（默认仍执行工具审批）",
    )
    parser.add_argument(
        "--approval",
        choices=("ask", "auto"),
        default=None,
        help="覆盖本次运行的工具审批；auto 仅用于可信任务",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--json",
        action="store_true",
        help="向 stdout 输出单个最终 JSON 对象（仅与 -p 一起使用）",
    )
    output.add_argument(
        "--stream-json",
        action="store_true",
        help="向 stdout 输出逐行 NDJSON 事件（仅与 -p 一起使用）",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="本次运行所有模型请求的全局轮次上限",
    )
    parser.add_argument(
        "--timeout",
        dest="timeout_seconds",
        type=float,
        default=None,
        help="本次运行的总超时秒数",
    )
    parser.add_argument(
        "--max-budget-usd",
        type=Decimal,
        default=None,
        help="按配置模型单价估算的总费用上限（USD）",
    )
    return parser


def _exit_for_result(repl, *, timed_out: bool) -> ExitCode:
    if timed_out:
        return ExitCode.TIMEOUT
    if repl.last_error_kind == "config":
        return ExitCode.CONFIG_ERROR
    if repl.last_error_kind == "provider":
        return ExitCode.PROVIDER_ERROR
    result = repl.last_result
    if result is None:
        return ExitCode.EXECUTION_ERROR
    reason = result.stop_reason
    if reason in {"max_turns", "budget_exceeded", "budget_unavailable", "timeout"}:
        return exit_code_for_reason(reason)
    return {
        "end_turn": ExitCode.SUCCESS,
        "refusal": ExitCode.REFUSAL,
        "max_iterations": ExitCode.MAX_TURNS,
        "max_tokens": ExitCode.MAX_TOKENS,
        "error": ExitCode.PROTOCOL_ERROR,
        "interrupted": ExitCode.INTERRUPTED,
    }.get(reason, ExitCode.EXECUTION_ERROR)


def _error_text(repl, code: ExitCode, *, timed_out: bool) -> str | None:
    if repl.last_error:
        return repl.last_error
    if timed_out:
        return "run timed out"
    return {
        ExitCode.MAX_TURNS: "model request limit reached",
        ExitCode.BUDGET_EXCEEDED: "USD budget reached",
        ExitCode.CONFIG_ERROR: (
            "configuration error or model pricing unavailable"
        ),
        ExitCode.MAX_TOKENS: "model output reached max_tokens",
        ExitCode.PROTOCOL_ERROR: "model returned an unsafe protocol state",
        ExitCode.INTERRUPTED: "run interrupted",
    }.get(code)


def _build_final(repl, controller: RunController, code: ExitCode, timed_out: bool):
    result = repl.last_result
    reason = (
        "timeout"
        if timed_out
        else result.stop_reason
        if result
        else {
            ExitCode.INTERRUPTED: "interrupted",
            ExitCode.CONFIG_ERROR: "config_error",
            ExitCode.PROVIDER_ERROR: "provider_error",
            ExitCode.EXECUTION_ERROR: "execution_error",
        }.get(code)
    )
    route = repl.route
    model = route.split("/", 1)[1] if "/" in route else route
    return final_payload(
        status="success" if code is ExitCode.SUCCESS else "error",
        stop_reason=reason,
        exit_code=int(code),
        output=result.final_text if result else "",
        route=route,
        model=model,
        served_by=result.served_by if result else None,
        refusal=result.refusal if result else None,
        usage=controller.usage,
        cost_usd=controller.cost,
        turns=controller.turns,
        elapsed_seconds=controller.elapsed_seconds,
        session_id=repl.session_id,
        error=_error_text(repl, code, timed_out=timed_out),
    )


def _write_final(
    payload: dict,
    *,
    output_mode: str,
    sink: JsonEventSink | None,
) -> None:
    if output_mode == "json":
        sys.stdout.write(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        sys.stdout.flush()
    elif output_mode == "stream-json":
        assert sink is not None
        body = {
            key: value
            for key, value in payload.items()
            if key not in {"schema_version", "type"}
        }
        sink.emit("final", **body)


async def _one_shot(repl, prompt: str, controller: RunController) -> tuple[bool, bool]:
    timeout = controller.remaining_timeout
    if timeout is None:
        return await repl.one_shot(prompt), False
    try:
        ok = await asyncio.wait_for(repl.one_shot(prompt), timeout=timeout)
    except TimeoutError:
        return False, True
    timed_out = controller.remaining_timeout == 0
    return ok, timed_out


def _run_doctor(argv: Sequence[str]) -> int:
    """`emiya doctor [--json] [--cwd …]`：输出脱敏诊断报告。

    退出码：0 = 无 fail 级检查项；3 = 存在 fail 级检查项或配置无法加载。
    """
    parser = argparse.ArgumentParser(
        prog="emiya doctor",
        description="脱敏自检：配置、凭据来源、路由、Hook/MCP/LSP、会话（绝不输出凭据值）",
    )
    parser.add_argument("--cwd", type=Path, default=None, help="工作目录")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON 报告")
    args = parser.parse_args(argv)

    from emiya_agent.diagnostics import build_report, render_report

    try:
        config = load_config(args.cwd)
    except EmiyaError as exc:
        if args.json:
            sys.stdout.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "fail",
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
        else:
            print(f"配置错误: {exc}", file=sys.stderr)
        return int(ExitCode.CONFIG_ERROR)
    report = build_report(config)
    if args.json:
        sys.stdout.write(
            json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
    else:
        print(render_report(report))
    return int(
        ExitCode.CONFIG_ERROR if report["status"] == "fail" else ExitCode.SUCCESS
    )


def run(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a stable process exit code."""

    _harden_stdio()
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "doctor":
        return _run_doctor(list(argv[1:]))
    parser = _parser()
    args = parser.parse_args(argv)
    one_shot_only = any(
        (
            args.json,
            args.stream_json,
            args.max_turns is not None,
            args.timeout_seconds is not None,
            args.max_budget_usd is not None,
        )
    )
    if one_shot_only and args.prompt is None:
        parser.error(
            "--json/--stream-json and run limits require -p/--prompt"
        )
    try:
        limits = RunLimits(
            max_turns=args.max_turns,
            timeout_seconds=args.timeout_seconds,
            max_budget_usd=args.max_budget_usd,
        )
    except ValidationError as exc:
        parser.error(str(exc))

    output_mode = (
        "stream-json" if args.stream_json else "json" if args.json else "human"
    )
    sink = JsonEventSink(sys.stdout) if args.stream_json else None

    try:
        config = load_config(args.cwd)
    except EmiyaError as exc:
        payload = final_payload(
            status="error",
            stop_reason="config_error",
            exit_code=int(ExitCode.CONFIG_ERROR),
            error=str(exc),
        )
        _write_final(payload, output_mode=output_mode, sink=sink)
        if output_mode == "human":
            print(f"配置错误: {exc}", file=sys.stderr)
        return int(ExitCode.CONFIG_ERROR)
    except Exception as exc:
        if output_mode == "human":
            raise
        payload = final_payload(
            status="error",
            stop_reason="execution_error",
            exit_code=int(ExitCode.EXECUTION_ERROR),
            error=f"{type(exc).__name__}: {exc}",
        )
        _write_final(payload, output_mode=output_mode, sink=sink)
        return int(ExitCode.EXECUTION_ERROR)

    if args.approval is not None:
        config.tool_approval = args.approval
    if sink is not None:
        sink.emit(
            "run_start",
            cwd=str(config.cwd),
            route=config.route("executor"),
            limits=limits,
        )
    if args.prompt is not None:
        for warning in config.load_warnings:
            print(f"配置警告: {warning}", file=sys.stderr)

    from emiya_agent.ui.repl import Repl

    if args.prompt is None:
        try:
            asyncio.run(Repl(config).run())
            return int(ExitCode.SUCCESS)
        except KeyboardInterrupt:
            return int(ExitCode.INTERRUPTED)

    controller = RunController(config, limits)
    try:
        repl = Repl(
            config,
            automation=True,
            output_mode=output_mode,
            event_sink=sink,
            controller=controller,
        )
    except (ConfigError, ProviderError) as exc:
        code = (
            ExitCode.CONFIG_ERROR
            if isinstance(exc, ConfigError)
            else ExitCode.PROVIDER_ERROR
        )
        payload = final_payload(
            status="error",
            stop_reason=("config_error" if isinstance(exc, ConfigError) else "provider_error"),
            exit_code=int(code),
            route=config.routes.get("executor"),
            error=str(exc),
        )
        _write_final(payload, output_mode=output_mode, sink=sink)
        if output_mode == "human":
            print(str(exc), file=sys.stderr)
        return int(code)
    except Exception as exc:
        if output_mode == "human":
            raise
        payload = final_payload(
            status="error",
            stop_reason="execution_error",
            exit_code=int(ExitCode.EXECUTION_ERROR),
            route=config.routes.get("executor"),
            error=f"{type(exc).__name__}: {exc}",
        )
        _write_final(payload, output_mode=output_mode, sink=sink)
        return int(ExitCode.EXECUTION_ERROR)
    timed_out = False
    try:
        _, timed_out = asyncio.run(_one_shot(repl, args.prompt, controller))
    except KeyboardInterrupt:
        repl.last_result = None
        repl.last_error = "run interrupted"
        code = ExitCode.INTERRUPTED
    except (ConfigError, ProviderError) as exc:
        repl.last_error = str(exc)
        repl.last_error_kind = "config" if isinstance(exc, ConfigError) else "provider"
        code = (
            ExitCode.CONFIG_ERROR
            if isinstance(exc, ConfigError)
            else ExitCode.PROVIDER_ERROR
        )
    except Exception as exc:
        repl.last_error = f"{type(exc).__name__}: {exc}"
        repl.last_error_kind = "execution"
        code = ExitCode.EXECUTION_ERROR
    else:
        code = _exit_for_result(repl, timed_out=timed_out)

    payload = _build_final(repl, controller, code, timed_out)
    _write_final(payload, output_mode=output_mode, sink=sink)
    return int(code)


def main() -> None:
    code = run()
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
