"""需求管线状态机：INTAKE → SPEC_DRAFT → CONFIRM → EXECUTE → [REPORT] → DONE。

语言感知：
- 中文输入 + 英文基座执行模型（Claude/GPT）→ translate 模式：
  英文 spec + 中文对照确认 + 执行 + 报告回译（REPORT）
- 中文输入 + 中文基座执行模型（DeepSeek/GLM/MiniMax…）→ native 模式：
  中文 spec 直接 plan 确认 + 执行，全程免翻译（跳过 REPORT）
- 非中文输入 → 直通执行

依赖全部注入（translator/confirm/executor 均为可替换 callable），
状态转移逻辑与网络、UI 完全解耦，可用假实现做单测。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Awaitable, Callable

from emiya_agent.pipeline.prompts import (
    REPORT_SYSTEM,
    SPEC_SYSTEM,
    SPEC_SYSTEM_NATIVE,
    spec_user_prompt,
)
from emiya_agent.providers.base import ChatRequest, MessageDone, Provider
from emiya_agent.types import Usage, user_message


class PipeState(Enum):
    INTAKE = auto()
    SPEC_DRAFT = auto()
    CONFIRM = auto()
    EXECUTE = auto()
    REPORT = auto()
    DONE = auto()
    ABORTED = auto()


class Decision(Enum):
    APPROVE = auto()  # 批准，执行英文 spec
    EDIT_EN = auto()  # 用户直接编辑了英文 spec，回到确认
    REVISE_ZH = auto()  # 用户给中文反馈，重新起草
    REJECT = auto()  # 放弃


@dataclass
class ConfirmResult:
    decision: Decision
    text: str = ""  # EDIT_EN: 编辑后的英文；REVISE_ZH: 中文反馈


@dataclass
class SpecDraft:
    spec: str  # 交给执行模型的规格（translate 模式为英文，native 模式为中文）
    mirror: str | None = None  # 用户确认用的对照译文；同语言时为 None（单栏展示）


@dataclass
class PipelineResult:
    state: PipeState
    spec: SpecDraft | None = None
    report_en: str = ""
    report_zh: str = ""

    @property
    def aborted(self) -> bool:
        return self.state is PipeState.ABORTED


# 注入点类型
Translator = Callable[[str, str], Awaitable[str]]  # (system, user_text) -> 完整回复文本
Confirmer = Callable[[SpecDraft], Awaitable[ConfirmResult]]
Executor = Callable[[str], Awaitable[str]]  # 英文输入 -> 最终英文回答


_SPEC_RE = re.compile(r"<english_spec>\s*(.*?)\s*</english_spec>", re.DOTALL)
_MIRROR_RE = re.compile(r"<chinese_mirror>\s*(.*?)\s*</chinese_mirror>", re.DOTALL)
_NATIVE_RE = re.compile(r"<spec>\s*(.*?)\s*</spec>", re.DOTALL)


def parse_spec_output(raw: str) -> SpecDraft:
    """translate 模式解析；标签缺失时整体视为英文 spec（对照退化为原文）。"""
    en = _SPEC_RE.search(raw)
    zh = _MIRROR_RE.search(raw)
    if en:
        return SpecDraft(spec=en.group(1), mirror=zh.group(1) if zh else raw)
    return SpecDraft(spec=raw.strip(), mirror=raw.strip())


def parse_native_spec(raw: str) -> SpecDraft:
    """native 模式解析：同语言规格，无对照镜像。"""
    m = _NATIVE_RE.search(raw)
    return SpecDraft(spec=m.group(1) if m else raw.strip(), mirror=None)


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


async def complete_text(
    provider: Provider, model: str, system: str, text: str, max_tokens: int = 16000
) -> tuple[str, Usage]:
    """非交互补全：消费流并返回 (完整文本, usage)。"""
    req = ChatRequest(
        model=model,
        system=system,
        messages=[user_message(text)],
        max_tokens=max_tokens,
    )
    done: MessageDone | None = None
    async for ev in provider.stream(req):
        if isinstance(ev, MessageDone):
            done = ev
    assert done is not None
    return done.message.text(), done.usage


class TranslationPipeline:
    """max_revisions 防御反复重draft的死循环。

    executor_lang: 执行模型的语言基座（config.model_lang(executor_route)）。
    "zh" → native 模式（免翻译直接 plan）；其他 → translate 模式。
    """

    def __init__(
        self,
        translator: Translator,
        confirmer: Confirmer,
        executor: Executor,
        *,
        enabled: bool = True,
        max_revisions: int = 5,
        executor_lang: str = "en",
    ) -> None:
        self.translator = translator
        self.confirmer = confirmer
        self.executor = executor
        self.enabled = enabled
        self.max_revisions = max_revisions
        self.executor_lang = executor_lang
        self.state = PipeState.INTAKE
        self.history: list[PipeState] = []

    def _goto(self, state: PipeState) -> None:
        self.state = state
        self.history.append(state)

    def should_plan(self, text: str) -> bool:
        return self.enabled and contains_cjk(text)

    @property
    def translate_mode(self) -> bool:
        """跨语言（中文输入 → 英文基座模型）才翻译；同语言只做 plan 扩写。"""
        return self.executor_lang != "zh"

    async def run(self, user_input: str) -> PipelineResult:
        self._goto(PipeState.INTAKE)

        if not self.should_plan(user_input):
            self._goto(PipeState.EXECUTE)
            report = await self.executor(user_input)
            self._goto(PipeState.DONE)
            return PipelineResult(state=self.state, report_en=report)

        translate = self.translate_mode
        spec_system = SPEC_SYSTEM if translate else SPEC_SYSTEM_NATIVE
        parse = parse_spec_output if translate else parse_native_spec

        # 起草 + 确认循环
        self._goto(PipeState.SPEC_DRAFT)
        raw = await self.translator(spec_system, spec_user_prompt(user_input))
        draft = parse(raw)

        revisions = 0
        while True:
            self._goto(PipeState.CONFIRM)
            result = await self.confirmer(draft)
            if result.decision is Decision.APPROVE:
                break
            if result.decision is Decision.REJECT:
                self._goto(PipeState.ABORTED)
                return PipelineResult(state=self.state, spec=draft)
            revisions += 1
            if revisions > self.max_revisions:
                self._goto(PipeState.ABORTED)
                return PipelineResult(state=self.state, spec=draft)
            if result.decision is Decision.EDIT_EN:
                draft = SpecDraft(spec=result.text, mirror=draft.mirror)
                continue  # 编辑后回到确认
            # REVISE_ZH：带反馈重新起草
            self._goto(PipeState.SPEC_DRAFT)
            raw = await self.translator(
                spec_system, spec_user_prompt(user_input, feedback_zh=result.text)
            )
            draft = parse(raw)

        self._goto(PipeState.EXECUTE)
        report_en = await self.executor(draft.spec)

        report_zh = ""
        if translate:
            # native 模式免回译（回答已是中文，直接流式展示过了）
            self._goto(PipeState.REPORT)
            # 执行被中断/失败时 report_en 为空，跳过回译
            if report_en.strip():
                report_zh = await self.translator(REPORT_SYSTEM, report_en)

        self._goto(PipeState.DONE)
        return PipelineResult(
            state=self.state, spec=draft, report_en=report_en, report_zh=report_zh
        )
