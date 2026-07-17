"""需求管线状态机：approve / edit / revise / reject / 直通 / native 免译 各路径。"""

import pytest

from emiya_agent.pipeline.prompts import REPORT_SYSTEM, SPEC_SYSTEM, SPEC_SYSTEM_NATIVE
from emiya_agent.pipeline.translation import (
    ConfirmResult,
    Decision,
    PipeState,
    SpecDraft,
    TranslationPipeline,
    contains_cjk,
    parse_native_spec,
    parse_spec_output,
)

RAW_SPEC = """\
<english_spec>
## Goal
Add a login button.
</english_spec>
<chinese_mirror>
## 目标
添加登录按钮。
</chinese_mirror>
"""

RAW_NATIVE_SPEC = """\
<spec>
## 目标
添加登录按钮（中文规格）。
</spec>
"""


class FakeTranslator:
    """按调用顺序记录 (system, text)；按 system 类型返回对应格式。"""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, system: str, text: str) -> str:
        self.calls.append((system, text))
        if system == SPEC_SYSTEM:
            return RAW_SPEC
        if system == SPEC_SYSTEM_NATIVE:
            return RAW_NATIVE_SPEC
        return f"[中文翻译] {text}"


class ScriptedConfirmer:
    def __init__(self, *results: ConfirmResult):
        self.queue = list(results)
        self.seen: list[SpecDraft] = []

    async def __call__(self, draft: SpecDraft) -> ConfirmResult:
        self.seen.append(draft)
        return self.queue.pop(0)


async def fake_executor(spec_en: str) -> str:
    return f"DONE: {spec_en.splitlines()[0]}"


def make_pipeline(confirmer, translator=None, **kw):
    return TranslationPipeline(
        translator or FakeTranslator(), confirmer, fake_executor, **kw
    )


# ------------------------------------------------------------------ 工具函数


def test_contains_cjk():
    assert contains_cjk("帮我写个爬虫")
    assert contains_cjk("fix bug in 配置")
    assert not contains_cjk("fix the login bug")


def test_parse_spec_output_with_tags():
    draft = parse_spec_output(RAW_SPEC)
    assert draft.spec.startswith("## Goal")
    assert "登录按钮" in draft.mirror


def test_parse_spec_output_missing_tags_falls_back():
    draft = parse_spec_output("plain text spec")
    assert draft.spec == "plain text spec"
    assert draft.mirror == "plain text spec"


def test_parse_native_spec():
    draft = parse_native_spec(RAW_NATIVE_SPEC)
    assert draft.spec.startswith("## 目标")
    assert draft.mirror is None
    # 标签缺失兜底
    fallback = parse_native_spec("裸文本规格")
    assert fallback.spec == "裸文本规格"
    assert fallback.mirror is None


# ------------------------------------------------------------------ 状态路径


@pytest.mark.asyncio
async def test_approve_path():
    translator = FakeTranslator()
    confirmer = ScriptedConfirmer(ConfirmResult(decision=Decision.APPROVE))
    p = make_pipeline(confirmer, translator)
    result = await p.run("加一个登录按钮")

    assert result.state is PipeState.DONE
    assert result.report_en.startswith("DONE:")
    assert result.report_zh.startswith("[中文翻译]")
    assert p.history == [
        PipeState.INTAKE,
        PipeState.SPEC_DRAFT,
        PipeState.CONFIRM,
        PipeState.EXECUTE,
        PipeState.REPORT,
        PipeState.DONE,
    ]
    # 两次 DeepSeek 调用：起草 + 回译
    assert [s for s, _ in translator.calls] == [SPEC_SYSTEM, REPORT_SYSTEM]


@pytest.mark.asyncio
async def test_english_input_bypasses_pipeline():
    translator = FakeTranslator()
    confirmer = ScriptedConfirmer()  # 不应被调用
    p = make_pipeline(confirmer, translator)
    result = await p.run("fix the login bug")

    assert result.state is PipeState.DONE
    assert result.report_en.startswith("DONE:")
    assert result.report_zh == ""
    assert translator.calls == []
    assert confirmer.seen == []
    assert PipeState.SPEC_DRAFT not in p.history


@pytest.mark.asyncio
async def test_disabled_pipeline_bypasses_even_chinese():
    p = make_pipeline(ScriptedConfirmer(), enabled=False)
    result = await p.run("中文需求")
    assert result.state is PipeState.DONE
    assert PipeState.CONFIRM not in p.history


@pytest.mark.asyncio
async def test_reject_aborts_without_execute():
    confirmer = ScriptedConfirmer(ConfirmResult(decision=Decision.REJECT))
    p = make_pipeline(confirmer)
    result = await p.run("加一个登录按钮")

    assert result.aborted
    assert result.state is PipeState.ABORTED
    assert PipeState.EXECUTE not in p.history
    assert result.spec is not None  # 保留草稿供查看


@pytest.mark.asyncio
async def test_edit_then_approve_uses_edited_spec():
    confirmer = ScriptedConfirmer(
        ConfirmResult(decision=Decision.EDIT_EN, text="## Goal\nEdited spec."),
        ConfirmResult(decision=Decision.APPROVE),
    )
    p = make_pipeline(confirmer)
    result = await p.run("加一个登录按钮")

    assert result.state is PipeState.DONE
    assert result.spec.spec == "## Goal\nEdited spec."
    assert result.report_en == "DONE: ## Goal"
    # 编辑不触发重新起草（无第二次 SPEC_DRAFT）
    assert p.history.count(PipeState.SPEC_DRAFT) == 1
    assert p.history.count(PipeState.CONFIRM) == 2


@pytest.mark.asyncio
async def test_revise_redrafts_with_feedback():
    translator = FakeTranslator()
    confirmer = ScriptedConfirmer(
        ConfirmResult(decision=Decision.REVISE_ZH, text="要支持手机号登录"),
        ConfirmResult(decision=Decision.APPROVE),
    )
    p = make_pipeline(confirmer, translator)
    result = await p.run("加一个登录按钮")

    assert result.state is PipeState.DONE
    assert p.history.count(PipeState.SPEC_DRAFT) == 2
    # 第二次起草的 prompt 应包含反馈
    assert "要支持手机号登录" in translator.calls[1][1]


@pytest.mark.asyncio
async def test_max_revisions_guard():
    endless_edit = [
        ConfirmResult(decision=Decision.EDIT_EN, text=f"spec v{i}") for i in range(10)
    ]
    confirmer = ScriptedConfirmer(*endless_edit)
    p = make_pipeline(confirmer, max_revisions=3)
    result = await p.run("加一个登录按钮")
    assert result.aborted


@pytest.mark.asyncio
async def test_empty_report_skips_back_translation():
    async def failing_executor(spec_en: str) -> str:
        return ""  # 模拟执行被中断

    translator = FakeTranslator()
    p = TranslationPipeline(
        translator,
        ScriptedConfirmer(ConfirmResult(decision=Decision.APPROVE)),
        failing_executor,
    )
    result = await p.run("加一个登录按钮")
    assert result.state is PipeState.DONE
    assert result.report_zh == ""
    # 只有起草一次调用，没有回译调用
    assert [s for s, _ in translator.calls] == [SPEC_SYSTEM]


# ------------------------------------------------------------------ native 模式（同语免译）


@pytest.mark.asyncio
async def test_native_mode_skips_translation_entirely():
    """中文输入 + 中文基座执行模型：中文 spec 直接 plan，无回译。"""
    translator = FakeTranslator()
    executed: list[str] = []

    async def executor(spec: str) -> str:
        executed.append(spec)
        return "已完成：登录按钮添加完毕。"

    confirmer = ScriptedConfirmer(ConfirmResult(decision=Decision.APPROVE))
    p = TranslationPipeline(translator, confirmer, executor, executor_lang="zh")
    result = await p.run("加一个登录按钮")

    assert result.state is PipeState.DONE
    # 执行模型收到的是中文规格
    assert executed == ["## 目标\n添加登录按钮（中文规格）。"]
    # 确认界面拿到的草稿无对照镜像（单栏）
    assert confirmer.seen[0].mirror is None
    # 全程只调了一次 DeepSeek（起草），且用的是 native 提示词；无回译
    assert [s for s, _ in translator.calls] == [SPEC_SYSTEM_NATIVE]
    assert result.report_zh == ""
    # REPORT 状态被跳过
    assert PipeState.REPORT not in p.history
    assert p.history == [
        PipeState.INTAKE,
        PipeState.SPEC_DRAFT,
        PipeState.CONFIRM,
        PipeState.EXECUTE,
        PipeState.DONE,
    ]


@pytest.mark.asyncio
async def test_native_mode_revise_uses_native_prompt():
    translator = FakeTranslator()
    confirmer = ScriptedConfirmer(
        ConfirmResult(decision=Decision.REVISE_ZH, text="要支持手机号登录"),
        ConfirmResult(decision=Decision.APPROVE),
    )
    p = TranslationPipeline(translator, confirmer, fake_executor, executor_lang="zh")
    result = await p.run("加一个登录按钮")

    assert result.state is PipeState.DONE
    assert [s for s, _ in translator.calls] == [SPEC_SYSTEM_NATIVE, SPEC_SYSTEM_NATIVE]
    assert "要支持手机号登录" in translator.calls[1][1]


@pytest.mark.asyncio
async def test_translate_mode_still_default():
    """默认 executor_lang=en 保持原有翻译行为。"""
    p = make_pipeline(ScriptedConfirmer(ConfirmResult(decision=Decision.APPROVE)))
    assert p.translate_mode is True
    result = await p.run("加一个登录按钮")
    assert PipeState.REPORT in p.history
    assert result.report_zh.startswith("[中文翻译]")
