"""初始化向导：多选解析、逐家配置、角色指派、写回与协议分发。"""

import io
from pathlib import Path

import pytest
from rich.console import Console

from emiya_agent.config import load_config, user_config_path
from emiya_agent.providers.anthropic_provider import (
    AnthropicProvider,
    anthropic_endpoint_identity,
)
from emiya_agent.providers.registry import ProviderRegistry
from emiya_agent.ui.init_wizard import (
    build_catalog,
    parse_selection,
    run_init_wizard,
)


class FakeRepl:
    """向导所需的最小 repl 接口：console/read_line/config/providers/route。"""

    def __init__(self, config, answers: list[str]):
        self.console = Console(file=io.StringIO(), width=100)
        self.config = config
        self.providers = ProviderRegistry(config)
        self.route = config.route("executor")
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.secret_prompts: list[str] = []

    async def read_line(self, prompt: str = "") -> str:
        self.prompts.append(prompt)
        if not self.answers:
            return ""  # 回车兜底
        return self.answers.pop(0)

    async def read_secret(self, prompt: str = "") -> str:
        self.secret_prompts.append(prompt)
        if not self.answers:
            return ""
        return self.answers.pop(0)

    def output(self) -> str:
        return self.console.file.getvalue()


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """把用户配置目录隔离到 tmp，避免污染真实 ~/.emiyacode。"""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    return fake_home


# ------------------------------------------------------------------ 选择解析


def test_parse_selection_variants():
    catalog = build_catalog()
    assert parse_selection("", catalog) == []
    assert [e.name for e in parse_selection("1", catalog)] == ["deepseek"]
    assert [e.name for e in parse_selection("deepseek,glm", catalog)] == [
        "deepseek",
        "glm",
    ]
    assert [e.name for e in parse_selection("1，2", catalog)] == ["deepseek", "glm"]
    assert len(parse_selection("all", catalog)) == len(catalog)
    # 非法 token 忽略、去重
    assert [e.name for e in parse_selection("1,1,99,nope", catalog)] == ["deepseek"]


def test_catalog_deepseek_has_anthropic_alt():
    (deepseek,) = [e for e in build_catalog() if e.name == "deepseek"]
    assert deepseek.base_url == "https://api.deepseek.com"
    assert deepseek.example_models == ["deepseek-v4-flash", "deepseek-v4-pro"]
    (alt,) = deepseek.alt_urls
    assert alt[1] == "https://api.deepseek.com/anthropic"
    assert alt[2] == "anthropic"


# ------------------------------------------------------------------ 完整流程


@pytest.mark.asyncio
async def test_wizard_full_flow_deepseek(tmp_path, isolated_home, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config = load_config(tmp_path)
    repl = FakeRepl(
        config,
        answers=[
            "deepseek",  # 选供应商
            "",  # base_url 回车 = 默认
            "sk-ds-test",  # api key（写入配置）
            "deepseek-v4-flash, deepseek-v4-pro",  # 模型
            "",  # executor 回车 = 建议（第一个候选）
            "",  # translator
            "",  # compactor
            "",  # subagent_explore
            "",  # subagent_plan
            "",  # subagent_general
        ],
    )
    await run_init_wizard(repl)

    # 写回了用户配置
    saved = load_config(tmp_path)
    pc = saved.providers["deepseek"]
    assert pc.api_key == "sk-ds-test"
    assert pc.base_url == "https://api.deepseek.com"
    assert pc.protocol == "openai"
    assert "deepseek/deepseek-v4-flash" in saved.models
    assert "deepseek/deepseek-v4-pro" in saved.models
    # 全部角色指派到了新模型
    for role in saved.routes:
        assert saved.routes[role] == "deepseek/deepseek-v4-flash"
    # key 来自配置文件（无环境变量）
    assert saved.credential_source("deepseek") == "config.toml"
    # 会话内立即生效
    assert repl.route == "deepseek/deepseek-v4-flash"
    assert len(repl.secret_prompts) == 1
    assert "sk-ds-test" not in repl.output()


@pytest.mark.asyncio
async def test_wizard_anthropic_compat_endpoint(tmp_path, isolated_home, monkeypatch):
    """选 DeepSeek 的 Anthropic 兼容端点 → protocol=anthropic → AnthropicProvider。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config = load_config(tmp_path)
    repl = FakeRepl(
        config,
        answers=[
            "1",  # deepseek
            "2",  # 备选端点 [2] Anthropic 兼容
            "sk-ds-anthropic",
            "deepseek-v4-pro",
            # 角色全部回车
        ],
    )
    await run_init_wizard(repl)

    saved = load_config(tmp_path)
    pc = saved.providers["deepseek"]
    assert pc.base_url == "https://api.deepseek.com/anthropic"
    assert pc.protocol == "anthropic"

    provider, model = ProviderRegistry(saved).get("deepseek/deepseek-v4-pro")
    assert isinstance(provider, AnthropicProvider)
    assert provider.name == "deepseek"
    assert provider._endpoint_identity == anthropic_endpoint_identity(
        "deepseek", "https://api.deepseek.com/anthropic"
    )
    assert provider._official_endpoint is False
    assert model == "deepseek-v4-pro"


@pytest.mark.asyncio
async def test_wizard_skip_writes_template_on_first_run(tmp_path, isolated_home):
    config = load_config(tmp_path)
    repl = FakeRepl(config, answers=[""])  # 供应商选择直接回车
    await run_init_wizard(repl, first_run=True)

    path = user_config_path()
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "/init" in text  # 模板注释里有指引
    # 模板不改变实际配置
    assert load_config(tmp_path).routes["executor"] == "anthropic/claude-fable-5"


@pytest.mark.asyncio
async def test_wizard_role_assignment_by_number(tmp_path, isolated_home, monkeypatch):
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    config = load_config(tmp_path)
    repl = FakeRepl(
        config,
        answers=[
            "glm",
            "",  # base_url 默认
            "",  # 跳过 key（用环境变量）
            "glm-4.6, glm-5",
            "2",  # executor → 第 2 个候选 glm/glm-5
            "1",  # translator → glm/glm-4.6
            # 其余回车
        ],
    )
    await run_init_wizard(repl)
    saved = load_config(tmp_path)
    assert saved.routes["executor"] == "glm/glm-5"
    assert saved.routes["translator"] == "glm/glm-4.6"
    # 跳过 key → 提醒缺 key
    assert "GLM_API_KEY" in repl.output()


@pytest.mark.asyncio
async def test_wizard_custom_url_with_anthropic_protocol(
    tmp_path, isolated_home, monkeypatch
):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    config = load_config(tmp_path)
    repl = FakeRepl(
        config,
        answers=[
            "kimi",
            "https://api.moonshot.ai/anthropic",  # 自定义 URL
            "a",  # anthropic 协议
            "",  # 跳过 key
            "kimi-k2",
        ],
    )
    await run_init_wizard(repl)
    saved = load_config(tmp_path)
    assert saved.providers["kimi"].protocol == "anthropic"
    assert saved.providers["kimi"].base_url == "https://api.moonshot.ai/anthropic"
