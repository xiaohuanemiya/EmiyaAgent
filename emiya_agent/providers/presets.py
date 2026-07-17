"""常见 OpenAI 兼容供应商预设 —— /provider add <名称> 一步接入。

端点地址以各家官方文档为准（可能随时间变化）；预设只是省去查文档的默认值，
用户可用 /provider add <名称> <base_url> [KEY_ENV] 覆盖，或事后改配置文件。
模型 ID 请查对应供应商文档后用 /route 或 /model 指定（预设不预埋模型，
避免误导到已下线的模型名）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    base_url: str
    api_key_env: str
    note: str  # 展示给用户的一句话说明
    example_model: str  # 仅用于提示文案，不写入配置
    lang: str = "en"  # 语言基座：zh 模型免翻译直接 plan


PRESETS: dict[str, ProviderPreset] = {
    "minimax": ProviderPreset(
        name="minimax",
        base_url="https://api.minimaxi.com/v1",
        api_key_env="MINIMAX_API_KEY",
        note="MiniMax（国内端点；海外用 https://api.minimax.io/v1）",
        example_model="MiniMax-M2",
        lang="zh",
    ),
    "kimi": ProviderPreset(
        name="kimi",
        base_url="https://api.moonshot.cn/v1",
        api_key_env="MOONSHOT_API_KEY",
        note="月之暗面 Kimi（国内端点；海外用 https://api.moonshot.ai/v1）",
        example_model="kimi-k2",
        lang="zh",
    ),
    "qwen": ProviderPreset(
        name="qwen",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        note="阿里通义千问（DashScope OpenAI 兼容模式）",
        example_model="qwen3-max",
        lang="zh",
    ),
    "openrouter": ProviderPreset(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        note="OpenRouter 聚合网关（可在 [models] 里按模型覆盖 lang）",
        example_model="anthropic/claude-opus-4.8",
        lang="en",
    ),
    "siliconflow": ProviderPreset(
        name="siliconflow",
        base_url="https://api.siliconflow.cn/v1",
        api_key_env="SILICONFLOW_API_KEY",
        note="硅基流动（国产模型聚合）",
        example_model="deepseek-ai/DeepSeek-V3.2",
        lang="zh",
    ),
    "ollama": ProviderPreset(
        name="ollama",
        base_url="http://localhost:11434/v1",
        api_key_env="OLLAMA_API_KEY",
        note="本地 Ollama（不校验 key，但环境变量需设任意非空值，如 ollama）",
        example_model="qwen3:32b",
        lang="zh",
    ),
}
