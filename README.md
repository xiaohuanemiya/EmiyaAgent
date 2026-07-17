# EmiyaAgent

多智能体 AI 编程助手（终端 REPL）。核心特性：**语言感知的需求管线**——

- 执行模型是**英文基座**（Claude/GPT）：中文需求 → DeepSeek 扩写英文详细 spec（plan 形式）→ 中英对照确认 → 执行 → 报告回译中文
- 执行模型是**中文基座**（DeepSeek/GLM/MiniMax/Kimi/Qwen…）：**免翻译**，中文需求直接扩写为中文 spec → 单栏 plan 确认 → 执行，全程零翻译开销

同语言的模型间不翻译，跨语言才翻译；语言基座在 provider 级标注（model 级可覆盖，聚合网关适用）。

功能一览：

- **需求管线（plan 模式）**：中文输入自动扩写为结构化 spec（目标/需求/假设/验收标准），确认（批准/编辑/中文反馈重写/放弃）后执行；执行模型为英文基座时自动加翻译层（英文 spec + 中英对照 + 报告回译），中文基座时免翻译
- **Agent 工具循环**：shell（PowerShell）、文件读写/精确编辑、glob/grep 搜索、todo 进度、中途提问；并行工具调用；Ctrl+C 干净中断
- **多智能体**：`spawn_subagent` 派生隔离上下文的子 agent —— explore（DeepSeek 只读搜索）/ plan（opus 方案设计）/ general（完整工具）
- **Skills**：`.emiyacode/skills/<名>/SKILL.md`（用户级+项目级），描述进系统提示词，内容按需加载
- **项目记忆**：`.emiyacode/memory/`（MEMORY.md 索引 + 一事一文件），跨会话持久
- **上下文压缩**：超过执行模型窗口 70% 自动用 DeepSeek 摘要旧轮次；`/compact` 手动触发
- **MCP 接入**：stdio / streamable-HTTP 服务器，工具以 `mcp__服务器__工具` 注册
- **会话持久化**：JSONL 转录，`/resume` 恢复

## 快速开始

```powershell
uv sync
uv run emiya
```

**首次运行会自动弹出初始化向导**：选择要配置的供应商（可多选、可跳过）→ 每家依次
设置 base_url（带默认值与备选端点，如 DeepSeek 的 Anthropic 兼容端点）、API key、
模型名 → 给每个智能体角色指派模型 → 自动写入 `~/.emiyacode/config.toml`。
跳过则生成带注释的配置模板；`/init` 可随时重跑向导。

API key 两种给法（环境变量优先）：

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."     # 方式一：环境变量
# 方式二：向导中直接输入，明文写入用户配置（注意文件保管）
```

直接输入中文需求即走需求管线；`/lang en` 切换直通模式。

## 多供应商路由（纯 API）

内置四家 provider：**anthropic（Claude）/ openai（GPT）/ deepseek / glm（智谱）**，
全部走各家官方 API 直连。**每一个智能体角色都可以独立路由到任意一家**：

| 角色 | 职责 | 默认路由 |
|---|---|---|
| `executor` | 主执行 agent（工具循环） | `anthropic/claude-fable-5` |
| `translator` | 中文需求扩写 / 报告回译 | `deepseek/deepseek-chat` |
| `compactor` | 上下文压缩摘要 | `deepseek/deepseek-chat` |
| `subagent_explore` | 只读搜索子 agent | `deepseek/deepseek-chat` |
| `subagent_plan` | 方案设计子 agent | `anthropic/claude-opus-4-8` |
| `subagent_general` | 通用执行子 agent | `anthropic/claude-opus-4-8` |

运行时切换：`/route`（列出全部）、`/route translator glm/glm-4.6`、
`/route executor openai/gpt-5.6`；`/model` 是 `/route executor` 的快捷方式。
持久化写进配置文件 `[routes]` 即可。

Anthropic 凭据除 `ANTHROPIC_API_KEY` 外也支持 `ant auth login` 账号 OAuth /
`ANTHROPIC_AUTH_TOKEN`（同为 API 按量计费）。凭据状态见 `/config`。

## 接入更多供应商（MiniMax / Kimi / Qwen 等）

无需改代码，REPL 里一条命令（自动写入 `~/.emiyacode/config.toml`，保留已有注释）：

```
/provider add minimax                 # 内置预设一步接入
$env:MINIMAX_API_KEY = "..."          # 设 key
/route translator minimax/MiniMax-M2  # 任意角色路由过去
```

内置预设：`minimax`、`kimi`（月之暗面）、`qwen`（通义/DashScope）、`openrouter`、
`siliconflow`、`ollama`（本地）。预设之外的任何 OpenAI 兼容端点：

```
/provider add 某家 https://api.example.com/v1 EXAMPLE_API_KEY
/provider remove 某家        # 移除（内置四家不可删）
/provider                    # 查看全部供应商与凭据状态
```

新模型无需注册即可路由（如 `deepseek/deepseek-reasoner`）；要精确的压缩触发阈值
和费用估算，在配置 `[models."provider/model"]` 补 `context_window` / 单价即可。

## 命令

| 命令 | 说明 |
|---|---|
| `/help` | 列出全部命令 |
| `/init` | 初始化向导（选供应商→端点/key/模型→角色指派） |
| `/model [route]` | 查看/切换执行模型（如 `glm/glm-4.6`） |
| `/route [角色] [route]` | 查看/设置任意角色的模型路由 |
| `/provider [add\|remove]` | 管理供应商（预设一步接入 MiniMax/Kimi 等） |
| `/lang zh\|en` | 切换中文管线 / 直通模式 |
| `/mcp` | MCP 服务器状态与工具 |
| `/skills` | 列出可用技能 |
| `/memory` | 查看项目记忆 |
| `/compact` | 立即压缩上下文 |
| `/sessions` `/resume [ID]` | 历史会话 / 恢复会话 |
| `/config` | 生效配置与 key 状态 |
| `/clear` | 清空对话上下文 |
| `/exit` | 退出 |

## 配置

叠加顺序：内置默认 ← `~/.emiyacode/config.toml` ← 项目 `.emiyacode/config.toml`。

```toml
language = "zh"          # zh: 中文管线；en: 直通
max_tokens = 16000
effort = "high"          # Anthropic effort: low|medium|high|xhigh|max

[routes]                 # 各角色的模型路由 "provider/model"，四家任意组合
executor         = "anthropic/claude-fable-5"
translator       = "deepseek/deepseek-chat"
compactor        = "glm/glm-4.6"
subagent_explore = "deepseek/deepseek-chat"
subagent_plan    = "anthropic/claude-opus-4-8"
subagent_general = "openai/gpt-5.6"

[providers.glm]                # 内置四家；OpenAI 兼容端点可自行加更多
api_key_env = "GLM_API_KEY"
base_url = "https://open.bigmodel.cn/api/paas/v4"
lang = "zh"                    # 语言基座：zh 模型作执行器时中文需求免翻译直接 plan

[providers.deepseek-a]         # 同一家也可再加 anthropic 协议端点
api_key_env = "DEEPSEEK_API_KEY"
base_url = "https://api.deepseek.com/anthropic"
protocol = "anthropic"         # 调用协议：openai（默认）| anthropic
lang = "zh"
# api_key = "sk-..."           # 可选：明文 key（环境变量优先）

[models."openrouter/deepseek-chat"]  # 聚合网关下可按模型覆盖语言基座
lang = "zh"

[models."openai/gpt-5.6"]      # 模型元数据（窗口/单价，用于压缩触发与费用估算）
context_window = 400000
input_price = 0.0              # USD / 1M tokens
output_price = 0.0

[mcp_servers.everything]       # stdio MCP 服务器
command = "npx"                # Windows 下 .cmd 命令自动加 cmd /c 垫片
args = ["-y", "@modelcontextprotocol/server-everything"]

[mcp_servers.linear]           # HTTP MCP 服务器
url = "https://mcp.linear.app/mcp"
```

MCP 服务器的 stderr 日志写入项目 `.emiyacode/logs/mcp-<名>.log`。

## Skills 格式

```
.emiyacode/skills/deploy/SKILL.md
```

```markdown
---
name: deploy
description: 部署本项目到生产环境的完整流程
---

# 部署步骤
...
```

## 开发

```powershell
uv run pytest          # 95 项单测：格式转换/循环不变量/管线状态机/压缩切分等
```

架构要点见 [AGENT.md](AGENT.md)；完整设计计划见开发会话的 plan 文件。
