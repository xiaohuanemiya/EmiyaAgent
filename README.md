# EmiyaAgent

多智能体 AI 编程助手（终端 REPL）。普通输入始终原样交给主 Agent；主 Agent
根据任务复杂度和独立性，自主决定直接回答、使用工具，或按需派生 `explore`、
`plan`、`general` 子 Agent。输入语言不参与是否规划的判断。

跨语言交接由主 Agent 按收益调用 `translate_for_model`。该工具从下一角色当前的
模型路由解析目标语言，再调用 translator Agent 做等价适配；调用方不能自报目标语言。
短文本、同语言或代码为主的交接不必翻译，也不存在固定的“翻译后再 plan”流程。
模型主要语言在 provider 级标注（model 级可覆盖，聚合网关适用）。

功能一览：

- **自主任务编排**：问候、简单问答和清晰的一步任务直接处理；只有复杂、高风险、跨模块或可独立并行的工作才按收益选择子 Agent，不执行固定的 explore → plan → general 流程
- **模型语言感知翻译**：`translate_for_model` 动态读取 executor/explore/plan/general 的下一跳模型与主要语言，仅在复杂或精度敏感的跨语言交接中调用 translator Agent；只翻译，不规划、不扩写
- **Agent 工具循环**：shell（PowerShell）、文件读写/精确编辑、glob/grep 搜索、todo 进度、中途提问；并行工具调用；Ctrl+C 干净中断
- **多智能体**：`spawn_subagent` 派生隔离上下文的子 agent —— explore（只读搜索）/ plan（只读设计）/ general（完整工具）；支持有界后台队列、独立取消与 Git worktree 隔离
- **Skills**：`.emiyacode/skills/<名>/SKILL.md`（用户级+项目级），描述进系统提示词，内容按需加载
- **项目记忆**：`.emiyacode/memory/`（MEMORY.md 索引 + 一事一文件），跨会话持久
- **上下文压缩**：超过执行模型窗口 70% 自动用 DeepSeek 摘要旧轮次；`/compact` 手动触发
- **MCP 接入**：stdio / streamable-HTTP 服务器，工具以 `mcp__服务器__工具` 注册
- **策略 Hook**：有界 `PreToolUse` / `PostToolUse` 命令钩子，可提升审批或阻断工具，子 agent 同样生效
- **LSP 语义导航**：按需连接本地 language server，提供诊断、定义、引用、悬停和符号查询
- **会话持久化**：容错 JSONL 转录、`/resume` 恢复、不可变命名检查点与带 lineage 的会话分叉
- **自动化接口**：`--json` 单对象 / `--stream-json` NDJSON，稳定退出码与全局轮次、超时、USD 预算

## 快速开始

```powershell
uv sync
uv run emiya
```

**首次运行会自动弹出初始化向导**：选择要配置的供应商（可多选、可跳过）→ 每家依次
设置 base_url（带默认值与备选端点，如 DeepSeek 的 Anthropic 兼容端点）、API key（隐藏输入且不写终端历史）、
模型名 → 给每个智能体角色指派模型 → 自动写入 `~/.emiyacode/config.toml`。
跳过则生成带注释的配置模板；`/init` 可随时重跑向导。

API key 两种给法（环境变量优先）：

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."     # 方式一：环境变量
# 方式二：向导中隐藏输入，明文写入用户配置（环境变量仍是更推荐的方式）
```

中英文普通输入都会先交给主 Agent；是否规划、派生或翻译均由 Agent 根据任务收益决定。

## 多供应商路由（纯 API）

内置四家 provider：**anthropic（Claude）/ openai（GPT）/ deepseek / glm（智谱）**，
全部走各家官方 API 直连。**每一个智能体角色都可以独立路由到任意一家**：

| 角色 | 职责 | 默认路由 |
|---|---|---|
| `executor` | 主执行 agent（工具循环） | `anthropic/claude-fable-5` |
| `translator` | 下一跳模型语言适配 | `deepseek/deepseek-chat` |
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
| `/mcp` | MCP 服务器状态与工具 |
| `/hooks` | 审核当前用户级命令 Hook |
| `/lsp` | Language Server 配置与活动连接 |
| `/skills` | 列出可用技能 |
| `/memory` | 查看项目记忆 |
| `/compact` | 立即压缩上下文 |
| `/tasks` | 列出后台子 agent 任务 |
| `/task show\|wait\|cancel\|cleanup <ID>` | 查看、等待、取消任务或清理其 worktree |
| `/sessions` `/resume [ID]` | 历史会话 / 恢复会话 |
| `/checkpoint <名称>` `/checkpoints` | 保存不可变命名检查点 / 列出当前会话检查点 |
| `/fork [检查点名称]` | 从当前活跃上下文或命名检查点创建独立子会话 |
| `/approval ask\|auto` | 工具审批：ask 时 shell/写文件/MCP 执行前须确认（默认） |
| `/trust` | 信任当前项目的 `.emiyacode/config.toml`（检查后重启生效） |
| `/config` | 生效配置与 key 状态 |
| `/clear` | 清空对话上下文 |
| `/exit` | 退出 |

### 会话检查点与分叉

```text
/checkpoint before-refactor  # 保存当前活跃上下文，不改变恢复基线
/fork before-refactor        # 从该检查点创建子会话并切换过去
/fork                        # 从当前活跃上下文创建子会话
```

检查点在所属会话中不可覆盖；名称区分大小写，支持 Unicode 字母和数字以及
`.`、`_`、`-`。分叉不会改写父会话，子会话在 `/sessions` 中显示直接父会话和检查点来源。
如需从旧会话分叉，先 `/resume <ID>`。每个分叉拥有独立的消息副本，后续 `/clear`、
`/compact` 和追加消息互不影响。

## 配置

叠加顺序：内置默认 ← `~/.emiyacode/config.toml` ← **已信任项目**的
`.emiyacode/config.toml`。项目配置可声明 provider 端点和 stdio MCP 命令，因此首次打开
仓库时默认整层忽略；检查文件后运行 `/trust` 并重启。审批模式、外部目录访问等安全策略
始终只能由用户级配置控制。

```toml
max_tokens = 16000
effort = "high"          # Anthropic effort: low|medium|high|xhigh|max
tool_approval = "ask"    # 用户级安全策略；项目配置不能覆盖
allow_external_paths = false  # 文件工具默认只能访问工作目录
trusted_projects = ["C:/code/my-trusted-repo"]  # 通常用 /trust 维护
shell_inherit_env = ["PATH", "SYSTEMROOT", "HOME", "TEMP", "VIRTUAL_ENV"]
mcp_inherit_env = ["PATH", "SYSTEMROOT", "HOME", "TEMP"]
hook_inherit_env = ["PATH", "SYSTEMROOT", "HOME", "TEMP"]
lsp_enabled = false            # 用户级显式开启；项目配置不能静默启动进程
lsp_inherit_env = ["PATH", "SYSTEMROOT", "HOME", "TEMP"]

[mcp_server_inherit_env]       # 仅把 token 暴露给指定的可信 stdio MCP
linear = ["LINEAR_API_KEY"]

# 用户级命令规则：按顺序对完整命令做正则 full-match；首条匹配生效。
[[shell_rules]]
action = "deny"                 # allow | ask | deny；规则优先于 tool_approval
pattern = '(?i)(rm|del) .*secret.*'
description = "保护敏感文件"

[[shell_rules]]
action = "allow"
pattern = 'git (status|diff)( --[a-z-]+)*'

# Hook 仅从用户配置加载。命令是 argv，不经过 shell；JSON 事件从 stdin 输入。
[[hooks]]
id = "guard-production"
event = "PreToolUse"            # PreToolUse | PostToolUse
matcher = "^(shell|write_file|edit_file)$"
command = ["python", "C:/Users/me/.emiyacode/hooks/guard.py"]
timeout = 5
on_error = "block"              # warn | block

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
lang = "zh"                    # 模型主要语言；翻译 Agent 用作下一跳目标语言

[providers.deepseek-a]         # 同一家也可再加 anthropic 协议端点
api_key_env = "DEEPSEEK_API_KEY"
base_url = "https://api.deepseek.com/anthropic"
protocol = "anthropic"         # 调用协议：openai（默认）| anthropic
lang = "zh"
# api_key = "sk-..."           # 可选：明文 key（环境变量优先）

[models."openrouter/deepseek-chat"]  # 聚合网关下可按模型覆盖主要语言
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

[lsp_servers.basedpyright]     # 只连接已安装的本地服务器，不会自动下载
command = "basedpyright-langserver"
args = ["--stdio"]
extensions = { ".py" = "python", ".pyi" = "python" }
root_markers = ["pyrightconfig.json", "pyproject.toml", ".git"]
priority = 100
```

MCP 服务器的 stderr 日志写入项目 `.emiyacode/logs/mcp-<名>.log`。
stdio MCP 与 shell 默认只继承 PATH、HOME、TEMP 等运行必需环境；API key、云凭据等变量
必须在用户配置的对应 `*_inherit_env` 白名单中显式加入。项目配置不能放宽这些白名单。
stdio MCP 的裸命令只在该白名单环境中的绝对 PATH 目录解析，不会从当前仓库或相对 PATH
目录隐式加载同名程序；其进程生命周期由 MCP SDK 管理，不具备 shell/Hook/LSP 的进程树
收容保证。

Hook 进程收到单行 JSON，成功时可不输出，或输出
`{"decision":"allow|ask|deny","reason":"..."}`；退出码 `2` 也表示阻断。
多个匹配 Hook 并发执行但按配置顺序聚合，优先级固定为 `deny > ask > allow`，因此
Hook 的 `allow` 永远不能绕过 shell deny 规则或已有用户审批。stdout/stderr、超时和环境
均有界；继承同一 Windows Job / POSIX 进程组的进程会在结束时尽力回收。Hook 命令及
`hook_inherit_env` 都是用户级策略，可信项目也不能覆盖。裸 Hook 命令同样只从白名单环境
中的绝对 PATH 目录解析。

LSP 默认关闭，`lsp_enabled` 和 `lsp_inherit_env` 只能在用户配置中设置；可信项目可以声明
服务器参数，但不能自行启用。查询按文件扩展名和优先级选择首个已安装的候选，不会运行
包管理器或自动下载。裸命令只搜索白名单环境中的绝对 `PATH` 目录，忽略 cwd、空项和
相对项；Windows 只直启原生 `.exe/.com`，Node 工具应显式配置 `node.exe` 与脚本参数，
不会经 `.cmd/.bat` shell 包装器启动。每个 server/root 首次启动会依照审批模式展示绝对命令和
项目根；冷启动审批签发一次性快照票据，并在启动锁内重新绑定 server/root/argv、环境、
初始化配置和可执行文件指纹，marker、PATH 或二进制变化会拒绝启动并要求重新审批。
后续查询复用连接。language server 可能执行项目插件或构建配置，因此审批不等同于沙箱。
文档与服务器返回的 URI 均限制在工作区内，位置统一转换为一基、Unicode 码点列；结果、
帧、通知、stderr 和超时都有上限，连接关闭时会尽力回收同一 Windows Job / POSIX
进程组内的服务器进程。

## Skills 格式

```
.emiyacode/skills/deploy/SKILL.md
```

也兼容用户级和项目级的 `.agents/skills/`、`.claude/skills/`、`.codex/skills/`；
同名技能按“项目覆盖用户、`.emiyacode` 覆盖兼容目录”解析。项目指令优先使用标准
`AGENTS.md`，并从 Git 根逐层合并到当前工作目录；`AGENTS.override.md` 可覆盖同层文件，
旧的 `AGENT.md` 仍作为兼容回退。项目指令与技能索引也会传给隔离上下文的子 agent。

```markdown
---
name: deploy
description: 部署本项目到生产环境的完整流程
---

# 部署步骤
...
```

## 一次性执行模式

```powershell
uv run emiya -p "列出本目录的 Python 文件并统计行数"
```

执行一轮后退出，适合脚本化调用与冒烟验证。一次性模式不再隐式关闭审批：非交互环境中，
需要新审批的 shell/写文件/MCP 调用会失败关闭。只有在任务与仓库都可信时才显式使用：

```powershell
uv run emiya --approval auto -p "运行测试并修复失败"
```

一次性模式不会读取 stdin 做审批或 `ask_user`；需要写操作时必须显式传
`--approval auto`。stdout 可切换为机器协议：

```powershell
# 最终只输出一个 schema_version=1 的 JSON 对象
uv run emiya -p "检查项目" --json --max-turns 12 --timeout 300

# 逐行输出 run_start / text_delta / tool_result / message_done / final 等 NDJSON
uv run emiya -p "运行测试" --stream-json --approval auto --max-budget-usd 0.50
```

`--max-turns`、`--timeout`、`--max-budget-usd` 都是整次调用的全局上限，执行器、
翻译/压缩请求与所有子 agent 共用同一计数器。预算按 `[models."provider/model"]`
中的 input/output/cache 单价估算；启用预算却缺少当前路由定价时会 fail closed，退出码为 3。
单次模型请求可能轻微越过预算，收到该请求的权威 usage 后不会再发请求或执行其工具调用。

稳定退出码：

| 代码 | 含义 |
|---:|---|
| `0` | 成功 |
| `1` | 一般执行错误 |
| `2` | CLI 参数/用法错误 |
| `3` | 配置错误或预算缺少定价 |
| `4` | Provider/API 错误 |
| `5` | 模型拒绝 |
| `6` | 达到全局请求/内部迭代上限 |
| `7` | 达到 USD 预算 |
| `8` | 模型输出达到 `max_tokens` |
| `9` | 非法工具协议状态 |
| `124` | 总超时 |
| `130` | 用户中断 |

## 安全

- 默认 `tool_approval = "ask"`：shell / 写文件 / 编辑 / MCP，以及项目记忆的写入和删除，
  执行前展示参数；y 允许（回车）/ n 拒绝 / a 本会话全部放行。`/approval auto` 或配置可关闭。
- `shell_rules` 可对完整命令强制 allow/ask/deny；即使处于 auto，显式 ask/deny 仍优先。
- shell stdin 默认关闭、输出捕获上限 2 MB。Windows 中真实 PowerShell 以悬挂状态
  启动，分配到 kill-on-close Job Object 后才恢复；分配或恢复失败则未执行命令并关闭进程。
  POSIX 使用独立进程组；正常退出、超时和中断都会进行有界清理。这些是生命周期收容，
  不是进程、文件系统或网络级 OS 沙箱；具有足够权限且主动脱离进程组/作业的代码仍可逃逸。
- 后台 explore/plan 可在共享工作区只读运行；后台 general 必须使用独立 Git worktree，且仅在
  `tool_approval = "auto"` 时启动。任务静默渲染、使用独立取消信号，并发上限为 2、排队上限为 16。
  为避免遗漏本地改动，创建 worktree 时要求源工作区干净；含改动的任务 worktree 默认拒绝清理，
  只有显式 `/task cleanup <ID> --force` 才会强制移除。
- 内部 worktree 操作解析 PATH 中 Git 的绝对路径，关闭 Git hooks/fsmonitor、终端凭据提示和
  LFS 自动下载；清理只通过 `git worktree remove` 完成。worktree 是代码目录隔离，不是 OS 沙箱。
- 只读工具（读文件/搜索/读取记忆）与低危工具（todo）不询问。
- 文件工具默认限制在当前工作目录，解析后的 `..`、绝对外部路径及越界符号链接会被拒绝；
  必须在用户配置显式设置 `allow_external_paths = true` 才能放宽。
- `.emiyacode` 的 memory/sessions/logs 目录及其运行时文件拒绝符号链接和 Windows
  reparse point；POSIX 文件打开额外使用 no-follow，防止持久化数据被重定向到工作区外。
- 新会话先在隐藏文件完整写入、同步并回读，再以不覆盖的硬链接原子发布；当前交互会话持有
  跨进程单写者租约，避免旧压缩/清空快照覆盖另一进程的新消息。会话文件和锁文件同时拒绝
  硬链接。JSONL 按有界记录逐行恢复，坏 UTF-8、损坏 JSON、未知版本或不完整检查点只跳过
  对应记录；分叉 lineage 与初始上下文写在同一条记录中。
- 项目 `AGENTS.md` 与技能文件解析实际路径，拒绝通过符号链接读取工作区外文件。
- 未信任项目的 `.emiyacode/config.toml` 不会启动 MCP、改写 provider 端点或关闭审批。
- 截断的工具调用只写入配对错误结果；非法 JSON、空/重复 ID 的整条助手响应不会进入活跃历史，
  也不会执行副作用。
- 拒绝会作为错误结果反馈给模型，它会调整方案而不是中断任务。

## 开发

```powershell
uv run pytest -q
uv run python -m evals.run_contracts --json
uv run ruff check .
uv build
uv run python scripts/verify_release_artifacts.py --dist-dir dist
```

GitHub Actions 会在 Windows、Ubuntu、macOS，以及 Python 3.12/3.13/3.14 上运行 lint、测试、
离线契约评测与 CLI 冒烟；全部通过后，独立发布门禁会构建 sdist/wheel、把 wheel 安装到
干净环境，并审计制品内容、源码一致性、依赖、包元数据和已安装 CLI。

维护与审计资料见 [贡献指南](CONTRIBUTING.md)、[架构说明](docs/ARCHITECTURE.md)、
[安全政策](SECURITY.md)、[对标分析](docs/COMPARISON.md)、[路线图](docs/ROADMAP.md) 和
[变更日志](CHANGELOG.md)；项目级 agent 指令见 [AGENTS.md](AGENTS.md)。

## 开源发布状态

仓库目前尚未选择 `LICENSE`，因此不应被假定为允许复制、修改或再分发的开源发行版。
许可证是维护者的法律与治理选择，不能由自动化代选；完成 SPDX 许可证选择仍是
[路线图](docs/ROADMAP.md) 中的公开发布阻断项。
