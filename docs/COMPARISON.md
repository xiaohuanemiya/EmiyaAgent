# 与主流编码代理的差距（2026-07-20 快照）

本文比较 EmiyaAgent 与 OpenCode、Claude Code、Codex 的**可验证能力与工程边界**，用于确定后续优化顺序，不做品牌、模型质量或流行度排名。EmiyaAgent 的事实口径来自当前仓库的 [README](../README.md)、[架构说明](ARCHITECTURE.md)、[路线图](ROADMAP.md)、[安全政策](../SECURITY.md) 和代码；外部事实只引用各项目的官方文档或官方仓库。

## 比较口径与非同类说明

- OpenCode 与 Codex 有公开源码仓库和明确许可证（[OpenCode LICENSE](https://github.com/anomalyco/opencode/blob/dev/LICENSE)、[Codex repository](https://github.com/openai/codex)）；Claude Code 在本文中作为产品能力基线参与比较，不把它当作同一治理模式的开源仓库。
- “已有”表示代码中存在可用路径，不等于 API 已稳定、安全审计已完成或真实项目质量已被证明。
- 不比较易变的具体模型版本、榜单分数、价格、star 数或未经统一任务集复现的成功率。
- 三个竞品也不是同一种产品：OpenCode 强调多 provider 与客户端/服务器架构（[Models](https://opencode.ai/docs/models/)、[Server](https://opencode.ai/docs/server/)），Claude Code 覆盖 CLI、IDE 和其他产品界面（[Platforms](https://code.claude.com/docs/en/platforms)），Codex 同时有开源 CLI、IDE、桌面与云端形态（[official repository](https://github.com/openai/codex)）。因此，下表的“竞品基线”是对应维度中已被官方资料证实的能力集合，不表示三者逐项都相同。

## 能力矩阵

| 维度 | EmiyaAgent 已有 | 竞品基线 | 判断 |
|---|---|---|---|
| Provider 与语言交接 | Anthropic、OpenAI 兼容协议及多 provider/多角色路由；普通输入由主 Agent 自主处理，翻译 Agent 按下一跳角色的实时模型语言做可选等价适配 | OpenCode 官方说明支持大量 provider 与本地模型，并使用 `provider/model` 选择模型（[Models](https://opencode.ai/docs/models/)） | 目标模型感知的可选翻译是差异化能力；provider 覆盖、认证方式和兼容性矩阵仍落后于 OpenCode 生态 |
| 工具循环与审批 | content-block 单一内部消息模型；严格 JSON、工具调用/结果配对、写操作默认审批、shell 规则及单调 Hook 决策 | OpenCode 提供可按工具、命令和子 agent 配置的 `allow/ask/deny` 权限（[Agents](https://opencode.ai/docs/agents/)）；Claude Code 将工具权限与 OS 级 Bash 沙箱组合（[Permissions](https://code.claude.com/docs/en/permissions)）；Codex 把审批策略与技术沙箱边界分开（[Sandboxing](https://developers.openai.com/codex/sandboxing/)） | 协议正确性和默认审批已达到可用基线；缺少 OS 沙箱是明确安全差距 |
| 子 agent 与并行任务 | `explore`、`plan`、`general` 角色；父子共享轮次、超时、usage 与 USD 预算；有界后台队列、取消及 general 任务 worktree | OpenCode 有 primary/subagent 与逐 agent 权限（[Agents](https://opencode.ai/docs/agents/)）；Claude Code 提供 subagent、后台会话、agent teams 与 worktree 并行方式（[Parallel agents](https://code.claude.com/docs/en/agents)）；Codex 可并行派生专用 agent 并汇总结果（[Subagents](https://developers.openai.com/codex/agent-configuration/subagents/)），桌面端以 worktree 支持独立后台聊天（[Worktrees](https://developers.openai.com/codex/environments/git-worktrees/)） | 核心并行路径已缩小差距；持久队列、重启恢复、跨 agent 通信、结果交接与审阅 UX 仍不足 |
| 会话、检查点与分叉 | 容错 JSONL、跨进程单写者、`/resume`、不可变命名检查点和 lineage 分叉 | Claude Code 持续保存、恢复和分支会话（[Sessions](https://code.claude.com/docs/en/sessions/)），并可将代码/会话回退到检查点（[Checkpointing](https://code.claude.com/docs/en/checkpointing)）；Codex CLI 支持恢复本地会话（[CLI features](https://developers.openai.com/codex/cli/features/)） | 本地会话完整性是当前强项；仍缺导出、迁移、保留策略和显式修复工具 |
| MCP | stdio 与 streamable HTTP；工具统一进入审批和 Hook；环境继承白名单、裸命令安全解析 | OpenCode 支持本地/远程 MCP 与远程 OAuth（[MCP](https://opencode.ai/docs/mcp-servers/)）；Claude Code 支持 MCP 工具接入（[MCP](https://code.claude.com/docs/en/mcp)）；Codex 支持 stdio、streamable HTTP、Bearer 与 OAuth（[MCP](https://developers.openai.com/codex/mcp/)） | 传输基线已具备；OAuth、逐 server/tool 能力审计和协议兼容性覆盖不足；stdio MCP 后代回收仍依赖 MCP SDK |
| Hook 与策略扩展 | 用户级命令 Hook 覆盖 9 个事件：工具事件（`PreToolUse`/`PostToolUse`）加 `UserPromptSubmit`、`SessionStart`、`SessionEnd`、`Stop`、`SubagentStop`、`PreCompact`、`PostCompact`；决策型/观测型分离，有界输入输出、超时、进程树回收和 `deny > ask > allow` 聚合 | OpenCode 插件暴露工具、会话、权限、LSP 等大量 JS 事件（[Plugins](https://opencode.ai/docs/plugins/)）；Claude Code Hook 覆盖 30+ 生命周期事件，且支持修改工具输入/输出、注入上下文与 HTTP hook（[Hooks](https://code.claude.com/docs/en/hooks)） | 生命周期事件面已明显缩小差距；仍缺输入/输出改写、上下文注入、进程内插件与调试工具，观测型事件不赋予 Hook 控制权是有意的安全取舍 |
| LSP | 默认关闭；仅连接已安装 server，不调用包管理器；提供诊断、定义、引用、悬停和符号查询，并限制帧、结果与工作区路径 | OpenCode 有内置 LSP 列表，启用后部分 server 可自动安装，也允许禁用自动下载和配置自定义 server（[LSP](https://opencode.ai/docs/lsp/)）；Claude Code 插件可携带 LSP 配置（[Plugins](https://code.claude.com/docs/en/plugins)） | “不自动下载、不静默执行”是供应链与本机执行风险上的主动取舍，不应仅记为缺陷；真正差距在发现体验、健康状态、多根工作区和兼容测试 |
| 机器自动化 | 一次性 CLI、单对象 JSON、NDJSON、稳定退出码，以及全局轮次/超时/USD 预算 | OpenCode 提供 headless HTTP/OpenAPI server（[Server](https://opencode.ai/docs/server/)）；Claude Code 非交互模式支持 JSON/stream-JSON 输入输出（[CLI reference](https://code.claude.com/docs/en/cli-usage)）；Codex `exec --json` 输出 JSONL，并支持 JSON Schema 约束最终结果（[Non-interactive mode](https://developers.openai.com/codex/non-interactive-mode/)） | CLI 自动化已到基础基线；尚无稳定 SDK/服务端接口，`schema_version = "1"` 也未完成兼容性冻结 |
| 扩展生态 | Skills、MCP、Hook 和内部 Python 工具注册；尚无版本化公共插件接口 | OpenCode 支持本地及 npm 插件并自动加载依赖（[Plugins](https://opencode.ai/docs/plugins/)）；Claude Code 插件可打包 skills、agents、hooks、MCP 与 LSP（[Plugins](https://code.claude.com/docs/en/plugins)）；Codex CLI 提供插件安装和 marketplace 管理（[CLI reference](https://developers.openai.com/codex/cli/reference/#codex-plugin)） | 这是明显生态差距；在定义权限、数据流、生命周期和兼容版本前，不应把内部 Python API 宣称为稳定插件 API |
| 产品界面与远程运行 | 终端 REPL 与一次性 CLI | OpenCode 有 TUI、桌面（[official repository](https://github.com/anomalyco/opencode)）、[IDE](https://opencode.ai/docs/ide/)、[Web](https://opencode.ai/docs/web/) 及 [headless server](https://opencode.ai/docs/server/)；Claude Code 覆盖 CLI、桌面、VS Code、JetBrains 与 Web 等界面（[Platforms](https://code.claude.com/docs/en/platforms)）；Codex 官方仓库列出 CLI、IDE、桌面和云端入口（[official repository](https://github.com/openai/codex)） | 终端内可用，但编辑器审阅、远程任务、服务 API 与跨端接续存在较大产品化差距 |
| 沙箱与进程边界 | 工作区路径约束、环境白名单、Windows Job Object/POSIX 进程组和 worktree；**无 OS sandbox** | Claude Code 的 Bash 沙箱使用 macOS Seatbelt、Linux/WSL2 bubblewrap 并约束文件系统与网络（[Sandboxing](https://code.claude.com/docs/en/sandboxing)）；Codex 本地命令默认运行于受限环境，越界由审批接管（[Sandboxing](https://developers.openai.com/codex/sandboxing/)） | Job、进程组、路径校验和 worktree 都不能等同 OS 沙箱；处理不可信仓库时必须依赖外部低权限环境、容器或虚拟机 |
| 发布与质量证据 | 跨平台 CI、wheel/sdist 内容与源码一致性审计、干净 wheel 安装门禁、单元测试和离线契约评测（**9 个 contract**：含并行只读、未知工具、取消配对、审批放行/拒绝）；仓库**无 `LICENSE`** | OpenCode 官方仓库采用 MIT（[LICENSE](https://github.com/anomalyco/opencode/blob/dev/LICENSE)）；Codex 官方仓库采用 Apache-2.0（[repository](https://github.com/openai/codex)） | 许可证是公开发布 P0 阻断项；9 个 contract 覆盖核心循环不变量，但仍不含检索/编辑/恢复等端到端案例，不能支撑与成熟产品的质量结论 |

## 已缩小的差距

- 工具循环已把严格 JSON、重复/空调用 ID、截断、拒绝、取消和预算终止纳入同一套配对不变量，避免“展示能跑、异常路径失控”。
- 默认审批、用户级安全策略、环境白名单、可信项目分层、Hook 单调权限以及 shell/Hook/LSP 进程树清理形成了可审计的纵深防御。
- 子 agent 不再只是同步调用：已有全局预算、并发上限、后台取消和 general 任务 worktree 隔离。
- 会话具备容错恢复、单写者租约、命名检查点与显式分叉，已接近成熟 CLI 的日常会话管理体验。
- JSON/NDJSON、退出码和离线 contract runner 使核心循环可以进入 CI，而不必依赖终端文本抓取。
- Hook 事件面从 2 个工具事件扩展到 9 个（含提示词提交、会话起止、停止、子 agent 结束与压缩前后），并以"决策型/观测型"分离防止 Hook 事后改写结果或强迫继续运行。
- `/doctor` 与 `emiya doctor [--json]` 提供与 Claude Code `/doctor` 对应的第一层自检：凭据只报来源、URL 去 userinfo/query、命令只留可执行名，可直接用于 issue 报告。

## 仍明显不足

### P0：公开发布与可信基线

1. **选择并添加 SPDX 许可证。** 当前没有 `LICENSE`，因此不能把仓库描述为允许复制、修改和再分发的开源发行版；许可证必须由维护者决定。
2. **把评测从 9 个 contract 扩为端到端回归集。** 取消、并行、审批双向与未知工具已入集；仍需覆盖检索、精确编辑、会话恢复、后台任务、MCP/LSP 坏帧与机器输出兼容性，并持续报告已知失败模式。
3. **冻结机器接口。** 明确 `schema_version = "1"` 的字段语义、事件顺序、退出码和迁移策略，并用兼容性测试约束。
4. **完成安全发布治理。** 制品内容与源码一致性审计已经进入 CI；仍需演练私密漏洞报告，并明确签名、provenance、安全公告和回滚流程。
5. **保持边界披露为发布门禁。** 在实现更强隔离前，所有入口都应明确提示“无 OS sandbox”；stdio MCP 后代进程是否彻底回收仍依赖 MCP SDK，不能沿用 shell/Hook/LSP 的清理保证。

### P1：可靠性、诊断与日常工作流

1. 脱敏诊断已有第一层（`/doctor` 与 `emiya doctor --json`）；仍需可导出诊断包、协议兼容矩阵、本地 fake provider/MCP/LSP server、故障注入和性能基线。
2. 为后台任务增加重启后元数据恢复、孤立 worktree 检测、安全接管，以及“查看差异—验证—显式应用/合并”的完整流程。
3. 为会话提供只读检查、备份/导出、修复预览、迁移和保留策略；为预算显示定价来源并区分已承诺与已结算成本。
4. 补齐 MCP OAuth、逐 server/tool 权限与数据流审计；Hook 生命周期事件面已扩展，仍缺输入/输出改写、上下文注入、事件日志与调试体验。
5. 改进 LSP 发现、健康状态、多根工作区和版本兼容性，但继续保持默认关闭、首次启动审批和**绝不隐式下载**。

### P2：生态与产品形态

1. 设计版本化插件接口，声明权限、数据流、生命周期和兼容版本；再考虑 marketplace、签名或受控安装流程。
2. 提供可选外部沙箱后端接口（用户自备容器、虚拟机或受限执行器），逐后端写明实际保证；不把 worktree 或进程组改名为“沙箱”。
3. 在核心协议稳定后评估 IDE diff/review、headless server/SDK、远程任务和跨端接续，而不是把 TUI 功能直接复制到新界面。
4. 增加跨 agent 通信、持久队列、结果引用和人工审阅后的应用流程；自动合并、推送或部署必须单独授权。
5. 若面向组织使用，再补集中策略、审计事件、身份/凭据边界和兼容性支持周期；这些不是当前单用户 Alpha 的既有承诺。

## 设计取舍

- **安全优先不是功能缺失的同义词。** 用户级策略不能被项目配置放宽、冷启动 LSP 要审批、裸命令只在绝对 PATH 目录解析，这些限制会增加配置成本，但减少恶意仓库静默启动本地代码的机会。
- **不自动下载 LSP 是有意选择。** OpenCode 的[自动安装路径](https://opencode.ai/docs/lsp/)提升开箱体验；EmiyaAgent 选择只连接已安装 server。后续优化目标应是更好的发现、解释和配置预览，而不是未经授权地恢复自动下载。
- **多 provider 与模型语言感知交接优先于绑定单一产品生态。** 这是项目定位，不代表每个 provider 的协议细节、认证方式和质量都已达到相同成熟度。
- **审批不是沙箱。** `tool_approval = "auto"` 只改变交互门槛；worktree 只隔离 Git 文件视图；Job Object/进程组主要解决生命周期清理，均不提供文件系统、系统调用、凭据或网络隔离。

## 维护规则

更新本文时，应先核对当前代码和上述仓库内文档，再逐项访问官方来源；更新快照日期并删除已经失效的结论。外部能力必须在对应句子附近给出直接链接；不要用搜索结果、社区帖子、宣传榜单或记忆补全事实，也不要写入容易过时的具体模型版本。优先维护“边界、证据、下一步”，而不是维护一张静态功能打勾表。
