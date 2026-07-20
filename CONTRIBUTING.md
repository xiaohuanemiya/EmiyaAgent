# 参与贡献

感谢你愿意改进 EmiyaAgent。项目仍处于 `0.x` Alpha 阶段；当前最重要的目标是让代理在可审计、可取消、可恢复的边界内可靠工作，而不是只增加更多工具。

提交代码前，请先阅读 [架构说明](docs/ARCHITECTURE.md)、[安全政策](SECURITY.md) 和 [路线图](docs/ROADMAP.md)。如果变更涉及未知或模糊的安全边界，先开 issue 讨论会比直接提交大型 PR 更高效。

## 开发环境

要求：

- Python 3.12 或更高版本；
- [uv](https://docs.astral.sh/uv/)；
- Windows、macOS 或 Linux。涉及子进程的改动应至少考虑 Windows Job Object 和 POSIX 进程组两条路径。

初始化并运行检查：

```powershell
uv sync --all-groups
uv lock --check
uv run ruff check .
uv run pytest -q
uv run python -m evals.run_contracts --json
uv run emiya --version
uv build
uv run python scripts/verify_release_artifacts.py --dist-dir dist
```

日常开发可只运行相关测试，但提交 PR 前应运行完整测试。测试由 `tests/conftest.py` 自动隔离真实的 `~/.emiyacode`；不要移除或绕过这项隔离，也不要让测试依赖真实 API key、网络服务、全局 Git 配置或已安装的 language server。

## 建议流程

1. 搜索已有 issue 和 PR，确认问题尚未被覆盖。
2. 对较大的功能、安全策略或公共协议改动，先提交一个说明问题、边界和验收标准的 issue。
3. 从最新主分支创建短生命周期分支，保持改动聚焦。
4. 先补能复现问题的测试，再实现修复；新增功能应覆盖成功、拒绝、超时、取消和损坏输入等相关路径。
5. 更新面向用户或维护者的文档，并在 `CHANGELOG.md` 的 `Unreleased` 下记录可见变化。
6. 提交 PR，说明设计取舍、风险、验证命令和人工检查结果。

请勿在 issue、测试夹具、终端输出或提交历史中包含 API key、OAuth token、私有仓库内容、真实会话 JSONL 或未经脱敏的模型输入/输出。

## 架构不变量

以下约束属于兼容性与安全边界。修改它们需要在 PR 中明确说明迁移方案和新增测试：

- `emiya_agent/types.py` 的 content-block 模型是内部消息格式的唯一真源；OpenAI/DeepSeek 的格式转换只放在 `providers/openai_compat.py`。
- 系统提示词在会话内保持字节级稳定；工具规格按名称排序，避免无意义的缓存失效。
- 每个 `tool_use` 必须有配对的 `tool_result`。拒绝、取消、解析失败或中断时也必须生成错误结果；并行结果合并到同一条 user 消息。
- `claude-fable-5` 不发送 `thinking`；现役 Claude 模型不发送 `temperature`、`top_p` 或 `top_k`；非 Claude 的第三方 Anthropic 兼容端点不发送 Claude 专用的 `effort`/`thinking`。
- Windows 保持 Proactor 事件循环；项目直接管理的 shell/Hook/LSP 子进程使用 Job Object，POSIX 使用独立进程组。超时、取消和关闭路径应尽力终止对应进程树；stdio MCP 的生命周期由 MCP SDK 管理，不具备同等后代收容保证。
- 所有运行时数据写入用户级或项目级 `.emiyacode`，并遵守安全路径、符号链接/reparse point 与文件大小限制。
- 文件和遍历 I/O 不得阻塞事件循环；使用 `asyncio.to_thread` 承载同步磁盘工作。
- 压缩判断读取执行器路由的最近 usage，不能被翻译器或其他角色的调用覆盖。
- 写操作默认进入审批；只有明确标为低风险、只读的工具才能豁免。
- `tool_approval`、环境继承白名单、Hook、LSP 启用状态、外部路径权限等安全策略只由用户配置控制。项目配置只有在用户显式信任后才整体参与叠加。

## 按领域测试

除通用的单元测试外，请根据改动补充以下验证：

| 领域 | 最低验证要求 |
|---|---|
| Provider / 消息转换 | content-block 往返、流式分片、usage、拒绝和无效工具参数 |
| Agent 工具循环 | 每个工具调用都有结果；并行顺序确定；拒绝、取消和预算停止不产生副作用 |
| Shell / Hook / LSP | 环境白名单、输出上限、总超时、进程树清理和 Windows/POSIX 分支 |
| stdio MCP | 环境白名单、绝对命令解析、连接/调用超时、SDK 关闭路径和错误映射 |
| 配置 | 默认值、用户配置、未信任项目、已信任项目及用户级键不可覆盖 |
| 会话 | 损坏 JSONL 恢复、原子发布、单写者租约、检查点不可变及分叉隔离 |
| 后台任务 | 队列上限、独立取消、只读共享模式、general worktree 隔离与保守清理 |
| JSON CLI / 预算 | schema 版本、事件顺序、稳定退出码、总轮次/超时/费用在子 agent 间共享 |

涉及并发的测试应使用事件、可控 fake 和有界超时，避免依赖 `sleep` 猜测时序。涉及外部协议的测试默认使用本地 fake server，不访问真实 provider、MCP 或 LSP 服务。

## 安全与数据边界

EmiyaAgent 以当前用户权限执行，不是 OS 沙箱。审批、路径限制、环境过滤、worktree 和进程树清理是纵深防御，不应被描述为对恶意代码的强隔离。改变这些边界时，请在 PR 中列出：

- 新增或扩大的文件、进程、网络与环境变量访问；
- 哪一层配置可以启用该能力；
- 首次执行是否需要审批，审批预览能否显示实际目标；
- 超时、取消、输出/帧大小、并发和队列上限；
- 哪些数据会离开本机，以及发送到 provider、HTTP MCP 或其他第三方的内容；
- 失败是否会关闭权限，是否可能在失败后继续产生副作用。

不要通过“只读工具”包装进程启动、网络写入或隐式下载。LSP 即使提供只读查询，首次启动本地服务器仍可能执行项目插件或构建配置，因此必须保留启动审批边界。Hook 的 `allow` 也不能降低已有的 `ask`/`deny` 决策。

如果发现漏洞，请不要提交公开 issue；按照 [SECURITY.md](SECURITY.md) 私下报告。

## 文档与兼容性

- 公共 CLI 参数、退出码、JSON/NDJSON 字段和 `schema_version` 都视为自动化接口。破坏性修改需要迁移说明和相应测试。
- 配置项、默认安全策略、会话记录和 Hook/LSP/MCP 行为变化应同步更新文档。
- 错误信息应说明如何修复，同时避免泄露密钥、完整环境变量或不必要的模型内容。
- 注释优先解释约束与原因，不重复代码表面行为。

## PR 规模与评审

一个 PR 尽量只解决一个问题。纯重构应与行为变化分开；机械格式化不要混入功能改动。维护者可能要求拆分涉及 Provider、工具执行、安全策略和持久化的跨层改动，以便逐层审计。

PR 合并前通常需要：相关测试通过、完整测试通过、Ruff 通过、用户可见变化已记录，以及安全/数据边界已在模板中回答。项目处于 Alpha 阶段，接口仍可能调整；这不降低对兼容性说明和迁移路径的要求。
