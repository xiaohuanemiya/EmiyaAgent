# 变更日志

本文件记录 EmiyaAgent 的重要用户可见变化。项目处于 Alpha 阶段；版本计划遵循[语义化版本](https://semver.org/lang/zh-CN/)，但 `0.x` 期间公共接口仍可能调整，破坏性变化会在此说明。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased]

### Added

- 用户级完整命令 `shell_rules`，支持有序的 `allow`、`ask`、`deny` 决策。
- 有界 `PreToolUse` / `PostToolUse` 命令 Hook；Hook 只可维持或提高现有权限要求，并传递到子 agent。
- 可选的本地 LSP 查询工具，支持诊断、定义、引用、悬停、文档符号和工作区符号；按需启动、复用连接并限制结果范围。
- 有界后台子 agent 队列、独立取消和 general 任务的 Git worktree 隔离与保守清理。
- 容错 JSONL 会话恢复、不可变命名检查点、带 lineage 的会话分叉和跨进程单写者租约。
- `--json` 单对象与 `--stream-json` NDJSON 自动化输出、版本化 schema 和稳定退出码。
- 跨执行器、翻译/压缩请求与子 agent 共享的全局轮次、总超时和 USD 预算控制。
- Windows Job Object / POSIX 进程组清理、环境继承白名单和运行时安全路径检查。
- 直接驱动真实 Agent 循环的离线确定性契约评测，覆盖工具执行、截断、预算、重复调用 ID 与审批拒绝。
- 跨平台 CI 测试矩阵，以及构建、干净 wheel 安装、依赖、元数据和 CLI 冒烟发布门禁。
- wheel/sdist 离线内容审计，拒绝路径穿越、链接、敏感运行时文件、跨平台重名以及与源码不一致的制品。
- 每周检查 uv 锁定依赖与 GitHub Actions 的 Dependabot 配置；CI 外部 Actions 固定到完整提交、uv 固定到明确版本，并禁用持久化检出凭据。

### Changed

- 普通 REPL 与一次性输入现在统一原样交给主 Agent；是否调用 `explore`、`plan`、`general` 或 translator 由 Agent 按任务收益自主决定，不再因中文字符自动进入 plan。
- 删除 `/lang`、`/spec`、顶层 `language` 配置及旧的固定规格/翻译管线；旧配置必须移除顶层 `language`。provider/model 的 `lang` 现作为下一跳模型主要语言，由 `translate_for_model` 动态解析并传给 translator Agent，调用方不能覆盖目标语言。
- 未信任项目的 `.emiyacode/config.toml` 现在整层忽略；用户显式 `/trust` 后才参与配置叠加。
- 审批、外部路径、环境继承、Hook 和 LSP 启用等安全策略限定为用户级配置，可信项目也不能覆盖。
- 一次性执行模式保留审批语义；非交互场景需要新审批时会失败关闭，自动执行必须显式选择 `--approval auto`。
- MCP、shell、Hook 和 LSP 子进程仅继承显式白名单环境，并采用有界输出与超时；shell、Hook 和 LSP 额外使用有界的进程树回收。

### Security

- 文件、会话、记忆、技能、项目指令和 worktree 路径增加符号链接、junction/reparse point、硬链接与越界检查。
- 工具协议在拒绝、解析失败和中断路径补齐配对错误结果，非法或重复工具调用 ID 不再进入活跃历史或执行副作用。
- 会话创建采用不覆盖的原子发布；恢复会跳过单条损坏/未知记录，并修复完整历史中的悬空工具调用。
- Hook、LSP 与 stdio MCP 的裸命令只在白名单环境的绝对 `PATH` 目录中解析，避免从当前仓库隐式加载同名程序。

### Documentation

- 新增贡献指南、安全政策、架构说明、主流编码代理对标、路线图和结构化 GitHub issue/PR 模板。
- 明确外部 provider、MCP、Hook、LSP、后台 worktree 和本地持久化的数据与信任边界，以及当前并非 OS 沙箱。

## [0.1.0] - 2026-07-17

### Added

- 初始 Alpha 版本：多 provider 路由、中文需求管线、流式工具循环、前台子 agent、Skills、项目记忆、上下文压缩、MCP、会话持久化和终端 REPL。
