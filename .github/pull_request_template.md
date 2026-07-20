# Pull Request

## 概述

<!-- 说明这个 PR 解决什么问题，以及用户可观察到的结果。 -->

关联 issue：<!-- Closes #123 -->

## 变更内容

- <!-- 列出主要改动。 -->

## 设计与取舍

<!-- 说明选择该方案的原因、考虑过的替代方案，以及没有纳入本 PR 的范围。 -->

## 验证

<!-- 列出实际运行过的命令和结果；删除未运行的项目。 -->

- [ ] `uv run ruff check .`
- [ ] `uv run pytest -q`
- [ ] `uv run python -m evals.run_contracts --json`
- [ ] `uv run emiya --version`
- [ ] `uv build`
- [ ] `uv run python scripts/verify_release_artifacts.py --dist-dir dist`
- [ ] 已完成必要的 Windows / macOS / Linux 人工验证（说明平台与结果）

补充测试或人工步骤：

```text

```

## 架构与协议检查

<!-- 不适用的项目可以标记为 N/A，并在下方解释。 -->

- [ ] 内部消息仍以 `emiya_agent/types.py` 的 content block 为唯一真源，Provider 格式转换未泄漏到其他层。
- [ ] 每个 `tool_use` 在成功、拒绝、取消、异常和中断路径都有配对 `tool_result`。
- [ ] 系统提示词保持会话内字节稳定，工具列表顺序确定。
- [ ] 同步磁盘/遍历 I/O 未阻塞事件循环。
- [ ] 项目直接管理的 shell/Hook/LSP 子进程在 Windows Job Object 与 POSIX 进程组路径均有有界超时、取消和清理设计；stdio MCP 的 SDK 生命周期边界未被混淆。
- [ ] 配置、退出码、JSON/NDJSON schema、会话格式或工具 schema 的兼容性影响已说明并测试。

## 安全、权限与数据流

<!-- 明确回答；“没有变化”也请写明。EmiyaAgent 不是 OS 沙箱。 -->

- 新增或扩大的文件、进程、网络、环境变量或持久化访问：
- 启用能力的配置层（默认 / 用户 / 已信任项目）：
- 审批与 Hook 决策如何生效，失败是否关闭权限：
- 超时、取消、输出/帧、并发和队列上限：
- 会离开本机的数据，以及接收它的 provider、HTTP MCP 或其他第三方：
- worktree、路径限制或进程组之外仍存在的风险：

## 用户可见变化

- [ ] 已更新 `CHANGELOG.md` 的 `Unreleased`（纯内部改动可说明 N/A）。
- [ ] 已更新相关文档、配置示例或迁移说明（如适用）。
- [ ] 新增错误信息可操作且不会泄露密钥、完整环境或敏感会话。
- [ ] 新增测试不访问真实 provider/MCP/LSP，不读取真实 `~/.emiyacode`，不依赖开发者机器状态。

## 评审提示

<!-- 指出最需要评审者关注的文件、竞态、安全边界或尚未解决的问题。 -->
