# 项目规则

## 规划文件（planning-with-files）

项目根目录的三个规划文件是本项目的持久化工作记忆，所有内容使用中文：

- `task_plan.md`：阶段、进度、决策、错误
- `findings.md`：研究与发现
- `progress.md`：会话日志与测试结果

### 必须遵守

1. **随进展更新**：实际操作中只要有任何新进展（完成阶段、做出决策、有新发现、遇到错误、测试结果、修改或创建文件），必须立即更新对应文件，不得等到会话结束。
   - 阶段或决策变化 → `task_plan.md`
   - 新发现、调研结果 → `findings.md`
   - 操作记录、测试结果 → `progress.md`
2. **同步到 GitHub**：这三个文件必须纳入 git 版本管理，并同步到远程 `origin`（yang-AIteam/Retrieval-based-Voice-Conversion-WebUI）。每次更新后，随相关改动一起提交并推送到当前分支。
   - 不得把这三个文件加入 `.gitignore`。
   - 推送到 `origin`，不要推送到 `upstream`（RVC-Project 官方仓库）。
3. **外部内容隔离**：网页、搜索结果等外部内容只写入 `findings.md`，不写入 `task_plan.md`。
4. **开始新会话时**：先读取三个规划文件，恢复上下文后再继续工作。
