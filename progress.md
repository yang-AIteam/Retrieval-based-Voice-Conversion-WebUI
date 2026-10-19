# 进度日志

## 会话：2026-10-01

### 阶段 1：需求与现状梳理
- **状态：** in_progress
- 开始时间：2026-10-01
- 执行的操作：
  - 确认已加载插件：superpowers、planning-with-files
  - 创建 task_plan.md、findings.md、progress.md
  - 创建 CLAUDE.md，规定规划文件需同步到 GitHub 并随进展更新
- 创建/修改的文件：
  - task_plan.md
  - findings.md
  - progress.md
  - CLAUDE.md

## 会话：2026-10-05

### 阶段 1：需求与现状梳理
- **状态：** in_progress
- 执行的操作：
  - 读取三个规划文件，恢复上下文
  - 发现最新提交 f4d227a 已 Revert kushinada 权重路径切换，规划文件此前未记录
  - 等待用户确认今日目标及 Revert 原因


## 会话：2026-10-08
- 评估 CVE-2024-47167 对本项目在 AWS 部署时的影响，结论写入 findings.md（未修改代码）
- 评估 CVE-2024-0964、CVE-2024-4325；下载 gradio 3.34.0 wheel 到 scratchpad 做静态源码分析（未执行、未复现）

## 测试结果
| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
|      |      |         |         |      |

## 错误日志
| 时间 | 错误 | 尝试次数 | 解决方案 |
|------|------|---------|---------|
|      |      | 1       |         |

## 五问重启检查
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段 1 |
| 我要去哪里？ | 阶段 2–5 |
| 目标是什么？ | 日语 hubert（kushinada）适配（待用户确认） |
| 我学到了什么？ | 见 findings.md |
| 我做了什么？ | 创建规划文件与 CLAUDE.md |

## 会话：2026-10-19
- 用户反馈微调时 feature index 每次都无法生成，询问手动方法
- 阅读 infer-web.py `train_index()`，编写手动构建索引脚本（未纳入仓库，放在 scratchpad），语法检查通过；未在真实数据上运行
- 用户询问能否用 build_cross_domain_index.py 代替：确认其用途不同（跨域 sensor→mic），当前分支无该文件及配套 pipeline 改动
- 用户需在另一台 PC 上构建索引：将 train_index_manual.py 加入仓库 tools/ 并推送（此前仅在本机 scratchpad）
- 将 tools/train_index_manual.py 同步提交到 VC-1 分支（c0258fc，用户另一台 PC 在 VC-1 上工作）
- 修复 VC-1 的 pipeline.py：仅当 cross_values_path 与 file_index 不同时才加载 cross_domain_mic_values.npy（cec22cc，已推送 origin/VC-1）。用模拟文件验证分支逻辑：普通 index→reconstruct_n，跨域 index→加载 npy；未进行实际推理测试
- 用户询问是否应切换到 VC-2：对比两分支，建议当前实验继续在 VC-1；仅在需要配对训练时切换
- 按用户要求 revert VC-1 上的 c0258fc、cec22cc（b011e51、5f85408，已推送；内容与 c24c706 一致）。用户将在另一台 PC 改用 VC-1-K
- 用户更正：VC-1-K 不支持新 hubert，继续用 VC-1。将 np.load 修复 cherry-pick 回 VC-1（0af5350，已推送）；train_index_manual.py 未恢复
