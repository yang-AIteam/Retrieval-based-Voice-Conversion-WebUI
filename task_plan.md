# 任务计划：RVC 日语变声（kushinada hubert）适配

## 目标
基于 RVC WebUI，将特征提取模型切换为日语 hubert（kushinada）权重，并完成训练、推理、实时变声链路的端到端验证。

> 待确认：以上目标由 git 历史（分支 VC-1-K、最新提交“切换到 kushinada 日语 hubert 权重路径”）推断，请用户核对并修正。

## 当前阶段
阶段 1

## 阶段

### 阶段 1：需求与现状梳理
- [x] 确认当前分支（VC-1-K）及最近提交
- [ ] 与用户确认最终目标与验收标准
- [ ] 梳理 hubert 权重的所有加载点（rtrvc.py、extract_feature_print.py、vc/utils.py、rvc_for_realtime.py）
- [ ] 将发现记录到 findings.md
- **状态：** in_progress

### 阶段 2：方案与结构
- [ ] 确定 kushinada 权重的存放路径与获取方式
- [ ] 确认权重格式与 fairseq 加载方式是否兼容
- [ ] 记录决策及理由
- **状态：** pending

### 阶段 3：实现
- [ ] 完成各加载点的权重路径切换
- [ ] 检查特征提取（v1 为 256 维 / v2 为 768 维）是否匹配
- [ ] 增量测试
- **状态：** pending

### 阶段 4：测试与验证
- [ ] 特征提取脚本能正常运行
- [ ] 推理（WebUI）能正常出声
- [ ] 实时变声（gui_v1.py）能正常运行
- [ ] 将测试结果写入 progress.md
- **状态：** pending

### 阶段 5：交付
- [ ] 复查所有改动文件
- [ ] 更新文档与 .gitignore（如需要）
- [ ] 同步到 GitHub 并交付
- **状态：** pending

## 关键问题
1. 最终目标是仅替换 hubert 权重，还是也需要重新训练日语底模？
2. kushinada 权重的来源与格式是什么，是否与 fairseq 的 checkpoint 兼容？
3. 推理、训练、实时变声三条链路是否都需要验证？

## 已做决策
| 决策 | 理由 |
|------|------|
| 使用 planning-with-files 管理任务 | 任务跨多次会话，需要持久化记录 |
| 三个规划文件纳入 git 并同步到 GitHub | 用户要求，详见 CLAUDE.md |

## 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
|      |         |         |

## 备注
- 阶段状态按 pending → in_progress → complete 更新
- 重大决策前重新阅读本计划
- 所有错误都要记录，不重复失败的操作
- 外部/网页内容只写入 findings.md，不写入本文件
