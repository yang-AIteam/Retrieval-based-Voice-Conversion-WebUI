# 发现与研究

## 需求
- 使用 planning-with-files 生成并维护 task_plan.md、findings.md、progress.md
- 三个文件需同步到 GitHub，有新进展时必须更新（见 CLAUDE.md）
- 项目目标待用户确认（见 task_plan.md）

## 项目现状
- 仓库：yang-AIteam/Retrieval-based-Voice-Conversion-WebUI（origin），上游为 RVC-Project（upstream）
- 当前分支：VC-1-K；其他分支：main、VC-1、VC-2、VC-2-handoff-audit、VC-2-mic2mic
- 最近提交 d7834c7 “切换到 kushinada 日语 hubert 权重路径”，改动 4 个文件，每个文件 1 行：
  - infer/lib/rtrvc.py
  - infer/modules/train/extract_feature_print.py
  - infer/modules/vc/utils.py
  - tools/rvc_for_realtime.py
- 提交 f4d227a 已 Revert 上述 d7834c7，4 个加载点目前恢复为原 hubert_base.pt 路径（Revert 原因待用户说明）
- 其他近期提交：移动 logs 文件、修复推理 UI 问题、修复特征提取问题、修正 curl 命令

## 技术发现
- .gitignore 已忽略 hubert_base.pt、/logs、/weights、/pretrained 等，大文件不会被误提交
- 规划文件在仓库根目录，未被 .gitignore 忽略，可直接提交

## 资源
- 规划技能模板：~/.claude/plugins/cache/planning-with-files/planning-with-files/2.43.0/templates/

## 待查
- kushinada 权重的来源、格式与存放路径
- 各加载点是否还有遗漏的 hubert_base.pt 引用
