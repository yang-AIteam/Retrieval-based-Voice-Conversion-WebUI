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

## 安全：CVE-2024-47167（Gradio SSRF）评估（2026-10-08）
- 漏洞：Gradio `/queue/join` 中 `async_save_url_to_cache` 存在 SSRF，影响 gradio < 5.0（来源：用户提供的 CVE 描述）
- 本项目锁定 `gradio==3.34.0`（requirements*.txt、pyproject.toml），属于受影响范围；未在 3.34 源码中逐行核实具体触发路径
- 暴露面：infer-web.py:1614 以 `server_name="0.0.0.0"` 监听、无认证；Colab 模式（iscolab）用 `share=True` 生成公网链接
- 使用了 gr.File / gr.Audio 等文件类组件
- AWS 上的主要风险：通过 SSRF 访问 EC2 元数据服务 169.254.169.254，若允许 IMDSv1 可窃取 IAM 角色临时凭证；也可能访问 VPC 内部服务
- 缓解：强制 IMDSv2（HttpTokens=required，容器内 hop limit=1）；IAM 角色最小权限或不挂角色；安全组只对可信 IP 开放 WebUI 端口（默认 7865）；不使用 share=True；必要时前置带认证的反向代理或使用 SSH 隧道 / SSM 端口转发访问
- 升级到 gradio 5 需要大量改 UI 代码（如 queue(concurrency_count=...) 在 4.x 已移除），工作量较大
- 补充（2026-10-08）：项目中启动 Gradio 服务的入口只有 infer-web.py 和 tools/app.py（`app.launch()` 未指定 server_name，默认只监听 127.0.0.1:7860）
- Dockerfile（CMD python3 infer-web.py，EXPOSE 7865）和 run.sh 会自动启动 infer-web.py
- api_231006.py / api_240604.py 是 FastAPI 服务，监听 0.0.0.0:6242、无认证（/config、/start、/stop 等）。它们不受该 CVE 影响，但同样需要用安全组限制访问
