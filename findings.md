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

## 安全：CVE-2024-0964（Gradio 本地文件包含 / 路径穿越）评估（2026-10-08）
- 漏洞：API 请求中用户可控的 JSON 值可远程触发本地文件包含，可读取服务器上的任意文件（来源：用户提供的 CVE 描述；osv.dev、GitLab Advisory）
- 影响版本 gradio < 4.9.0，4.9.0 修复；CVSS 7.5（部分来源 9.4）。本项目 gradio==3.34.0，属于受影响范围
- AWS 上的主要风险：读取 ~/.aws/credentials、.env、SSH 私钥、/proc/self/environ（环境变量中的 AWS 密钥）、模型与训练数据
- 与 CVE-2024-47167 的区别：IMDSv2 对这个漏洞无效，关键是服务器上不要以文件或环境变量形式存放长期密钥
- 缓解：与 CVE-2024-47167 相同的网络隔离（不对外开放端口、只用 SSH 隧道或 SSM 访问、不用时不启动 WebUI）；使用 IAM 角色而不是 Access Key；以非 root 用户运行；Docker 只挂载必要目录
- 修复版本核实（2026-10-08）：GitHub Advisory GHSA-f3h9-8phc-6gvh（经 OSV API 读取）写明 PyPI gradio 影响范围 introduced 0、fixed 4.9.0；修复提交 d76bcaa “Fix api event drops (#6556)”，提交时间 2023-12-12 23:24 UTC，PyPI 上 4.9.0 发布于 2023-12-13 02:37 UTC，时间上吻合。CVE 原始记录（huntr 提交）本身没有给出版本号；没有在 git 里直接确认该提交属于 4.9.0 标签（GitHub compare API 返回 diverged）

## 安全：CVE-2024-4325（Gradio SSRF，/queue/join + save_url_to_cache）评估（2026-10-08）
- 漏洞：用户传入的 path 被当作 URL 发起请求，没有校验，可访问内网或 AWS 元数据服务；NVD CVSS 8.6 High（来源：用户提供的 NVD 描述）
- OSV（GHSA-973g-55hp-3frw / PYSEC-2026-1413）范围为 introduced 0、last_affected 4.36.0（没有标注修复版本）；用 gradio 3.34.0 查询 OSV 时会命中这个 CVE
- 静态阅读 gradio 3.34.0 wheel 源码（未实际复现）：
  - 没有 save_url_to_cache 函数（这是 4.x 的代码），但有作用相同的旧实现
  - gradio/components.py `Audio.preprocess`：is_file=True 且 name 是 URL 时，先调用 utils.validate_url（HEAD，遇到 403/405 再 GET），再调用 download_temp_copy_if_needed（requests.get 下载到临时目录），没有任何 IP 或域名限制 → 完整的 SSRF，并且内容会落盘
  - gradio/routes.py `/file={path_or_url}`：对任意用户输入调用 validate_url，服务器会向该 URL 发起 HEAD/GET，再 302 跳转 → 盲 SSRF（看不到响应内容，但可探测内网），所有 3.34 应用都存在，不依赖具体组件；只有设置了 auth 时才需要登录
- 本项目中的可达性：
  - infer-web.py：gr.Audio 只作为输出（vc_output2，第 944 行），不会触发 Audio.preprocess；gr.File 在 3.34 中不会下载 URL；但 /file= 的盲 SSRF 仍然存在
  - tools/app.py：vc_input3 = gr.Audio 作为输入 → 完整 SSRF 可达；但 launch() 默认只监听 127.0.0.1
- AWS 上的风险：通过完整 SSRF 访问 IMDSv1 可拿到 IAM 凭证；盲 SSRF 只能探测，读不到凭证。IMDSv2 对这两条路径都有效（它们只发 HEAD/GET，没有 PUT）
- 结论：缓解措施与 CVE-2024-47167 相同（网络隔离 + 强制 IMDSv2 + 最小权限的 IAM 角色）
- 这些代码分析同样适用于 CVE-2024-47167 在 3.34 上的触发路径

## 安全：CVE-2024-0964 在 gradio 3.34.0 上的源码分析（2026-10-09，静态阅读，未复现）
- gr.File.preprocess（components.py:2733）：is_file=True 时直接对用户传入的 name 调用 make_temp_copy_if_needed → shutil.copy2，可以把服务器上的任意文件复制到 /tmp/gradio/<文件内容的 sha1>/<文件名>，没有路径限制。这和 CVE 描述的“通过用户提供的 JSON 值包含本地文件”是同一类问题
  - 要取回复制出来的文件，需要知道文件内容的 sha1，事先不知道内容就无法算出路径 → 直接读取较难
  - infer-web.py 有 3 个 gr.File 输入（f0_file:921、inputs:1076、wav_inputs:1126），所以这条路径可达
  - f0_file 解析失败时只在服务器端打印 traceback，不会返回给前端（pipeline.py:349）
- /file= 路由（routes.py:342 起）：app.cwd（=os.getcwd()，即项目根目录）下的文件，只要路径中没有以 . 开头的部分，就允许未认证下载 → 模型权重（assets/weights）、logs/ 下的训练数据和特征文件、配置文件等，谁都能下载。这是 3.34 本身的设计，不依赖 CVE
  - .env、~/.aws 等以 . 开头的路径会被 /file= 拒绝（403）
- WebUI 自身的功能允许指定服务器上的文件夹路径（批量转换的输入/输出、训练数据目录等），设计上就不适合对外公开
- 结论修正：CVE-2024-0964 这条路径本身“直接读取任意文件较难”，但项目目录下的模型和训练语音数据在当前公开状态下可以被直接下载，整体判断为高风险

## 安全：CVE-2024-47871（Gradio share=True 时 FRP 隧道未加密）评估（2026-10-09）
- 漏洞：share=True 时，FRP 客户端与服务器之间没有强制 HTTPS，通信内容（包括上传的文件）可被窃听或篡改；NVD CVSS 9.1 Critical（来源：用户提供的 NVD 描述）
- OSV（GHSA-279j-x4gx-hfrh / PYSEC-2024-219）：影响范围 introduced 0、fixed 5.0.0；用 3.34.0 查询会命中
- gradio 3.34.0 tunneling.py `_start_tunnel`：frpc 启动参数中没有 TLS 相关选项，和 CVE 描述一致（静态阅读）
- 本项目只有 infer-web.py:1612 使用 share=True，条件是 config.iscolab，而它只在加了 --colab 命令行参数时为真（configs/config.py:82）
  - --colab 只出现在两个 Colab 笔记本（Retrieval_based_Voice_Conversion_WebUI*.ipynb）中；run.sh、Dockerfile、go-web*.bat 都不带这个参数
  - tools/app.py 的 launch() 也没有使用 share
- 结论：在 AWS 上按常规方式启动（不加 --colab）时，不会建立 FRP 隧道，不受影响 → 低风险
- 另外：7865 端口本身是明文 HTTP，如果直接对外公开，也会被窃听。这不属于本 CVE，但属于同类问题；改用 SSH 隧道或 SSM 访问可以一并解决
