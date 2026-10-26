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
- 用户反馈 VC-1 的 tools 里找不到 train_index_manual.py：原因是之前按"VC-1 回退"要求执行的 revert 5f85408 一并删除了该脚本，之后重新应用时只恢复了 np.load 修复（0af5350）。已从 c0258fc 恢复脚本并推送到 origin/VC-1（bdd3a30）。
- 按用户要求将 split_channels_new.py 从 VC-1-K 复制到 VC-1 并推送 origin/VC-1（语法检查通过，未实际运行）
- 用户在 VC-1-K 训练报错 `FileNotFoundError: /app/logs/mute/0_gt_wavs/mute48k.wav`。原因：VC-1-K 的提交 dc8d949（2026-06-16 "move logs files"）删除了 logs/mute/ 下 8 个文件，而训练生成 filelist 时会加入 mute 样本。已从 dc8d949^ 恢复 logs/mute（与 VC-1 相同）并推送
- 用户确认：logs/<实验名>/ 有 total_fea.npy、无 trained_*.index，WebUI 卡在 faiss index.train。建议直接用 total_fea.npy（已是 KMeans 后的 10000 中心）在单线程下构建 index，无需重跑 KMeans

## 2026-10-23 模型交付清单（VC-1 SensorHubert 推理）
- 确认 VC-1 推理加载链路：`infer/modules/vc/utils.py` / `infer/lib/rtrvc.py` → `rvc_hubert_loader.load_sensor_hubert(pth="assets/hubert/sensor_hubert_rvc.pth", config="assets/hubert/hf_model")`，只用 `HubertConfig.from_pretrained`，`hf_model/` 下需要 `config.json`。
- 交付给他人需要：VC-1 分支代码（含 rvc_hubert_loader.py、transformers 依赖）、`assets/hubert/sensor_hubert_rvc.pth`、`assets/hubert/hf_model/config.json`、`assets/weights/<exp>.pth`、`logs/<exp>/added_*.index`（可选但推荐）、`assets/rmvpe/rmvpe.pt`（使用 rmvpe 提取 F0 时）。预训练底模、logs 中间文件不需要。

## 2026-10-23 无历史交付 VC-1 代码
- 建议用 `git archive --format=zip --prefix=RVC-VC1/ -o RVC-VC1.zip origin/VC-1` 导出最终快照（只含被跟踪文件，不含 .git 历史）。
- 已确认 origin/VC-1 跟踪了 `.env`（仅路径配置，无敏感信息）和 `logs/mute/`（训练用，推理不需要，保留无妨），不含规划文件。
- 权重、hubert、index、rmvpe 不在 git 中，需解压后另行放入对应目录。

## 2026-10-26 assets/weights 位置说明
- 训练小模型由 `infer/lib/train/process_ckpt.py` 的 `savee()` 保存到 `assets/weights/<name>.pth`（.env 中 weight_root 也指向此处）。
- VC-1 的 docker-compose.yml 把宿主机 `./weights` 挂载到容器 `/app/assets/weights`，所以在宿主机上模型出现在仓库根目录的 `weights/`，而不是 `assets/weights/`；容器内路径仍为 `/app/assets/weights`。
- git 中 `assets/weights/` 只跟踪一个隐藏的 `.gitignore`，文件管理器默认看不到，容易误以为该文件夹不存在。

## 2026-10-26 VC-1-K 模型交付清单
- VC-1-K 推理加载 `assets/hubert/hubert_base.pt`（fairseq，路径写死在 `infer/modules/vc/utils.py` 与 `infer/lib/rtrvc.py`），无 rvc_hubert_loader.py，不使用 hf_model/config.json。
- 交付需要：VC-1-K 代码、`assets/hubert/hubert_base.pt`（自训 HuBERT 须为 fairseq 格式并以此文件名替换）、`assets/weights/<exp>.pth`（Docker 下在宿主机 `weights/`）、`added_*.index`（推荐）、`assets/rmvpe/rmvpe.pt`（选 rmvpe 时）。
- VC-1-K 跟踪了 CLAUDE.md 与三个规划文件，`git archive` 导出时需排除。

## 2026-10-26 新建分支 VC-1-J
- 从 VC-1（b9dccd9）分出 `VC-1-J`，已推送到 origin 并设置跟踪 `origin/VC-1-J`。未推送到 upstream。

## 2026-10-26 VC-1-J 注释日语化
- VC-1-J：`infer/lib/rtrvc.py` 中 6 行中文注释/文档字符串译为日语；`infer/modules/vc/utils.py` 无中文，未改动。
- 注意：rtrvc.py 为 CRLF 换行，首次用文本模式写回导致全文件 diff，已改为二进制读写保留换行后重做。py_compile 通过。

## 2026-10-26 VC-1-J requirements 检查
- requirements.txt 与 requirements_rvc.txt 内容完全相同（后者由 701f558 复制加入）；Dockerfile 使用 requirements.txt。
- 扫描全部 import：WebUI 推理/训练路径缺 `transformers`（rvc_hubert_loader.py）。已在 VC-1-J 两个文件末尾追加，提交并推送。
- torch/torchaudio 未显式列出，由 fairseq==0.12.2 依赖间接安装。
- 风险：load_sensor_hubert 使用严格 load_state_dict，transformers/torch 版本若与导出 sensor_hubert_rvc.pth 时不同，pos_conv weight_norm 的键名（weight_g/weight_v vs parametrizations）可能不一致，建议固定为训练机上的版本。
- 实时 GUI/API 额外需要 sounddevice、FreeSimpleGUI；UVR5 需要 opencv-python；ONNX 导出需要 onnx、onnxsim。均未加入。

## 2026-10-26 VC-1-J 固定依赖版本
- 用户提供训练机版本：transformers 4.57.6、torch 2.8.0、torchaudio 2.8.0。已写入 VC-1-J 的 requirements.txt / requirements_rvc.txt 并推送。
- 注意：Linux 上 pip 默认安装的 torch 2.8.0 为 CUDA 12.8 版，需要较新的 NVIDIA 驱动；Dockerfile 基础镜像仍为 cuda:11.6.2，但 pip 版 torch 自带 CUDA 运行库，主要受宿主机驱动限制。

## 2026-10-26 实时（online）推理入口调查（VC-1-J）
- 实时变声 GUI：`gui_v1.py`（Windows 用 go-realtime-gui.bat），引擎为 `infer/lib/rtrvc.py`，已改为加载 SensorHubert，可直接用。
- API 版：`api_231006.py` / `api_240604.py`（FastAPI，端口 6242）使用 `tools/rvc_for_realtime.py`，其中仍写死 `hubert_base.pt`，未适配 SensorHubert。
- gui_v1.py 依赖 FreeSimpleGUI、sounddevice（不在 requirements.txt；requirements-win-for-realtime_vc_gui.txt 写的是 PySimpleGUI，与代码不一致）。
- 实时版需要本机声卡输入输出，Docker 容器内一般无法使用，应在接 sensor 麦克风的本地电脑上运行。
