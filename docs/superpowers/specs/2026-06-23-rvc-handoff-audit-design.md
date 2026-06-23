# RVC sensor2mic 配对训练实现审计 + #1 修复 — 设计文档

**日期**：2026-06-23
**分支**：`VC-2-handoff-audit`（从 `VC-2` 拉出）
**对照基准**：[rvc_paired_training_handoff.md](../../../rvc_paired_training_handoff.md)

---

## 1. 背景与动机

`VC-2` 分支已按 handoff 实现了配对训练（sensor 内容 + sensor F0 → 干净 mic 48k），
目的是消除 VC-1 的训练/推理 OOD。但实测结果**并不好**（progress.md 记的"效果良好"判定不成立）。

本工作的目标：**逐条对照 handoff，审计 `VC-2` 的实现，找出"看起来做了但其实没做对"的偏离并修正**，
让 sensor2mic 真正满足 handoff 的一致性约束，从而得到一个**干净、可信**的实验结论。

---

## 2. 头号发现（#1）：训练特征提取 vs 推理，sensor 预处理不一致

handoff §4.3 的真实要求不是"对 sensor 什么都不做"，而是
**"训练时喂给 SensorHubert 的 sensor，与推理时喂给它的，逐字节同款"**。

实际两侧不一致：

| 阶段 | sensor 进 SensorHubert 前的处理链 | 代码位置 |
|---|---|---|
| **推理** | `load_audio(16k)` → **peak-norm**(`÷ max/0.95`, 仅 `max>1` 触发) → **48Hz 高通 `filtfilt`**(零相位) → SensorHubert | `infer/modules/vc/modules.py:165-168` + `infer/modules/vc/pipeline.py:326` |
| **训练特征提取** | `load_audio(16k)` → 直接写 `1_16k_wavs` → `sf.read` → SensorHubert（**无归一化、无高通**） | `infer/modules/train/preprocess.py:195,209` + `infer/modules/train/extract_feature_print.py:65` |

`load_audio` 本身是裸 ffmpeg 解码（`infer/lib/audio.py:33-52`），不归一化、不滤波，
所以推理侧那两步确实是训练侧没有的额外处理。

**后果**：生成器是在"裸 sensor 特征"上训练的，推理时却喂"归一化+高通后的 sensor 特征"
→ 残余 OOD 没有真正消除，只是换了形式。这与"结果不好"高度吻合。

**量级评估（工程判断，非实测）**：
- peak-norm：几乎可忽略（多数为 no-op 或极小标量；HuBERT 对标量增益鲁棒）。
- 48Hz 高通：真实但偏小（48Hz 在语音基频之下，主要切 DC/隆隆声）。
- 综合：中小量级。**因此它是一个"混淆变量(confound)"，让 sensor2mic 实验不干净，但不等于音频是垃圾。**

**修复方案**：在训练侧，对 **sensor**（且仅 sensor）补上与推理**逐字节相同**的
peak-norm + 48Hz 高通 `filtfilt`，**但仍绝不加 `F.layer_norm`**。
两种落点二选一（实现时定，倾向 A）：
- **A. 在 preprocess 写 `1_16k_wavs` 之前处理**：sensor 在 `pipeline_paired` 里先 peak-norm + filtfilt 再写盘，使盘上文件即为"推理同款"。优点：特征提取脚本无需改、`1_16k_wavs` 同时也是 F0 来源，保持单一真相。
- **B. 在 `extract_feature_print.py:readwave` 内处理**：读盘后再 norm+filter。缺点：F0 提取读的是未处理的盘文件，两条下游不一致，易再踩坑。→ **不采用**。

> ⚠️ 非对称陷阱：**sensor 要补高通（对齐推理）；mic(gt) 绝不能加高通（保 48k 全频带目标，§4.1）**。
> 修 #1 时不得顺手给 mic 加滤波。

---

## 3. 逐条审计清单

层 = 在哪验证：**代码**=本开发机可确证；**运行**=需运行机真实权重/数据，产出命令清单交用户执行。

| # | handoff 要求 | 检查点 | 层 | 预判 |
|---|---|---|---|---|
| 1 | §4.3 sensor 训练/推理预处理逐字节一致 | peak-norm + 48Hz 高通两侧对齐；layer_norm 两侧都无 | 代码 | 🔴 偏离 → 修复（见 §2） |
| 2 | §2 sensor→`1_16k_wavs` 16k mono | `load_audio(...,16000)` 写出，确保 mono | 代码 | ✅ |
| 3 | §4.1 gt=`0_gt_wavs` 必须 48k 全频带 mic | 写出用 `self.sr`；且**源 mic 本身是真 48k**（非 16k 升采样） | 代码+运行 | ⚠️ 核源数据 |
| 4 | §2 不做静音切片 | `pipeline_paired` 无 slicer 调用 | 代码 | ✅ |
| 5 | §4.2 每对 mic/sensor 同时长 | 0.05s 阈值仅 WARNING 不阻断；抽样实测时长 | 运行 | ⚠️ |
| 6 | §4.4 stem 1:1、孤儿跳过 | `discover_pairs` 逻辑 + 两目录文件名集合相等、数量相等 | 代码+运行 | ✅代码/⚠️数据 |
| 7 | §3 特征提取 output_layer=12、同一权重 | 训练与推理都 layer=12、同 `sensor_hubert_rvc.pth` | 代码 | ✅ |
| 8 | §4.5 只用目标说话人配对 | `prepare_rvc_trainset` 抽取正确、无串说话人 | 运行 | ⚠️ |
| 9 | §3 F0=rmvpe from sensor、SR=48k、spk=0 | F0 读 `1_16k_wavs`(sensor)；训练配置 spk_id=0、sr=48k | 代码+运行 | ⚠️ |
| 10 | §2 mic(gt) 不加高通 | 确认 `pipeline_paired`/`norm_write_paired` 对 mic 无滤波 | 代码 | ✅ |
| 11 | 跨域 FAISS 索引是否同踩 #1 | 建索引(extract_sensor_features) 用裸特征 vs 查询用归一化特征 | 代码 | ⚠️ 同性质待查 |

---

## 4. 开发机 / 运行机分工

| | 开发机（本机） | 运行机 |
|---|---|---|
| 代码审计 + 修复 | ✅ | — |
| 权重 `sensor_hubert_rvc.pth` | ❌ | ✅ |
| Vibravox 配对数据 | ❌ | ✅ |
| 跑 preprocess/特征/F0/训练/推理 | ❌ | ✅ |

**交付物**：
1. 代码级：在 `VC-2-handoff-audit` 上修复 #1（及其它代码级偏离），附逐条审计结论。
2. 运行级：一份**可直接粘贴执行的命令 + 判据**清单（对应 #3/#5/#6/#8/#9/#11 的运行验证），由用户在运行机执行后回贴结果。

---

## 5. 验证逻辑（修完怎么确认有没有用）

1. 修 #1 后，在运行机重跑：preprocess → 特征提取 → F0 → 训练（实验名建议 `vibravox_spk1_handoff_fix`）。
2. 用**训练集之外**的 sensor 做 A/B：新模型 vs 旧 sensor2mic 模型。
3. 判定：
   - 结果**明显变好** → #1 是主因，旧负面结论确系被它带偏，作废重写。
   - 结果**几乎不变** → #1 非主因，转查非 bug 成因（数据量 126 对~10min、SensorHubert 质量）。

---

## 6. 非 bug 成因（备择解释，本轮不在代码层面追）

- 目标说话人训练数据约 126 对（~10min），处于 RVC 训练数据边界线。
- SensorHubert checkpoint 本身质量上限。
- 法语 Vibravox 数据集与目标音色差异。

仅在 §3 代码审计未发现可解释结果差的偏离时，才回到这些。

---

## 7. 范围与默认

- 审计范围 = §3 的 11 条 + #1 修复。
- 修复尺度 = 代码级可确证偏离直接改并自验；运行级出命令清单交用户；存疑项先标记、等确认再动。
- 不做：训练循环/生成器/损失改动；SensorHubert 重训（那是另一仓库 Stage B）；推理参数 sweep（那是 stage-A 另一交接单）。
