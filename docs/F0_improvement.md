# 振动 Sensor 音频的 F0（音高）提取改进方案

## 背景与问题

链路现状：

- HuBERT 已经重新训练，完成 sensor→mic 的内容特征域适配；
- cross-domain FAISS 索引已建好（`build_cross_domain_index.py` / `3_feature768_sensor`）；
- 拥有振动 sensor 与 mic 的 **paired data**（同步录制）。

剩余瓶颈：RVC 做音色转换时，F0 是从**源音频**（推理时即振动 sensor 输入）提取的。而现有 RMVPE 模型是在干净 mic 音频上训练的，对 sensor 的频谱特性可能提取不准。

问题：如何微调现有模型，使其能从振动 sensor 提取到正确的 pitch？

## 关键认知：F0 在两个信号里是物理上同一个量

声带振动的基频是发声器官的物理属性。振动 sensor（骨传导/接触式）和 mic 在同一时刻观测的是**同一个声门振动源**，所以理论上 sensor 和 mic 的 F0 序列应当完全相等——差别只在谐波结构和频谱包络不同，导致**学习型** pitch 提取器（RMVPE）因输入频谱长得不像 mic 而判断失准。

推论：

- paired data 提供的是一个**精确的**监督信号（而非近似），蒸馏方案理论上能学得很好；
- 骨传导/接触 sensor 通常基频和低次谐波能量很强，F0 信息其实很充分。

## Step 0：先测量，别急着训练

很可能不需要微调。RVC 本身支持多种 F0 方法（RMVPE / Harvest / Crepe / PM / Dio）。先做量化对比：

1. 取一批 held-out 的 paired 片段，都重采样到 16k；
2. 把 **mic 音频经 RMVPE 提取的 F0 当作 ground truth**；
3. 分别对 **sensor 音频**跑 RMVPE、Harvest、Crepe，计算 RPA（±50 cents 内的帧比例）和 voicing 准确率。

判断：

- 算法型方法（Harvest/PYIN/Crepe）只找周期性、不假设 mic 频谱。若 sensor 基频强，**可能开箱即用**——直接换 `f0_method` 即可。
- 若 RMVPE 在 sensor 上的误差主要是**八度跳变**或阈值问题，调整 `thred`（`infer/lib/rmvpe.py:587` 的 `decode` 阈值）也许就够。

只有当所有现成方法误差都明显偏大时，才走下面的微调。

### 评测脚本

仓库根目录的 `eval_sensor_f0.py` 实现了上述对比，直接可用：

1. 编辑脚本顶部 `PAIRED_DATA_DIR` 填入存放 paired wav 的目录；
2. 确认 `assets/rmvpe/rmvpe.pt` 存在（RVC 预训练 RMVPE 权重，需先下载）；
3. 在 RVC 的 python 环境里执行 `python eval_sensor_f0.py`。

脚本行为：

- **配对**：递归扫描目录，按 `*_sensor` / `*_mic` 去后缀的前缀做 key 匹配（支持 wav/flac/mp3/ogg）；
- **抽样**：`random.seed(SEED)` 固定后随机取 `N_PAIRS`（默认 30）；
- **基准**：用 RMVPE 跑 mic 音频得到 ground-truth F0；
- **对比**：对 sensor 跑 RMVPE / Harvest / Crepe，统一到 16k、10ms 帧（`f0_min=50` / `f0_max=1100`，与 `pipeline.py` 一致）；
- **时间对齐**：每个方法在 ±10 帧内搜索最优常数 lag 再打分，并报告 lag 中位数，用于验证 sensor/mic 固定延迟；
- **指标**：RPA、RCA（八度无关）、VR、cents 中位误差、lag，帧加权汇总并附解读。

可调参数（脚本顶部）：`N_PAIRS`、`SEED`、`METHODS`、`MAX_LAG_FRAMES`、`RMVPE_THRED`、`DEVICE` 等。Crepe 在 mps 上不稳，脚本会自动对其退回 CPU。

### 评测集规模与覆盖度

Step 0 只是评测、不是训练，量级不需要很大——目标是让 RPA/voicing 指标的统计噪声足够小，能可靠区分各方法优劣。

**总时长比片段数更重要。** RPA 是在所有 voiced 帧（10ms/帧）上的平均，真正决定置信度的是 voiced 帧总数而非文件数：

- **10~20 分钟的 voiced 语音**通常足够（约 6 万~12 万帧），方法间几个百分点的差异即可稳定区分；
- 换算成片段：每段 5~10 秒时，约 **100~200 个 paired 片段**即可。

**比数量更关键的是覆盖度**，务必照顾：

1. **音高范围**——低音、高音、自然语调起伏都要有；八度错误往往只在音域边缘暴露。
2. **不同说话人/录音条件**——若实际使用涉及多人或多种佩戴位置，评测集要覆盖，否则结论不可迁移。
3. **清浊音都有**——voicing 准确率需要足够的 unvoiced 帧（静音、清辅音）才能测出来。
4. **真正 held-out**——这些片段不能参与任何 HuBERT / 未来 RMVPE 的训练，否则指标虚高。

**实操节奏**：先拿 **20~30 段** 快速跑一轮看趋势（哪个方法明显好/差、是否八度跳变），趋势清楚后再扩到 **100+ 段** 出正式数字。三种方法差距大时小样本即可定论；若咬得很近，才需要更多帧压低噪声。

## Step 1（如需）：把 RMVPE 蒸馏到 sensor 域

RMVPE 是个 E2E 模型（`infer/lib/rmvpe.py:373`）：

```
log-mel(128, 16k, hop=160, fmin=30, fmax=8000)
  → DeepUnet → CNN → BiGRU → Linear(512→360) → Sigmoid
  → 360 维 cents salience
```

微调要点：

**1. 生成标签**：对每个 pair，用 RMVPE 跑 **mic** 音频得到 F0 序列（teacher）。建议再用 Harvest / median filter 交叉清洗、去八度跳，提高标签质量。

**2. 关键：时间对齐**。虽同步录制，但 sensor 与 mic 间常有固定声学/电路延迟（几 ms）。用两条 F0 contour（或能量包络）做互相关估出常数偏移并补偿——错几帧就会显著拖累训练，这一步别省。

**3. 构造训练目标**：把 teacher F0(Hz) 映射到 RMVPE 的 360-bin cents 网格。bin 中心为 `cents = 20*i + 1997.379`（`infer/lib/rmvpe.py:566`），反推 `f0 = 10*2^(cents/1200)`（`infer/lib/rmvpe.py:589`）。在真值 bin 上放高斯软标签（σ ≈ 1~几个 bin，CREPE/RMVPE 原始训练即高斯模糊 one-hot），unvoiced 帧给全零向量。Loss 用 BCE（匹配 Sigmoid 输出）。

**4. 训练**：

- **从预训练 `rmvpe.pt` 初始化**（这才是"微调"，数据量有限不要从头训）；
- 小学习率（~1e-4 或更低），可先冻结 encoder 早期层、主要训 decoder/GRU/FC，再解冻；
- 数据增强：`MelSpectrogram` 自带 `keyshift`（`infer/lib/rmvpe.py:452`）可做 pitch-shift 增强，提升泛化；
- 验证用 RPA / voicing 指标。

**5. 部署**：存成同格式 state_dict，把 `rmvpe_root` / model_path 指过去即可，**RVC 其余代码一行不用改**。

## 一个更弱的备选（不推荐做主方案）

直接拿已适配的 sensor HuBERT 特征接一个小 GRU/MLP 头去回归 mic F0。好处是复用了已对齐的特征空间；但 HuBERT 是 content 特征、设计上对音高近似不变，F0 信息残留不足，效果通常不如直接从波形提取。可作对照，别当主力。

## 推荐路径

1. **先做 Step 0 量化对比**（成本最低，很可能 Harvest/Crepe 直接能用）；
2. 不行再上 **Step 1 蒸馏**。

## 相关代码位置

- RMVPE 类：`infer/lib/rmvpe.py:495`
- E2E 模型：`infer/lib/rmvpe.py:373`
- MelSpectrogram（含 keyshift 增强）：`infer/lib/rmvpe.py:418`
- cents 网格映射：`infer/lib/rmvpe.py:566`
- decode / 阈值：`infer/lib/rmvpe.py:587`
- sensor 特征提取脚本：`extract_sensor_features.py`
