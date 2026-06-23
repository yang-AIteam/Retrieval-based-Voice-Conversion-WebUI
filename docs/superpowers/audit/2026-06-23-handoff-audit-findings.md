# Handoff 审计结论 (2026-06-23, 分支 VC-2-handoff-audit)

> 逐条对照 rvc_paired_training_handoff.md 审计 VC-2 配对训练实现。代码级证据如下。

| # | handoff 要求 | 证据(文件:行) | 结论 | 备注 |
|---|---|---|---|---|
| 1 | §4.3 sensor 训练/推理预处理逐字节一致 | preprocess.py:195,209 / modules.py:165-168 / pipeline.py:326 | 🔴 偏离 | Task 2/3 修复 |
| 2 | §2 sensor->1_16k_wavs 16k mono | preprocess.py:195,210 | ✅符合 | load_audio(sensor_path, 16000) 写入 1_16k_wavs |
| 3 | §4.1 gt 必须真 48k 全频带 mic | preprocess.py:150 / pipeline.py:24 | ⚠️ 需运行机验证 | norm_write_paired 用 self.sr 写，但管道内其他路径未过滤 |
| 4 | §2 配对不切片 | preprocess.py:185-216 | ✅符合 | pipeline_paired 无 slicer 调用；slicer 仅在 pipeline 行163 |
| 5 | §4.2 mic/sensor 同时长 | preprocess.py:200 仅 WARNING 不阻断 | ⚠️ 需运行机验证 | 运行机抽样实测 |
| 6 | §4.4 stem 1:1 孤儿跳过 | preprocess.py:35-84 | ✅符合 | discover_pairs 检测 mic/sensor 同名，孤儿记 log 跳过 |
| 7 | §3 layer=12 同权重 | extract_feature_print.py:118 / extract_sensor_features.py:105 | ✅符合 | 均指定 output_layer=12，都用 sensor_hubert_rvc.pth |
| 8 | §4.5 只用目标说话人 | — | ⚠️ 需运行机验证 | 运行机核 prepare_rvc_trainset |
| 9 | §3 F0=rmvpe(sensor) SR=48k spk=0 | preprocess.py:195 读 1_16k_wavs(=sensor) | ⚠️ 需运行机验证 | 运行机核训练配置 |
| 10 | §2 mic(gt) 不加高通 | preprocess.py:133-153 / 160,326 | 🔴 偏离 | norm_write_paired 未过滤；但 pipeline 行160 过滤 (非配对路径) |
| 11 | 跨域 FAISS 是否同踩 #1 | extract_sensor_features.py:76-83 | 🔴 偏离 | 详见下 |

## #11 详情

**结论：extract_sensor_features.py 不符合推理侧预处理，Task 5 应触发。**

对比证据：
- **推理侧** (modules.py:165-168)：`load_audio() → peak-norm(÷max/0.95) → high-pass filtfilt(bh,ah)`
- **FAISS 训练侧** (extract_sensor_features.py:76-83)：
  ```python
  def readwave(wav_path):
      wav, sr = sf.read(wav_path)          # 仅读取
      assert sr == 16000
      feats = torch.from_numpy(wav).float()
      if feats.dim() == 2:
          feats = feats.mean(-1)
      assert feats.dim() == 1
      return feats.view(1, -1)
  ```
  **无 peak-norm、无高通滤波，直接送模型**。

这与 #1 同源：extract_sensor_features.py 逐字节复刻 preprocess.py 的 pipeline_paired 读 sensor，都缺少推理侧的规范化处理。训练 FAISS 时的 sensor 特征与推理时的 sensor 特征因此产生域偏移。

**Task 5 条件成立：需修复 extract_sensor_features.py readwave() 以应用 peak-norm + 48Hz 高通，使 FAISS 训练与推理一致。**
