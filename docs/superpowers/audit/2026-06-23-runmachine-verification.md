# 运行机验证 runbook (2026-06-23, #1 修复后)

> 在**运行机**执行（有 SensorHubert 权重 + Vibravox 配对数据）。`<exp>` 替换为实验目录，建议 `vibravox_spk1_handoff_fix`。每步附判据，跑完回贴结果。

## 0. 同步代码
```bash
git fetch && git checkout VC-2-handoff-audit && git pull
```

## 1. 重跑预处理 → 特征 → F0（WebUI 或命令行）
```bash
python infer/modules/train/preprocess.py /app/data/rvc_trainset 48000 4 logs/<exp> False 3.7
```
**判据：**
- `0_gt_wavs/` 与 `1_16k_wavs/` 文件名集合完全相同、数量相同（#6）：
  `diff <(ls logs/<exp>/0_gt_wavs/) <(ls logs/<exp>/1_16k_wavs/)` 无输出。
- `0_gt_wavs/` 抽一条确认是 **48k**（#3/#4.1）：
  `python -c "import soundfile as sf,glob; f=sorted(glob.glob('logs/<exp>/0_gt_wavs/*.wav'))[0]; print(f, sf.info(f).samplerate)"` → 48000。
- `1_16k_wavs/` 抽一条确认是 **16k**：同上换目录 → 16000。
- 抽一对同名文件确认时长一致（#5，差 < 0.05s）：
  `python -c "import soundfile as sf; a=sf.info('logs/<exp>/0_gt_wavs/STEM.wav'); b=sf.info('logs/<exp>/1_16k_wavs/STEM.wav'); print(a.frames/a.samplerate, b.frames/b.samplerate)"`.

## 2. 确认源 mic 是真 48k 全频带（#3 关键坑）
```bash
python -c "import soundfile as sf,glob; f=sorted(glob.glob('/app/data/rvc_trainset/**/*mic*.wav', recursive=True))[0]; i=sf.info(f); print(f, i.samplerate, i.frames)"
```
**判据：** 源 mic 采样率为 48000，且频谱有 8kHz 以上能量（非 16k 升采样的假 48k）。
可选频谱核对：`python -c "import librosa,numpy as np; y,_=librosa.load('FILE',sr=48000); S=np.abs(librosa.stft(y)); print('hi-band energy', S[300:].mean())"`。

## 3. 特征提取（#7）
**判据：** `3_feature768/` 的 .npy 数量 = 样本数；抽一个确认 768 维：
`python -c "import numpy as np,glob; a=np.load(sorted(glob.glob('logs/<exp>/3_feature768/*.npy'))[0]); print(a.shape)"` → (T, 768)。
提取日志含 SensorHubert 加载、无 layer_norm 报错。

## 4. F0 提取（#9）
**判据：** `2a_f0/`、`2b-f0nsf/` 数量与样本数一致（F0 读的是 `1_16k_wavs`=sensor）。

## 5. 只用目标说话人（#8）
**判据：** `ls /app/data/rvc_trainset` 全部来自同一说话人 id，无串说话人；数量 ≈ 126 对。

## 6. 开训 + A/B
- 实验名 `vibravox_spk1_handoff_fix`，与旧 sensor2mic 模型同超参（便于对照）。
- 用**训练集之外**的 sensor 推理，A/B：新模型 vs 旧 sensor2mic 模型。
- **判据/判定：**
  - 明显变好 → #1 是主因，旧负面结论作废重写。
  - 几乎不变 → 转查非 bug 成因（数据量 126 对~10min、SensorHubert 质量，见 spec §6）。
