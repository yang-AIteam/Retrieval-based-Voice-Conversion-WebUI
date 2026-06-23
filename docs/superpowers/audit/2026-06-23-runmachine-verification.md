# 运行机验证 runbook (2026-06-23, #1 修复后)

> 在**运行机**执行（有 SensorHubert 权重 + Vibravox 配对数据）。`<exp>` 替换为实验目录，建议 `vibravox_spk1_handoff_fix`。每步附判据，跑完回贴结果。

## 0. 同步代码
```bash
git fetch && git checkout VC-2-handoff-audit && git pull
```

## 1. 重跑预处理（命令行）
> 命令行直接跑时必须**先手动建实验目录**——preprocess.py 在模块加载时就 open 日志文件，但建目录的 makedirs 在更靠后，目录不存在会 `FileNotFoundError`。走 WebUI 不会遇到（WebUI 先建目录）。
> `<inp_root>` = 同时含 mic+sensor 的配对目录（子目录 `mic/`+`sensor/` 或后缀 `*_mic.wav`/`*_sensor.wav`）。
```bash
mkdir -p logs/<exp>
python infer/modules/train/preprocess.py <inp_root> 48000 4 logs/<exp> False 3.7
```
参数：`<inp_root> sr n_p exp_dir noparallel per`。配对模式下 `per`(3.7) 不生效（不切片）、只为占位；`noparallel=False` 表示用多进程并行。

**必须先确认配对生效：** preprocess 日志出现 `detected paired dataset: <N> pairs`（preprocess.py:226）。
若没有这行 → 退化成单目录切片、配对没生效、#1 修复白做 → 停下检查 `<inp_root>` 摆法。

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
读 `1_16k_wavs/`（已修 #1 的 sensor）→ SensorHubert → 写 `3_feature768/`。
```bash
python infer/modules/train/extract_feature_print.py cuda 1 0 logs/<exp> v2 True
```
参数：`device n_part i_part exp_dir version is_half`（device 会自动检测 cuda/mps/cpu，除 DirectML 外只是占位）。
- `n_part=1 i_part=0`：单进程；多 GPU 才分片。
- **`version=v2` 必填** → 768 维到 `3_feature768/`（v1 是 256 维）。
- `is_half=True`：CUDA 上可用；报错改 `False`（mps/cpu 会自动不 half）。

**判据：** `3_feature768/` 的 .npy 数量 = 样本数；抽一个确认 768 维：
`python -c "import numpy as np,glob; a=np.load(sorted(glob.glob('logs/<exp>/3_feature768/*.npy'))[0]); print(a.shape)"` → (T, 768)。
提取日志须显示加载 `assets/hubert/sensor_hubert_rvc.pth`（SensorHubert）、无 layer_norm 报错。

## 4. F0 提取（rmvpe, #9）
读 `1_16k_wavs/`（=sensor）→ rmvpe → 写 `2a_f0/`、`2b-f0nsf/`。注意脚本在 `extract/` 子目录下。
```bash
python infer/modules/train/extract/extract_f0_rmvpe.py 1 0 0 logs/<exp> True
```
参数：`n_part i_part i_gpu exp_dir is_half`（单 GPU 填 `1 0 0`）。

**判据：** `2a_f0/`、`2b-f0nsf/` 数量与样本数一致（F0 读的是 `1_16k_wavs`=sensor）。

## 5. 只用目标说话人（#8）
**判据：** `ls /app/data/rvc_trainset` 全部来自同一说话人 id，无串说话人；数量 ≈ 126 对。

## 6. 开训 + A/B
- 实验名 `vibravox_spk1_handoff_fix`，与旧 sensor2mic 模型同超参（便于对照）。
- 用**训练集之外**的 sensor 推理，A/B：新模型 vs 旧 sensor2mic 模型。
- **判据/判定：**
  - 明显变好 → #1 是主因，旧负面结论作废重写。
  - 几乎不变 → 转查非 bug 成因（数据量 126 对~10min、SensorHubert 质量，见 spec §6）。
