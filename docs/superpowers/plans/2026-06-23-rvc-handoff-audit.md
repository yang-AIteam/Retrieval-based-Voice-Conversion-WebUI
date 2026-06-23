# RVC sensor2mic 配对训练审计 + #1 修复 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 逐条对照 [rvc_paired_training_handoff.md](../../../rvc_paired_training_handoff.md) 审计 `VC-2` 的配对训练实现，并修复头号偏离 #1（训练特征提取喂给 SensorHubert 的 sensor 未做与推理相同的 peak-norm + 48Hz 高通），消除残余 OOD。

**Architecture:** 把"与推理逐字节对齐的 sensor 预处理"抽成一个**纯函数模块** `infer/lib/sensor_preprocess.py`（仅依赖 numpy/scipy，可在开发机单测），在 `preprocess.py` 的配对路径写盘前调用它。其余审计项产出两份文档：代码级审计结论 + 运行机验证 runbook。**不动**推理路径、训练循环、生成器、损失、SensorHubert。

**Tech Stack:** Python, numpy, scipy.signal, pytest。参照标准来源：`infer/modules/vc/modules.py:165-168`（peak-norm）、`infer/modules/vc/pipeline.py:24,326`（48Hz 高通 filtfilt）。

**分支:** `VC-2-handoff-audit`（已从 `VC-2` 拉出）。

---

## 文件结构

| 文件 | 角色 | 动作 |
|---|---|---|
| `infer/lib/sensor_preprocess.py` | 与推理对齐的 sensor 预处理纯函数（训练侧单一真相） | 新建 |
| `tests/test_sensor_preprocess.py` | helper 单元测试（开发机可跑） | 新建 |
| `infer/modules/train/preprocess.py` | 配对路径 `pipeline_paired` 写 `1_16k_wavs` 前调用 helper | 修改 |
| `docs/superpowers/audit/2026-06-23-handoff-audit-findings.md` | 11 条审计结论（代码级证据） | 新建 |
| `docs/superpowers/audit/2026-06-23-runmachine-verification.md` | 运行机验证 runbook（命令+判据） | 新建 |
| `xmodal_hubert/scripts/extract_sensor_features.py` | 跨域 FAISS 特征提取（#11，条件修复） | 条件修改 |

---

## Task 1: 代码级审计 pass（产出 findings 文档）

**Files:**
- Create: `docs/superpowers/audit/2026-06-23-handoff-audit-findings.md`

本任务是**调查 + 记录**，不改业务代码、无测试。逐条核对并把证据（文件:行号）写进 findings 文档。

- [ ] **Step 1: 建目录**

Run: `mkdir -p docs/superpowers/audit`

- [ ] **Step 2: 核对每条审计项**（逐条执行命令，记录结论）

```bash
# #2 sensor->1_16k_wavs 16k mono
grep -nE "load_audio\(sensor_path, 16000\)|wavs16k_dir" infer/modules/train/preprocess.py
# #4 不切片: pipeline_paired 内不应出现 slicer
grep -n "slicer" infer/modules/train/preprocess.py   # 应只在 pipeline()(原版) 出现, pipeline_paired 内无
# #7 output_layer=12 + 同一权重
grep -n "output_layer\|sensor_hubert_rvc.pth\|hf_model" infer/modules/train/extract_feature_print.py
grep -n "output_layer\|sensor_hubert_rvc.pth" infer/modules/vc/utils.py infer/modules/vc/pipeline.py
# #10 mic(gt) 不加高通: norm_write_paired / pipeline_paired 对 mic 无 filtfilt/lfilter
grep -nE "filtfilt|lfilter|norm_write_paired|def pipeline_paired" infer/modules/train/preprocess.py
# #6 孤儿跳过逻辑
grep -n "orphan\|common\|discover_pairs" infer/modules/train/preprocess.py
# #11 跨域 FAISS 提取是否对 sensor 做了与推理一致的预处理
sed -n '1,80p' xmodal_hubert/scripts/extract_sensor_features.py 2>/dev/null | grep -nE "normaliz|filtfilt|butter|layer_norm|load_audio|sf.read|0.95|max\(\)"
```

- [ ] **Step 3: 写 findings 文档**

把下表填入 `docs/superpowers/audit/2026-06-23-handoff-audit-findings.md`，每行附 Step 2 得到的 `文件:行号` 证据，结论用 ✅符合 / 🔴偏离 / ⚠️需运行机验证：

```markdown
# Handoff 审计结论 (2026-06-23, 分支 VC-2-handoff-audit)

| # | handoff 要求 | 证据(文件:行) | 结论 | 备注 |
|---|---|---|---|---|
| 1 | §4.3 sensor 训练/推理预处理逐字节一致 | preprocess.py:195,209 / modules.py:165-168 / pipeline.py:326 | 🔴 偏离 | Task 2/3 修复 |
| 2 | §2 sensor->1_16k_wavs 16k mono | (填) | | |
| 3 | §4.1 gt 必须真 48k 全频带 mic | (填) | ⚠️ | 运行机核源数据采样率 |
| 4 | §2 配对不切片 | (填) | | |
| 5 | §4.2 mic/sensor 同时长 | preprocess.py:200 仅 WARNING | ⚠️ | 运行机抽样实测 |
| 6 | §4.4 stem 1:1 孤儿跳过 | (填) | | 运行机核两目录集合 |
| 7 | §3 layer=12 同权重 | (填) | | |
| 8 | §4.5 只用目标说话人 | — | ⚠️ | 运行机核 prepare_rvc_trainset |
| 9 | §3 F0=rmvpe(sensor) SR=48k spk=0 | (填) | ⚠️ | 运行机核训练配置 |
| 10 | §2 mic(gt) 不加高通 | (填) | | |
| 11 | 跨域 FAISS 是否同踩 #1 | (填) | | Task 5 条件修复 |
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/audit/2026-06-23-handoff-audit-findings.md
git commit -m "docs: 代码级审计结论 (handoff 11 项)"
```

---

## Task 2: sensor 预处理共享 helper + 单元测试（TDD）

**Files:**
- Create: `infer/lib/sensor_preprocess.py`
- Test: `tests/test_sensor_preprocess.py`

- [ ] **Step 1: 写失败测试**

`tests/test_sensor_preprocess.py`:

```python
import numpy as np
from scipy import signal

from infer.lib.sensor_preprocess import match_inference_sensor_preprocess


def _reference_inference_preprocess(audio):
    """独立复刻推理侧操作 (modules.py:165-168 + pipeline.py:24,326), 交叉验证用。"""
    bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)
    audio = np.asarray(audio, dtype=np.float64)
    audio_max = np.abs(audio).max() / 0.95
    if audio_max > 1:
        audio = audio / audio_max
    audio = signal.filtfilt(bh, ah, audio)
    return audio.astype(np.float32)


def test_matches_inference_when_peak_above_one():
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(16000) * 1.5).astype(np.float32)  # 峰值>1, 触发归一化
    out = match_inference_sensor_preprocess(audio)
    ref = _reference_inference_preprocess(audio)
    np.testing.assert_allclose(out, ref, rtol=0, atol=0)


def test_matches_inference_when_peak_below_one():
    rng = np.random.default_rng(1)
    audio = (rng.standard_normal(16000) * 0.1).astype(np.float32)  # 峰值<1, 不归一化
    out = match_inference_sensor_preprocess(audio)
    ref = _reference_inference_preprocess(audio)
    np.testing.assert_allclose(out, ref, rtol=0, atol=0)


def test_highpass_removes_dc():
    audio = np.full(16000, 0.5, dtype=np.float32)  # 纯 DC
    out = match_inference_sensor_preprocess(audio)
    assert np.abs(out.mean()) < 1e-3  # 48Hz 高通滤除 DC
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_sensor_preprocess.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'infer.lib.sensor_preprocess'`

- [ ] **Step 3: 写最小实现**

`infer/lib/sensor_preprocess.py`:

```python
"""Sensor 波形预处理: 与推理侧逐字节对齐 (训练特征提取的单一真相)。

喂给 SensorHubert 的 sensor, 训练提特征时必须与推理时经过相同预处理, 否则残余 OOD。
推理侧等价操作 (本模块即复刻它们, 改动这里务必同步核对推理):
  - infer/modules/vc/modules.py:165-168  peak-norm: max=|x|.max()/0.95; if max>1: x/=max
  - infer/modules/vc/pipeline.py:24,326   48Hz 高通 filtfilt @ fs=16000 (零相位)
绝不施加 F.layer_norm —— SensorHubert 是 normalize=False 训练, 加 layer_norm 是踩过的坑。

注: 训练侧最终把结果写成 float32 wav 再读回 (RVC 既有设计), 与推理的内存路径存在
一次 float32 量化差, 该差可忽略且原版 RVC 同样存在; 本模块只保证"操作一致"。
"""
import numpy as np
from scipy import signal

# 与 infer/modules/vc/pipeline.py:24 完全相同的系数
_BH16, _AH16 = signal.butter(N=5, Wn=48, btype="high", fs=16000)


def match_inference_sensor_preprocess(audio):
    """对 16k mono sensor 波形施加与推理逐字节相同的 peak-norm + 48Hz 高通。

    audio: 1-D numpy array @ 16000 Hz
    return: 处理后的 1-D float32 array
    """
    audio = np.asarray(audio, dtype=np.float64)
    audio_max = np.abs(audio).max() / 0.95
    if audio_max > 1:
        audio = audio / audio_max
    audio = signal.filtfilt(_BH16, _AH16, audio)
    return audio.astype(np.float32)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_sensor_preprocess.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add infer/lib/sensor_preprocess.py tests/test_sensor_preprocess.py
git commit -m "fix(#1): 新增与推理对齐的 sensor 预处理 helper + 单测"
```

---

## Task 3: 接入 `pipeline_paired`（修复 #1），并确认 mic(gt) 不受影响

**Files:**
- Modify: `infer/modules/train/preprocess.py`（import + `pipeline_paired` 内 1 行）

- [ ] **Step 1: 加 import**

在 `infer/modules/train/preprocess.py` 的 `from infer.lib.slicer2 import Slicer` 之后新增一行：

```python
from infer.lib.sensor_preprocess import match_inference_sensor_preprocess
```

- [ ] **Step 2: 在写 `1_16k_wavs` 前调用 helper**

在 `pipeline_paired` 内，把（约 preprocess.py:195）：

```python
            sensor_audio = load_audio(sensor_path, 16000)
```

改为：

```python
            sensor_audio = load_audio(sensor_path, 16000)
            # #1 修复: 对齐推理侧 sensor 预处理 (peak-norm + 48Hz 高通), 绝不 layer_norm
            sensor_audio = match_inference_sensor_preprocess(sensor_audio)
```

> ⚠️ 只改 sensor 分支。**不要**对 `mic_audio` 加任何滤波（#10：gt 必须保 48k 全频带）。
> `norm_write_paired(mic_audio, stem)` 维持原样。

- [ ] **Step 3: 静态核对改动正确**

Run:
```bash
grep -n "match_inference_sensor_preprocess" infer/modules/train/preprocess.py
grep -nE "filtfilt|lfilter" infer/modules/train/preprocess.py
python -c "import ast; ast.parse(open('infer/modules/train/preprocess.py').read()); print('syntax OK')"
```
Expected:
- `match_inference_sensor_preprocess` 出现 2 次（import + 调用）；
- `filtfilt`/`lfilter` 仅出现在原版 `pipeline()`（mic 切片流程）相关行，**不在 `pipeline_paired`/`norm_write_paired` 内**；
- 打印 `syntax OK`。

> 注：无法在开发机做端到端运行（缺 ffmpeg 数据/SensorHubert 权重）。端到端验证在 Task 4 的运行机 runbook 里完成。

- [ ] **Step 4: Commit**

```bash
git add infer/modules/train/preprocess.py
git commit -m "fix(#1): pipeline_paired 写 1_16k_wavs 前对 sensor 做推理同款预处理"
```

---

## Task 4: 运行机验证 runbook（文档）

**Files:**
- Create: `docs/superpowers/audit/2026-06-23-runmachine-verification.md`

本任务产出"可直接粘贴执行的命令 + 判据"清单，由用户在**运行机**执行后回贴结果。无测试。

- [ ] **Step 1: 写 runbook 文档**

`docs/superpowers/audit/2026-06-23-runmachine-verification.md`（`<exp>` 替换为实验目录，建议 `vibravox_spk1_handoff_fix`）：

````markdown
# 运行机验证 runbook (2026-06-23, #1 修复后)

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
  `python -c "import soundfile as sf; a=sf.info('logs/<exp>/0_gt_wavs/STEM.wav'); b=sf.info('logs/<exp>/1_16k_wavs/STEM.wav'); print(a.frames/a.samplerate, b.frames/b.samplerate)"`。

## 2. 确认源 mic 是真 48k 全频带（#3 关键坑）
```bash
python -c "import soundfile as sf,glob; f=sorted(glob.glob('/app/data/rvc_trainset/**/*mic*.wav', recursive=True))[0]; i=sf.info(f); print(f, i.samplerate, i.frames)"
```
**判据：** 源 mic 采样率为 48000，且频谱有 8kHz 以上能量（非 16k 升采样的假 48k）。
可选频谱核对：`python -c "import librosa,numpy as np; y,_=librosa.load(F,sr=48000); S=np.abs(librosa.stft(y)); print('hi-band energy', S[300:].mean())"`。

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
  - 几乎不变 → 转查非 bug 成因（数据量/SensorHubert 质量，见 spec §6）。
````

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/audit/2026-06-23-runmachine-verification.md
git commit -m "docs: 运行机验证 runbook (#1 修复后)"
```

---

## Task 5（条件）: 跨域 FAISS 特征提取同步修 #1

**触发条件：** Task 1 Step 2 的 #11 核对显示 `extract_sensor_features.py` 读 sensor 后**未**做与推理一致的 peak-norm+高通（大概率如此）。若用户本轮不关心跨域 FAISS 索引，则跳过本任务。

**Files:**
- Modify: `xmodal_hubert/scripts/extract_sensor_features.py`

- [ ] **Step 1: 定位 sensor 读取处**

Run: `grep -nE "load_audio|sf.read|readwave|16000|extract_features" xmodal_hubert/scripts/extract_sensor_features.py`

- [ ] **Step 2: 在喂 SensorHubert 前调用同一 helper**

在读出 16k sensor 波形之后、转 tensor 之前，插入：

```python
from infer.lib.sensor_preprocess import match_inference_sensor_preprocess
# ... 读出 wav 为 1-D numpy @16k 后:
wav = match_inference_sensor_preprocess(wav)
```

（确保此处不再有任何 `F.layer_norm`/独立归一化，统一走 helper。）

- [ ] **Step 3: 语法核对**

Run: `python -c "import ast; ast.parse(open('xmodal_hubert/scripts/extract_sensor_features.py').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 4: Commit**

```bash
git add xmodal_hubert/scripts/extract_sensor_features.py
git commit -m "fix(#11): 跨域 FAISS 特征提取的 sensor 也走推理同款预处理"
```

> 跨域索引需在运行机重建后才生效；重建命令见 findings 中 `build_cross_domain_index.py` 用法。

---

## 完成判据

- [ ] `tests/test_sensor_preprocess.py` 3 项全过（开发机）。
- [ ] `pipeline_paired` 写 `1_16k_wavs` 前调用了 helper；`pipeline_paired`/`norm_write_paired` 内无 mic 滤波。
- [ ] findings 文档 11 项均有结论与证据。
- [ ] runbook 文档可被运行机直接执行。
- [ ] 运行机回贴 §1-§5 判据全绿后，方可进入 §6 的 A/B 判定（该步由用户在运行机完成，不属本机交付）。
