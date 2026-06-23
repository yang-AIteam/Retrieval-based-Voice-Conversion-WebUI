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

## 6. 开训

训练前必须先有 `logs/<exp>/filelist.txt` 和 `logs/<exp>/config.json`。走 WebUI「训练」按钮会自动生成；命令行则用下面脚本复刻 `infer-web.py click_train` 的生成逻辑（v2 / 48k / f0 / spk_id=0 / 末尾追加 2 行 mute 参考）。

### 6.1 生成 filelist + config（命令行）
> 把 `EXP` 改成你的实验名（如 `stage9_handoff_fix-48k`）。依赖 `logs/mute/`（仓库自带）。
```bash
EXP=stage9_handoff_fix-48k python - <<'PY'
import os, shutil
from random import shuffle
now = os.getcwd(); exp = os.environ["EXP"]
sr, spk, ver, fea = "48k", 0, "v2", 768
d   = f"{now}/logs/{exp}"
gt  = f"{d}/0_gt_wavs"; fe = f"{d}/3_feature768"; f0 = f"{d}/2a_f0"; f0n = f"{d}/2b-f0nsf"
names = (set(n.split('.')[0] for n in os.listdir(gt))
         & set(n.split('.')[0] for n in os.listdir(fe))
         & set(n.split('.')[0] for n in os.listdir(f0))
         & set(n.split('.')[0] for n in os.listdir(f0n)))
opt = [f"{gt}/{n}.wav|{fe}/{n}.npy|{f0}/{n}.wav.npy|{f0n}/{n}.wav.npy|{spk}" for n in names]
for _ in range(2):  # mute 参考行 x2 (RVC 惯例)
    opt.append(f"{now}/logs/mute/0_gt_wavs/mute{sr}.wav|{now}/logs/mute/3_feature{fea}/mute.npy|"
               f"{now}/logs/mute/2a_f0/mute.wav.npy|{now}/logs/mute/2b-f0nsf/mute.wav.npy|{spk}")
shuffle(opt)
open(f"{d}/filelist.txt", "w").write("\n".join(opt))
cfg = f"{d}/config.json"
if not os.path.exists(cfg):
    shutil.copy(f"{now}/configs/{ver}/{sr}.json", cfg)
print("filelist lines =", len(opt), "(应 = 样本数 + 2)")
PY
```
**判据：** `filelist lines` = 样本数 + 2；`logs/<exp>/config.json` 存在。

### 6.2 启动训练
> flag 定义见 `infer/lib/train/utils.py:300-365`。`-e` 只填实验**名**（train.py 自动加 `logs/` 前缀）。超参**必须与旧 sensor2mic 模型一致**（旧：200 epoch、bs=4）以便公平对照。
```bash
python infer/modules/train/train.py \
  -e stage9_handoff_fix-48k -sr 48k -f0 1 -bs 4 -g 0 \
  -te 200 -se 50 \
  -pg assets/pretrained_v2/f0G48k.pth -pd assets/pretrained_v2/f0D48k.pth \
  -l 1 -c 1 -sw 1 -v v2
```
含义：`-sr 48k -v v2`（与数据一致）、`-f0 1`（有 F0）、`-bs` 批大小、`-te` 总 epoch、`-se 50`（每 50 epoch 存一次，= WebUI save frequency=50）、`-pg/-pd` 48k v2 预训练 G/D、`-l 1`（save only latest ckpt：只保留最新训练 ckpt `G_*/D_*` 省盘，**不影响** `-sw` 存到 `assets/weights/` 的推理权重）、`-c 1`（cache all training sets to GPU memory，小数据集提速；显存不够改 0）、`-sw 1`（每 50 epoch 存推理权重到 `assets/weights/<exp>_eXXX.pth`，A/B 就用这些）。
**判据：** 训练日志 filelist 行数 = 样本数 + 2（mute）；loss 正常下降；`assets/weights/` 出现以实验名命名的 `.pth`。

### 6.3 （可选）构建标准特征索引
> **判定 #1 不需要索引**（§7 用 `--index_rate 0` 隔离生成器）。只有想跑 `--index_rate>0` 的检索增强对照时才建。
> 复刻 `infer-web.py: train_index()`（v2，读 `3_feature768/`）。key=`SensorHubert(sensor)` 训练特征，与推理 query 同域、且已被 #1 修复对齐——与跨域索引（#11，需 `3_feature768`=mic）是两码事，**当前 paired 流程不要建跨域索引**。
```bash
EXP=stage9_handoff_fix-48k python - <<'PY'
import os, faiss, numpy as np
exp = os.environ["EXP"]; ver = "v2"
exp_dir = f"logs/{exp}"; feat_dir = f"{exp_dir}/3_feature768"
big = np.concatenate([np.load(f"{feat_dir}/{n}") for n in sorted(os.listdir(feat_dir))], 0)
idx = np.arange(big.shape[0]); np.random.shuffle(idx); big = big[idx]
np.save(f"{exp_dir}/total_fea.npy", big)
n_ivf = min(int(16*np.sqrt(big.shape[0])), big.shape[0]//39)
index = faiss.index_factory(768, f"IVF{n_ivf},Flat")
faiss.extract_index_ivf(index).nprobe = 1
index.train(big)
faiss.write_index(index, f"{exp_dir}/trained_IVF{n_ivf}_Flat_nprobe_1_{exp}_{ver}.index")
for i in range(0, big.shape[0], 8192):
    index.add(big[i:i+8192])
out = f"{exp_dir}/added_IVF{n_ivf}_Flat_nprobe_1_{exp}_{ver}.index"
faiss.write_index(index, out); print("wrote", out)
PY
```
**判据：** 生成 `logs/<exp>/added_IVF*_..._<exp>_v2.index`。该文件即推理 `--index_path` 的入参。

## 7. A/B 推理（判定 #1 是否为主因）

用**训练集之外**的同一段 sensor，分别喂**新模型**和**旧 sensor2mic 模型**，**其余推理参数完全一致**（只换模型），听哪个更接近干净 mic。命令行推理用 `tools/infer_cli.py`（参数见其 argparse）。

```bash
# 新模型 (本次 #1 修复后训练得到的)
python tools/infer_cli.py --f0method rmvpe \
  --input_path /path/to/heldout_sensor.wav --opt_path out_NEW.wav \
  --model_name <new_exp_weight>.pth --index_path "" \
  --index_rate 0 --protect 0.33 --filter_radius 3

# 旧模型 (之前结果不好的那个), 参数与上完全相同, 只改 model_name / opt_path
python tools/infer_cli.py --f0method rmvpe \
  --input_path /path/to/heldout_sensor.wav --opt_path out_OLD.wav \
  --model_name <old_sensor2mic_weight>.pth --index_path "" \
  --index_rate 0 --protect 0.33 --filter_radius 3
```
> `--model_name` 是 `assets/weights/` 下的 .pth 文件名。`--index_rate 0` 关掉 FAISS 检索，**隔离生成器本身**（避免索引混淆"是不是 #1 起作用"）；若要含索引另作一组、两边同样设置即可。两条命令除 model/opt 外**逐字相同**是公平对照的前提。

**判定：**
- `out_NEW` 明显比 `out_OLD` 更接近干净 mic → **#1 是主因**，旧负面结论作废重写。
- 几乎不变 → #1 非主因，转查非 bug 成因（数据量 126 对~10min、SensorHubert 质量，见 design spec §6）。
