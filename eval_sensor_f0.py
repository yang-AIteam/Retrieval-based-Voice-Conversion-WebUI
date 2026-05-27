"""
Step 0 quantitative comparison for sensor F0 extraction.

Goal: decide whether an existing F0 method (RMVPE / Harvest / Crepe) already works
well enough on vibration-sensor audio, before investing in fine-tuning RMVPE.

Method:
    - Ground truth = RMVPE F0 on the *mic* audio of each pair (mic is clean; F0 is
      physically the same source as the sensor, so this is an exact reference).
    - For each candidate method we run it on the *sensor* audio and compare against
      the ground truth, frame-by-frame, on a common 10ms grid.
    - A small constant time-lag (sensor vs mic recording delay + per-method framing)
      is searched per method and corrected before scoring, so the comparison is fair.

Metrics (pooled over all frames of all sampled pairs):
    RPA   Raw Pitch Accuracy   - % of GT-voiced frames within 50 cents of GT.
    RCA   Raw Chroma Accuracy  - same but octave-collapsed; RPA << RCA means octave errors.
    VR    Voicing Recall       - % of GT-voiced frames the method also calls voiced.
    VFA   Voicing False Alarm  - % of GT-unvoiced frames the method wrongly calls voiced.
    cents median/mean abs error on frames where both are voiced.

Usage:
    1. Edit PAIRED_DATA_DIR below to point at your folder of *_sensor / *_mic wavs.
    2. Make sure assets/rmvpe/rmvpe.pt exists (RVC's pretrained RMVPE).
    3. Run inside your RVC python env:  python eval_sensor_f0.py
"""

import os
import random
import glob
import sys
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG  -- edit these
# ---------------------------------------------------------------------------
PAIRED_DATA_DIR = "/app/datasets/F0"          # <-- 手动填入存放 2000+ 对 paired wav 的目录
N_PAIRS = 30                  # 随机抽取多少对来评测 (先 20~30 看趋势, 后续可调到 100+)
SEED = 0                      # 随机种子, 固定以便复现
METHODS = ["rmvpe", "harvest", "crepe"]   # 要对比的 sensor F0 方法

F0_MIN = 50                   # 与 infer/modules/vc/pipeline.py 一致
F0_MAX = 1100
HOP = 160                     # 16kHz 下 10ms 一帧
SR = 16000
MAX_LAG_FRAMES = 10           # 时间对齐时搜索的最大帧偏移 (±, 1 帧 = 10ms)
RMVPE_THRED = 0.03            # 与 pipeline 默认一致
CREPE_PD_THRESHOLD = 0.1      # crepe periodicity 阈值
RMVPE_MODEL_PATH = os.environ.get("rmvpe_root", "assets/rmvpe") + "/rmvpe.pt"
DEVICE = "auto"               # auto / cpu / cuda / mps
# ---------------------------------------------------------------------------

import librosa
import soundfile as sf
import torch


def pick_device(arg):
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_16k_mono(path):
    wav, sr = sf.read(path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float64)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav


def discover_pairs(root):
    """Return list of (id, sensor_path, mic_path) for files named *_sensor.* / *_mic.*"""
    sensors = {}
    mics = {}
    for path in glob.glob(os.path.join(root, "**", "*.*"), recursive=True):
        low = path.lower()
        if not low.endswith((".wav", ".flac", ".mp3", ".ogg")):
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.endswith("_sensor"):
            sensors[stem[: -len("_sensor")]] = path
        elif stem.endswith("_mic"):
            mics[stem[: -len("_mic")]] = path
    keys = sorted(set(sensors) & set(mics))
    return [(k, sensors[k], mics[k]) for k in keys]


# ----------------------------- F0 extractors --------------------------------
_rmvpe_model = None


def get_rmvpe():
    global _rmvpe_model
    if _rmvpe_model is None:
        if not os.path.exists(RMVPE_MODEL_PATH):
            sys.exit(f"RMVPE model not found: {RMVPE_MODEL_PATH} (download it first)")
        from infer.lib.rmvpe import RMVPE

        is_half = DEV.startswith("cuda")
        _rmvpe_model = RMVPE(RMVPE_MODEL_PATH, is_half=is_half, device=DEV)
    return _rmvpe_model


def f0_rmvpe(wav):
    return get_rmvpe().infer_from_audio(wav.astype(np.float32), thred=RMVPE_THRED)


def f0_harvest(wav):
    import pyworld

    f0, t = pyworld.harvest(
        wav, SR, f0_floor=F0_MIN, f0_ceil=F0_MAX, frame_period=1000 * HOP / SR
    )
    return pyworld.stonemask(wav, f0, t, SR)


def f0_crepe(wav):
    import torchcrepe

    # crepe is unreliable on mps; fall back to cpu for the crepe call only
    dev = "cpu" if DEV == "mps" else DEV
    audio = torch.tensor(np.copy(wav))[None].float()
    f0, pd = torchcrepe.predict(
        audio, SR, HOP, F0_MIN, F0_MAX, "full",
        batch_size=512, device=dev, return_periodicity=True,
    )
    pd = torchcrepe.filter.median(pd, 3)
    f0 = torchcrepe.filter.mean(f0, 3)
    f0[pd < CREPE_PD_THRESHOLD] = 0
    return f0[0].cpu().numpy()


EXTRACTORS = {"rmvpe": f0_rmvpe, "harvest": f0_harvest, "crepe": f0_crepe}


# ------------------------------- metrics ------------------------------------
def to_cents(f0):
    out = np.full_like(f0, np.nan, dtype=np.float64)
    voiced = f0 > 0
    out[voiced] = 1200.0 * np.log2(f0[voiced] / 10.0)
    return out, voiced


def score_with_lag(gt, pred, lag):
    """Shift pred by `lag` frames, align lengths, return (rpa, rca, vr, vfa, mae, n_voiced)."""
    if lag > 0:
        pred = pred[lag:]
    elif lag < 0:
        pred = np.concatenate([np.zeros(-lag), pred])
    n = min(len(gt), len(pred))
    gt, pred = gt[:n], pred[:n]

    gt_c, gt_v = to_cents(gt)
    pr_c, pr_v = to_cents(pred)

    n_gt_voiced = int(gt_v.sum())
    n_gt_unvoiced = int((~gt_v).sum())
    if n_gt_voiced == 0:
        return None

    both = gt_v & pr_v
    diff = np.abs(pr_c - gt_c)              # cents error where defined
    diff_both = diff[both]
    chroma = np.abs(((pr_c - gt_c + 600) % 1200) - 600)[both]

    # RPA: correct-pitch frames / all GT-voiced frames (pred-unvoiced counts as wrong)
    correct = np.zeros(n, dtype=bool)
    correct[both] = diff_both < 50
    rpa = correct.sum() / n_gt_voiced

    correct_ch = np.zeros(n, dtype=bool)
    correct_ch[both] = chroma < 50
    rca = correct_ch.sum() / n_gt_voiced

    vr = (gt_v & pr_v).sum() / n_gt_voiced
    vfa = ((~gt_v) & pr_v).sum() / max(1, n_gt_unvoiced)
    mae = float(np.median(diff_both)) if diff_both.size else float("nan")
    return rpa, rca, vr, vfa, mae, n_gt_voiced


def best_over_lags(gt, pred):
    """Search constant lag in [-MAX_LAG, +MAX_LAG], pick the one with highest RPA."""
    best = None
    best_lag = 0
    for lag in range(-MAX_LAG_FRAMES, MAX_LAG_FRAMES + 1):
        r = score_with_lag(gt, pred, lag)
        if r is None:
            continue
        if best is None or r[0] > best[0]:
            best, best_lag = r, lag
    return best, best_lag


# --------------------------------- main -------------------------------------
def main():
    global DEV
    DEV = pick_device(DEVICE)
    print(f"device: {DEV}")

    if not PAIRED_DATA_DIR:
        sys.exit("请先在脚本顶部填写 PAIRED_DATA_DIR")

    pairs = discover_pairs(PAIRED_DATA_DIR)
    if not pairs:
        sys.exit(f"未在 {PAIRED_DATA_DIR} 找到任何 *_sensor / *_mic 配对")
    print(f"发现 {len(pairs)} 对 paired data")

    random.seed(SEED)
    sample = random.sample(pairs, min(N_PAIRS, len(pairs)))
    print(f"随机抽取 {len(sample)} 对进行评测 (seed={SEED})\n")

    # accumulate frame-weighted metrics per method
    acc = {m: {"rpa": 0.0, "rca": 0.0, "vr": 0.0, "vfa_num": 0.0,
               "vfa_den": 0.0, "mae": [], "nv": 0, "lags": []} for m in METHODS}

    for i, (pid, sensor_path, mic_path) in enumerate(sample, 1):
        try:
            sensor = load_16k_mono(sensor_path)
            mic = load_16k_mono(mic_path)
        except Exception as e:
            print(f"[{i}/{len(sample)}] {pid}  跳过 (读取失败: {e})")
            continue

        gt = f0_rmvpe(mic)  # ground truth from clean mic

        line = [f"[{i}/{len(sample)}] {pid}"]
        for m in METHODS:
            try:
                pred = EXTRACTORS[m](sensor)
            except Exception as e:
                line.append(f"{m}:ERR({e})")
                continue
            res, lag = best_over_lags(gt, pred)
            if res is None:
                line.append(f"{m}:no-voiced")
                continue
            rpa, rca, vr, vfa, mae, nv = res
            acc[m]["rpa"] += rpa * nv
            acc[m]["rca"] += rca * nv
            acc[m]["vr"] += vr * nv
            acc[m]["nv"] += nv
            acc[m]["mae"].append(mae)
            acc[m]["lags"].append(lag)
            # vfa weighted by unvoiced frames -> recompute denom via score at best lag
            line.append(f"{m}:RPA={rpa*100:4.1f} lag={lag:+d}")
        print("  ".join(line))

    print("\n================= 汇总 (帧加权) =================")
    print(f"{'method':<10}{'RPA%':>8}{'RCA%':>8}{'VR%':>8}{'cents(med)':>12}{'lag(med)':>10}")
    for m in METHODS:
        nv = acc[m]["nv"]
        if nv == 0:
            print(f"{m:<10}{'n/a':>8}")
            continue
        rpa = acc[m]["rpa"] / nv * 100
        rca = acc[m]["rca"] / nv * 100
        vr = acc[m]["vr"] / nv * 100
        mae = np.median(acc[m]["mae"]) if acc[m]["mae"] else float("nan")
        lag = int(np.median(acc[m]["lags"])) if acc[m]["lags"] else 0
        print(f"{m:<10}{rpa:>8.1f}{rca:>8.1f}{vr:>8.1f}{mae:>12.1f}{lag:>10d}")

    print("\n解读:")
    print("  - RPA 越高越好; >~90% 通常可直接用, 无需微调.")
    print("  - RPA 明显低但 RCA 高 = 八度错误为主, 可先尝试调阈值/后处理.")
    print("  - 各方法 lag 中位数若稳定非零, 说明 sensor/mic 间存在固定延迟, 训练时务必对齐.")


if __name__ == "__main__":
    main()
