"""
Extract HuBERT features from sensor audio files for cross-domain FAISS.

This mirrors what extract_feature_print.py does for mic audio, but reads
from 1_16k_wavs_sensor/ and writes to 3_feature768_sensor/.

Usage:
    python extract_sensor_features.py <exp_name> [device]

    exp_name: experiment name (same as used in standard RVC preprocessing)
    device:   cpu / cuda / mps  (default: auto-detect)

Prerequisites:
    - Sensor wav files resampled to 16kHz must be in logs/<exp_name>/1_16k_wavs_sensor/
      (copy or symlink your *_sensor.wav files there after 16k resampling)
    - assets/hubert/sensor_hubert_rvc.pth and assets/hubert/hf_model/ must exist
"""

import os
import sys
import traceback

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from infer.lib.sensor_preprocess import match_inference_sensor_preprocess

exp_name = sys.argv[1]
device_arg = sys.argv[2] if len(sys.argv) > 2 else "auto"

exp_dir = os.path.join("logs", exp_name)
wav_dir = os.path.join(exp_dir, "1_16k_wavs_sensor")
out_dir = os.path.join(exp_dir, "3_feature768_sensor")
os.makedirs(out_dir, exist_ok=True)

log_path = os.path.join(exp_dir, "extract_sensor_feature.log")
f = open(log_path, "a+")


def log(msg):
    print(msg)
    f.write(msg + "\n")
    f.flush()


if device_arg == "auto":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
else:
    device = device_arg

log(f"device: {device}")
log(f"wav_dir: {wav_dir}")
log(f"out_dir: {out_dir}")

pth_path = "assets/hubert/sensor_hubert_rvc.pth"
config_path = "assets/hubert/hf_model"
if not os.access(pth_path, os.F_OK):
    log(f"Error: {pth_path} not found. Copy your exported sensor HuBERT checkpoint here.")
    sys.exit(1)

log(f"Loading sensor HuBERT from {pth_path}")
sys.path.insert(0, os.getcwd())
from rvc_hubert_loader import load_sensor_hubert
model = load_sensor_hubert(pth_path=pth_path, config_path=config_path, device=device)
log("Model loaded.")


def readwave(wav_path):
    wav, sr = sf.read(wav_path)
    assert sr == 16000, f"Expected 16kHz, got {sr}Hz: {wav_path}"
    # stereo → mono (numpy 层完成, 与 helper 调用保持同层)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=-1)
    assert wav.ndim == 1
    # 施加与推理侧等价的预处理 (peak-norm + 48Hz 高通), 消除 OOD
    wav = match_inference_sensor_preprocess(wav)
    feats = torch.from_numpy(wav).float()
    return feats.view(1, -1)


todo = sorted(f for f in os.listdir(wav_dir) if f.endswith(".wav"))
if not todo:
    log(f"No wav files in {wav_dir}")
    sys.exit(1)

log(f"Processing {len(todo)} files ...")
for idx, fname in enumerate(todo):
    try:
        wav_path = os.path.join(wav_dir, fname)
        out_path = os.path.join(out_dir, fname.replace(".wav", ".npy"))
        if os.path.exists(out_path):
            continue

        feats = readwave(wav_path).to(device)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False).to(device)
        with torch.no_grad():
            logits = model.extract_features(
                source=feats,
                padding_mask=padding_mask,
                output_layer=12,  # v2
            )
            out = logits[0].squeeze(0).float().cpu().numpy()

        if np.isnan(out).sum() == 0:
            np.save(out_path, out, allow_pickle=False)
            if idx % max(1, len(todo) // 10) == 0:
                log(f"[{idx+1}/{len(todo)}] {fname} → {out.shape}")
        else:
            log(f"WARNING: {fname} contains NaN, skipped.")
    except Exception:
        log(traceback.format_exc())

log("Done.")
f.close()
