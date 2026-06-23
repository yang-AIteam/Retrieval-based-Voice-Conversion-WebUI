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


def test_empty_input_returns_empty_float32():
    out = match_inference_sensor_preprocess(np.array([], dtype=np.float32))
    assert out.dtype == np.float32
    assert out.size == 0
