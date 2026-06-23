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
