"""配对模式 (mic + sensor) 的同步切片: 两侧用同一组时间切点。

不切片时, 超过约 6.75s 的整条样本会被 train.py 的 DistributedBucketSampler
(boundaries 100..900) 静默丢弃, 因此配对模式也要和普通模式一样切成短片段。
切法与 PreProcess.pipeline 的固定窗口切片相同 (per 秒一段, 重叠 overlap 秒),
但以「秒」为单位计算, 再分别换算成 mic (模型 SR) 和 sensor (16k) 的采样点, 保证对齐。
"""


def paired_segments(duration, per, overlap):
    """返回 [(start_s, end_s), ...]。

    duration: 两侧共同时长 (秒, 取 mic/sensor 较短者)
    per, overlap: 与 PreProcess 相同 (默认 3.7 / 0.3)
    与 PreProcess.pipeline 一致: 剩余长度 > per+overlap 时切 per 秒, 否则剩余整段作为最后一段。
    """
    if duration <= 0:
        return []
    tail = per + overlap
    segs = []
    i = 0
    while 1:
        start = (per - overlap) * i
        i += 1
        if duration - start > tail:
            segs.append((start, start + per))
        else:
            segs.append((start, duration))
            break
    return segs


def to_samples(seg, sr):
    """把 (start_s, end_s) 换算成 sr 下的采样点区间。"""
    return int(seg[0] * sr), int(seg[1] * sr)
