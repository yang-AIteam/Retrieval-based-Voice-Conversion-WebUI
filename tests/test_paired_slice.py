from infer.lib.paired_slice import paired_segments, to_samples


def test_short_clip_single_segment():
    assert paired_segments(3.0, 3.7, 0.3) == [(0.0, 3.0)]


def test_empty():
    assert paired_segments(0.0, 3.7, 0.3) == []


def test_long_clip_all_segments_within_bucket_limit():
    # 20s 录音: 每段都不超过 per+overlap, 远低于 bucket 上限 (~6.75s)
    segs = paired_segments(20.0, 3.7, 0.3)
    assert len(segs) > 1
    assert segs[0][0] == 0.0 and segs[-1][1] == 20.0
    for s, e in segs:
        assert 0 < e - s <= 3.7 + 0.3 + 1e-9
    # 相邻段重叠 overlap 秒
    for (s0, e0), (s1, _) in zip(segs, segs[1:]):
        assert abs((e0 - s1) - 0.3) < 1e-9


def test_matches_preprocess_pipeline_cut_points():
    # 与 PreProcess.pipeline 的采样点切法一致 (sr=16000)
    sr, per, overlap, n = 16000, 3.7, 0.3, 16000 * 11
    ref, i = [], 0
    while 1:
        start = int(sr * (per - overlap) * i)
        i += 1
        if n - start > (per + overlap) * sr:
            ref.append((start, start + int(per * sr)))
        else:
            ref.append((start, n))
            break
    got = [to_samples(s, sr) for s in paired_segments(n / sr, per, overlap)]
    assert len(got) == len(ref)
    for (a0, a1), (b0, b1) in zip(got, ref):
        assert abs(a0 - b0) <= 1 and abs(a1 - b1) <= 1


def test_mic_sensor_alignment():
    # 同一切点换算到 48k 与 16k 后, 时间误差 < 1 个 16k 采样
    for s, e in paired_segments(12.34, 3.7, 0.3):
        m0, m1 = to_samples((s, e), 48000)
        s0, s1 = to_samples((s, e), 16000)
        assert abs(m0 / 48000 - s0 / 16000) < 1 / 16000
        assert abs(m1 / 48000 - s1 / 16000) < 1 / 16000
