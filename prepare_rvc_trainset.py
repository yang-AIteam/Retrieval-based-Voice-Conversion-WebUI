#!/usr/bin/env python3
"""统计 Vibravox 混合目录里各说话人的录音数，并为数据最多的说话人建 symlink 训练目录。

文件命名约定（由 extract_parquet_to_wav.py 生成）:
    {speaker_id}_{sentence_id:04d}_mic.wav
    {speaker_id}_{sentence_id:04d}_sensor.wav

用法:
    # 只统计，不建目录
    python prepare_rvc_trainset.py --data_dir /app/data/vibravox

    # 统计后自动为数据最多的说话人建 symlink 目录
    python prepare_rvc_trainset.py --data_dir /app/data/vibravox --out_dir /app/data/rvc_trainset

    # 指定说话人（不自动选最多的）
    python prepare_rvc_trainset.py --data_dir /app/data/vibravox --out_dir /app/data/rvc_trainset --speaker _3bGP7YNz
"""

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

PAIR_RE = re.compile(r'^(.+)_(\d{4})_(mic|sensor)\.wav$')


def count_speakers(data_dir: Path) -> Counter:
    counts: Counter = Counter()
    for name in os.listdir(data_dir):
        m = PAIR_RE.match(name)
        if m and m.group(3) == "mic":
            counts[m.group(1)] += 1
    return counts


def make_symlink_dir(data_dir: Path, out_dir: Path, speaker_id: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for name in os.listdir(data_dir):
        if name.startswith(speaker_id + "_") and name.endswith((".wav",)):
            src = (data_dir / name).resolve()
            dst = out_dir / name
            if not dst.exists():
                os.symlink(src, dst)
                created += 1
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Vibravox 说话人统计 & RVC 训练集准备")
    parser.add_argument("--data_dir", required=True, help="Vibravox 混合 WAV 目录")
    parser.add_argument("--out_dir", default=None, help="输出 symlink 目录（不填则只统计）")
    parser.add_argument("--speaker", default=None, help="指定说话人 ID（不填则自动选数据最多的）")
    parser.add_argument("--top", type=int, default=10, help="显示前 N 名说话人（默认 10）")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"错误: {data_dir} 不存在或不是目录", file=sys.stderr)
        sys.exit(1)

    print(f"扫描目录: {data_dir}")
    counts = count_speakers(data_dir)

    if not counts:
        print("未找到符合命名规范的 mic WAV 文件，请确认目录正确。")
        sys.exit(1)

    print(f"\n共 {len(counts)} 个说话人，前 {min(args.top, len(counts))} 名：")
    print(f"{'录音数':>8}  说话人 ID")
    print("-" * 36)
    for spk, n in counts.most_common(args.top):
        marker = "  ← 最多" if spk == counts.most_common(1)[0][0] else ""
        print(f"{n:8d}  {spk}{marker}")

    if args.out_dir is None:
        print("\n（未指定 --out_dir，跳过建目录步骤）")
        return

    speaker = args.speaker or counts.most_common(1)[0][0]
    if speaker not in counts:
        print(f"\n错误: 说话人 '{speaker}' 不在数据中", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    print(f"\n为说话人 {speaker}（共 {counts[speaker]} 对）建 symlink 目录: {out_dir}")
    n = make_symlink_dir(data_dir, out_dir, speaker)
    total = len(list(out_dir.iterdir()))
    print(f"新建 symlink: {n}，目录总文件数: {total}（应为 {counts[speaker] * 2}）")
    print(f"\n下一步：把 {out_dir} 传给 RVC preprocess.py 即可。")


if __name__ == "__main__":
    main()
