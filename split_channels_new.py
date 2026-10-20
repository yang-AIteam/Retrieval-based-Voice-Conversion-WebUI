#!/usr/bin/env python3
"""
stereo_split.py — Split a stereo WAV (L=mic, R=sensor) into two mono WAV files.

Usage:
    python split_channels.py <input_file_or_folder> [--out <output_folder>]

Examples:
    # single file, output to same folder
    python split_channels.py audio/1027辻-01.wav

    # single file, specify output folder
    python split_channels.py audio/1027辻-01.wav --out output/

    # process all .wav in a folder
    python split_channels.py audio/ --out output/
"""

import argparse
import os
import sys
import warnings
from typing import Optional, Tuple
import numpy as np
import scipy.io.wavfile as wf

def split_stereo(input_path: str, output_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    Split a stereo WAV into _mic (L) and _sensor (R) mono files.

    Returns
    -------
    (mic_path, sensor_path) : paths of the two output files
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rate, data = wf.read(input_path)

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"{input_path}: expected stereo (2-ch), got shape {data.shape}")

    left  = data[:, 0]   # mic
    right = data[:, 1]   # sensor

    basename = os.path.splitext(os.path.basename(input_path))[0]
    ext = ".wav"
    dest = output_dir if output_dir else os.path.dirname(os.path.abspath(input_path))
    os.makedirs(dest, exist_ok=True)

    mic_path    = os.path.join(dest, f"{basename}_mic{ext}")
    sensor_path = os.path.join(dest, f"{basename}_sensor{ext}")

    wf.write(mic_path,    rate, left)
    wf.write(sensor_path, rate, right)

    return mic_path, sensor_path


def process_path(target: str, output_dir: Optional[str] = None):
    if os.path.isfile(target):
        files = [target]
    elif os.path.isdir(target):
        files = sorted(
            os.path.join(target, f)
            for f in os.listdir(target)
            if f.lower().endswith(".wav")
        )
        if not files:
            print(f"No .wav files found in {target}", file=sys.stderr)
            return
    else:
        print(f"Not found: {target}", file=sys.stderr)
        sys.exit(1)

    for path in files:
        try:
            mic, sensor = split_stereo(path, output_dir)
            print(f"✓  {os.path.basename(path)}")
            print(f"     mic    → {mic}")
            print(f"     sensor → {sensor}")
        except Exception as e:
            print(f"✗  {os.path.basename(path)}: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split L/R channels of a stereo WAV")
    parser.add_argument("input",  help="Input WAV file or folder containing WAV files")
    parser.add_argument("--out",  metavar="DIR", default=None,
                        help="Output folder (default: same as input file)")
    args = parser.parse_args()

    process_path(args.input, args.out)
