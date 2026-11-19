import wave
import struct
import os
import glob

def split_stereo(input_path, output_dir=None, swap=False):
    if output_dir is None:
        output_dir = os.path.dirname(input_path)

    base = os.path.splitext(os.path.basename(input_path))[0]

    with wave.open(input_path, 'rb') as src:
        channels = src.getnchannels()
        rate = src.getframerate()
        width = src.getsampwidth()
        n_frames = src.getnframes()

        if channels != 2:
            print(f"スキップ: {input_path} はステレオではありません (channels={channels})")
            return

        frames = src.readframes(n_frames)

    # インターリーブされたサンプルを分離
    fmt = {1: 'B', 2: 'h', 4: 'i'}[width]
    samples = struct.unpack(f'<{len(frames) // width}{fmt}', frames)

    left_samples  = samples[0::2]
    right_samples = samples[1::2]
    # デフォルト: Lch = mic, Rch = sensor / swap=True: Lch = sensor, Rch = mic
    mic_samples, sensor_samples = (right_samples, left_samples) if swap else (left_samples, right_samples)

    def write_mono(path, samples_mono):
        packed = struct.pack(f'<{len(samples_mono)}{fmt}', *samples_mono)
        with wave.open(path, 'wb') as dst:
            dst.setnchannels(1)
            dst.setframerate(rate)
            dst.setsampwidth(width)
            dst.writeframes(packed)
        print(f"書き出し: {path}")

    write_mono(os.path.join(output_dir, f"{base}_mic.wav"),    mic_samples)
    write_mono(os.path.join(output_dir, f"{base}_sensor.wav"), sensor_samples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ステレオ WAV を Lch(mic) / Rch(sensor) に分割")
    parser.add_argument("-i", "--input", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets"),
                        help="入力: WAV ファイル or ディレクトリ (default: ./datasets)")
    parser.add_argument("-o", "--output", default=None,
                        help="出力ディレクトリ (default: 入力ファイルと同じ場所)")
    parser.add_argument("--swap", action="store_true",
                        help="Lch = sensor, Rch = mic として分割 (default: Lch = mic, Rch = sensor)")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        wav_files = glob.glob(os.path.join(args.input, "*.wav"))
        # 既に分割済みのファイルは除外
        wav_files = [f for f in wav_files if not (f.endswith("_mic.wav") or f.endswith("_sensor.wav"))]
    elif os.path.isfile(args.input):
        wav_files = [args.input]
    else:
        raise SystemExit(f"入力が見つかりません: {args.input}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    if not wav_files:
        print(f"{args.input} に WAV ファイルが見つかりません。")
    else:
        for wav in sorted(wav_files):
            split_stereo(wav, output_dir=args.output, swap=args.swap)
