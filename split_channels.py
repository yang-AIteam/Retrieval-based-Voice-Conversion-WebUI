import wave
import struct
import os
import glob

def split_stereo(input_path, output_dir=None):
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

    left_samples  = samples[0::2]  # Lch = 話筒
    right_samples = samples[1::2]  # Rch = sensor

    def write_mono(path, samples_mono):
        packed = struct.pack(f'<{len(samples_mono)}{fmt}', *samples_mono)
        with wave.open(path, 'wb') as dst:
            dst.setnchannels(1)
            dst.setframerate(rate)
            dst.setsampwidth(width)
            dst.writeframes(packed)
        print(f"書き出し: {path}")

    write_mono(os.path.join(output_dir, f"{base}_mic.wav"),    left_samples)
    write_mono(os.path.join(output_dir, f"{base}_sensor.wav"), right_samples)


if __name__ == "__main__":
    datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")
    wav_files = glob.glob(os.path.join(datasets_dir, "*.wav"))

    # 既に分割済みのファイルは除外
    wav_files = [f for f in wav_files if not (f.endswith("_mic.wav") or f.endswith("_sensor.wav"))]

    if not wav_files:
        print("datasets/ に WAV ファイルが見つかりません。")
    else:
        for wav in sorted(wav_files):
            split_stereo(wav, output_dir=datasets_dir)
