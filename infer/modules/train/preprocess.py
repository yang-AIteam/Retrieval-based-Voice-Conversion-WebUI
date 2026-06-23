import multiprocessing
import os
import sys

from scipy import signal

now_dir = os.getcwd()
sys.path.append(now_dir)
print(*sys.argv[1:])
inp_root = sys.argv[1]
sr = int(sys.argv[2])
n_p = int(sys.argv[3])
exp_dir = sys.argv[4]
noparallel = sys.argv[5] == "True"
per = float(sys.argv[6])
import os
import traceback

import librosa
import numpy as np
from scipy.io import wavfile

from infer.lib.audio import load_audio
from infer.lib.slicer2 import Slicer
from infer.lib.sensor_preprocess import match_inference_sensor_preprocess

f = open("%s/preprocess.log" % exp_dir, "a+")


def println(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


def discover_pairs(inp_root):
    """检测配对数据集, 返回 [(mic_path, sensor_path, stem), ...] 或 None。

    优先级:
      1. inp_root/mic/ 与 inp_root/sensor/ 子目录, 同名文件配对 (stem = 文件名去扩展名)
      2. inp_root 下 *_mic.wav / *_sensor.wav 后缀成对 (stem = 去掉 _mic/_sensor 后的部分)

    缺任一侧的样本跳过 (孤儿文件), 并记录 log。
    都没检测到则返回 None -> 回退原版切片流程。
    """
    audio_exts = (".wav", ".flac", ".mp3", ".m4a", ".ogg")

    def _stem_map(dir_path):
        m = {}
        for name in sorted(os.listdir(dir_path)):
            base, ext = os.path.splitext(name)
            if ext.lower() in audio_exts:
                m[base] = os.path.join(dir_path, name)
        return m

    mic_dir = os.path.join(inp_root, "mic")
    sensor_dir = os.path.join(inp_root, "sensor")
    if os.path.isdir(mic_dir) and os.path.isdir(sensor_dir):
        mic_map = _stem_map(mic_dir)
        sensor_map = _stem_map(sensor_dir)
    else:
        mic_map, sensor_map = {}, {}
        try:
            names = sorted(os.listdir(inp_root))
        except OSError:
            return None
        for name in names:
            base, ext = os.path.splitext(name)
            if ext.lower() not in audio_exts:
                continue
            full = os.path.join(inp_root, name)
            if base.endswith("_mic"):
                mic_map[base[: -len("_mic")]] = full
            elif base.endswith("_sensor"):
                sensor_map[base[: -len("_sensor")]] = full
        if not mic_map and not sensor_map:
            return None

    common = sorted(set(mic_map) & set(sensor_map))
    orphans = (set(mic_map) ^ set(sensor_map))
    for stem in sorted(orphans):
        println("orphan-skipped (缺另一侧): %s" % stem)
    if not common:
        return None
    return [(mic_map[s], sensor_map[s], s) for s in common]


class PreProcess:
    def __init__(self, sr, exp_dir, per=3.7):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.exp_dir = exp_dir
        self.gt_wavs_dir = "%s/0_gt_wavs" % exp_dir
        self.wavs16k_dir = "%s/1_16k_wavs" % exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def norm_write(self, tmp_audio, idx0, idx1):
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            print("%s-%s-%s-filtered" % (idx0, idx1, tmp_max))
            return
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio
        wavfile.write(
            "%s/%s_%s.wav" % (self.gt_wavs_dir, idx0, idx1),
            self.sr,
            tmp_audio.astype(np.float32),
        )
        tmp_audio = librosa.resample(
            tmp_audio, orig_sr=self.sr, target_sr=16000
        )  # , res_type="soxr_vhq"
        wavfile.write(
            "%s/%s_%s.wav" % (self.wavs16k_dir, idx0, idx1),
            16000,
            tmp_audio.astype(np.float32),
        )

    def norm_write_paired(self, mic_audio, stem):
        """配对模式: mic 整段(不切片)做响度归一化写入 0_gt_wavs/<stem>.wav (模型 SR)。

        与 norm_write 的 gt 写入幅度处理保持一致, 但不切片、文件名用 stem。
        """
        tmp_max = np.abs(mic_audio).max()
        if tmp_max > 2.5:
            println("%s-%s-filtered (mic over range)" % (stem, tmp_max))
            return False
        if tmp_max <= 0:
            println("%s-silent-skipped" % stem)
            return False
        mic_audio = (mic_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * mic_audio
        wavfile.write(
            "%s/%s.wav" % (self.gt_wavs_dir, stem),
            self.sr,
            mic_audio.astype(np.float32),
        )
        return True

    def pipeline(self, path, idx0):
        try:
            audio = load_audio(path, self.sr)
            # zero phased digital filter cause pre-ringing noise...
            # audio = signal.filtfilt(self.bh, self.ah, audio)
            audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            for audio in self.slicer.slice(audio):
                i = 0
                while 1:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio[start:]) > self.tail * self.sr:
                        tmp_audio = audio[start : start + int(self.per * self.sr)]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio[start:]
                        idx1 += 1
                        break
                self.norm_write(tmp_audio, idx0, idx1)
            println("%s\t-> Success" % path)
        except:
            println("%s\t-> %s" % (path, traceback.format_exc()))

    def pipeline_mp(self, infos):
        for path, idx0 in infos:
            self.pipeline(path, idx0)

    def pipeline_paired(self, mic_path, sensor_path, stem):
        """配对模式: 一对同步录音 (mic 全频带 + sensor) 直接写出, 不做静音切片。

        - mic   -> 重采样到模型 SR (self.sr, 通常 48k), 响度归一化, 写 0_gt_wavs/<stem>.wav
        - sensor-> 确保 16k mono, 写 1_16k_wavs/<stem>.wav (不做 layer_norm 等推理侧没有的处理)

        下游 f0/feature/filelist 都靠 stem 配对, 故两侧 stem 必须完全一致。
        """
        try:
            mic_audio = load_audio(mic_path, self.sr)
            sensor_audio = load_audio(sensor_path, 16000)
            # #1 修复: 对齐推理侧 sensor 预处理 (peak-norm + 48Hz 高通), 绝不 layer_norm
            sensor_audio = match_inference_sensor_preprocess(sensor_audio)

            # 时长一致性核对 (同步录音、切点相同, 应满足)
            mic_dur = len(mic_audio) / float(self.sr)
            sensor_dur = len(sensor_audio) / 16000.0
            if abs(mic_dur - sensor_dur) > 0.05:
                println(
                    "%s\t-> WARNING mic/sensor 时长不一致 mic=%.3fs sensor=%.3fs"
                    % (stem, mic_dur, sensor_dur)
                )

            if not self.norm_write_paired(mic_audio, stem):
                return

            wavfile.write(
                "%s/%s.wav" % (self.wavs16k_dir, stem),
                16000,
                sensor_audio.astype(np.float32),
            )
            println("%s\t-> Success (paired)" % stem)
        except:
            println("%s\t-> %s" % (stem, traceback.format_exc()))

    def pipeline_paired_mp(self, infos):
        for mic_path, sensor_path, stem in infos:
            self.pipeline_paired(mic_path, sensor_path, stem)

    def pipeline_mp_inp_dir(self, inp_root, n_p):
        try:
            paired = discover_pairs(inp_root)
            if paired is not None:
                println("detected paired dataset: %s pairs" % len(paired))
                self._run_mp(paired, self.pipeline_paired_mp, n_p)
                return

            infos = [
                ("%s/%s" % (inp_root, name), idx)
                for idx, name in enumerate(sorted(list(os.listdir(inp_root))))
            ]
            self._run_mp(infos, self.pipeline_mp, n_p)
        except:
            println("Fail. %s" % traceback.format_exc())

    def _run_mp(self, infos, worker, n_p):
        if noparallel:
            for i in range(n_p):
                worker(infos[i::n_p])
        else:
            ps = []
            for i in range(n_p):
                p = multiprocessing.Process(target=worker, args=(infos[i::n_p],))
                ps.append(p)
                p.start()
            for i in range(n_p):
                ps[i].join()


def preprocess_trainset(inp_root, sr, n_p, exp_dir, per):
    pp = PreProcess(sr, exp_dir, per)
    println("start preprocess")
    pp.pipeline_mp_inp_dir(inp_root, n_p)
    println("end preprocess")


if __name__ == "__main__":
    preprocess_trainset(inp_root, sr, n_p, exp_dir, per)
