"""
命令行版「训练特征索引」——等价于 WebUI 里那个按钮（infer-web.py 的 train_index()）。

用法:
    python build_feature_index.py <实验名>
    python build_feature_index.py <实验名> --no-kmeans          # 跳过 kmeans（最常见的"卡住"原因）
    python build_feature_index.py <实验名> --feature-dir logs/xx/3_feature768_sensor

产物（写到 logs/<实验名>/ 下，文件名和 WebUI 完全一致，推理端能直接认）:
    trained_IVF<n>_Flat_nprobe_1_<实验名>_<版本>.index
    added_IVF<n>_Flat_nprobe_1_<实验名>_<版本>.index    ← 推理时用的就是这个
    total_fea.npy
"""

import argparse
import os
import sys
import time

import faiss
import numpy as np


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def resolve_feature_dir(exp_dir, version, override):
    if override:
        return override
    for name in ("3_feature768", "3_feature256"):
        d = os.path.join(exp_dir, name)
        if os.path.isdir(d) and os.listdir(d):
            return d
    sys.exit("找不到特征目录，请先做特征提取，或用 --feature-dir 指定")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_name", help="实验名，即 logs/ 下的目录名")
    ap.add_argument("--version", default="v2", choices=["v1", "v2"])
    ap.add_argument("--feature-dir", default=None, help="覆盖特征目录（例如 sensor 特征）")
    ap.add_argument("--no-kmeans", action="store_true", help="跳过 MiniBatchKMeans 降采样")
    ap.add_argument(
        "--kmeans-threshold",
        type=int,
        default=200000,
        help="超过多少帧才做 kmeans，默认 2e5（和 WebUI 一致）",
    )
    args = ap.parse_args()

    exp_dir = os.path.join("logs", args.exp_name)
    if not os.path.isdir(exp_dir):
        sys.exit("实验目录不存在: %s" % exp_dir)

    feature_dir = resolve_feature_dir(exp_dir, args.version, args.feature_dir)
    names = sorted(f for f in os.listdir(feature_dir) if f.endswith(".npy"))
    if not names:
        sys.exit("特征目录是空的: %s" % feature_dir)
    log("特征目录 %s，共 %d 个文件" % (feature_dir, len(names)))

    npys = []
    for i, name in enumerate(names, 1):
        npys.append(np.load(os.path.join(feature_dir, name)))
        if i % 200 == 0 or i == len(names):
            log("  已加载 %d/%d" % (i, len(names)))

    big_npy = np.concatenate(npys, 0).astype("float32")
    del npys
    idx = np.arange(big_npy.shape[0])
    np.random.shuffle(idx)
    big_npy = big_npy[idx]
    log("拼接完成 shape=%s dtype=%s" % (big_npy.shape, big_npy.dtype))

    if big_npy.shape[0] > args.kmeans_threshold and not args.no_kmeans:
        from sklearn.cluster import MiniBatchKMeans

        n_cpu = os.cpu_count() or 4
        log("帧数 %d 超阈值，开始 MiniBatchKMeans 降到 10k 中心（这一步最慢）..." % big_npy.shape[0])
        big_npy = (
            MiniBatchKMeans(
                n_clusters=10000,
                verbose=True,
                batch_size=256 * n_cpu,
                compute_labels=False,
                init="random",
            )
            .fit(big_npy)
            .cluster_centers_
        ).astype("float32")
        log("kmeans 完成 shape=%s" % (big_npy.shape,))
    elif args.no_kmeans:
        log("按要求跳过 kmeans")

    np.save(os.path.join(exp_dir, "total_fea.npy"), big_npy)

    dim = big_npy.shape[1]
    n_total = big_npy.shape[0]
    n_ivf = max(min(int(16 * np.sqrt(n_total)), n_total // 39), 1)
    log("建索引 dim=%d n_total=%d n_ivf=%d" % (dim, n_total, n_ivf))

    index = faiss.index_factory(dim, "IVF%s,Flat" % n_ivf)
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1

    log("training ...")
    index.train(big_npy)
    trained_path = os.path.join(
        exp_dir,
        "trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, args.exp_name, args.version),
    )
    faiss.write_index(index, trained_path)

    log("adding ...")
    for i in range(0, n_total, 8192):
        index.add(big_npy[i : i + 8192])
    added_path = os.path.join(
        exp_dir,
        "added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, args.exp_name, args.version),
    )
    faiss.write_index(index, added_path)

    log("完成:")
    log("  %s" % trained_path)
    log("  %s   ← 推理用这个" % added_path)


if __name__ == "__main__":
    main()
