# infer-web.py の train_index() を単体で実行するスクリプト
# 使い方（リポジトリ直下で）:
#   python tools/train_index_manual.py <実験名> [v1|v2] [--no-kmeans]
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import faiss

exp_name = sys.argv[1]
version = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in ("v1", "v2") else "v2"
use_kmeans = "--no-kmeans" not in sys.argv

exp_dir = "logs/%s" % exp_name
dim = 256 if version == "v1" else 768
feature_dir = "%s/3_feature%d" % (exp_dir, dim)
if not os.path.isdir(feature_dir) or not os.listdir(feature_dir):
    sys.exit("特徴量がありません: %s（先に特徴抽出を実行）" % feature_dir)

npys = []
for name in sorted(os.listdir(feature_dir)):
    arr = np.load(os.path.join(feature_dir, name))
    if arr.ndim != 2 or arr.shape[1] != dim:
        print("skip (shape %s): %s" % (arr.shape, name))
        continue
    npys.append(arr)
big_npy = np.concatenate(npys, 0).astype("float32")
np.random.shuffle(big_npy)
print("features:", big_npy.shape)

if use_kmeans and big_npy.shape[0] > 2e5:
    from sklearn.cluster import MiniBatchKMeans

    print("kmeans -> 10000 centers")
    big_npy = (
        MiniBatchKMeans(n_clusters=10000, batch_size=256 * os.cpu_count(),
                        compute_labels=False, init="random", verbose=True)
        .fit(big_npy)
        .cluster_centers_.astype("float32")
    )

np.save("%s/total_fea.npy" % exp_dir, big_npy)
n_ivf = max(1, min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39))
print("n_ivf:", n_ivf)

index = faiss.index_factory(dim, "IVF%s,Flat" % n_ivf)
index_ivf = faiss.extract_index_ivf(index)
index_ivf.nprobe = 1
print("training...")
index.train(big_npy)
print("adding...")
for i in range(0, big_npy.shape[0], 8192):
    index.add(big_npy[i : i + 8192])

out = "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index" % (
    exp_dir, n_ivf, index_ivf.nprobe, exp_name, version)
faiss.write_index(index, out)
print("done:", out)
