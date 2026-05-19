"""
Cross-domain FAISS index builder for sensor→mic voice conversion.

Standard RVC FAISS: index on mic features, search with mic-like query
Cross-domain FAISS: index on sensor features (search key), store mic features (retrieve value)

This means: at inference, sensor HuBERT features retrieve the corresponding
mic-domain HuBERT features, eliminating domain mismatch during retrieval.

Usage:
    python build_cross_domain_index.py <exp_name>

Prerequisites:
    - Run standard preprocessing on mic audio → logs/<exp_name>/3_feature768/
    - Run sensor feature extraction  → logs/<exp_name>/3_feature768_sensor/
      (use extract_feature_print.py pointed at sensor wavs, output to 3_feature768_sensor/)
    - File names in both dirs must match (e.g., "abc_mic.npy" ↔ "abc_sensor.npy"
      or same base name if extracted separately)

Output:
    logs/<exp_name>/cross_domain_sensor_keys.index  — FAISS index keyed by sensor features
    logs/<exp_name>/cross_domain_mic_values.npy     — paired mic features (same row order)
"""

import os
import sys
import logging
from multiprocessing import cpu_count

import faiss
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_npy_dir(directory: str) -> tuple[list[str], np.ndarray]:
    names = sorted(f for f in os.listdir(directory) if f.endswith(".npy"))
    if not names:
        raise FileNotFoundError(f"No .npy files found in {directory}")
    arrays = [np.load(os.path.join(directory, n)) for n in names]
    return names, np.concatenate(arrays, axis=0).astype("float32")


def strip_suffix(name: str) -> str:
    base = os.path.splitext(name)[0]
    for suffix in ("_mic", "_sensor"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


def build_cross_domain_index(exp_name: str):
    exp_dir = os.path.join("logs", exp_name)
    mic_dir = os.path.join(exp_dir, "3_feature768")
    sensor_dir = os.path.join(exp_dir, "3_feature768_sensor")

    for d in (mic_dir, sensor_dir):
        if not os.path.isdir(d):
            logger.error(f"Directory not found: {d}")
            sys.exit(1)

    mic_names = sorted(f for f in os.listdir(mic_dir) if f.endswith(".npy"))
    sensor_names = sorted(f for f in os.listdir(sensor_dir) if f.endswith(".npy"))

    # Match files by base name (strip _mic / _sensor suffixes)
    mic_map = {strip_suffix(n): n for n in mic_names}
    sensor_map = {strip_suffix(n): n for n in sensor_names}

    common_keys = sorted(set(mic_map) & set(sensor_map))
    if not common_keys:
        logger.error(
            "No matching file pairs found between mic and sensor feature dirs. "
            "Check that base names match after stripping _mic/_sensor suffixes."
        )
        logger.info(f"  mic files:    {mic_names[:5]}")
        logger.info(f"  sensor files: {sensor_names[:5]}")
        sys.exit(1)

    logger.info(f"Found {len(common_keys)} paired files: {common_keys}")

    sensor_parts, mic_parts = [], []
    for key in common_keys:
        s = np.load(os.path.join(sensor_dir, sensor_map[key])).astype("float32")
        m = np.load(os.path.join(mic_dir, mic_map[key])).astype("float32")

        # Time-align: both from the same synchronized recording, but may have
        # slight frame count differences; trim to the shorter one.
        min_len = min(len(s), len(m))
        if abs(len(s) - len(m)) > 5:
            logger.warning(
                f"{key}: sensor={len(s)} frames, mic={len(m)} frames — "
                f"trimming to {min_len}. Large difference may indicate misalignment."
            )
        sensor_parts.append(s[:min_len])
        mic_parts.append(m[:min_len])

    sensor_feats = np.concatenate(sensor_parts, axis=0)  # (N, 768)
    mic_feats = np.concatenate(mic_parts, axis=0)        # (N, 768)

    logger.info(f"Total frames: {sensor_feats.shape[0]}, dim={sensor_feats.shape[1]}")
    assert sensor_feats.shape == mic_feats.shape

    dim = sensor_feats.shape[1]
    n_total = sensor_feats.shape[0]

    # IVF cell count: same heuristic as standard RVC train-index-v2.py
    n_ivf = min(int(16 * np.sqrt(n_total)), n_total // 39)
    n_ivf = max(n_ivf, 4)  # need at least a few cells

    logger.info(f"Building IVF{n_ivf},Flat index on sensor features ...")
    index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1

    index.train(sensor_feats)
    index.add(sensor_feats)

    out_index = os.path.join(exp_dir, "cross_domain_sensor_keys.index")
    out_values = os.path.join(exp_dir, "cross_domain_mic_values.npy")

    faiss.write_index(index, out_index)
    np.save(out_values, mic_feats)

    logger.info(f"Saved index  → {out_index}")
    logger.info(f"Saved values → {out_values}")
    logger.info("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <exp_name>")
        sys.exit(1)
    build_cross_domain_index(sys.argv[1])
