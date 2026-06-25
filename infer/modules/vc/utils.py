import os

from rvc_hubert_loader import load_sensor_hubert


def consist_off_enabled():
    """oldgen_consistOFF 臂开关：置位时旁路推理侧喂给 SensorHubert 的两步预处理
    （peak-norm + 48Hz 高通 filtfilt），使推理输入回到旧/已交付生成器训练时所见的
    裸特征分布（在「裸」上一致）。默认关闭 → 维持正常 ON 行为，不影响 anchor/newgen。

    通过环境变量 ``rvc_consist_off`` 控制（接受 1/true/yes，大小写不敏感）。
    注意：只旁路 sensor 输入路径；绝不影响 gt/输出归一化，绝不加 layer_norm。
    """
    return os.environ.get("rvc_consist_off", "").strip().lower() in ("1", "true", "yes")


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def load_hubert(config):
    print("[sensor] load_hubert: loading sensor_hubert_rvc.pth")
    model = load_sensor_hubert(
        pth_path="assets/hubert/sensor_hubert_rvc.pth",
        config_path="assets/hubert/hf_model",
        device=config.device,
    )
    if config.is_half and config.device not in ("cpu", "mps"):
        model = model.half()
    return model
