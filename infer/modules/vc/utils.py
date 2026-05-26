import os

from rvc_hubert_loader import load_sensor_hubert


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
