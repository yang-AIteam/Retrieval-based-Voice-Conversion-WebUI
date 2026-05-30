import os
import sys
import traceback

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

device = sys.argv[1]
n_part = int(sys.argv[2])
i_part = int(sys.argv[3])
if len(sys.argv) == 7:
    exp_dir = sys.argv[4]
    version = sys.argv[5]
    is_half = sys.argv[6].lower() == "true"
else:
    i_gpu = sys.argv[4]
    exp_dir = sys.argv[5]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    version = sys.argv[6]
    is_half = sys.argv[7].lower() == "true"
import numpy as np
import soundfile as sf
import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

if "privateuseone" not in device:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
else:
    import torch_directml

    device = torch_directml.device(torch_directml.default_device())

f = open("%s/extract_f0_feature.log" % exp_dir, "a+")


def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


printt(" ".join(sys.argv))
# 训练特征提取必须与推理 (infer/modules/vc/utils.py load_hubert) 完全一致,
# 都用自训练的 SensorHubert, 才能消除训练/推理 OOD。
pth_path = "assets/hubert/sensor_hubert_rvc.pth"
config_path = "assets/hubert/hf_model"

printt("exp_dir: " + exp_dir)
wavPath = "%s/1_16k_wavs" % exp_dir
outPath = (
    "%s/3_feature256" % exp_dir if version == "v1" else "%s/3_feature768" % exp_dir
)
os.makedirs(outPath, exist_ok=True)


# wave must be 16k, hop_size=320
# 与推理 (infer/modules/vc/pipeline.py) 一致: sensor 波形只读成 mono float,
# 绝不施加 F.layer_norm —— SensorHubert 是 normalize=False 训练, 加 layer_norm 是踩过的坑。
def readwave(wav_path):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    feats = feats.view(1, -1)
    return feats


# SensorHubert model (与推理同一套权重/接口)
printt("load model(s) from {}".format(pth_path))
if os.access(pth_path, os.F_OK) == False:
    printt(
        "Error: Extracting is shut down because %s does not exist. "
        "请放置自训练的 SensorHubert 权重 (sensor_hubert_rvc.pth) 与 hf_model/ 配置。"
        % pth_path
    )
    exit(0)
from rvc_hubert_loader import load_sensor_hubert

model = load_sensor_hubert(pth_path=pth_path, config_path=config_path, device=device)
printt("move model to %s" % device)
if is_half:
    if device not in ["mps", "cpu"]:
        model = model.half()
model.eval()

todo = sorted(list(os.listdir(wavPath)))[i_part::n_part]
n = max(1, len(todo) // 10)  # 最多打印十条
if len(todo) == 0:
    printt("no-feature-todo")
else:
    printt("all-feature-%s" % len(todo))
    for idx, file in enumerate(todo):
        try:
            if file.endswith(".wav"):
                wav_path = "%s/%s" % (wavPath, file)
                out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))

                if os.path.exists(out_path):
                    continue

                feats = readwave(wav_path)
                padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                inputs = {
                    "source": (
                        feats.half().to(device)
                        if is_half and device not in ["mps", "cpu"]
                        else feats.to(device)
                    ),
                    "padding_mask": padding_mask.to(device),
                    "output_layer": 12,  # v2: SensorHubert layer-12 (768-d)
                }
                with torch.no_grad():
                    logits = model.extract_features(**inputs)
                    feats = logits[0]

                feats = feats.squeeze(0).float().cpu().numpy()
                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                else:
                    printt("%s-contains nan" % file)
                if idx % n == 0:
                    printt("now-%s,all-%s,%s,%s" % (len(todo), idx, file, feats.shape))
        except:
            printt(traceback.format_exc())
    printt("all-feature-done")
