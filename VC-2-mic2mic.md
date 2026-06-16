# VC-2-mic2mic 分支说明

## 基点
本分支以 `VC-2` 为基点创建（`git checkout -b VC-2-mic2mic VC-2`）。

## 改了什么
把 RVC 训练预处理的 **input 从 sensor 改成 mic**，**output 仍然是 mic**，**特征提取器仍是自训练的 SensorHubert**。

| 项目 | VC-2 (sensor2mic) | VC-2-mic2mic (本分支) |
| --- | --- | --- |
| input（`1_16k_wavs/`，特征提取来源） | **sensor** 重采样 16k | **mic** 重采样 16k |
| output（`0_gt_wavs/`，重建目标） | mic（模型 SR） | mic（模型 SR）— 不变 |
| 特征提取器 | SensorHubert (`sensor_hubert_rvc.pth`) | SensorHubert (`sensor_hubert_rvc.pth`) — 不变 |

唯一的变量是「特征提取的输入波形」：sensor → mic。`0_gt_wavs/`（mic 目标）与 VC-2 完全一致，因此本分支与 VC-2 构成**干净的对照实验**（mic2mic vs sensor2mic）。

## 代码改动
仅改动 [infer/modules/train/preprocess.py](infer/modules/train/preprocess.py) 的 `pipeline_paired`：
- 删除对 `sensor_path` 的读取与时长核对；
- `1_16k_wavs/<stem>.wav` 的来源由 `load_audio(sensor_path, 16000)` 改为 `load_audio(mic_path, 16000)`；
- 日志标记由 `Success (paired)` 改为 `Success (mic2mic)`。

`extract_feature_print.py`（SensorHubert 加载与特征提取）**未改动**，与 VC-2 完全相同。

## 数据集用法
- **复用 VC-2 的配对数据集**（`inp_root/mic/` + `inp_root/sensor/`，或 `*_mic.wav` / `*_sensor.wav`）：
  `discover_pairs` 仍按配对检测并对齐 stem，但 `pipeline_paired` 只读取 mic 侧，sensor 侧被忽略。这样 stem 集合、mic 目标都与 sensor2mic 完全一致，是最干净的对照写法。
- **只有普通 mic 目录**（无配对）：`discover_pairs` 返回 `None`，自动回退到原版切片流程（`pipeline`），input=output=mic 切片，特征仍由 SensorHubert 提取，也成立。

## 目的
在控制其他变量（mic 目标 + SensorHubert 提取器）不变的前提下，对比「特征提取输入用 sensor」与「用 mic」对训练/推理效果的影响，评估 sensor 输入引入的训练/推理一致性与质量差异。
