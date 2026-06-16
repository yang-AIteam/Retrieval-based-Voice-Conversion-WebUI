# VC-2-mic2mic 分支说明

## 基点
本分支以 `VC-2` 为基点创建（`git checkout -b VC-2-mic2mic VC-2`）。

## 改了什么
RVC 训练的 **input = mic，output = mic**（与原生 RVC 数据流一致），但 **特征提取器是自训练的 SensorHubert**。

也就是：**原版数据流（单一 mic 目录 + 切片）+ SensorHubert 特征提取器**。

| 项目 | VC-2 (sensor2mic) | VC-2-mic2mic (本分支) |
| --- | --- | --- |
| 数据集 | 配对（mic + sensor） | 单一 mic 目录 |
| 预处理 | paired，不切片，按 stem 配对 | 原版静音切片流程 |
| input（`1_16k_wavs/`，特征提取来源） | **sensor** 重采样 16k | **mic** 切片重采样 16k |
| output（`0_gt_wavs/`，重建目标） | mic（模型 SR） | mic 切片（模型 SR） |
| 特征提取器 | SensorHubert (`sensor_hubert_rvc.pth`) | SensorHubert (`sensor_hubert_rvc.pth`) — 不变 |

## 代码改动（相对 VC-2）
- [infer/modules/train/preprocess.py](infer/modules/train/preprocess.py)：**移除 paired 逻辑**，还原为原版单一来源切片流程（VC-1 版本）。
  - 删除 `discover_pairs` / `norm_write_paired` / `pipeline_paired` / `pipeline_paired_mp` / `_run_mp`；
  - `pipeline_mp_inp_dir` 不再检测配对数据集，直接对单一目录切片；
  - `norm_write` 把每个 mic 切片同时写出 `0_gt_wavs/`（模型 SR）与 `1_16k_wavs/`（16k），input=output 同源 mic。
- [infer/modules/train/extract_feature_print.py](infer/modules/train/extract_feature_print.py)：**未改动**，仍是 VC-2 的 SensorHubert 加载与 layer-12 特征提取。

## 数据集用法
直接放一个普通的 mic 音频目录即可（与原生 RVC 训练相同），**不需要配对的 sensor 数据**。预处理会做静音切片，input 和 output 都来自同一段 mic 切片。

## 目的
作为 VC-2（sensor2mic）的对照基线：在特征提取器同为 SensorHubert 的前提下，对比「input 用 mic」与「input 用 sensor」对训练/推理效果的影响。
本分支也可视为 VC-1 数据流 + SensorHubert 提取器，用来验证「换成 SensorHubert 后、纯 mic→mic 训练」是否消除了 VC-1 中训练（原版 hubert_base.pt）与推理（SensorHubert）不一致带来的 OOD。
