# ONNX 验证与合并可视化

在 3090 的 SparseDrive 根目录，激活 sparsedrive 环境。默认使用 onnx/s1.onnx 和同名 s1.calibration.npz，数据 data/nuscenes/，验证索引 data/infos/nuscenes_infos_val.pkl，OCC 真值 data/nuscenes/gts。无需 PTH。

## 完整验证

```bash
PYTHONPATH=. python tools/val_onnx.py
```

输出 val/onnx/metrics.json：nuScenes 官方检测 mAP、NDS、各类 AP 和定位/尺度/朝向/速度等误差；OCC camera-visible mIoU 及 18 类 IoU。OCC mIoU 排除 free 类17，对有效相机 mask 统计，忽略255，无有效 union 类写 null。指标是0～1比例。输出 occ_confusion.npy、detections.pkl、occ/<sample_token>.npy，官方检测 JSON/报告由数据集评估器写入同目录。

run.json 记录运行帧数、provider 和每帧标定与固定 ONNX 标定的最大差。当前 ONNX 将 sample-0 标定固化，nuScenes 各帧标定变化，因此这些是固定标定部署图在数据集上的实测指标，不能当作原 PTH 精度或导出精度损失。若需与 PTH 公平比较，应提供动态标定 ONNX，或采用标定一致的数据。这里没有伪造动态标定输入。

顺序遍历验证集，首帧/换场景重置缓存，场景内反馈真实 ONNX 缓存，采用真实 dt 和上一帧到当前帧的 lidar 位姿变换。检测使用项目原 SparseBox3DDecoder，保留 top300、多类别和 quality centerness 评分逻辑，不套用可视化的0.3阈值。

冒烟测试：

```bash
PYTHONPATH=. python tools/val_onnx.py --max-frames 2 --out-dir val/onnx_smoke
```

部分数据不运行官方 mAP/NDS，以免产生误导。不会自动启动全量验证。

## 可视化

```bash
# 默认全部 scene
PYTHONPATH=. python tools/visualization/visualize_onnx.py
# 只渲染第一个完整 scene
PYTHONPATH=. python tools/visualization/visualize_onnx.py --num-scenes 1
# 每5帧保存一次，仍推理所有中间帧以保持缓存连续
PYTHONPATH=. python tools/visualization/visualize_onnx.py --num-scenes 2 --interval 5
```

仅保存 vis/onnx/scene_000_<scene_token>/combine/0000.jpg 形式的合并图，没有单独六路图、BEV图或视频。上方两行六相机3D检测投影；下方左侧以车为中心的10～50m同心圈检测BEV（绿GT、红预测）；下方中/右为OCC真值/预测，统一相机可见mask，按最高非free体素投影到BEV；底部有类别颜色图例。OCC坐标为ego，检测BEV为lidar，分别标注，未混作同一坐标。未知/不可见区域与free共用浅色，图例已注明。--score-threshold 默认0.3，仅影响绘图。

共用参数 --onnx、--config、--data-root、--threads、--max-frames、--out-dir。所有命令从项目根目录运行，先执行 PYTHONPATH=.。

## 文件与验证范围

- tools/onnx_stage1.py：ONNX Runtime 自动 CUDA/CPU 会话、数据预处理、历史状态和检测解码。
- tools/val_onnx.py：流式OCC混淆矩阵和官方检测评估，OCC预测落盘，避免把全量体素留在内存。
- tools/visualization/visualize_onnx.py：直接推理并仅保存合并图。

已对实际 onnx/s1.onnx 完成一帧验证及两帧可视化冒烟测试，检查了生成图片。尚未运行6019帧全量验证，不提供未经测量的mAP/NDS结论。


## 指定模型和输出目录

```bash
PYTHONPATH=. python tools/val_onnx.py --onnx onnx/s1.onnx --output val/s1
PYTHONPATH=. python tools/visualization/visualize_onnx.py --onnx onnx/s1.onnx --output vis/s1 --num-scenes 1
```

--output 与旧 --out-dir 为同一参数；路径必须是输出目录。更换 ONNX 时同时提供同名 .calibration.npz，例如 model2.onnx 对应 model2.calibration.npz。


## 速度与后端（2026-11-17）

已安装 onnxruntime-gpu 1.14.1（与原 ORT 版本一致），使用现有 PyTorch CUDA 11/cuDNN8 库。默认 --device auto，有 CUDA provider 时选 GPU；可显式 --device cuda，初始化失败时报错，避免悄悄变慢；--device cpu 可复查原 CPU 结果。CUDA 仍允许部分算子由 CPU 执行，不代表全图都在 GPU。

```bash
PYTHONPATH=. python tools/val_onnx.py --onnx onnx/s1.onnx --output val/s1 --device cuda
PYTHONPATH=. python tools/visualization/visualize_onnx.py --onnx onnx/s1.onnx --output vis/s1 --device cuda
```

前三帧同输入与自身历史反馈：CPU 9.52秒（推理8.85秒），CUDA 5.17秒（推理4.55秒），包括首帧开销，约1.84倍提速。随后八帧CUDA验证跑通，累计预处理1.67秒、推理8.86秒、解码0.008秒；OCC读取/落盘不包含在这三个分项内。run.json 会保存各阶段耗时。单次小样本测试不能承诺全量同等提速。

CUDA和CPU不逐元素等价：前三帧OCC类别一致率99.93%～99.94%，排序后检测分数最大差约0.00259；排序后分数差不代表匹配同一目标的检测误差，也不能证明mAP等价。真实后端指标以完整数据验证为准。这里未修改模型、历史顺序、采样、检测解码或mIoU计算口径。

升级代码/运行时后需要重新启动验证进程，已有进程不会自动切换GPU。不应向仍在使用的输出目录同时启动第二份验证。
