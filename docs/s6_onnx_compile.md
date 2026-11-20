# S6 固定与动态外参导出及编译

以下命令在 `ssh 3090` 上执行。两种 ONNX 都固定 `image_wh`，都使用同一套时序输入。固定模式把检测投影和 OCC 投影写进 ONNX 权重区，运行时共 7 个输入。动态模式额外接收 `projection_mat` 和 `occ_ego2img`，共 9 个输入。

## 1. 导出 ONNX

```bash
cd /home/fjm/tong/SparseDrive
conda activate sparsedrive
export PYTHONPATH=.
CFG=projects/configs/ep1/s6_det_occ_decoder4_fpn3_fastbev.py
PTH=ckpt/iter_3512.pth
PYTHONPATH=. python tools/export_ep1_npu.py --config "$CFG" --checkpoint "$PTH" \
  --calibration-mode dynamic --output onnx/s6/s6_dynamic.onnx
PYTHONPATH=. python tools/export_ep1_npu.py --config "$CFG" --checkpoint "$PTH" \
  --calibration-mode fixed --sample-index 0 --output onnx/s6/s6_fixed.onnx
```

命令以已经用于动态编译验证的 S6 中间权重为例。要使用训练最终权重，将 `PTH` 换成 `ckpt/s6.pth`。固定模式的 sample 0 标定用于验证流程。部署到自己的车时，通过 `--calibration /绝对路径/rig.npz` 传入实际六相机标定，NPZ 包含 `projection_mat`、`occ_img2ego` 和 `image_wh`，分辨率与图像预处理必须对应。动态模式由每帧预处理提供两组投影，其中 `occ_ego2img` 是 ego→image，数据集原有的 `occ_img2ego` 由 `tools.ep1_common.occ_forward_projection` 转换。

每次导出同时生成同名 `.validation.txt` 和 `.calibration.npz`。检查 `validation_passed=True`，并保留 NPZ 供校准及验证入口读取。首帧或切换场景时，传入空缓存、单位位姿变换和 `history_valid=0`。连续有效帧使用上次输出的缓存、实际相对位姿与时间间隔。

## 2. 生成校准输入

```bash
cd /home/fjm/tong/SparseDrive
APP=/home/fjm/tong/Compilation/app/sparse
for mode in dynamic fixed; do
  PYTHONPATH=. python "$APP/prepare_calibration.py" \
    --config "$CFG" --onnx "onnx/s6/s6_${mode}.onnx" \
    --frames 4 --output "$APP/calibration/$mode" || break
done
```

校准输入的顺序由各自的 ONNX 决定。固定和动态的 `dataset.txt` 分开存放。4 个连续帧用于验证编译流程，全量 mAP、NDS、mIoU 需要另行评估。

## 3. 依次编译两份模型

```bash
cd /home/fjm/tong/Compilation/app/sparse
for mode in dynamic fixed; do
  export S6_VARIANT="s6_$mode"
  export S6_CALIBRATION_DIR="$PWD/calibration/$mode"
  bash 00_init.sh "/home/fjm/tong/SparseDrive/onnx/s6/${S6_VARIANT}.onnx" &&
  bash 01_check.sh &&
  bash 02_preprocess.sh &&
  bash 03_omg.sh || break
done
```

- `00` 读取 ONNX 的输入顺序与形状，生成 `input_cfg.yml`。
- `01` 检查原始图的算子支持。
- `02` 应用 SDK 图改写并在 ORT 比对连续帧，再检查编译图。特征默认 INT8，选中的几何张量使用 FP16。
- `03` 调用 OMG 进行 LQ560ProV200 编译。成功后在 `models/` 下生成 `s6_dynamic.ortm` 和 `s6_fixed.ortm`。

两种模型和配套 NPZ 使用独立文件名。同一目录的暂存图、`input_cfg.yml` 和 `results/` 日志由两轮共用，按顺序执行。以 OMG 成功日志和对应 ORTM 文件为准。实体板运行验证需要把 ORTM 和对应的运行时输入放到板上。

## 4. 验证 ONNX

```bash
cd /home/fjm/tong/SparseDrive
PYTHONPATH=. python tools/val_ep.py onnx/s6/s6_dynamic.onnx --config "$CFG" --output val/s6_dynamic
PYTHONPATH=. python tools/val_ep.py onnx/s6/s6_fixed.onnx --config "$CFG" --output val/s6_fixed
```

nuScenes 各帧标定会变化，固定模式的指标包含固定标定与真实标定不匹配的影响。

固定版的 `00_init.sh` 会在 `input_cfg.yml` 写入顶层 `passes` 设置，关闭 `FuseMatMulAddPass`。SDK 2.1.6 把固定投影路径融合为 Gemm 时会拒绝 FP16 输入和 INT8 逐通道权重的组合。动态版保留 SDK 默认融合设置。

## 本次验证

使用 `iter_3512.pth` 和 nuScenes sample 0 标定，固定版通过 ONNX 数值校验、四帧编译图比对和 OMG 实际编译，输出 `app/sparse/models/s6_fixed.ortm`。动态版此前以同一权重完成 OMG 编译，现有产物为 `app/sparse/models/s6.ortm`，按本文重跑则输出 `s6_dynamic.ortm`。两份结果均为 LQ560ProV200 的编译产物，实体板推理仍待测试。
