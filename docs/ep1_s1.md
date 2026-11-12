# EP1 S1：SparseDrive 检测 + FlashOCC

配置：`projects/configs/ep1/s1_det_occ_b0_nose.py`。`s1` 表示 stage 1；后续 stage 2 再新增 `s2` 配置，本次没有 motion/planning。

## 模型与数据

| 部分 | 本次实现 |
| --- | --- |
| 共享图像骨干 | torchvision EfficientNet-B0，移除所有 SqueezeExcitation；不含分类器 |
| 骨干输出 | stride 4/8/16/32，通道 24/40/112/1280 |
| 检测 FPN | 输入上述四级特征，输出四级 256 通道 |
| 检测 query | 共 450；首轮单帧解码后使用历史 300 + 当前 top 150；首帧保持 450 个当前候选 |
| 检测解码器 | 6 层，256 维，保留 quality 分支；最终最多输出 300 个检测框 |
| 检测注意力 | 训练/普通 GPU 推理使用原 MultiheadFlashAttention；导出用同一套打包 QKV/输出权重执行标准算子 |
| map/tracking/motion/planning | 不构建对应头，不加载 map/future 标注，不分配 tracking ID |
| OCC 图像 neck | 共享骨干的 112/1280 两级特征融合，输出 stride 16、256 通道 |
| OCC 视图变换 | LSS 深度分布 × 64 通道 context，按相机投影累加到 BEV |
| OCC BEV encoder | ResNet stages 128/256/512；FPN-LSS 融合及上采样后卷积均为 128 通道 |
| OCC predictor | `Linear(128,128) -> Softplus -> Linear(128,18*16)` |
| OCC 输出 | `[B,128,128,16,18]` logits；普通推理同时返回 `[128,128,16]` uint8 类别 |
| OCC 网格 | x/y `[-25.6,25.6,0.4]`；z `[-1,5.4,0.4]`，16 格 |
| LSS 配置 | z `[-1,5.4,6.4]` 合并为单层 BEV；depth `[1,45,1]`，44 格 |

按照本机《LQ560V200 NPU工具使用指南》FlashOCC 章节（117–123 页）实现 128 通道 BEV neck、128×128 网格和不扩张隐藏维度的 Softplus predictor。LSS/2D-BEV/channel-to-height 结构参考 [FlashOCC](https://github.com/Yzichen/FlashOCC)。本实现使用本仓库的 mmdet registry，不依赖安装另一个项目的 mmdet3d，也不直接复用其他项目的权重。

检测仍在 lidar 坐标下。Occ3D 标签在关键帧 ego 坐标下，通过 `occ_ego2lidar` 和增强后的 `lidar2img` 求逆得到 `occ_img2ego`，避免直接把 lidar 网格当成 ego 网格。图像 resize/crop/flip/rotation 都体现在投影矩阵中；本配置保持 `rot3d_range=[0,0]`，防止 3D 框与未旋转的 OCC 标签失配。

现有 `gts/scene-name/sample-token/labels.npz` 是 200×200×16、范围 [-40,40] 的标签，训练/评测会取 `[36:164,36:164,:]` 中心裁剪，保留 0.4 m 体素及类别编号。不要把类别图双线性缩放到 128×128。已经预裁剪到 128 的其他数据，需要同步调整 `occ_source_grid`，不能混用。

检测范围仍沿用原配置的 55 m 过滤；±25.6 m 仅约束 OCC，二者不是同一个范围。

## 四卡租赁服务器训练

目标硬件为用户指定的 4 张、每张 32 GB GPU。默认每卡 batch=4，总 batch=16，AdamW 学习率 1e-4（从原全局 batch 64 / lr 4e-4 线性缩放）。每卡 2 个 DataLoader worker；训练 100 个等效 epoch，每 epoch 1758 iter，每 10 epoch 验证/存档，保留 3 份 checkpoint。实际显存占用需在租赁机上确认，3090 的单样本测试不等于四卡容量测量。

在租赁机上将代码和以下数据放在项目目录：

```text
data/nuscenes/                 # 可软链接到实际磁盘
  samples/ sweeps/ maps/ v1.0-trainval/
  gts/scene-xxxx/<sample-token>/labels.npz
data/infos/nuscenes_infos_train.pkl
data/infos/nuscenes_infos_val.pkl
data/kmeans/kmeans_det_900.npy
```

`infos` 使用本仓库的数据生成格式，而不是 BEVDet/FlashOCC 的 `bevdetv*-nuscenes_infos_*.pkl`。图像和 lidar 路径从 infos 中读取，迁移后需保证这些路径有效。`data/`、`ckpt/` 和 `work_dirs*/` 被 Git 忽略，git clone 不会带上数据/权重。

当前验证环境：Python 3.8.20、torch 1.13.0+cu116、torchvision 0.14.0+cu116、mmcv-full 1.7.1、mmdet 2.28.2、numpy 1.23.5。新服务器先安装匹配的 torch/torchvision，再运行：

```bash
conda activate sparsedrive
cd /path/to/SparseDrive
python -m pip install -r requirement.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python tools/prepare_ep1_anchors.py
bash scripts/train_ep1_s1.sh
```

本配置的特征采样仍使用 `use_deformable_func=False`，不要求编译 `projects/mmdet3d_plugin/ops`。检测注意力已恢复原来的 `MultiheadFlashAttention`，训练和普通 GPU 推理需要 flash-attn 2.3.2；`requirement.txt` 已加入对应 Python 3.8 / torch 1.13 / CUDA 11.6 的 wheel。

导出不训练第二套注意力。`forward_export` 在调用期间将原 FlashMHA 切到标准 MatMul/Softmax，再恢复原内核；权重沿用原 `attn.in_proj_weight/bias` 和 `attn.out_proj.*`。FP16 Flash 内核与 FP32 导出路径存在正常的舍入差异，自注意力和历史交叉注意力均有输出/梯度对比测试。之前 EP1 的独立 `q_proj/k_proj/v_proj` checkpoint 会在加载时拼接为原打包 QKV 权重；旧 `out_proj` 同时迁移，支持严格加载。

默认 `pretrained=None`，可完全离线构建，但从头训练。建议正式训练前准备匹配的 no-SE 骨干预训练权重，并在配置中填写 `model.img_backbone.pretrained`。也支持 torchvision B0 初始化匹配层，SE/classifier 参数会被忽略；这只是初始化，不是指南里的已训练 no-SE 权重。官方 ResNet50 stage1/stage2 checkpoint 与本模型不兼容，不能作为完整模型直接恢复。

`prepare_ep1_anchors.py` 对全部 900 个原始中心重新聚为 450 个，避免简单截断前 450 个导致空间分布偏置。该文件是初始化；若重新生成自己的数据，建议直接在训练集 GT 上重新做 450 类聚类。工具会保留已存在的 450 文件。

恢复训练：

```bash
bash scripts/train_ep1_s1.sh --resume-from work_dirs/ep1/s1_det_occ_b0_nose/iter_XXXX.pth
```

如果改 batch/GPU 数，需同步修改配置顶部 `num_gpus`、`batch_size`、优化器 lr，并重算迭代/验证间隔；仅通过 `--cfg-options data.samples_per_gpu=...` 不会自动重算顶部变量。若显存和数据吞吐允许，下一步可试每卡 8（总 32，lr 2e-4），需实测后决定。

## 一个 ONNX 处理首帧和历史帧

```bash
python -m pip install 'numpy<1.24' onnx==1.14.1 onnxruntime==1.14.1
PYTHONPATH=. python tools/export_ep1_onnx.py \
  projects/configs/ep1/s1_det_occ_b0_nose.py \
  --checkpoint /path/to/your/ep1-checkpoint.pth
```

仅验证图结构时用 `--random-init` 替代 `--checkpoint`，生成的文件没有训练效果。不要将这份随机权重 ONNX 用作有效模型。

固定 batch 1 / 6 cameras / 256×704 / 450 queries / 300 history 的 ONNX，opset 16。输入：

| 输入 | 形状/含义 |
| --- | --- |
| img | `[1,6,3,256,704]`，已按配置归一化的 RGB 图像 |
| projection_mat | `[1,6,4,4]`，增强后的 lidar→image |
| image_wh | `[1,6,2]`，每相机 `[704,256]` |
| occ_img2ego | `[1,6,4,4]`，增强后 image ray→keyframe ego |
| history_feature | `[1,300,256]` |
| history_anchor | `[1,300,11]`，上一帧 lidar 下编码框 |
| history_confidence | `[1,300]` |
| history_transform | `[1,4,4]`，上一帧 lidar→当前帧 lidar |
| time_interval | `[1]`，当前减上一帧时间，秒 |
| history_valid | `[1]` bool，当前样本是否有合法历史 |

首帧/新场景：history 三个张量补 0，transform 用单位矩阵，time_interval 可用 0.5，history_valid=False。后续将 `next_history_feature/anchor/confidence` 回传，提供实际变换/时间，history_valid=True。模型内部还会拒绝 dt≤0 或超过 2 秒的历史。无效 history 同时屏蔽缓存替换及 temporal attention 残差，避免“补零但 attention bias 仍影响首帧”。普通 PyTorch 模式也检查 scene_token，支持同一个 batch 中部分样本切换场景。

ONNX 输出依次为 raw det logits `[1,450,10]`、encoded boxes `[1,450,11]`、quality `[1,450,2]`、occ logits `[1,128,128,16,18]` 和三个 next_history 张量。框解码/top-k 阈值/类别解码在图外，OCC logits 最后一维 argmax 得到类别。框编码沿用原项目 `[x,y,z,log(w),log(l),log(h),sin(yaw),cos(yaw),vx,vy,vz]`。

PyTorch 中所有相关 bilinear resize 都是 align_corners=False；旧版导出器通常写 half_pixel，导出脚本会将本图（所有 resize 输出空间维度均大于 1）的模式写为 `pytorch_half_pixel`，两者在这里数值等价，再用 ORT 实测验证，而非仅依赖参数名。

当前 ONNX 是完整、标准算子的参考图，包含 GridSample 和 ScatterElements/add（LSS pooling），没有 FlashAttention/自定义 DAF/按帧分支的 If。它已能在 ORT 上执行，但不等于已通过 LQ560 NPU 编译。NPU 工具链对 pooling、ScatterElements/add、注意力矩阵形状及量化的支持仍需板端验证；可优先用固定标定预计算 pooling 索引，并适配厂商 BEVPool/自定义算子。相机标定或图像增强变化时，索引不能直接复用。

OCC 的 scatter_add 在 CUDA 有浮点加法顺序差异，所以测试对 OCC logits 使用小容差；检测状态一致性测试不依赖这部分随机顺序。FP32 对齐 ORT 时关闭 TF32，避免细小卷积误差使相近置信度的 top-k 缓存交换顺序。

## 验证与后续压缩

```bash
PYTHONPATH=. python -m unittest discover -s tests -p test_ep1.py -v
PYTHONPATH=. python -m unittest discover -s tests -p test_ep1_attention.py -v
```

短验证覆盖真实图像/检测/OCC 标签的前后向梯度、无效缓存隔离、正常历史影响、超时重置，以及标签中心裁剪、ego-motion、混合 batch 场景重置、LSS 累加/坐标轴/梯度、空可见掩码和替换 box block 的数学等价性。ONNX 脚本对同一个文件验证首帧、历史帧（包含非单位 ego-motion）和新场景重置。

默认模型参数量 40,533,182；原 stage1（检测+map，ResNet50）为 75,675,882，任务组合不同，不能将参数比当作同任务速度比。本模型检测 head 21,182,895、OCC 分支 12,297,484，是下一步压缩重点。

恢复 FlashAttention 后在 3090 的单样本验证：真实训练数据两步反向传播通过，8 项回归测试通过，包括原 Flash 内核调用、共享权重输出/梯度比较、旧权重加载和导出后内核恢复。随机权重 ONNX 的首帧/历史帧/新场景 ORT 最大绝对误差约 2.51e-4，6 个 Resize 均为 pytorch_half_pixel；这不代表训练后精度或 LQ560 板端性能，尚未进行长训和 NPU 编译。

建议按以下顺序做消融，每次都看检测 mAP/NDS、小目标召回、OCC mIoU（17 个非 free-space 类）以及目标板延迟：

1. 将检测 embed_dims 256→128，联动 FPN、anchor encoder、attention、FFN、refine。attention score 矩阵大小不随通道变，但 QKV/FFN 参数量和计算量会明显下降。不能只改 FPN 输出通道。
2. 解码器 6→3/4 层，或者浅层候选过滤。前者有直接延迟收益，后者可能引入动态 shape，需同时设计导出接口。默认先保留 6 层，减少同时改动导致的精度归因困难。
3. OCC BEV encoder 的 128/256/512→128/128/128，neck 已经是 128，但 encoder 本身仍占较多参数。该项需要单独验证 OCC 精度。
4. 检测减少 stride-4 大特征图或采样点（13→7），并同步更改 DenseDepth 与训练 gt_depth；重点检查远处小目标和行人的召回。
5. INT8/QAT、Conv-BN 融合、固定标定的投影/池化索引预计算。优先 profile 真正的板端算子，尤其 GridSample、pooling、attention/softmax，不以参数量代替板端测速。
