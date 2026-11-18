# EP1：S1～S5 改动与训练关系

参照：`../sparsedrive_small_stage1_4gpu_32g.py` 及其原版 Stage1 基础配置。
S3、S4、S5 都是 S2 的独立分支，**并非逐个累积**。文件名中的 s1/s2 也不是原 SparseDrive 的 Stage1/Stage2；全部只训练感知 Stage1。

## 结构对照

| 项目 | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|
| 基础 | 轻量检测 + OCC | S1 减解码轮数 | S2 缩 FFN | S2 缩 OCC BEV | S2 减 FPN 特征层 |
| Backbone | EfficientNetB0-noSE | 同 S1 | 同 S1 | 同 S1 | 同 S1 |
| 检测特征通道 | 256 | 256 | 256 | 256 | 256 |
| 解码轮数 | 6 | 4 | 4 | 4 | 4 |
| FFN 隐层 | 1024 | 1024 | 512 | 1024 | 1024 |
| FPN strides | 4/8/16/32 | 同 S1 | 同 S1 | 同 S1 | 8/16/32 |
| OCC BEV 主干通道 | 128/256/512 | 同 S1 | 同 S1 | 64/128/256 | 同 S1 |
| 总 query / 历史 | 450 / 300 | 同 S1 | 同 S1 | 同 S1 | 同 S1 |
| 参数量约 | 41.909M | 34.298M | 32.723M | 24.673M | 33.595M |

## S1 相对原版 Stage1

- 关闭 map、motion/planning、跟踪 ID 输出；保留目标检测，增加共享 backbone 的 FlashOCC 分支。
- ResNet50 替换为 SDK 风格 EfficientNetB0-noSE，移除 SE，使用 ReLU6；检测 FPN 输入通道改为 `[24,40,112,1280]`，输出仍 256。
- 检测 anchor 从 900（300 新实例+600 历史）减为 450（150+300）。使用 450 聚类 anchor，而非直接截断原文件。
- `PaddedInstanceBank/PaddedSparse4DHead` 统一有无历史的接口；补齐历史缓存并检查场景、时间间隔有效性；无有效历史采用首帧注意力路径。补零缓存不等于让无效历史参与正常历史注意力。
- 训练保留 `MultiheadFlashAttention` 与 `use_deformable_func=True` 的原生 CUDA 采样；动态采样点没有删除。`match_cuda_sampling=True` 对齐标准算子导出回退的边界行为，不是取消动态采样。
- 采样点生成和 box refinement 用可导出的等价表达；框速度标签清洗处理 NaN/Inf，公共后处理兼容修复。它们并非新的检测分支；标签清洗与首帧规则会影响异常样本/首帧行为，不能笼统宣称完全无影响。
- `OccImageNeck` 对 stride16/32 特征融合，输出 256；LSS view channels=64；BEV encoder 为 SDK CustomResNet + FPN_LSS，最终特征通道 128。
- OCC 网格 X/Y 为 `[-25.6,25.6,0.4]`，128×128；高度 16、18 类。原始 Occ3D 标签裁剪到对应区域。预测器为 `Linear(128,128) → Softplus → Linear(128,18*16)`。
- OCC 的双线性 Resize 使用 align_corners=False，导出时转换为 `pytorch_half_pixel`；检测 FPN 仍为 nearest，二者不是等价插值。
- CUDA 训练 OCC 使用 BEVPoolv2；标准 ONNX/固定标定池化属于独立部署路径。
- 初始化融合旧 BEVDet 权重和原 SparseDrive Stage1 检测权重，使用 `ckpt/s1_det_occ_init.pth`；不是完全随机 backbone。
- 当前四卡每卡 batch16、workers8，禁用 OpenCV 内部线程并限制 BLAS/OMP 线程；backbone checkpointing 开启，cudnn benchmark 开启。
- 当前基础 LR=1e-4、backbone LR=5e-5，warmup1000，动态 FP16 loss scale，梯度裁剪25，100轮；不再按 batch 自动线性放大学习率。
- 每439步约1轮保存，保留最近3份；训练入口传 `--no-validate`，训练期间不验证。

## S2

6轮解码改为4轮，保留单帧初始轮和后续时序注意力逻辑。FPN、query、通道、OCC 分支均继承 S1。
从 S1 最终 `iter_43900.pth` 复用保留层参数，初始化为 `ckpt/s2_from_s1_final_init.pth`，不恢复 S1 优化器。
当前 LR=2e-5（backbone1e-5），warmup500，20轮/8780步。

## S3

在 S2 上仅把 FFN `feedforward_channels` 从1024减为512。原 S2 中形状匹配参数复用；12个变化张量重新初始化后已完成训练。
`load_from=ckpt/s3_from_s2_final_init.pth`；LR2e-5、warmup1000、20轮。当前完成权重位于 `work_dirs/ep1/s3_det_occ_decoder4_ffn512/iter_8780.pth`。

## S4

在 S2 上仅把 OCC BEV 主干 `[128,256,512]` 改为 `[64,128,256]`。view_channels仍64，最终head输入仍128，网格/高度/类别不变。BEV neck 的输入通道随主干自动变化。
从 S2 复用匹配参数；67个变化的参数/缓冲张量重新初始化后已完成训练。
`load_from=ckpt/s4_from_s2_final_init.pth`；LR2e-5、warmup1000、20轮。当前完成权重位于 `work_dirs/ep1/s4_det_occ_decoder4_occsmall/iter_8780.pth`。

## S5

在 S2 上去掉 stride4 检测 FPN 输出：start_level=1、num_outs=3，动态采样层数 num_levels=3。同步将深度监督尺度改为8/16/32，避免只改顶层 strides 而训练 pipeline 仍沿用旧值。
FPN 初始化按特征尺度映射（S5第0层对应S2第1层），不是简单按列表下标加载。8个采样权重预测张量因输出维度变化重新初始化，其他匹配参数复用，后续已完成训练。
`load_from=ckpt/s5_from_s2_final_init.pth`；LR2e-5、warmup500、20轮。完成权重：`work_dirs/ep1/s5_det_occ_decoder4_fpn3/iter_8780.pth`。
去掉高分辨率特征会影响小/远目标，需用统一验证比较精度；静态MAC下降不代表相同比例的板端耗时下降。

## 部署改写和验证边界

固定标定导出、单图历史缓存接口、数值掩码、相机投影二维化、高维广播压缩、OCC固定索引池化属于部署适配，不是 S2～S5 的结构缩减。
此前流式采样布局实验未通过原数值容差，**没有采用到训练模型**。
固定 nuScenes 第0帧标定的 ONNX 不能代表按每帧真实几何验证的 PTH 精度；应以原生PTH验证模型本体，再逐层检查导出误差。mAP、NDS、地图mAP是不同指标。

## EP2 组合与 QAT

- `../ep2/q1.py`：S1+S2+S4+S5，小 OCC BEV + 3层FPN + 4轮decoder，FFN仍1024。
- `../ep2/q2.py`：q1 再加 S3，小 FFN=512。
- 已训练分支按模块组合，q1来自S5+S4，q2来自S5+S4+S3；不是一份已经联合训练的模型。后续QAT同时承担组合后的适应训练。
- 使用官方 Oritek 2.1.6 卷积伪量化，具体量化范围、训练/验证和导出约束见 `../ep2/README.md`。
