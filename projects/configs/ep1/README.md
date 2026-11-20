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

## S6：基于 S5 的 FastBEV 式 OCC 体素构建

位置：3090 `/home/fjm/tong/SparseDrive`。只新增训练配置和体素转换模块，未启动正式训练。

### 为什么改、改了什么

官方参考：[Sense-GVT/Fast-BEV 的 fastbev.py](https://github.com/Sense-GVT/Fast-BEV/blob/dev/mmdet3d/models/detectors/fastbev.py)，重点是 `get_points`、`backproject_inplace`。官方方法从预定义的三维体素点投影到图像，按最近邻获取图像特征，多相机重叠处按相机顺序覆盖。它与 LSS 的深度概率展开、落点 scatter 求和不是等价运算。

S6 继承 `s5_det_occ_decoder4_fpn3.py`，保持共享 backbone、3层检测FPN、4轮decoder、时序缓存、OCC图像neck、BEV编码器和18类预测头。仅将 OCC 的 LSSViewTransform 替换为 FastBEVViewTransform：

1. X/Y投影网格为 `[-25.6,25.6,0.4]`。当前已训练S6继承的投影Z配置是 `[-1,5.4,6.4]`，因此只有 `z=-1` 一个128×128采样平面。预测标签另有16个高度层，两者不是同一个网格。与官方 get_points 一致用网格下角点，未增加半个体素偏移。
2. 将图像特征由256通道投影为64通道，取消 OCC 深度概率预测/展开。
3. 从每帧 `occ_img2ego` 计算逆映射，将体素点投影到六路图像。3×3伴随矩阵公式实现仿射逆，不引入 ONNX Inverse 或自定义算子。假定输入是本项目生成的可逆仿射标定矩阵，不能传零矩阵。
4. 按图像/特征尺寸缩放坐标、round 最近邻、检查正深度和图像范围。相机列表后面的有效相机优先，保持官方 inplace 的覆盖语义。用 GridSample 和数值掩码实现，无 scatter 或布尔索引写入。
5. **本项目的适配**：按配置的采样高度层取均值，得到64×128×128，再接原二维BEV encoder；当前仅一个采样平面，因此均值没有额外作用。官方FastBEV保留三维volume并由其后续neck处理，本实现不是完整复刻。若以后改成16个采样高度，属于结构变化，需要重新训练，不能直接拿当前checkpoint冒充该结构。

这改变了 OCC 学习问题，需要重新训练。当前单平面采样不等同于原LSS的深度聚合，也不等同于16层FastBEV体素采样。检测模块初值不变，但共享backbone联合训练后检测精度也可能变化。

### 坐标和动态输入

`projection_mat` 是检测分支的 lidar→image 映射。OCC 网格使用 ego 坐标，不能直接用它替代 OCC 投影；数据pipeline的 `occ_img2ego` 包含 image→ego 映射，PTH训练时S6从这个矩阵恢复ego→image。无需改训练数据pipeline、标签或Collect。ONNX部署改为由输入预处理计算正向 `occ_ego2img` 并逐帧传入，避免图内逆矩阵子图，检测的projection_mat仍独立逐帧输入。

固定的是体素网格和图像尺寸，**不是相机标定**。本实现没有固定sample-0标定，也没有提前缓存相机采样索引。

### 文件作用

- `projects/mmdet3d_plugin/models/ep1/fastbev.py`：仿射逆、体素点、最近邻反投影、相机覆盖、高度均值；注册 FastBEVOccBranch，复用原loss和编码器。
- `projects/mmdet3d_plugin/models/ep1/__init__.py`：导入并注册新增分支。
- `projects/configs/ep1/s6_det_occ_decoder4_fpn3_fastbev.py`：继承S5，仅切换OCC分支类型、初始化权重和输出路径。
- `ckpt/s6_from_s5_init.pth`：从 `ckpt/s5.pth` 转移的训练初始化，**不是训练完成的S6**。保留761个匹配张量；新context_net的weight/bias来自S5 depth_net最后64个context输出通道，丢弃其前面的深度输出通道；不携带优化器状态。完整state_dict已严格加载验证。
- `tests/test_ep1_fastbev.py`：与独立的官方风格索引覆盖参考比较、梯度检查、小网格ONNX检查及两组动态标定的ORT对比。无需checkpoint，不会生成初始化权重。
- `projects/mmdet3d_plugin/models/ep1/deployment.py`：固定LSS导出适配器明确拒绝S6。S6动态适配器使用正向occ_ego2img输入和同一个FastBEV采样方法，避免图内求逆。

### 训练

```bash
cd /home/fjm/tong/SparseDrive
conda activate sparsedrive
# 与S5相同的四卡、每卡batch16：20轮，8780 iter。
bash tools/dist_train.sh projects/configs/ep1/s6_det_occ_decoder4_fpn3_fastbev.py 4 --no-validate
```

初始化文件已生成，无需先跑权重准备脚本。将工程迁到训练机时同时复制 `ckpt/s6_from_s5_init.pth`。配置保留S5的LR=2e-5、warmup500、动态loss scale、梯度裁剪及数据增强。

**单卡3090不要直接把四卡配置当成相同20轮。**若显存允许batch16，单卡20轮用如下显式覆盖（若OOM需降低batch，并按 `28130 // batch` 重算每轮iter和总iter）：

```bash
PYTHONPATH=. python tools/train.py projects/configs/ep1/s6_det_occ_decoder4_fpn3_fastbev.py \
  --no-validate --cfg-options data.samples_per_gpu=16 data.workers_per_gpu=4 \
  runner.max_iters=35160 checkpoint_config.interval=1758
```

四卡和单卡的有效batch不同，上述并非保证等价的收敛设置。尚未测量S6全模型batch16的显存峰值。

训练完成后评估：

```bash
PYTHONPATH=. python tools/val_ep.py work_dirs/ep1/s6_det_occ_decoder4_fpn3_fastbev/iter_8780.pth \
  --config projects/configs/ep1/s6_det_occ_decoder4_fpn3_fastbev.py --output val/s6_pth
```

单卡训练则替换为实际最终checkpoint，例如iter_35160.pth。比较S5/S6的检测mAP、NDS与OCC mIoU，不能仅看训练loss。

### 当前验证边界和编译

已通过：独立参考数值一致；反向梯度有限；小网格ONNX checker及两组标定ORT一致；完整OCC分支在真实sample-0标定下输出 `[1,128,128,16,18]`，loss和输入特征梯度有限；S5迁移后的全模型state_dict严格加载。

小规模导出未包含 ScatterElements、ScatterND 或矩阵Inverse。比较有效性采用数值掩码相乘，避免And和GreaterOrEqual。当前按一个高度/六相机展开为6次GridSample。完整导出、标准算子部署改写、FP16几何保护及实际编译流程见下节；量化后的全量精度和实体板性能需要独立验证。

复现小规模检查：

```bash
PYTHONPATH=. python tests/test_ep1_fastbev.py
```

### S6 动态标定部署（iter_3512.pth）

```bash
cd /home/fjm/tong/SparseDrive
PYTHONPATH=. python tools/export_ep1_npu.py \
  --config projects/configs/ep1/s6_det_occ_decoder4_fpn3_fastbev.py \
  --checkpoint ckpt/iter_3512.pth --output onnx/s6/s6.onnx
PYTHONPATH=. python /home/fjm/tong/Compilation/app/sparse/prepare_calibration.py \
  --onnx onnx/s6/s6.onnx --frames 4
cd /home/fjm/tong/Compilation/app/sparse
bash 00_init.sh /home/fjm/tong/SparseDrive/onnx/s6/s6.onnx
bash 01_check.sh
bash 02_preprocess.sh
bash 03_omg.sh
```

默认导出参数即为上述S6配置、checkpoint和路径，以后换权重用 `--checkpoint`、换输出用 `--output`。保留参数 `--no-simplify` 用于排查，默认onnxsim简化；`--check-only`只复核已有图。输出ONNX、匹配的calibration.npz和validation.txt；不再生成临时JSON报告。

**接口变化**：ONNX输入是 `img, projection_mat, occ_ego2img, history_feature, history_anchor, history_confidence, history_transform, time_interval, history_valid`。两个投影均为每帧真实输入，image_wh固定。OCC改为ego→image的正向矩阵；原数据的image→ego矩阵由 `tools.ep1_common.occ_forward_projection` 在输入准备阶段转换，运算与训练FP32公式一致。`tools/onnx_stage1.py`验证/可视化及 `tools/export_ep_geometry.py`已支持新接口。不要将旧occ_img2ego原值或检测的projection_mat直接填到occ_ego2img输入。无需重新训练就能使用这个部署接口。

导出比较覆盖首帧、历史、重置和不同标定；第二样本1个体素类别变化，前两类logit分差约4.77e-7、最大logit误差约9.92e-5，报告会记录而不隐藏。要求logits继续满足3e-3原容差，并限制类别翻转处的logit分差。部署图改写的四帧OCC logits误差为0，检测/缓存最大误差约1.42e-4。

编译仅使用标准算子：坐标Clip改为Min/Max，避免SDK误判QAT；浮点比较掩码改为有限输入下等价的ceil/Min/Max表达，绕过FP16比较+Cast后端限制；相机MatMul二维化；边界mask移到分组采样权重以减少重复乘法。学习特征INT8、几何坐标及必要常量FP16；配置和文件作用详见已有 [Compilation/app/sparse/README.md](/home/fjm/tong/Compilation/app/sparse/README.md)。S2的自定义CPU池化不再使用。

四帧校准只验证编译流程，不能替代代表性量化校准集、全量mAP/NDS/mIoU评估或实体板测速。

本次 `iter_3512.pth` 已实际生成 `/home/fjm/tong/Compilation/app/sparse/models/s6.ortm`，1437个算子全部NPU，无自定义CPU插件。SDK单核模拟约343.19ms/2.91383 FPS，非实体板实测。首帧SDK fake-quant输出均有限，但体素类别一致率约91.23%、检测仍有明显量化差异，INT8精度未验收；不能将该比例作为mIoU。
