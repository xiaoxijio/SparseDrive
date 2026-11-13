# EP1 S1：SparseDrive 检测 + SDK 优化 FlashOCC

配置：`projects/configs/ep1/s1_det_occ_b0_nose.py`。只训练 stage1；检测和 OCC 共享骨干，无 map、tracking ID、motion/planning。

## 网络与 SDK 对应关系

参考 LQ560 SDK `SPC060/samples/app/vision_occ/evaluate/FlashOCC_Board/` 的优化配置 `configs/flashocc/flashocc-r50-M0-opt.py`，以及《LQ560V200 NPU工具使用指南》的优化要求。

| S1 模块 | SDK 对应模块 / 配置 |
| --- | --- |
| `EfficientNetB0NoSE` | `EfficientNetB0HZ`，无 SE、ReLU6，无 stochastic depth；增加 stride 4/8 两个中间输出供检测使用 |
| 检测 FPN | 原 SparseDrive FPN，输入 24/40/112/1280，输出四级 256；保持 nearest |
| `OccImageNeck` | SDK `CustomFPN`，输入 112/1280，输出 stride16、256 通道；bilinear、align_corners=False |
| `LSSViewTransform` | SDK LSS 深度/context 预测，44 个深度格、64 通道 context；训练使用原 BEVPoolv2 CUDA 聚合 |
| `BEVEncoder.backbone` | SDK `CustomResNet`，三个阶段 128/256/512，各两个 BasicBlock；下采样分支为带 bias 的 3×3 卷积 |
| `BEVEncoder.neck` | SDK `FPN_LSS`，输入 640，输出128，channels_factor=1；保留原 Conv/BN/ReLU 结构 |
| `BEVOccHead2D` | 3×3 final_conv，`Linear(128,128) -> Softplus -> Linear(128,18*16)`，按指南和 SDK 优化 ONNX 的预测结构 |

`sdk_blocks.py` 中 CustomFPN、CustomResNet、FPN_LSS 从上述 SDK 源码移植，只适配 import 和移除 registry 装饰器，类主体保持一致。预测器采用用户明确指定的两层 Linear，而不是某些 SDK 源文件中的实验性单 Linear 版本。

骨干四级输出为 stride 4/8/16/32，通道24/40/112/1280。检测仍使用原 MultiheadFlashAttention 和融合 CUDA 特征采样；450个候选，首轮单帧解码后使用历史300 + 当前top150，6层decoder、256维、13个采样点，最终最多输出300个框。

OCC 网格为 x/y `[-25.6,25.6,0.4]`，128×128；标签 z `[-1,5.4,0.4]`，16格、18类。LSS z `[-1,5.4,6.4]` 折叠成一层 BEV，depth `[1,45,1]`。输出 `[B,128,128,16,18]`。保留 camera mask 下交叉熵，loss_weight=1，无类别平衡。

## 组合预训练初始化

默认配置 `load_from='ckpt/s1_det_occ_init.pth'`，不再只加载骨干。该文件包含本架构的完整初始状态，可由 runner 正常加载；`resume_from` 仍优先恢复训练。骨干的独立 `pretrained` 设为 None，避免重复加载或迁移时依赖旧的14MB骨干文件。

| 目标 | 来源 | 迁移张量 |
| --- | --- | --- |
| 共享 SDK B0-noSE 骨干 | G5 | 294，全部严格匹配 |
| OCC 图像 neck、BEV CustomResNet/FPN_LSS、final_conv | G5 | 106，按语义映射并逐项检查形状 |
| OCC 深度/context 卷积 | G5 | 2，取对应44个深度bin及完整64个context通道 |
| 检测 head（不含anchor和初始instance feature） | SparseDrive stage1 | 477 |
| 检测 FPN 输出卷积 | SparseDrive stage1 | 8 |
| 辅助深度分支 | SparseDrive stage1 | 6 |

合计893个张量，41,360,968 / 41,908,542个参数（98.69%）来自预训练；此计数包含非训练参数，不是精度提升比例。剩余14个张量保持本架构初始化：检测FPN输入lateral的4组weight/bias、450个anchor和初始instance feature、OCC预测器的两组Linear weight/bias。即使bias形状匹配，也不脱离不兼容的所属层单独迁移。

G5深度配置为 `[1,60,1]`（59个depth+64个context），当前为 `[1,45,1]`（44+64）；转换选择源行 `[0:44]` 和 `[59:123]`，不是简单截前108行。softmax会在新的44个bin上重新归一化，属于迁移初始化，不是源深度分布的完全等价推理。OCC预测MLP原为128→256→288，现为128→128→288，不任意截取隐藏通道。

生成脚本：

```bash
PYTHONPATH=. python tools/prepare_ep1_weights.py \
  --g5 /path/to/BEVDet/ckpt/g5.pth \
  --stage1 ckpt/sparsedrive_stage1.pth
```

默认输出 `ckpt/s1_det_occ_init.pth`（约161MB）及 `ckpt/s1_det_occ_init.report.json`。报告逐层记录来源、源/目标形状、转换方式、保留初始化的理由、源文件SHA256及参数覆盖率。源缺失、意外尺寸不一致或没有明确迁移策略的目标层会报错；检测类别顺序也会检查。生成使用seed=0，可通过 `--seed` 修改。

`--source-depth-grid` 默认1 60 1，对应本次已核对的G5配置；换用其他来源必须明确检查其深度定义。G5/D1的骨干都兼容，但本次组合工具按G5的单帧结构映射，不支持将D1完整分支混入。

导入的骨干和BEV编码器保留G5折叠BN的running statistics（各自norm_eval=True），卷积和BN affine仍训练。这个策略不会冻结检测LayerNorm，也不会冻结OCC预测MLP。G5的BEV物理范围/分辨率与当前不同，迁移后仍需微调，不能视为源模型的精度复现。

原 `tools/prepare_ep1_backbone.py` 保留作单独骨干检查工具，不是默认训练初始化入口。正式推理/导出应加载训练后的完整S1 checkpoint；不使用原SparseDrive stage1作为完整模型恢复文件。

## 数据与坐标

检测使用 lidar 坐标，OCC 使用关键帧 ego 坐标。保留 `occ_ego2lidar` 和增强后的 `lidar2img`，得到 `occ_img2ego`；该适配接口没有照搬 SDK 数据加载器，但几何含义相同。

使用本仓库 SparseDrive 格式的 `data/infos/nuscenes_infos_{train,val}.pkl`，以及 `data/nuscenes/gts/scene-xxxx/<sample-token>/labels.npz`。200×200×16 的 Occ3D 标签在 x/y 上中心裁剪 `[36:164]`，不插值类别标签。预裁剪数据需同步调整 `occ_source_grid`。

保持原配置 `rot3d_range=[0,0]`；检测的55m过滤与 OCC 的±25.6m范围相互独立。其他图像增强和投影关系保留。

## 四卡训练与迁移

默认4 GPU，每卡batch4，总batch16，AdamW lr=1e-4，100个等效epoch（每epoch1758 iteration）。保留当前每卡2 worker、每10epoch评估/保存、保留3份checkpoint的工程设置。新增 OCC 的实际四卡容量和吞吐需在租赁机验证。

环境为 Python3.8.20、torch1.13.0+cu116、torchvision0.14.0+cu116、mmcv-full1.7.1、mmdet2.28.2、flash-attn2.3.2。

```bash
conda activate sparsedrive
cd /path/to/SparseDrive
python -m pip install -r requirement.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
(cd projects/mmdet3d_plugin/ops && python setup.py build_ext --inplace)
python tools/prepare_ep1_anchors.py
bash scripts/train_ep1_s1.sh
```

同一条 build_ext 命令现在同时编译检测聚合和 BEVPoolv2。需要与 torch 匹配的 CUDA toolkit、兼容的 C++ 编译器；例如 CUDA11.6 环境可使用 gcc/g++11。当前3090机已在 sparsedrive 环境中编译并验证。

迁移到租赁机时，`git clone` 不会带上被忽略的权重、数据和二进制：必须额外复制 `ckpt/s1_det_occ_init.pth`、anchor文件和训练数据（迁移报告建议一并保存），在目标机器重新编译两个算子。正常训练无需原G5/SparseDrive源checkpoint、BEVDet项目目录或SDK安装目录；只有重新生成组合初始化时才需要两个源checkpoint。无需依赖SDK安装目录；SDK 所需网络和算子源码已经包含在本项目内。

恢复训练使用本架构的 checkpoint：

```bash
bash scripts/train_ep1_s1.sh --resume-from /path/to/ep1_checkpoint.pth
```

## 首帧、历史帧与导出

首帧仍补齐300个历史槽位，但无有效历史时，temp_gnn 恢复原 SparseDrive 的当前帧自注意力；有历史时使用历史交叉注意力。保留场景切换、dt>0、最大间隔2秒的有效性检查。普通 GPU 模式在整个batch均有效/均无效时只计算对应一路；混合batch计算两路后按样本选择。

同一个 ONNX 使用显式历史输入/输出。导出时通过张量选择两种注意力结果，不导出两个首帧/历史模型；目前可能多算一路，后续再优化。训练与导出都遵循相同首帧语义。

现有 `forward_export` 的标准算子路线暂时保留：FlashAttention 转共享权重的 MatMul/Softmax，检测 CUDA 聚合转 GridSample，OCC pooling 保留 ScatterElements/add 参考实现。**网络结构已对齐 SDK，但导出的 pooling 图尚未替换为厂商 ONNX 的固定索引 Gather/CumSum/ScatterND 路线，也未实现 NPU 自定义算子。**

检测 FPN 的 nearest Resize 保留原坐标规则；只有 OCC 的 linear Resize 写为 `pytorch_half_pixel`。PyTorch 对应 bilinear、align_corners=False；当前输出空间维均大于1，half_pixel 与 pytorch_half_pixel 数值等价。该坐标模式本身不是 nearest/bilinear 的性能比较结论。

```bash
PYTHONPATH=. python tools/export_ep1_onnx.py \
  projects/configs/ep1/s1_det_occ_b0_nose.py --checkpoint /path/to/ep1_checkpoint.pth
```

输入/输出定义见 `tools/ep1_common.py`。历史特征 `[1,300,256]`、框 `[1,300,11]`、置信度 `[1,300]`，首帧补0并设置history_valid=False；后续回传输出状态和正确时间/位姿变换。框解码/筛选在图外。

## 验证

```bash
PYTHONPATH=. OMP_NUM_THREADS=2 python -m unittest discover -s tests -p 'test_ep1*.py' -v
```

16项测试通过，包括严格骨干迁移、BN统计保持、四级输出、BEVPoolv2与参考实现的输出及输入/权重梯度、空池化、首帧/历史帧/混合batch与原注意力语义对比，以及原有采样/几何测试。G5 初始化的骨干与 SDK 原始骨干执行结果逐位一致；移植的三个 SDK 网络类经 AST 比较与原类主体一致。

当前总参数41,908,542。真实数据连续两帧前向/反向通过，配置中的FP16 loss_scale=32反向也通过，每步调用6次检测CUDA聚合、10次FlashAttention和1次BEVPoolv2，历史有效标志由False转True。没有进行完整训练或四卡容量验证，也没有 LQ560 编译、量化和板端测速结果。

统一 ONNX 的首帧、历史帧及场景重置均通过 ORT 对比（rtol/atol均3e-3，缓存按框/特征/置信度共同对应后比较）。接近置信度的 TopK 排序可因浮点误差交换；校验只对齐缓存行，不修改模型输出，也不独立重排特征。导出图没有 If，3个检测nearest Resize保持原规则，3个OCC linear Resize为pytorch_half_pixel。该测试使用未训练检测/OCC头，仅验证导出计算。

组合初始化另外验证了893个迁移张量与对应源权重/转换结果逐项一致，450个anchor保持当前文件初始化；真实连续两帧FP16反向传播通过，192个导入BN状态张量保持不变。权重迁移不改变ONNX计算结构，尚无本次初始化后的完整训练精度结果。
