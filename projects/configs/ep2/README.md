# EP2：组合轻量结构的 Oritek QAT

## 配置

| 配置 | 组合 | 初始化来源 | 参数量（FP模型） |
|---|---|---|---:|
| q1.py | S1+S2+S4+S5 | S5全部匹配参数，OCC BEV encoder取S4 | 23.969M |
| q2.py | S1+S2+S3+S4+S5 | q1组合基础上，FFN取S3 | 22.394M |

两者均为4轮检测解码、450 query/300历史、3层FPN(stride8/16/32)、小OCC BEV编码器[64,128,256]，最终OCC head128通道、128×128×16、18类。q1 FFN隐层1024，q2为512。
全部参数按名称和形状严格核对后复用，初始化无随机缺失层。来源为 `work_dirs/ep1/{s3,s4,s5}*/iter_8780.pth`。
`ckpt/q1_fp_init.json`、`q2_fp_init.json`记录各分支来源。
这些分支分别训练过，但组合后的结构尚未联合收敛；后续QAT同时完成适应训练。不要把它当成已经验证精度的联合模型。

## 官方 SDK 和量化范围

使用 `oritek_qat_pytorch-2.1.6-py3-none-any.whl`，已安装到 `sparsedrive` 环境。
与 `/home/tong/LQ560_SDK_V200R001C00SPC060/npu_toolchain/linux/mct/` 指定包SHA256相同：
`2184be4dde05427d709c67e9e406cf3381951f4dd2e3b82569bbcc4a9d13d703`。
参考官方《LQ560V200 NPU ORITEK-QAT-Pytorch工具使用指南》及SDK 2.1.6 API。
采用SDK默认 `quantize_tool.initialize()`：activation INT8非对称per-tensor，weight INT8对称per-channel，fake_quantization（非LSQ）。chip_type=LQ560ProV200。

量化范围：img_backbone、img_neck、occ_img_neck、occ_branch 中的 Conv2d 叶模块。对每个Conv2d建立SDK FX量化岛，用真实数据捕获输入并调用 `create_quant_model`/`optimize_and_initialize_model`；不对整个含控制流/自定义CUDA的网络做FX追踪。
**注意力、Linear（含OCC predictor）、Softplus、动态采样、BEVPoolv2、几何、索引、归一化、损失保持浮点。不是整网INT8，不保证与后续OMG全INT8 PTQ的量化范围一致。**
训练中原生FlashAttention/动态采样CUDA算子保留。SDK伪量化与主网络采用FP32；仅FlashMHA内部使用局部FP16 autocast（参数名不变、输出还原FP32）。全局fp16=None，避免全局AMP下已观察到的非有限梯度。
量化标尺scale/zero_point注册为buffer，随GPU移动、checkpoint保存并跨rank同步。转化在DDP和optimizer创建前完成，原FP权重在转化前严格加载；不会在量化模块重命名后再次加载FP权重。

## 训练参数

每卡batch12，4卡global48，每轮586步，6轮3516步。关闭backbone checkpointing，避免反向重算重复更新量化observer。
LR1e-5、backbone5e-6，warmup167，cosine最小LR比例0.1。关闭全局FP16 loss scaling，梯度裁剪25。
BN从开始冻结；observer在第586步后冻结，后续仅优化带固定伪量化的模型。冻结依据iter而非IterBasedRunner的epoch计数。
每586步保存，保留最近3份。训练不自动验证。这个batch值需要以实际显存检查为准，不承诺持续满卡。

## 启动

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sparsedrive
cd /root/SparseDrive
bash scripts/train_ep1.sh projects/configs/ep2/q1.py
# q1完成后，独立训练q2（q2不是接着q1的优化器继续训练）
bash scripts/train_ep1.sh projects/configs/ep2/q2.py
```

不需要重新生成初始化，已经准备好。若需要自行重建（输出文件必须不存在）：
```bash
PYTHONPATH=. python tools/prepare/prepare_ep2_weights.py projects/configs/ep2/q1.py
PYTHONPATH=. python tools/prepare/prepare_ep2_weights.py projects/configs/ep2/q2.py
```
真实样本QAT检查（并非完整训练）：
```bash
PYTHONPATH=. python tools/check_ep2_qat.py --config projects/configs/ep2/q1.py
```

显式断点恢复：
```bash
bash scripts/train_ep1.sh projects/configs/ep2/q1.py --resume-from work_dirs/ep2/q1/latest.pth
```
恢复时按同一recipe重建量化模块并严格恢复量化状态和优化器。不要用FP模型直接加载QAT state_dict，也不要把QAT checkpoint误写为FP load_from。
`tools/val_ep.py` 支持带 oritek_qat 元数据的 QAT PTH：从 checkpoint recipe 重建卷积量化岛，严格恢复所有参数、observer范围、scale/zero_point，验证期间冻结observer且保持fake quant。无需FP初始化文件。必须安装与checkpoint相同版本SDK。原EP1 ONNX exporter仍不能直接用于QAT checkpoint；需要量化模块/QDQ与标尺对齐。

## 已完成的检查

q1/q2均通过真实训练样本前向、反向、一次优化器更新，梯度有限；320个量化scale/zero_point缓冲严格保存/加载检查通过。q1通过四卡DDP训练入口、checkpoint保存、重新构建量化模型后严格断点恢复，以及实际每卡batch4的两步短测，日志显存峰值约17462MiB。记录：`work_dirs/ep2/qat_validation.json`。这些是启动/状态正确性检查，不代表完整训练收敛、INT8精度或板端部署已经验证。

## 吞吐调整（2026-11-18）

实测每卡batch4约28.9样本/秒，batch8约43.6，batch10约47.2，batch12约48.8（四卡合计，短测稳定区间；不含启动）。选batch12，显存日志峰值28144MiB。正式单步吞吐会随样本与observer阶段变化，GPU利用率仍可能波动。
Q1/Q2均保持LR1e-5、不按batch放大。每轮586步、6轮3516步；observer在586步冻结，warmup167步对应原来近似相同样本数。observer开关仅在状态变化时更新，避免每个卷积重复递归设置。
原Q1在第一个checkpoint前停止，重启从组合初始化训练，原日志归档于work_dirs/ep2/q1_before_batch12_*。本轮使用后台shell直接执行q1 && q2，q1成功后才会启动q2，无额外排队脚本。

## QAT PTH 验证（3090）
```bash
PYTHONPATH=. python tools/val_ep.py ckpt/q1.pth --config projects/configs/ep2/q1.py --output val/q1_pth
```
已通过实际q1 checkpoint的两帧CUDA检查，未运行全量验证。普通PTH、ONNX入口保留。SDK模块结构用临时合成张量建立，随后全部量化状态被训练checkpoint严格覆盖；不会利用验证集更新训练标尺。

## Q3：以 S6 FastBEV 为基础的 QAT

`q3.py`继承EP1 `s6_det_occ_decoder4_fpn3_fastbev.py`，从最终S6权重初始化，不再使用S5的LSS结构。保留4轮decoder、FFN1024、3层FPN、大OCC编码器[128,256,512]。FastBEV当前在z=-1单平面反投影，输出预测仍为128×128×16、18类。

`load_from=ckpt/s6.pth`。本次服务器上该路径指向 `work_dirs/ep1/s6_det_occ_decoder4_fpn3_fastbev/iter_8780.pth`，使用训练最终权重，不是iter_3512中间权重。不恢复S6优化器，QAT从第0步开始独立训练。

采用SDK 2.1.6 Conv2d混合精度QAT：量化范围是backbone、检测FPN、OCC图像neck、OCC分支中的卷积，包括FastBEV的context_net；几何求逆/投影、GridSample、掩码、注意力、Linear、Softplus及损失保持浮点。它不等于整个网络INT8，也不能直接保证上一轮OMG全网PTQ的量化误差被解决。后续QAT导出与编译必须对齐相同量化范围和训练标尺，不能再直接套用普通FP checkpoint导出器。

四卡每卡batch8、global32，每轮879步，6轮5274步；LR1e-5、backbone5e-6，warmup250步，第一轮后冻结observer，BN从开始冻结。全局fp16=None，只让FlashAttention局部FP16，关闭backbone checkpointing。每879步保存，保留最近3份。

QAT加载时只在严格核对生成网格完全一致后忽略FP checkpoint中的voxel_points缓存。转换后把网格显式持久化，确保PyTorch/MMCV保存和恢复使用相同state_dict；QAT checkpoint同时保存量化标尺及recipe。

```bash
cd /root/SparseDrive
conda activate sparsedrive
bash scripts/train_ep1.sh projects/configs/ep2/q3.py
```

后台日志 `work_dirs/ep2/q3/launch.log`，输出权重在同一目录。断点续训使用 `--resume-from work_dirs/ep2/q3/latest.pth`，不要将QAT checkpoint作为普通FP load_from。

本次真实样本检查已通过前向、反向、一次优化器更新和量化状态严格保存/加载：565个梯度张量均有限，320个scale/zero_point缓冲一致。四卡正式训练已跑到50/5274步，loss_occ约0.2125、总loss约7.5111，日志显存峰值约27607MiB。仅确认启动和初期训练正常，未代表最终精度或收敛。

## Q4：Q3 FastBEV QAT + S3 FFN512

`q4.py`继承Q3，只将4个检测FFN的hidden从1024改为512。其余结构、FastBEV单平面采样、大OCC编码器、3层FPN、query/历史数量和QAT范围不变。FFN的Linear仍为浮点，这里是S3的结构缩减，不是新增Linear INT8量化。

初始化由 `tools/prepare/prepare_ep2_weights.py projects/configs/ep2/q4.py` 生成 `ckpt/q4_qat_init.pth`：保留最终Q3 `work_dirs/ep2/q3/iter_5274.pth` 的卷积QAT权重、320个scale/zero_point缓冲和其他匹配状态；四个FFN完整使用已训练S3 `work_dirs/ep1/s3_det_occ_decoder4_ffn512/iter_8780.pth` 对应参数，不随意截取Q3权重。结构变化需要联合适应训练，不能当作等价推理替换。

`qat_init_from`表示量化模型权重暖启动，区别于普通FP load_from和断点resume_from：先按recipe重建量化模块，再严格加载全部初始化状态；不恢复旧优化器、epoch或iter。训练仍为四卡每卡batch8、6轮5274步，LR1e-5，warmup250；BN立即冻结，observer重新观察一轮后冻结。checkpoint每879步保存，保留最近3份。

```bash
cd /root/SparseDrive
conda activate sparsedrive
PYTHONPATH=. python tools/prepare/prepare_ep2_weights.py projects/configs/ep2/q4.py
bash scripts/train_ep1.sh projects/configs/ep2/q4.py
```

初始化文件已存在时不要重复生成。日志和权重位于 `work_dirs/ep2/q4/`。Q4需独立比较mAP/NDS/mIoU，不能因参数量下降就推断精度或板端速度。

本次Q4通过真实样本前向/反向/一次更新和量化状态严格保存加载检查，565个梯度张量有限，320个标尺缓冲一致。四卡正式训练已到50/5274步，loss_occ约0.1946、总loss约8.8246、grad_norm约14.1393，显存日志峰值27532MiB；仅表示初期训练正常。
