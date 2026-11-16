# S1 ONNX / LQ560 编译预检（2026-11-13）

后续固定标定部署版本已将30项不支持节点消除，详见 [部署适配说明](ep1_s1_npu_deployment.md)。本页保留原始通用导出图的检查记录；新版仍有ORT数值校验超容差，未作为部署成品。

本次不修改训练网络。使用当前 S1 配置和组合初始化 `ckpt/s1_det_occ_init.pth`，未进行 S1 训练。检查工具为本机 SDK SPC060 中 OMG 2.1.6，target=LQ560ProV200（SDK示例默认值）；其他板卡变体需重测。

## 实际结论

1. 生成单一 opset16 ONNX，固定 batch1、6相机、256×704、450候选、300历史槽位，首帧/后续帧共用图，无 If。
2. 原始6348节点图通过ONNX语法检查、可在ORT执行，但OMG在根部Concat的形状推导失败。
3. onnxsim静态简化校验通过，降至2947节点，随后通过OMG模型加载，平台检查完成。
4. OMG优化后的检查表共1975项，1945项NPU Supported、30项NULL Unsupported。**当前不具备全部节点受支持的完整部署图，未生成ORTM。** `--check`返回0也不能替代检查Unsupported行。
5. ORT对比在首帧缓存特征出现1/76800元素超出rtol/atol=3e-3容差，最大绝对差约0.004165。未放宽容差；本次组合权重下的完整数值一致性验证尚未通过。此前其他初始化下的通过结果不能覆盖本次结果。
6. 未进行PTQ/QAT、离线编译、仿真/板端测速或精度验证。平台支持检查不是完整编译或性能保证。

## 输入输出

输入（FP32，history_valid为bool）：

| 输入 | 形状 |
| --- | --- |
| img | [1,6,3,256,704] |
| projection_mat / occ_img2ego | 各[1,6,4,4] |
| image_wh | [1,6,2] |
| history_feature | [1,300,256] |
| history_anchor | [1,300,11] |
| history_confidence | [1,300] |
| history_transform | [1,4,4] |
| time_interval / history_valid | 各[1] |

输出：det_logits [1,450,10]、det_boxes [1,450,11]、det_quality [1,450,2]、occ_logits [1,128,128,16,18]，以及3个对应历史状态输出。OCC logits单次FP32输出约18MiB，部署时应评估图内/板端argmax，避免不必要的数据搬运；不在本次改变输出接口。

## 不支持节点（实际OMG结果）

| 类型 | 数量 | 所在逻辑 |
| --- | --- | --- |
| And | 13 | 历史/采样/OCC有效性掩码 |
| Not | 12 | 检测采样边界掩码等 |
| GreaterOrEqual | 3 | OCC网格有效性 |
| LessOrEqual | 1 | 历史时间有效性 |
| ScatterElements（reduction=add） | 1 | OCC pooling |

建议在导出/部署适配层保留数学语义，采用后端支持的等价掩码表达或由部署端计算。不能仅为了通过编译删除场景/边界规则，也不能把重复索引累加直接换成覆盖写入。

## 官方文档要求与当前结果

依据《LQ560V200 NPU工具使用指南》文档版本2.1：正文页21、39–43、46、53–61、117–123。PDF实际页码比正文多13页。

| 模块 | 适合的图形态 / 约束 | 当前实测 |
| --- | --- | --- |
| 骨干/卷积 | SDK B0-noSE、ReLU6；Conv/BN融合，减少不必要布局转换 | 本图81个Conv；骨干已对齐SDK |
| OCC Resize | linear + pytorch_half_pixel；PyTorch bilinear、align_corners=False | 3个OCC节点符合；3个检测nearest/asymmetric沿用原行为；6个均NPU支持 |
| OCC通道/网格 | BEV neck及head128通道，BEV128×128，x/y[-25.6,25.6,.4] | 已满足；BEV残差阶段仍为128/256/512，这是SDK配置，不等于所有层128 |
| OCC预测器 | Linear(128,128) → Softplus → Linear(128,18×16) | 已满足，Softplus标为NPU支持；文档还注明输入x<89，量化/训练后需验证数值范围 |
| OCC几何/池化 | SDK固定外参示例可预计算索引，Gather/CumSum/ScatterND分段累加 | 当前动态几何+ScatterElements(add)不支持，应优先适配；不是单纯改算子名称 |
| 检测采样 | 4D NCHW GridSample，bilinear/nearest，padding=zeros；align_corners与训练一致 | 24个GridSample均标为NPU支持，当前bilinear/zeros/align_corners=0 |
| 注意力 | MatMul/Softmax标准表达；减少转置和中间张量，关注实际profile | 22个Softmax、40个优化后MatMul均支持；其中6个Softmax来自采样权重、1个来自OCC深度，不能都视为时序注意力 |
| Linear | 编译器FuseMatMulAddPass默认开启，识别Gemm | 已产生258个NPU支持Gemm，无需先手动全改1×1 Conv |
| LayerNorm | 原生LayerNormalization axis=-1；保留epsilon及scale/bias | 编译器已识别143个NPU支持LayerNormalization，不能只按导出ReduceMean数量认定瓶颈 |
| TopK | 文档要求最后一维、固定K等；按实际平台检查核对 | 2个均NPU支持，暂不需要移出图；后续仍要测耗时 |
| 张量rank | 文档一般最大6维；固定形状并做常量折叠 | 简化图有28个节点涉及7维中间张量。当前优化后除列出的30节点外均标支持；仍应在完整编译中确认，必要时合并轴消除多余维度 |

固定相机标定和OCC池化索引不等于固定目标采样点、历史位姿或时间间隔；后者仍随目标/车辆运动变化。预计算索引必须对应部署相机标定和固定图像预处理。

原版SDK FlashOCC优化案例正文118页报告约246ms→59ms，仅是厂商OCC案例，不能当作当前检测+OCC整模型速度预测。nearest与pytorch_half_pixel属于不同属性；SDK也支持nearest/asymmetric，不存在将所有Resize坐标模式改成pytorch_half_pixel必然更快的结论。

## 优化顺序

1. 保留当前训练网络，补充独立部署适配：静态simplify、30个不支持节点的等价改写，优先处理OCC pooling与逻辑掩码。
2. 处理组合初始化下ORT一致性失败，不通过提高容差掩盖问题；核对不同后端误差及缓存后续帧传播。
3. 通过完整编译，再用训练后的代表性数据做量化。标定应覆盖正常历史、首帧/换场景及实际时间/位姿，不能把所有历史输入永远置零。
4. 板端profile再决定GridSample融合、自定义算子或注意力优化。当前导出为统一首帧/历史帧会多算一路时序注意力；可优化QKV准备/选择，但需保持原始首帧数学语义。
5. 若仍不达标，再讨论检测通道/decoder层数/采样点数量等需重新训练的结构变化。

## 保留文件

本地缓存目录：`/home/tong/.cache/sparsedrive/s1_npu_preflight_20261113/`。

- `s1_init_sim.onnx`：初始化权重下的简化参考图，约164MiB，**不可作为可部署成品**。
- `check.log`：原始图形状推导失败日志。
- `check_sim.log`、`check_results.csv`：完整平台检查结果。
- `unsupported.json`：30个不支持节点。
- `graph.json`：原始图输入/输出及算子统计。
- `input_cfg.yml`：仅用于本次`--check`，不是量化标定配置。

复现检查（先设置SDK omg/lib为LD_LIBRARY_PATH）：

```bash
/path/to/omg --model s1_init_sim.onnx --config input_cfg.yml \
  --target LQ560ProV200 --check
```

SDK正文3页的下一步ORTM离线转换，必须在兼容性适配、标定配置和数据准备之后进行。本次尚未执行该步骤。
