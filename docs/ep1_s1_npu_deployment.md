# S1 固定标定部署适配（2026-11-13）

本次只新增推理/导出适配器，不改训练网络、损失、数据增强或checkpoint结构；无需重新生成组合初始化权重，也无需按部署adapter训练。普通训练和评测继续逐样本使用相机参数及真实时间/位姿。

## 当前结果

- 原版简化参考图：OMG检查1945项NPU支持、30项不支持。
- 新版简化图：2702个ONNX节点，OMG 2.1.6 / LQ560ProV200优化后1810项全部标为NPU Supported，0 Unsupported；中间张量最高6维。
- 21项回归测试通过，覆盖单路注意力的输出/梯度、固定池化的重复射线累加/梯度、数值边界掩码、6维融合及原训练功能。
- 同一批固定标定、相同输入下，新adapter与原PyTorch导出路径：首帧/重置检测输出一致，历史帧最大检测logit差约2.24e-4；OCC类别图均一致。
- **ORT端到端数值检查尚未全部通过。** 在本次未训练S1的组合初始化下，首帧/重置缓存特征最大差约0.01082，超出原rtol/atol=3e-3；历史帧对比通过，三个场景OCC类别均一致。没有放宽容差。JSON记录validation_passed=false，导出命令会在保存报告后以非零状态退出。
- 完成的是平台支持检查，不是生成ORTM；未进行量化、板端测速或训练精度验证，不能作为部署成品。

## 标定与时间

当前使用nuScenes val样本0生成的标定，仅用于预检。保存在ONNX同名`.calibration.npz`，包含projection_mat、image_wh、occ_img2ego及provenance。实际车辆需替换真实标定，必须匹配相机顺序及固定resize/crop/归一化流程。

image_wh在固定输入大小时可固定；projection_mat和occ_img2ego只有在对应参考坐标、标定和图像预处理固定时才能固定。nuScenes跨样本/场景并不恒定：相邻样本CAM_FRONT的sensor2lidar旋转矩阵实测最大元素差约0.001525。不要用该样本的固定图直接评估整个nuScenes数据集并将结果视为原模型精度。

nuScenes标注关键帧为2Hz，但当前val infos中5869个同场景相邻间隔：min=0.398234s、median=0.499884s、max=0.650830s，恰好0.5s的记录为0。因此保留动态time_interval；.5只用于无历史的默认间隔，不硬编码有效历史间隔。历史位姿也保留动态，车辆运动补偿不能用单位矩阵代替。

## 推理图改动

1. 从输入删除projection_mat、image_wh、occ_img2ego，变为固定buffer并由onnxsim常量折叠。固定相机嵌入和初始anchor编码提前按PyTorch计算；只缓存首个初始anchor编码调用，不缓存历史anchor或回归后的动态anchor编码。
2. 历史场景/时间有效性检查移到部署主程序，使用`history_is_valid(dt, previous_scene, current_scene)`生成0/1浮点输入；训练侧规则不变。首次启动、换场景、dt<=0、dt>2、非有限时间必须重置历史。无效状态在图内仍清零，首帧仍做原版自注意力。
3. 采样边界规则保留，用数值掩码等价替换And/Not。不能删除边界规则，否则与训练CUDA采样的严格图像边界行为不同。
4. 历史K/V补至450，通过输入侧选择当前或历史K/V，只执行一次attention；有效历史的后150个padding key被屏蔽，避免改变softmax归一化。当前V仍保留原版的query+position语义，历史V仍经fc_before。没有新增训练参数。整体Softmax从22降至16，其中也包含固定标定产生的常量折叠。
5. 固定OCC池化按体素预计算射线分组，分桶padding到2的幂，Gather/ReduceSum累加，再用唯一索引ScatterND写入；没有重复覆盖写入，也没有用大规模全局CumSum减法引入额外消减误差。
6. 采样加权融合合并batch/query轴，保留原求和顺序，将7维中间表达改为最高6维。
7. OCC仅输出argmax类别图，int32 `[1,128,128,16]`，1MiB；原FP32 logits为18MiB，输出体积减少18倍。训练仍使用完整logits和CE损失。内部logits不一定因此免分配，实际内存优化需看编译器。

剩余动态目标采样点仍由网络特征/框预测并投影。用户Sparse4D仓库也保留动态points_2d，symbolic将DAF导为`custom::DeformableAttentionAggrPlugin`，由TensorRT插件实现；并没有把所有采样位置固定。该.so不能直接用于LQ560。本版24个GridSample均通过NPU支持检查，暂不需要为兼容性移植TensorRT插件。

## 使用

代码：`projects/mmdet3d_plugin/models/ep1/deployment.py`；工具：`tools/export_ep1_npu.py`。

```bash
# onnxsim为导出用的可选依赖，不影响训练环境。
python -m pip install onnxsim -i https://mirrors.aliyun.com/pypi/simple/
PYTHONPATH=. python tools/export_ep1_npu.py \
  --checkpoint /path/to/trained_s1.pth --simplify
```

没有实车标定时默认提取nuScenes样本0并标记PREFLIGHT ONLY；可用`--sample-index`改变参考样本。提供标定时：

```bash
PYTHONPATH=. python tools/export_ep1_npu.py \
  --checkpoint /path/to/trained_s1.pth \
  --calibration /path/to/rig.calibration.npz --simplify
```

`--check-only`验证已存在文件，会使用同名保存标定，避免误用另一个样本。无--checkpoint时按配置加载组合初始化，只能用于结构预检，不能当作训练完成的模型。

新输入（7个）：img、history_feature、history_anchor、history_confidence、history_transform、time_interval、history_valid。history_valid为FP32的0或1，其他输入保持FP32；前三个静态相机参数不再输入。新输出仍为7个，原occ_logits位置改为occ_classes（int32），其余检测和历史状态接口保留。

主程序先检查是否存在上一帧缓存、场景/会话是否一致、真实dt是否有效。无效时送history_valid=0，历史补0、位姿单位矩阵、默认dt=.5；有效时送真实dt和上一帧到当前帧变换。不要因为ONNX里删掉了逻辑算子，就在运行程序里也省略这些判断。

当前时间/位姿测试明确包含dt=.7和非单位变换。注意力padding使用负无穷mask，后续PTQ还需验证工具链对mask范围的处理，必要时让相关注意力节点保留浮点；本次未声称量化已通过。

## 当前文件

3090项目下 `work_dirs/ep1/npu/`：s1_fixed.onnx、s1_fixed.calibration.npz、s1_fixed.json、check_results.csv、check.log、input_cfg.yml。

ONNX已在本机工具环境简化后回传，SHA256：`b46e5571423878e11b4cbae5a2cab5f4727b4b1a4048b65fe4671d6902dcaf50`。本机缓存 `/home/tong/.cache/sparsedrive/s1_fixed_20261113/` 有同一文件和检查结果。

OMG配置只用于--check，不是量化标定配置；检查命令与原预检报告一致。该图尚有上述ORT数值检查失败，后续完整编译、量化和训练后精度验证需分别完成。

当前对比是同输入、同历史状态的一步比较，并非独立长序列误差传播验证。后续还需要训练后权重的长序列、量化与板端精度评估。
