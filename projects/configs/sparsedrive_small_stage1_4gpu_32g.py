"""Stage 1 on four GPUs with 32 GiB each; launch from the repository root.

Start with four samples per GPU and measure memory on the target machine.
If changing batch_size, edit it here so LR and iteration counts stay in sync.
Do not additionally pass --autoscale-lr to tools/train.py.
"""

_base_ = "./sparsedrive_small_stage1.py"

version = "trainval"
length = {"trainval": 28130, "mini": 323}
num_gpus = 4
batch_size = 4
total_batch_size = num_gpus * batch_size
num_iters_per_epoch = length[version] // total_batch_size
num_epochs = 100
checkpoint_epoch_interval = 20

work_dir = "work_dirs/sparsedrive_small_stage1_4gpu_32g"
load_from = None
resume_from = None

# Inherit the model, 704x256 input, FP16 and backbone checkpointing.
# Override derived values explicitly: MMCV does not recompute base variables.
data = dict(samples_per_gpu=batch_size, workers_per_gpu=4)
optimizer = dict(lr=4e-4 * total_batch_size / 64)
runner = dict(max_iters=num_iters_per_epoch * num_epochs)
checkpoint_config = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval,
)
evaluation = dict(interval=num_iters_per_epoch * checkpoint_epoch_interval)
