"""Single-RTX-3090 smoke-test/debug config for SparseDrive stage 1.

This keeps the official model and input resolution unchanged, but uses one
sample per iteration and no DataLoader subprocesses so IDE breakpoints in the
dataset/pipeline are reachable.  It is intentionally a 10-iteration debug run,
not a reproduction-training recipe.
"""

_base_ = ["./sparsedrive_small_stage1.py"]

# Hardware: 1 x RTX 3090 (24 GiB), 32 GiB system RAM.
num_gpus = 1
batch_size = 1

# Make a smoke test finish quickly and expose every iteration in the log.
runner = dict(type="IterBasedRunner", max_iters=10)
log_config = dict(interval=1)
checkpoint_config = dict(interval=10)
evaluation = dict(interval=10)

# workers_per_gpu=0 is deliberate for source-level debugging: dataset and
# pipeline code then run in the main Python process.  Change it to 2 after the
# data path is understood and breakpoints there are no longer needed.
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=0,
)

# Official stage-1 global batch is 64 (8 GPUs x 8 samples).  Linear scaling for
# a global batch of 1 gives 4e-4 / 64 = 6.25e-6.
optimizer = dict(lr=6.25e-6)
lr_config = dict(warmup_iters=5)

# Save debug artifacts separately and initialize from the ResNet-50 backbone,
# exactly like the official stage-1 config.
work_dir = "./work_dirs/sparsedrive_small_stage1_1gpu_debug"
load_from = None
resume_from = None
