"""S7: S5 with depth-weighted indexed OCC pooling."""
_base_ = './s5_det_occ_decoder4_fpn3.py'

work_dir = './work_dirs/ep1/s7_det_occ_decoder4_fpn3_indexpool'
load_from = 'ckpt/s5.pth'
resume_from = None
model = dict(occ_branch=dict(type='IndexedFlashOccBranch'))

# Standard PyTorch pooling needs more memory than the fused S5 CUDA kernel.
num_gpus = 4
batch_size = 4
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 6
data = dict(samples_per_gpu=batch_size, workers_per_gpu=8)
optimizer = dict(lr=1e-5)
lr_config = dict(warmup_iters=500)
runner = dict(max_iters=num_iters_per_epoch * num_epochs)
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
