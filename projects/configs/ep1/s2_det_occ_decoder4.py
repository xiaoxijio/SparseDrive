"""S2: four decoding rounds; retain 450 queries, 300 history and OCC.
Train/fine-tune separately. This is not an equivalent inference replacement.
"""
_base_ = './s1_det_occ_b0_nose.py'
work_dir = './work_dirs/ep1/s2_det_occ_decoder4'
load_from = 'ckpt/s2_from_s1_final_init.pth'
resume_from = None
num_decoder = 4
model = dict(head=dict(det_head=dict(num_decoder=4, operation_order=(
    ['gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine']
    + ['temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine'] * 3)[2:])))
# Fine-tune the converged S1: detector LR 2e-5, inherited backbone multiplier 0.5.
optimizer = dict(lr=2e-5)
lr_config = dict(warmup_iters=500)
fp16 = dict(_delete_=True, loss_scale=dict(init_scale=32.0, growth_interval=2000))
checkpoint_epoch_interval = 1
checkpoint_config = dict(interval=439, max_keep_ckpts=3)

# Same batch and schedule on the development and rental machines.
num_gpus = 4
batch_size = 16
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
data = dict(samples_per_gpu=batch_size, workers_per_gpu=8)
num_epochs = 20
runner = dict(max_iters=num_iters_per_epoch * num_epochs)
