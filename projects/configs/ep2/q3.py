"""Q3: trained S6 FastBEV architecture with official Oritek Conv2d INT8 QAT.
Retain OCC BEV channels [128,256,512]; no S3 FFN or S4 OCC reduction.
Default schedule targets four 32-GB GPUs, not the single 3090 host.
"""
_base_ = '../ep1/s6_det_occ_decoder4_fpn3_fastbev.py'
work_dir = './work_dirs/ep2/q3'
# Reuse final S6 iter_8780 weights; FastBEV geometry stays floating point.
load_from = 'ckpt/s6.pth'
resume_from = None
fp16 = None
model = dict(img_backbone=dict(with_cp=False))
qat = dict(algorithm='fake_quantization', chip_type='LQ560ProV200',
           scopes=['img_backbone', 'img_neck', 'occ_img_neck', 'occ_branch'],
           calibration_max_batch=1, freeze_observer_iter=879,
           freeze_bn_iter=0, coalesced_sync=True)
# Larger OCC encoder than q1/q2: retain more memory headroom.
num_gpus = 4
batch_size = 8
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 6
data = dict(samples_per_gpu=batch_size, workers_per_gpu=8)
optimizer = dict(lr=1e-5)
lr_config = dict(warmup_iters=250, warmup_ratio=0.1, min_lr_ratio=0.1)
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
runner = dict(type='IterBasedRunner', max_iters=num_iters_per_epoch * num_epochs)
checkpoint_epoch_interval = 1
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
find_unused_parameters = True
