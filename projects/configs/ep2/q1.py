"""Q1 = S1 + S2 + S4 + S5, official SDK mixed-precision INT8 QAT."""
_base_ = '../ep1/s5_det_occ_decoder4_fpn3.py'
work_dir = './work_dirs/ep2/q1'
load_from = 'ckpt/q1_fp_init.pth'
# Official Oritek 2.1.6 Conv2d fake-quant islands; other operations stay floating.
qat = dict(algorithm='fake_quantization', chip_type='LQ560ProV200',
           scopes=['img_backbone', 'img_neck', 'occ_img_neck', 'occ_branch'],
           calibration_max_batch=1, freeze_observer_iter=586,
           freeze_bn_iter=0, coalesced_sync=True)
# SDK fake quant in FP32; only native FlashAttention uses local autocast.
fp16 = None
model = dict(img_backbone=dict(with_cp=False),
             occ_branch=dict(bev_channels=[64,128,256]))
num_gpus = 4
batch_size = 12
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 6
data = dict(samples_per_gpu=batch_size, workers_per_gpu=8)
optimizer = dict(lr=1e-5)
lr_config = dict(warmup_iters=167, warmup_ratio=0.1, min_lr_ratio=0.1)
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
runner = dict(type='IterBasedRunner', max_iters=num_iters_per_epoch * num_epochs)
checkpoint_epoch_interval = 1
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
resume_from = None
find_unused_parameters = True
