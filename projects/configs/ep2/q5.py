"""Q5: S7 indexed depth-weighted OCC pooling with Oritek Conv2d QAT."""
_base_ = '../ep1/s7_det_occ_decoder4_fpn3_indexpool.py'

work_dir = './work_dirs/ep2/q5'
# S7 and S5 share the same parameter layout. Point this at a trained S7
# checkpoint once available.
load_from = 'ckpt/s5.pth'
resume_from = None
fp16 = None
model = dict(img_backbone=dict(with_cp=False))

# S7's four-card batch size is 4 per GPU, or 1758 steps per nominal epoch.
qat = dict(algorithm='fake_quantization', chip_type='LQ560ProV200',
           scopes=['img_backbone', 'img_neck', 'occ_img_neck', 'occ_branch'],
           calibration_max_batch=1, freeze_observer_iter=1758,
           freeze_bn_iter=0, coalesced_sync=True)
lr_config = dict(warmup_iters=500, warmup_ratio=0.1, min_lr_ratio=0.1)
find_unused_parameters = True
