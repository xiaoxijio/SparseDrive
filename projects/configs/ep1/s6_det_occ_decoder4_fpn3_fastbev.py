"""S6: S5 detection + FastBEV backward OCC sampling, height mean."""
_base_ = './s5_det_occ_decoder4_fpn3.py'
work_dir = './work_dirs/ep1/s6_det_occ_decoder4_fpn3_fastbev'
load_from = 'ckpt/s6_from_s5_init.pth'
resume_from = None
# Match the already trained S6: one sampled plane, 16 predicted height bins.
# Changing the sampling z step to 0.4 requires retraining.
model = dict(occ_branch=dict(type='FastBEVOccBranch',
                             grid_config=dict(z=[-1.0, 5.4, 6.4])))
