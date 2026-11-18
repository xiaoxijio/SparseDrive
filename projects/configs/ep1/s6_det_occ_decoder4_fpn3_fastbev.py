"""S6: S5 detection + FastBEV backward OCC sampling, height mean."""
_base_ = './s5_det_occ_decoder4_fpn3.py'
work_dir = './work_dirs/ep1/s6_det_occ_decoder4_fpn3_fastbev'
load_from = 'ckpt/s6_from_s5_init.pth'
resume_from = None
model = dict(occ_branch=dict(type='FastBEVOccBranch'))
