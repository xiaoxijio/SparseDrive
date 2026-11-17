"""S4: S2 plus BEV backbone widths [128,256,512] -> [64,128,256].
Keep view channels=64, final head=128, grid=128x128 and all 18 classes.
Requires retraining the changed BEV encoder; not an equivalent replacement.
"""
_base_ = './s2_det_occ_decoder4.py'
work_dir = './work_dirs/ep1/s4_det_occ_decoder4_occsmall'
load_from = 'ckpt/s4_det_occ_decoder4_occsmall_init.pth'
model = dict(occ_branch=dict(bev_channels=[64,128,256]))
