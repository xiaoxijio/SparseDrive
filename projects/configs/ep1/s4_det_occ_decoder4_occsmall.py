"""S4: S2 plus BEV backbone widths [128,256,512] -> [64,128,256].
Keep view channels=64, final head=128, grid=128x128 and all 18 classes.
Requires retraining the changed BEV encoder; not an equivalent replacement.
"""
_base_ = './s2_det_occ_decoder4.py'
work_dir = './work_dirs/ep1/s4_det_occ_decoder4_occsmall'
# Initialized from completed S2 iter_8780; incompatible tensors initialized anew.
load_from = 'ckpt/s4_from_s2_final_init.pth'
resume_from = None
optimizer = dict(lr=2e-5)
# More new parameters than S5: warm up conservatively.
lr_config = dict(warmup_iters=1000)
num_epochs = 20
runner = dict(max_iters=439 * num_epochs)
model = dict(occ_branch=dict(bev_channels=[64,128,256]))
