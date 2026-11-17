"""S3: S2 plus FFN hidden width 1024 -> 512. Requires fine-tuning.
Shape-compatible tensors are reused; incompatible FFN tensors are initialized.
"""
_base_ = './s2_det_occ_decoder4.py'
work_dir = './work_dirs/ep1/s3_det_occ_decoder4_ffn512'
# Initialized from completed S2 iter_8780; incompatible tensors initialized anew.
load_from = 'ckpt/s3_from_s2_final_init.pth'
resume_from = None
optimizer = dict(lr=2e-5)
# More new parameters than S5: warm up conservatively.
lr_config = dict(warmup_iters=1000)
num_epochs = 20
runner = dict(max_iters=439 * num_epochs)
model = dict(head=dict(det_head=dict(ffn=dict(feedforward_channels=512))))
