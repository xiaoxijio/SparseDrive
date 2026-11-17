"""S3: S2 plus FFN hidden width 1024 -> 512. Requires fine-tuning.
Shape-compatible tensors are reused; incompatible FFN tensors are initialized.
"""
_base_ = './s2_det_occ_decoder4.py'
work_dir = './work_dirs/ep1/s3_det_occ_decoder4_ffn512'
load_from = 'ckpt/s3_det_occ_decoder4_ffn512_init.pth'
model = dict(head=dict(det_head=dict(ffn=dict(feedforward_channels=512))))
