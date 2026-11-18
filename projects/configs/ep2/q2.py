"""Q2 = Q1 + S3 (FFN hidden 512), separate QAT experiment."""
_base_ = './q1.py'
work_dir = './work_dirs/ep2/q2'
load_from = 'ckpt/q2_fp_init.pth'
model = dict(head=dict(det_head=dict(ffn=dict(feedforward_channels=512))))
