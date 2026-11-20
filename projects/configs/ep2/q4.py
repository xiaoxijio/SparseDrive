"""Q4: trained Q3 FastBEV QAT + S3 FFN hidden width 1024 -> 512."""
_base_ = './q3.py'
work_dir = './work_dirs/ep2/q4'
# A QAT warm start preserves Q3 quantizer buffers, not its optimizer/iteration.
load_from = None
resume_from = None
qat_init_from = 'ckpt/q4_qat_init.pth'
qat_initialization = dict(
    source='work_dirs/ep2/q3/iter_5274.pth',
    ffn_source='work_dirs/ep1/s3_det_occ_decoder4_ffn512/iter_8780.pth')
model = dict(head=dict(det_head=dict(ffn=dict(feedforward_channels=512))))
