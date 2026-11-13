"""Shared setup for EP1 ONNX export."""
import logging
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmcv.utils import get_logger
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
import projects.mmdet3d_plugin  # noqa: F401 - register project modules


INPUT_NAMES = ['img', 'projection_mat', 'image_wh', 'occ_img2ego',
               'history_feature', 'history_anchor', 'history_confidence',
               'history_transform', 'time_interval', 'history_valid']
OUTPUT_NAMES = ['det_logits', 'det_boxes', 'det_quality', 'occ_logits',
                'next_history_feature', 'next_history_anchor', 'next_history_confidence']


def load_model(config_path, checkpoint=None, device='cuda:0'):
    get_logger('mmcv', log_level=logging.WARNING)
    logging.disable(logging.INFO)
    # CPU ONNX Runtime uses full FP32. TF32 convolution errors can reorder
    # almost-tied TopK cache entries even when final box predictions are close.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = Config.fromfile(config_path)
    initialization = checkpoint or cfg.get('load_from')
    if checkpoint:
        cfg.model.img_backbone.pretrained = None
    model = build_detector(cfg.model)
    model.init_weights()
    if initialization:
        # A combined S1 initialization and a trained S1 checkpoint both contain
        # the complete architecture. Never partially accept an old stage1 model.
        load_checkpoint(model, initialization, map_location='cpu', strict=True)
    model.to(device)
    return cfg, model


def sample_to_device(dataset, index=0, device=0):
    return scatter(collate([dataset[index]], samples_per_gpu=1), [device])[0]


def make_export_inputs(model, data):
    bank = model.head.det_head.instance_bank
    img = data['img']
    b = img.shape[0]
    return (img, data['projection_mat'], data['image_wh'], data['occ_img2ego'],
            img.new_zeros(b, bank.num_temp_instances, bank.embed_dims),
            img.new_zeros(b, bank.num_temp_instances, bank.anchor.shape[-1]),
            img.new_zeros(b, bank.num_temp_instances),
            torch.eye(4, device=img.device, dtype=img.dtype)[None].expand(b, -1, -1),
            img.new_full((b,), 0.5),
            torch.zeros(b, dtype=torch.bool, device=img.device))


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *inputs):
        return self.model.forward_export(*inputs)
