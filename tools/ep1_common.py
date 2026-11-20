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


def load_export_checkpoint(model, checkpoint):
    """Load a local training checkpoint; discard only the regenerated frustum."""
    saved = torch.load(checkpoint, map_location='cpu')
    state = saved.get('state_dict', saved).copy()
    if state and all(key.startswith('module.') for key in state):
        from collections import OrderedDict
        state = OrderedDict((key[7:], value) for key, value in state.items())
    key = 'occ_branch.view_transform.frustum'
    # This adapter expects frustum to be a nonpersistent, regenerated buffer.
    if key in model.state_dict():
        raise RuntimeError('Expected a nonpersistent frustum in the export model')
    removed = state.pop(key, None) is not None
    voxel_key = 'occ_branch.view_transform.voxel_points'
    if voxel_key in state:
        expected = model.occ_branch.view_transform.voxel_points.cpu()
        torch.testing.assert_close(state.pop(voxel_key), expected, rtol=0, atol=0)
    model.load_state_dict(state, strict=True)
    print('Export checkpoint loaded strictly; regenerated frustum removed:', removed,
          flush=True)


def load_model(config_path, checkpoint=None, device='cuda:0', export_checkpoint=False):
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
        if export_checkpoint:
            load_export_checkpoint(model, initialization)
        else:
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


def occ_forward_projection(img2ego):
    """Host FP32 calibration preparation matching the trained S6 arithmetic."""
    from projects.mmdet3d_plugin.models.ep1.fastbev import affine_inverse
    with torch.no_grad():
        projection = torch.zeros_like(img2ego, dtype=torch.float32)
        projection[..., :3, :] = affine_inverse(img2ego.float())
        projection[..., 3, 3] = 1
    if not torch.isfinite(projection).all():
        raise ValueError('OCC calibration must be invertible and finite')
    return projection
