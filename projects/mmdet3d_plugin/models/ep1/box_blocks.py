"""Sparse box blocks without the legacy exporter's advanced index_put path."""
import torch
from torch.nn import functional as F
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from ..detection3d.detection3d_blocks import (
    SparseBox3DRefinementModule, SparseBox3DKeyPointsGenerator)


@PLUGIN_LAYERS.register_module()
class ExportableBoxRefinement(SparseBox3DRefinementModule):
    def forward(self, instance_feature, anchor, anchor_embed, time_interval=1., return_cls=True):
        feature = instance_feature + anchor_embed
        delta = self.layers(feature)
        geometry = delta[..., :6] + anchor[..., :6]
        yaw = delta[..., 6:8]
        if self.refine_yaw:
            yaw = yaw + anchor[..., 6:8]
        if self.normalize_yaw:
            yaw = F.normalize(yaw, dim=-1)
        parts = [geometry, yaw]
        if self.output_dim > 8:
            if not isinstance(time_interval, torch.Tensor):
                time_interval = instance_feature.new_tensor(time_interval)
            if time_interval.dim() > 0:
                time_interval = time_interval[:, None, None]
            parts.append(delta[..., 8:] / time_interval + anchor[..., 8:])
        output = torch.cat(parts, -1)
        cls = self.cls_layers(instance_feature) if return_cls else None
        quality = self.quality_layers(feature) if return_cls and self.with_quality_estimation else None
        return output, cls, quality


@PLUGIN_LAYERS.register_module()
class ExportableBoxKeyPoints(SparseBox3DKeyPointsGenerator):
    def forward(self, anchor, instance_feature=None):
        size = anchor[..., None, 3:6].exp()
        points = self.fix_scale * size
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learned = self.learnable_fc(instance_feature).reshape(
                anchor.shape[0], anchor.shape[1], self.num_learnable_pts, 3).sigmoid() - .5
            points = torch.cat([points, learned * size], -2)
        sine, cosine = anchor[..., 6:7], anchor[..., 7:8]
        x = cosine * points[..., 0] - sine * points[..., 1]
        y = sine * points[..., 0] + cosine * points[..., 1]
        rotated = torch.stack([x, y, points[..., 2]], -1)
        return rotated + anchor[..., None, :3]
