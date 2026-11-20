"""Fixed-rig NPU export adapter. No changes to the training model or weights."""
import copy
from types import MethodType
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..blocks import DeformableFeatureAggregation
from ..detection3d.detection3d_head import Sparse4DHead

INPUT_NAMES = ['img', 'history_feature', 'history_anchor', 'history_confidence',
               'history_transform', 'time_interval', 'history_valid']
OUTPUT_NAMES = ['det_logits', 'det_boxes', 'det_quality', 'occ_classes',
                'next_history_feature', 'next_history_anchor', 'next_history_confidence']


def history_is_valid(dt, previous_scene, current_scene):
    """Host-side validity; call before feeding the numeric 0/1 flag to ONNX."""
    return (previous_scene is not None and current_scene is not None
            and previous_scene == current_scene and np.isfinite(dt) and 0 < dt <= 2.)


def deployment_bank_get(self, batch_size, metas=None, dn_metas=None):
    feature = self.instance_feature[None].expand(batch_size, -1, -1)
    anchor = self.anchor[None].expand(batch_size, -1, -1)
    self.mask = metas['history_valid'] > .5
    valid = self.mask[:, None, None]
    self.cached_feature = torch.where(valid, metas['history_feature'], torch.zeros_like(metas['history_feature']))
    history_anchor = torch.where(valid, metas['history_anchor'], torch.zeros_like(metas['history_anchor']))
    dt = metas['time_interval']
    safe_dt = torch.where(self.mask, dt, torch.zeros_like(dt))
    projected = self.anchor_handler.anchor_projection(
        history_anchor, [metas['history_transform']], time_intervals=[-safe_dt])[0]
    self.cached_anchor = torch.where(valid, projected, torch.zeros_like(projected))
    self.confidence = torch.where(self.mask[:, None], metas['history_confidence'],
                                  torch.zeros_like(metas['history_confidence']))
    interval = torch.where(self.mask, dt, torch.full_like(dt, self.default_time_interval))
    return feature, anchor, self.cached_feature, self.cached_anchor, interval


def single_temporal_attention(self, index, query, key=None, value=None,
                              query_pos=None, key_pos=None, **kwargs):
    if self.operation_order[index] != 'temp_gnn':
        return Sparse4DHead.graph_model(self, index, query, key, value,
                                        query_pos=query_pos, key_pos=key_pos, **kwargs)
    # Preserve the original no-history semantics: V is concatenated current
    # query+position, whereas history V goes through fc_before.
    attention = self.layers[index]
    mha = attention.attn
    q = torch.cat([query, query_pos], -1)
    hk = torch.cat([key, key_pos], -1)
    hv = self.fc_before(value)
    pad = q.shape[1] - hk.shape[1]
    hk, hv = F.pad(hk, (0, 0, 0, pad)), F.pad(hv, (0, 0, 0, pad))
    valid = self.instance_bank.mask[:, None, None]
    k, v = torch.where(valid, hk, q), torch.where(valid, hv, q)
    weights = mha.in_proj_weight.chunk(3)
    biases = (None, None, None) if mha.in_proj_bias is None else mha.in_proj_bias.chunk(3)
    q, k, v = [F.linear(x, w, bias) for x, w, bias in zip((q, k, v), weights, biases)]
    b, nq, channels = q.shape
    heads, dim = mha.num_heads, mha.head_dim
    q, k, v = [x.reshape(b, nq, heads, dim).transpose(1, 2) for x in (q, k, v)]
    scores = (q * (dim ** -.5)) @ k.transpose(-2, -1)
    # Only the first T history keys are valid. First frames use all Q current keys.
    keep = torch.cat([query.new_ones(b, key.shape[1]),
                      (1. - self.instance_bank.mask.to(query.dtype))[:, None].expand(b, pad)], 1)
    scores = torch.where(keep[:, None, None, :] > .5, scores,
                          torch.full_like(scores, float('-inf')))
    context = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, nq, channels)
    result = torch.cat([query, query_pos], -1) + mha.out_proj(context)
    return self.fc_after(result)


def numeric_boundary_sampling(feature_maps, key_points, projection_mat, image_wh=None,
                               strict_image_bounds=False):
    # Keep the original bilinear operation; replace unsupported logical mask ops.
    points = DeformableFeatureAggregation.project_points(key_points, projection_mat, image_wh)
    valid_xy = (points > 0).to(points.dtype) * (points < 1).to(points.dtype)
    valid = valid_xy[..., 0] * valid_xy[..., 1]
    grid = (points * 2 - 1).flatten(end_dim=1)
    features = torch.stack([F.grid_sample(f.flatten(end_dim=1), grid,
                                        mode='bilinear', padding_mode='zeros', align_corners=False)
                            for f in feature_maps], dim=1)
    b, cams = feature_maps[0].shape[:2]
    features = features.reshape(b, cams, len(feature_maps), feature_maps[0].shape[2],
                                key_points.shape[1], key_points.shape[2]).permute(0, 4, 1, 2, 5, 3)
    if strict_image_bounds:
        features = features * valid.permute(0, 2, 1, 3)[:, :, :, None, :, None]
    return features


def rank_six_fusion(self, features, weights):
    b, q, cams, levels, points, groups = weights.shape
    grouped = features.reshape(b * q, cams, levels, points, groups, self.group_dims)
    weighted = grouped * weights.reshape(b * q, cams, levels, points, groups, 1)
    return weighted.sum(1).sum(1).reshape(b, q, points, self.embed_dims)


class FixedRigPool(nn.Module):
    """Precomputed ragged voxel sums: Gather/ReduceSum + unique ScatterND.

    Buckets padded to powers of two avoid a huge dense max-rays-per-voxel table.
    Each output index is unique; ScatterND performs assignment, never accumulation.
    """
    def __init__(self, original, img2ego):
        super().__init__()
        self.depth_net = original.depth_net
        self.num_depth = original.num_depth
        self.out_channels = original.out_channels
        self.nx, self.ny = original.nx, original.ny
        with torch.no_grad():
            matrix = img2ego.detach().cpu()
            frustum = original.frustum.detach().cpu()
            points = (matrix[:, :, None, None, None, :3, :] @ frustum[None, None, ..., None]).squeeze(-1)
            xcfg, ycfg, zcfg = [original.grid_config[k] for k in ('x', 'y', 'z')]
            x = torch.floor((points[..., 0] - xcfg[0]) / xcfg[2]).long()
            y = torch.floor((points[..., 1] - ycfg[0]) / ycfg[2]).long()
            valid = ((x >= 0) & (x < self.nx) & (y >= 0) & (y < self.ny)
                     & (points[..., 2] >= zcfg[0]) & (points[..., 2] < zcfg[1]))
            _, cams, depth, h, w = x.shape
            rd = torch.arange(x.numel()).reshape_as(x)[valid].numpy()
            rf = torch.arange(cams * h * w).reshape(1, cams, 1, h, w).expand_as(x)[valid].numpy()
            rb = (y * self.nx + x)[valid].numpy()
            order = np.argsort(rb, kind='stable')
            rb, rd, rf = rb[order], rd[order], rf[order]
            voxels, starts, counts = np.unique(rb, return_index=True, return_counts=True)
            self.bucket_names = []
            output_indices = []
            sizes = sorted(set(1 << (int(c) - 1).bit_length() for c in counts))
            for width in sizes:
                select = np.flatnonzero(np.array([1 << (int(c) - 1).bit_length() for c in counts]) == width)
                dindex = np.full((len(select), width), cams * depth * h * w, dtype=np.int64)
                findex = np.zeros_like(dindex)
                for row, group in enumerate(select):
                    start, count = starts[group], counts[group]
                    dindex[row, :count] = rd[start:start + count]
                    findex[row, :count] = rf[start:start + count]
                name = 'bucket_' + str(width)
                self.register_buffer(name + '_depth', torch.from_numpy(dindex))
                self.register_buffer(name + '_feature', torch.from_numpy(findex))
                self.bucket_names.append(name)
                output_indices.extend(voxels[select].tolist())
            self.register_buffer('voxel_indices', torch.tensor(output_indices, dtype=torch.long))

    def forward(self, feature, img2ego=None):
        prediction = self.depth_net(feature.float())
        depth = prediction[:, :self.num_depth].softmax(1).reshape(-1)
        depth = torch.cat([depth, depth.new_zeros(1)])
        context = prediction[:, self.num_depth:].permute(0, 2, 3, 1).reshape(-1, self.out_channels)
        sums = []
        for name in self.bucket_names:
            d = depth[getattr(self, name + '_depth')]
            f = context[getattr(self, name + '_feature')]
            sums.append((d[..., None] * f).sum(1))
        values = torch.cat(sums, 0) if sums else context[:0]
        bev = feature.new_zeros(self.ny * self.nx, self.out_channels)
        bev = bev.index_put((self.voxel_indices,), values, accumulate=False)
        return bev.reshape(1, self.ny, self.nx, self.out_channels).permute(0, 3, 1, 2)


class CachedInitialEncoder(nn.Module):
    """First encoder call receives the checkpoint's fixed initial anchors."""
    def __init__(self, original, initial_anchor):
        super().__init__()
        self.original = original
        with torch.no_grad():
            self.register_buffer('initial_embedding', original(initial_anchor).detach())
        self.first = True

    def forward(self, anchor):
        if self.first:
            self.first = False
            return self.initial_embedding
        return self.original(anchor)


class CachedCameraEncoder(nn.Module):
    def __init__(self, original, projection):
        super().__init__()
        with torch.no_grad():
            self.register_buffer('embedding', original(projection[:, :, :3].reshape(1, projection.shape[1], -1)).detach())

    def forward(self, projection):
        return self.embedding


class FixedRigDeployment(nn.Module):
    def __init__(self, model, projection_mat, image_wh, occ_img2ego):
        super().__init__()
        self.model = copy.deepcopy(model).eval()
        self.register_buffer('projection_mat', projection_mat.detach().clone())
        self.register_buffer('image_wh', image_wh.detach().clone())
        self.register_buffer('occ_img2ego', occ_img2ego.detach().clone())
        if projection_mat.shape[0] != 1:
            raise ValueError('Fixed-rig export currently supports batch 1')
        head = self.model.head.det_head
        if not head.decouple_attn:
            raise ValueError('S1 deployment adapter requires decoupled attention')
        head.anchor_encoder = CachedInitialEncoder(head.anchor_encoder, head.instance_bank.anchor[None])
        head.graph_model = MethodType(single_temporal_attention, head)
        head.instance_bank.get = MethodType(deployment_bank_get, head.instance_bank)
        for module in self.model.modules():
            if isinstance(module, DeformableFeatureAggregation):
                if module.camera_encoder is not None:
                    module.camera_encoder = CachedCameraEncoder(module.camera_encoder, projection_mat)
                module.feature_sampling = numeric_boundary_sampling
                module.multi_view_level_fusion = MethodType(rank_six_fusion, module)
        if not hasattr(self.model.occ_branch.view_transform, 'depth_net'):
            raise ValueError('S6 FastBEV uses tools/export_ep1_npu.py for fixed or dynamic export')
        pool = FixedRigPool(self.model.occ_branch.view_transform, occ_img2ego)
        self.model.occ_branch.view_transform = pool.to(projection_mat.device)
        self.eval()

    def forward(self, img, history_feature, history_anchor, history_confidence,
                history_transform, time_interval, history_valid):
        if self.training:
            raise RuntimeError('FixedRigDeployment is inference-only; train the original S1 model')
        self.model.head.det_head.anchor_encoder.first = True
        outputs = self.model.forward_export(img, self.projection_mat, self.image_wh,
            self.occ_img2ego, history_feature, history_anchor, history_confidence,
            history_transform, time_interval, history_valid)
        # Int32 is more portable for OMG than int64; no logits leave this graph.
        return outputs[:3] + (outputs[3].argmax(-1).to(torch.int32),) + outputs[4:]


DYNAMIC_INPUT_NAMES = ['img', 'projection_mat', 'occ_ego2img'] + INPUT_NAMES[1:]


class DynamicRigDeployment(nn.Module):
    """S6 runtime camera geometry; only image dimensions are fixed."""
    def __init__(self, model, image_wh):
        super().__init__()
        self.model = copy.deepcopy(model).eval()
        self.register_buffer('image_wh', image_wh.detach().clone())
        head = self.model.head.det_head
        head.anchor_encoder = CachedInitialEncoder(head.anchor_encoder, head.instance_bank.anchor[None])
        head.graph_model = MethodType(single_temporal_attention, head)
        head.instance_bank.get = MethodType(deployment_bank_get, head.instance_bank)
        for module in self.model.modules():
            if isinstance(module, DeformableFeatureAggregation):
                module.feature_sampling = numeric_boundary_sampling
                module.multi_view_level_fusion = MethodType(rank_six_fusion, module)
        from .fastbev import FastBEVViewTransform
        if not isinstance(self.model.occ_branch.view_transform, FastBEVViewTransform):
            raise ValueError('Dynamic LQ560 export now supports S6 FastBEV only')
        view = self.model.occ_branch.view_transform
        view.forward = MethodType(FastBEVViewTransform.forward_projected, view)

    def forward(self, img, projection_mat, occ_ego2img, history_feature,
                history_anchor, history_confidence, history_transform,
                time_interval, history_valid):
        self.model.head.det_head.anchor_encoder.first = True
        outputs = self.model.forward_export(img, projection_mat, self.image_wh,
            occ_ego2img, history_feature, history_anchor, history_confidence,
            history_transform, time_interval, history_valid)
        return outputs[:3] + (outputs[3].argmax(-1).to(torch.int32),) + outputs[4:]
