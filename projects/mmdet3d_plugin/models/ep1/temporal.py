"""Fixed-size temporal inputs shared by the first and subsequent frames."""
import torch
from torch import nn
from mmcv.cnn.bricks.registry import ATTENTION, PLUGIN_LAYERS
from mmcv.runner import BaseModule
from mmdet.models import HEADS

from ..instance_bank import InstanceBank, topk
from ..detection3d.detection3d_head import Sparse4DHead


@ATTENTION.register_module()
class ExportableAttention(BaseModule):
    """Explicit MatMul/Softmax attention; no FlashAttention CUDA export op."""
    def __init__(self, embed_dims, num_heads, batch_first=True, dropout=0.0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if not batch_first or embed_dims % num_heads:
            raise ValueError('Expected batch_first and divisible head dimensions')
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key=None, value=None, identity=None,
                query_pos=None, key_pos=None, attn_mask=None, **kwargs):
        identity = query if identity is None else identity
        key = query if key is None else key
        value = key if value is None else value
        query = query if query_pos is None else query + query_pos
        key = key if key_pos is None else key + key_pos

        def split(x):
            return x.reshape(x.shape[0], -1, self.num_heads,
                             self.head_dim).transpose(1, 2)
        q, k, v = split(self.q_proj(query)), split(self.k_proj(key)), split(self.v_proj(value))
        scores = (q * (self.head_dim ** -0.5)) @ k.transpose(-2, -1)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float('-inf'))
            else:
                scores = scores + attn_mask
        out = self.dropout(scores.softmax(-1)) @ v
        out = out.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1)
        return identity + self.dropout(self.out_proj(out))


@PLUGIN_LAYERS.register_module()
class PaddedInstanceBank(InstanceBank):
    """Always supply T history slots, including on the first frame.

    Export callers pass external state tensors in ``metas['history_*']``.
    Normal PyTorch training/inference uses the ordinary internal frame cache.
    Invalid history is zeroed and its temporal attention contribution is gated
    in PaddedSparse4DHead. No row is masked entirely to -inf in a softmax.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < self.num_temp_instances < self.num_anchor:
            raise ValueError('Expected 0 < history slots < total anchors')

    def get(self, batch_size, metas=None, dn_metas=None):
        if 'history_feature' in metas:
            feature = self.instance_feature[None].expand(batch_size, -1, -1)
            anchor = self.anchor[None].expand(batch_size, -1, -1)
            dt = metas['time_interval'].to(feature)
            self.mask = (metas['history_valid'].bool() & (dt > 0)
                         & (dt.abs() <= self.max_time_interval))
            valid = self.mask[:, None, None]
            history_feature = torch.where(valid, metas['history_feature'],
                                          torch.zeros_like(metas['history_feature']))
            history_anchor = torch.where(valid, metas['history_anchor'],
                                         torch.zeros_like(metas['history_anchor']))
            safe_dt = torch.where(self.mask, dt, torch.zeros_like(dt))
            history_anchor = self.anchor_handler.anchor_projection(
                history_anchor, [metas['history_transform']],
                time_intervals=[-safe_dt])[0]
            self.cached_feature = history_feature
            self.cached_anchor = torch.where(valid, history_anchor,
                                              torch.zeros_like(history_anchor))
            self.confidence = torch.where(self.mask[:, None],
                                          metas['history_confidence'],
                                          torch.zeros_like(metas['history_confidence']))
            interval = torch.where(self.mask, dt,
                                   torch.full_like(dt, self.default_time_interval))
            return feature, anchor, self.cached_feature, self.cached_anchor, interval

        previous_metas = self.metas
        feature, anchor, history_feature, history_anchor, interval = super().get(
            batch_size, metas, dn_metas)
        if history_feature is None:
            history_feature = feature.new_zeros(batch_size, self.num_temp_instances,
                                                self.embed_dims)
            history_anchor = anchor.new_zeros(batch_size, self.num_temp_instances,
                                              anchor.shape[-1])
            self.mask = torch.zeros(batch_size, dtype=torch.bool, device=feature.device)
            self.confidence = feature.new_zeros(batch_size, self.num_temp_instances)
        else:
            dt = (metas['timestamp'] - previous_metas['timestamp']).to(feature)
            self.mask = self.mask & (dt > 0)
            # A scene change must reset history even if timestamps are close.
            if 'scene_token' in metas['img_metas'][0]:
                same_scene = feature.new_tensor([
                    now['scene_token'] == old['scene_token']
                    for now, old in zip(metas['img_metas'], previous_metas['img_metas'])
                ], dtype=torch.bool)
                self.mask = self.mask & same_scene
            interval = torch.where(self.mask, interval,
                                   torch.full_like(interval, self.default_time_interval))
            history_feature = torch.where(self.mask[:, None, None], history_feature,
                                          torch.zeros_like(history_feature))
            history_anchor = torch.where(self.mask[:, None, None], history_anchor,
                                         torch.zeros_like(history_anchor))
            self.confidence = torch.where(self.mask[:, None], self.confidence,
                                          torch.zeros_like(self.confidence))
        self.cached_feature, self.cached_anchor = history_feature, history_anchor
        return feature, anchor, history_feature, history_anchor, interval

    def cache(self, instance_feature, anchor, confidence, metas=None, feature_maps=None):
        # Functional concatenation rather than in-place writes also exports to ONNX.
        scores = confidence.detach().max(-1).values.sigmoid()
        t = self.num_temp_instances
        scores = torch.cat([torch.maximum(self.confidence * self.confidence_decay,
                                         scores[:, :t]), scores[:, t:]], dim=1)
        self.temp_confidence = scores
        self.confidence, (self.cached_feature, self.cached_anchor) = topk(
            scores, t, instance_feature.detach(), anchor.detach())
        self.metas = metas


@HEADS.register_module()
class PaddedSparse4DHead(Sparse4DHead):
    def graph_model(self, index, query, *args, **kwargs):
        out = super().graph_model(index, query, *args, **kwargs)
        if self.operation_order[index] == 'temp_gnn':
            out = torch.where(self.instance_bank.mask[:, None, None], out, query)
        return out
