"""SDK FlashOCC optimized network with SparseDrive image/ego interfaces.

Trainable modules match SDK CustomFPN, CustomResNet, FPN_LSS and the
128 -> 128 -> 18*16 Softplus predictor requested in the NPU guide.
CUDA training uses the official BEVPoolv2 kernel; the existing standard
ONNX pooling fallback is retained pending deployment-specific optimization.
"""
import torch
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import HEADS, NECKS


from .sdk_blocks import CustomFPN, CustomResNet, FPN_LSS


@NECKS.register_module()
class OccImageNeck(CustomFPN):
    """SDK CustomFPN: [112,1280] -> one stride-16, 256-channel map."""
    def __init__(self, in_channels=(112, 1280), out_channels=256, init_cfg=None):
        kwargs = dict(in_channels=list(in_channels), out_channels=out_channels,
                      num_outs=1, start_level=0, out_ids=[0],
                      upsample_cfg=dict(mode='bilinear', align_corners=False))
        if init_cfg is not None:
            kwargs['init_cfg'] = init_cfg
        super().__init__(**kwargs)

    def forward(self, features):
        return super().forward(features)[0]


class LSSViewTransform(nn.Module):
    def __init__(self, in_channels, out_channels, grid_config, input_size, downsample=16,
                 use_bev_pool=True):
        super().__init__()
        self.out_channels = out_channels
        self.use_bev_pool = use_bev_pool
        self.grid_config = grid_config
        self.nx = round((grid_config['x'][1] - grid_config['x'][0]) / grid_config['x'][2])
        self.ny = round((grid_config['y'][1] - grid_config['y'][0]) / grid_config['y'][2])
        depths = torch.arange(*grid_config['depth'], dtype=torch.float32)
        self.num_depth = len(depths)
        h, w = input_size[0] // downsample, input_size[1] // downsample
        # Same frustum convention as FlashOCC/BEVDet LSSViewTransformer.
        d, v, u = torch.meshgrid(depths,
                                torch.linspace(0, input_size[0] - 1, h),
                                torch.linspace(0, input_size[1] - 1, w), indexing='ij')
        frustum = torch.stack([u * d, v * d, d, torch.ones_like(d)], -1)
        self.register_buffer('frustum', frustum, persistent=False)
        self.depth_net = nn.Conv2d(in_channels, self.num_depth + out_channels, 1)

    def forward(self, feature, img2ego):
        b, n = img2ego.shape[:2]
        prediction = self.depth_net(feature.float())
        depth = prediction[:, :self.num_depth].softmax(1)
        context = prediction[:, self.num_depth:]
        c, h, w = context.shape[1:]
        points = torch.matmul(
            img2ego[:, :, None, None, None, :3, :],
            self.frustum[None, None, ..., None]).squeeze(-1)
        xcfg, ycfg, zcfg = (self.grid_config[k] for k in ('x', 'y', 'z'))
        x = torch.floor((points[..., 0] - xcfg[0]) / xcfg[2]).long()
        y = torch.floor((points[..., 1] - ycfg[0]) / ycfg[2]).long()
        valid = ((x >= 0) & (x < self.nx) & (y >= 0) & (y < self.ny)
                 & (points[..., 2] >= zcfg[0]) & (points[..., 2] < zcfg[1]))
        if self.use_bev_pool and feature.is_cuda and not torch.onnx.is_in_onnx_export():
            from ...ops.bev_pool_v2 import bev_pool_v2
            # SDK ranks layout: depth B,N,D,H,W; features B,N,H,W,C;
            # output B,Z,Y,X,C, with Z collapsed to one BEV plane.
            ranks_depth = torch.arange(valid.numel(), device=feature.device).reshape(valid.shape)
            ranks_feat = torch.arange(b * n * h * w, device=feature.device)
            ranks_feat = ranks_feat.reshape(b, n, 1, h, w).expand_as(valid)
            batch = torch.arange(b, device=feature.device)[:, None, None, None, None]
            ranks_bev = batch * (self.ny * self.nx) + y * self.nx + x
            ranks_depth, ranks_feat, ranks_bev = [r[valid] for r in
                                                  (ranks_depth, ranks_feat, ranks_bev)]
            if ranks_bev.numel() == 0:
                return (context.sum() + depth.sum()) * context.new_zeros(b, c, self.ny, self.nx)
            order = ranks_bev.argsort()
            ranks_depth, ranks_feat, ranks_bev = [r[order].int().contiguous() for r in
                                                  (ranks_depth, ranks_feat, ranks_bev)]
            kept = torch.ones_like(ranks_bev, dtype=torch.bool)
            kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
            starts = torch.where(kept)[0].int()
            lengths = torch.cat([starts[1:], starts.new_tensor([ranks_bev.numel()])]) - starts
            context_nhwc = context.reshape(b, n, c, h, w).permute(0, 1, 3, 4, 2)
            return bev_pool_v2(depth.reshape(b, n, self.num_depth, h, w), context_nhwc,
                               ranks_depth, ranks_feat, ranks_bev,
                               (b, 1, self.ny, self.nx, c), starts, lengths).squeeze(2)
        # Preserve the existing standard export path; optimize it separately.
        lifted = (depth[:, None] * context[:, :, None]).reshape(b, n, c, -1)
        lifted = lifted.permute(0, 2, 1, 3).reshape(b, c, -1)
        indices = torch.where(valid, y * self.nx + x, torch.zeros_like(x)).reshape(b, 1, -1)
        lifted = lifted * valid.reshape(b, 1, -1).to(lifted)
        bev = lifted.new_zeros(b, c, self.ny * self.nx).scatter_add(
            2, indices.expand(-1, c, -1), lifted)
        return bev.reshape(b, c, self.ny, self.nx)


class BEVEncoder(nn.Module):
    """Unmodified SDK CustomResNet + FPN_LSS learned operations."""
    def __init__(self, in_channels=64, channels=(128, 256, 512), out_channels=128):
        super().__init__()
        self.backbone = CustomResNet(numC_input=in_channels,
                                    num_channels=list(channels))
        self.neck = FPN_LSS(in_channels=channels[0] + channels[-1],
                            out_channels=out_channels, channels_factor=1)

    def forward(self, x):
        return self.neck(self.backbone(x))


class BEVOccHead2D(nn.Module):
    def __init__(self, in_dim=128, out_dim=128, Dz=16, num_classes=18):
        super().__init__()
        self.out_dim, self.Dz, self.num_classes = out_dim, Dz, num_classes
        self.final_conv = ConvModule(in_dim, out_dim, 3, padding=1, bias=True)
        self.predicter = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.Softplus(),
            nn.Linear(self.out_dim, num_classes * Dz))

    def forward(self, x):
        x = self.final_conv(x).permute(0, 3, 2, 1)  # B,C,Y,X -> B,X,Y,C
        return self.predicter(x).reshape(x.shape[0], x.shape[1], x.shape[2],
                                         self.Dz, self.num_classes)


@HEADS.register_module()
class FlashOccBranch(BaseModule):
    def __init__(self, grid_config, input_size=(256, 704), in_channels=256,
                 view_channels=64, bev_channels=(128, 256, 512), out_channels=128,
                 Dz=16, num_classes=18, use_mask=True, loss_weight=1.0, norm_eval=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.view_transform = LSSViewTransform(in_channels, view_channels,
                                               grid_config, input_size)
        self.bev_encoder = BEVEncoder(view_channels, bev_channels, out_channels)
        self.occ_head = BEVOccHead2D(out_channels, out_channels, Dz, num_classes)
        self.use_mask = use_mask
        self.loss_weight = loss_weight
        self.norm_eval = norm_eval

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            # Imported G5 encoder uses folded-BN statistics. Preserve these
            # moments while leaving all conv and BN affine parameters trainable.
            for module in self.bev_encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    @force_fp32(apply_to=('feature', 'img2ego'))
    def forward(self, feature, img2ego):
        return self.occ_head(self.bev_encoder(self.view_transform(feature, img2ego)))

    @force_fp32(apply_to=('logits',))
    def loss(self, logits, voxel_semantics, mask_camera):
        target = voxel_semantics.long()
        if tuple(logits.shape[:-1]) != tuple(target.shape):
            raise ValueError('OCC logits/labels shape mismatch: {} vs {}'.format(
                logits.shape, target.shape))
        valid = target != 255
        if self.use_mask:
            valid = valid & mask_camera.bool()
        losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 target.reshape(-1), ignore_index=255, reduction='none')
        # clamp avoids a NaN when a sample has no visible/valid voxels.
        loss = (losses * valid.reshape(-1)).sum() / valid.sum().clamp(min=1)
        return {'loss_occ': loss * self.loss_weight}
