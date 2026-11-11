"""FlashOCC-style LSS -> 2D BEV encoder -> channel-to-height OCC head.

The view transform is an explicit PyTorch reference implementation, keeping
gradients through depth/context and exporting with ONNX ScatterElements/add.
It does not require installing another mmdet3d package or BEVPool CUDA plugin.
"""
import torch
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import HEADS, NECKS


@NECKS.register_module()
class OccImageNeck(BaseModule):
    def __init__(self, in_channels=(112, 1280), out_channels=256, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        self.output = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, features):
        low, high = [conv(x) for conv, x in zip(self.lateral, features)]
        return self.output(low + F.interpolate(
            high, size=low.shape[-2:], mode='bilinear', align_corners=False))


class LSSViewTransform(nn.Module):
    def __init__(self, in_channels, out_channels, grid_config, input_size, downsample=16):
        super().__init__()
        self.out_channels = out_channels
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
        lifted = (depth[:, None] * context[:, :, None]).reshape(b, n, c, -1)
        lifted = lifted.permute(0, 2, 1, 3).reshape(b, c, -1)
        points = torch.matmul(
            img2ego[:, :, None, None, None, :3, :],
            self.frustum[None, None, ..., None]).squeeze(-1)
        xcfg, ycfg, zcfg = (self.grid_config[k] for k in ('x', 'y', 'z'))
        x = torch.floor((points[..., 0] - xcfg[0]) / xcfg[2]).long()
        y = torch.floor((points[..., 1] - ycfg[0]) / ycfg[2]).long()
        valid = ((x >= 0) & (x < self.nx) & (y >= 0) & (y < self.ny)
                 & (points[..., 2] >= zcfg[0]) & (points[..., 2] < zcfg[1]))
        indices = torch.where(valid, y * self.nx + x, torch.zeros_like(x)).reshape(b, 1, -1)
        lifted = lifted * valid.reshape(b, 1, -1).to(lifted)
        bev = lifted.new_zeros(b, c, self.ny * self.nx).scatter_add(
            2, indices.expand(-1, c, -1), lifted)
        return bev.reshape(b, c, self.ny, self.nx)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvModule(in_channels, out_channels, 3, stride=stride, padding=1,
                                norm_cfg=dict(type='BN'), bias=False)
        self.conv2 = ConvModule(out_channels, out_channels, 3, padding=1,
                                norm_cfg=dict(type='BN'), bias=False, act_cfg=None)
        self.shortcut = (nn.Identity() if stride == 1 and in_channels == out_channels
                         else ConvModule(in_channels, out_channels, 1, stride=stride,
                                         norm_cfg=dict(type='BN'), bias=False, act_cfg=None))

    def forward(self, x):
        return F.relu(self.conv2(self.conv1(x)) + self.shortcut(x))


class BEVEncoder(nn.Module):
    """ResNet BEV stages and FPN-LSS, with 128-channel neck convolutions."""
    def __init__(self, in_channels=64, channels=(128, 256, 512), out_channels=128):
        super().__init__()
        self.stages = nn.ModuleList()
        previous = in_channels
        for channel in channels:
            self.stages.append(nn.Sequential(
                ResidualBlock(previous, channel, stride=2), ResidualBlock(channel, channel)))
            previous = channel
        self.fusion = nn.Sequential(
            ConvModule(channels[0] + channels[-1], out_channels, 3, padding=1,
                       norm_cfg=dict(type='BN')),
            ConvModule(out_channels, out_channels, 3, padding=1, norm_cfg=dict(type='BN')))
        self.output = nn.Sequential(
            ConvModule(out_channels, out_channels, 3, padding=1, norm_cfg=dict(type='BN')),
            nn.Conv2d(out_channels, out_channels, 1))

    def forward(self, x):
        output_size = x.shape[-2:]
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        high = F.interpolate(features[-1], size=features[0].shape[-2:],
                             mode='bilinear', align_corners=False)
        x = self.fusion(torch.cat([features[0], high], 1))
        return self.output(F.interpolate(x, size=output_size,
                                         mode='bilinear', align_corners=False))


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
                 Dz=16, num_classes=18, use_mask=True, loss_weight=1.0, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.view_transform = LSSViewTransform(in_channels, view_channels,
                                               grid_config, input_size)
        self.bev_encoder = BEVEncoder(view_channels, bev_channels, out_channels)
        self.occ_head = BEVOccHead2D(out_channels, out_channels, Dz, num_classes)
        self.use_mask = use_mask
        self.loss_weight = loss_weight

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
