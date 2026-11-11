"""Torchvision EfficientNet-B0 features, with every SE block removed."""
import torch
from torch import nn
from torchvision.models import efficientnet_b0
from torchvision.ops import SqueezeExcitation
from mmcv.runner import BaseModule, load_checkpoint
from mmdet.models import BACKBONES


def _remove_se(module):
    for name, child in list(module.named_children()):
        if isinstance(child, SqueezeExcitation):
            setattr(module, name, nn.Identity())
        else:
            _remove_se(child)


@BACKBONES.register_module()
class EfficientNetB0NoSE(BaseModule):
    """Return strides 4/8/16/32, with channels 24/40/112/1280.

    A torchvision B0 checkpoint may initialize matching parameters; its SE and
    classifier parameters are intentionally unused. It is not a trained no-SE
    checkpoint. ``pretrained=None`` enables fully offline construction.
    """
    out_channels = (24, 40, 112, 1280)
    strides = (4, 8, 16, 32)

    def __init__(self, pretrained=None, frozen_stages=-1, norm_eval=False,
                 with_cp=False, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if pretrained is not None and init_cfg is not None:
            raise ValueError('Use pretrained or init_cfg, not both')
        self.features = efficientnet_b0(weights=None).features
        _remove_se(self.features)
        self.pretrained = pretrained
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self._freeze_stages()

    def init_weights(self):
        if self.pretrained:
            load_checkpoint(self, self.pretrained, map_location='cpu', strict=False)
        else:
            super().init_weights()

    def _freeze_stages(self):
        for i in range(self.frozen_stages + 1):
            self.features[i].eval()
            for param in self.features[i].parameters():
                param.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.features):
            if self.with_cp and self.training and x.requires_grad:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(layer, x)
            else:
                x = layer(x)
            if i in (2, 3, 5, 8):
                outputs.append(x)
        return tuple(outputs)
