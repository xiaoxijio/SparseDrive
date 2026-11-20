"""FastBEV-style backward sampling for the S6 OCC branch.
Reference: Sense-GVT/Fast-BEV dev, backproject_inplace/get_points.
Uses voxel lower-corner coordinates, nearest sampling and last-visible-camera
priority. Height mean is an explicit adaptation to the existing 2D OCC encoder.
"""
import torch
from torch import nn
from torch.nn import functional as F
from mmdet.models import HEADS
from .flashocc import FlashOccBranch


def affine_inverse(matrix):
    """Inverse of affine 4x4 projection without ONNX inverse/custom operators."""
    a, b, c = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    d, e, f = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    g, h, i = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    cof = torch.stack((e*i-f*h, c*h-b*i, b*f-c*e,
                       f*g-d*i, a*i-c*g, c*d-a*f,
                       d*h-e*g, b*g-a*h, a*e-b*d), -1)
    determinant = a*(e*i-f*h) + b*(f*g-d*i) + c*(d*h-e*g)
    inverse = cof.reshape(*matrix.shape[:-2], 3, 3) / determinant[..., None, None]
    translation = -torch.matmul(inverse, matrix[..., :3, 3:4])
    return torch.cat((inverse, translation), -1)


class FastBEVViewTransform(nn.Module):
    def __init__(self, in_channels, out_channels, grid_config, input_size):
        super().__init__()
        self.out_channels = out_channels
        self.input_size = tuple(input_size)
        self.context_net = nn.Conv2d(in_channels, out_channels, 1)
        axes = [torch.arange(*grid_config[k], dtype=torch.float32) for k in ('z', 'y', 'x')]
        z, y, x = torch.meshgrid(*axes, indexing='ij')
        self.nz, self.ny, self.nx = z.shape
        # Match official get_points: lower corner, not a half-cell shift.
        self.register_buffer('voxel_points', torch.stack((x, y, z, torch.ones_like(x)), -1), persistent=False)

    def forward(self, feature, img2ego):
        return self.forward_projected(feature, affine_inverse(img2ego.float()))

    def forward_projected(self, feature, ego2img):
        """Same sampler, accepting forward projection for deployment."""
        b, n = ego2img.shape[:2]
        context = self.context_net(feature.float())
        _, c, h, w = context.shape
        context = context.reshape(b, n, c, h, w)
        # OCC coordinates are ego coordinates; Sparse projection_mat uses lidar.
        # Invert OCC's affine mapping, never substitute the detection matrix.
        projection = ego2img[..., :3, :].float()
        planes = []
        for zi in range(self.nz):
            points = self.voxel_points[zi].reshape(-1, 4).t()
            xyz = torch.matmul(projection, points)
            depth = xyz[:, :, 2]
            safe_depth = depth.clamp(min=1e-5)
            # Official FastBEV scales image projection rows by feature stride,
            # then rounds to nearest integer pixel before checking visibility.
            u = torch.round(xyz[:, :, 0] / safe_depth * (w / self.input_size[1]))
            v = torch.round(xyz[:, :, 1] / safe_depth * (h / self.input_size[0]))
            valid = ((depth > 0).float() * (u > -1).float() * (u < w).float()
                     * (v > -1).float() * (v < h).float())
            grid = torch.stack((u.clamp(0, w-1) * (2.0 / max(w-1, 1)) - 1,
                                v.clamp(0, h-1) * (2.0 / max(h-1, 1)) - 1), -1)
            grid = grid.reshape(b, n, self.ny, self.nx, 2)
            # Reverse cumulative visibility implements last-valid-camera wins.
            # No boolean indexing assignment or scatter in the exported graph.
            seen = torch.zeros_like(valid[:, 0], dtype=torch.float32)
            plane = context.new_zeros(b, c, self.ny, self.nx)
            for camera in range(n-1, -1, -1):
                mask = valid[:, camera].float()
                weight = mask * (1 - seen)
                seen = torch.maximum(seen, mask)
                sampled = F.grid_sample(context[:, camera], grid[:, camera],
                                        mode='nearest', padding_mode='zeros', align_corners=True)
                plane = plane + sampled * weight.reshape(b, 1, self.ny, self.nx)
            planes.append(plane)
        return torch.stack(planes, 0).mean(0)


@HEADS.register_module()
class FastBEVOccBranch(FlashOccBranch):
    def __init__(self, grid_config, input_size=(256, 704), in_channels=256,
                 view_channels=64, **kwargs):
        super().__init__(grid_config, input_size, in_channels, view_channels, **kwargs)
        self.view_transform = FastBEVViewTransform(in_channels, view_channels, grid_config, input_size)
