"""Semantic checkpoint routing, depth-bin conversion and imported BN policy."""
import unittest
import torch
from torch import nn
from tools.prepare_ep1_weights import depth_channel_indices, merge_states
from projects.mmdet3d_plugin.models.ep1.flashocc import FlashOccBranch


class TransferChecks(unittest.TestCase):
    def test_depth_bins_keep_all_context_after_source_depth_range(self):
        indices = depth_channel_indices([1, 60, 1], [1, 45, 1], 64)
        self.assertEqual(indices.tolist(), list(range(44)) + list(range(59, 123)))
        subset = depth_channel_indices([1, 60, 1], [2, 10, 2], 64)
        self.assertEqual(subset[:4].tolist(), [1, 3, 5, 7])
        with self.assertRaises(ValueError):
            depth_channel_indices([1, 60, 1], [1.5, 4.5, 1], 64)

    def test_routing_never_reuses_unrelated_same_shape_biases(self):
        target = {
            'img_backbone.conv_stem.weight': torch.zeros(2),
            'head.det_head.layers.0.weight': torch.zeros(2),
            'img_neck.fpn_convs.0.conv.bias': torch.zeros(2),
            'depth_branch.depth_layers.0.bias': torch.zeros(2),
            'occ_img_neck.lateral_convs.0.conv.bias': torch.zeros(2),
            'occ_branch.bev_encoder.backbone.layers.0.0.bn1.weight': torch.zeros(2),
            'occ_branch.bev_encoder.neck.conv.1.weight': torch.zeros(2),
            'occ_branch.occ_head.final_conv.conv.bias': torch.zeros(2),
            'occ_branch.view_transform.depth_net.weight': torch.zeros(108, 2, 1, 1),
            'occ_branch.view_transform.depth_net.bias': torch.zeros(108),
            'img_neck.lateral_convs.0.conv.bias': torch.full((2,), -1.),
            'occ_branch.occ_head.predicter.2.bias': torch.full((2,), -2.),
            'head.det_head.instance_bank.anchor': torch.full((2,), -3.),
            'head.det_head.instance_bank.instance_feature': torch.full((2,), -4.),
        }
        g5 = {'module.' + k: torch.full((2,), 5.) for k in (
            'img_backbone.conv_stem.weight', 'img_neck.lateral_convs.0.conv.bias',
            'img_bev_encoder_backbone.layers.0.0.bn1.weight',
            'img_bev_encoder_neck.conv.1.weight', 'occ_head.final_conv.conv.bias',
            'occ_head.predicter.2.bias')}
        depth = torch.arange(123, dtype=torch.float32)
        g5['module.img_view_transformer.depth_net.bias'] = depth
        g5['module.img_view_transformer.depth_net.weight'] = depth[:, None, None, None].expand(123, 2, 1, 1)
        stage1 = {k: torch.full((2,), 9.) for k in (
            'head.det_head.layers.0.weight', 'img_neck.fpn_convs.0.conv.bias',
            'depth_branch.depth_layers.0.bias', 'img_neck.lateral_convs.0.conv.bias',
            'head.det_head.instance_bank.anchor', 'head.det_head.instance_bank.instance_feature')}
        merged, report = merge_states(target, dict(state_dict=g5), dict(state_dict=stage1),
                                       [1, 60, 1], [1, 45, 1], 64)
        for entry in report['fresh']:
            torch.testing.assert_close(merged[entry['target']], target[entry['target']], rtol=0, atol=0)
        torch.testing.assert_close(merged['img_backbone.conv_stem.weight'], torch.full((2,), 5.))
        torch.testing.assert_close(merged['head.det_head.layers.0.weight'], torch.full((2,), 9.))
        expected_depth = torch.cat([depth[:44], depth[59:]])
        torch.testing.assert_close(merged['occ_branch.view_transform.depth_net.bias'], expected_depth)
        self.assertEqual(len(report['loaded']), 10)
        self.assertEqual(len(report['fresh']), 4)
        bad = dict(stage1)
        bad['head.det_head.layers.0.weight'] = torch.zeros(3)
        with self.assertRaises(ValueError):
            merge_states(target, dict(state_dict=g5), dict(state_dict=bad), [1, 60, 1], [1, 45, 1], 64)
        bad = dict(g5)
        bad.pop('module.img_backbone.conv_stem.weight')
        with self.assertRaises(ValueError):
            merge_states(target, dict(state_dict=bad), dict(state_dict=stage1), [1, 60, 1], [1, 45, 1], 64)

    def test_imported_encoder_bn_moments_frozen_affine_trainable(self):
        branch = FlashOccBranch(
            dict(x=[-2, 2, 1], y=[-2, 2, 1], z=[0, 4, 4], depth=[1, 4, 1]),
            norm_eval=True)
        branch.train()
        bn = [m for m in branch.bev_encoder.modules() if isinstance(m, nn.BatchNorm2d)]
        self.assertTrue(bn)
        self.assertTrue(all(not m.training and m.weight.requires_grad and m.bias.requires_grad for m in bn))
        # Preserve caller-controlled evaluation/training mode transitions.
        branch.eval().train()
        self.assertTrue(all(not m.training for m in bn))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
