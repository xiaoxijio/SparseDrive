"""Geometry, label alignment and mixed-batch temporal regression checks."""
import unittest
import numpy as np
import torch
from projects.mmdet3d_plugin.datasets.ep1 import crop_occ
from projects.mmdet3d_plugin.models.ep1.flashocc import LSSViewTransform, FlashOccBranch
from projects.mmdet3d_plugin.models.ep1.temporal import PaddedInstanceBank
from projects.mmdet3d_plugin.models.ep1.box_blocks import (
    ExportableBoxRefinement, ExportableBoxKeyPoints)
from projects.mmdet3d_plugin.models.detection3d.detection3d_blocks import (
    SparseBox3DRefinementModule, SparseBox3DKeyPointsGenerator)


class EP1Checks(unittest.TestCase):
    def test_exportable_box_blocks_preserve_original_math(self):
        torch.manual_seed(0)
        feature, embedding, anchor = torch.randn(2, 4, 32), torch.randn(2, 4, 32), torch.randn(2, 4, 11)
        original = SparseBox3DRefinementModule(embed_dims=32, refine_yaw=True,
                                              with_quality_estimation=True).eval()
        replacement = ExportableBoxRefinement(embed_dims=32, refine_yaw=True,
                                               with_quality_estimation=True).eval()
        replacement.load_state_dict(original.state_dict())
        interval = torch.tensor([.5, .7])
        for expected, actual in zip(original(feature, anchor, embedding, interval),
                                     replacement(feature, anchor, embedding, interval)):
            torch.testing.assert_close(actual, expected)
        kwargs = dict(embed_dims=32, num_learnable_pts=2,
                      fix_scale=[[0, 0, 0], [.45, 0, 0]])
        original = SparseBox3DKeyPointsGenerator(**kwargs)
        replacement = ExportableBoxKeyPoints(**kwargs)
        replacement.load_state_dict(original.state_dict())
        torch.testing.assert_close(replacement(anchor, feature), original(anchor, feature))

    def test_semantic_crop_preserves_voxel_centers(self):
        source = dict(x=[-40., 40., .4], y=[-40., 40., .4], z=[-1., 5.4, .4])
        target = dict(x=[-25.6, 25.6, .4], y=[-25.6, 25.6, .4], z=[-1., 5.4, .4])
        label = np.arange(200 * 200 * 16).reshape(200, 200, 16)
        actual = crop_occ(label, source, target)
        np.testing.assert_array_equal(actual, label[36:164, 36:164, :])
        with self.assertRaises(ValueError):
            crop_occ(label, source, dict(target, x=[-25.5, 25.7, .4]))

    def test_lss_pooling_axis_order_camera_sum_and_gradient(self):
        module = LSSViewTransform(1, 1,
            dict(x=[-1, 1, 1], y=[-1, 1, 1], z=[0, 2, 2], depth=[1, 2, 1]),
            input_size=(16, 16))
        with torch.no_grad():
            module.depth_net.weight.zero_()
            module.depth_net.bias.zero_()
            module.depth_net.weight[1] = 1
        features = torch.tensor([[[[2.]]], [[[3.]]]], requires_grad=True)
        transforms = torch.eye(4)[None, None].repeat(1, 2, 1, 1)
        transforms[:, 1, 0, 3] = -1  # move camera 1 from x=0 to x=-1
        result = module(features, transforms)
        expected = torch.tensor([[[[0., 0.], [3., 2.]]]])  # B,C,Y,X
        torch.testing.assert_close(result, expected)
        result.sum().backward()
        torch.testing.assert_close(features.grad, torch.ones_like(features))
        transforms[:, 1, 0, 3] = 100  # camera 1 points fall outside the BEV
        torch.testing.assert_close(module(features, transforms),
                                   torch.tensor([[[[0., 0.], [0., 2.]]]]))
        transforms[:, 1, 0, 3] = 0
        self.assertEqual(module(features, transforms)[0, 0, 1, 1].item(), 5.)

    def test_mixed_batch_scene_reset_and_motion_compensation(self):
        anchors = np.zeros((4, 11), dtype=np.float32)
        anchors[:, 7] = 1
        bank = PaddedInstanceBank(num_anchor=4, embed_dims=4, anchor=anchors,
                                  num_temp_instances=2,
                                  anchor_handler=dict(type='SparseBox3DKeyPointsGenerator'))

        def metas(timestamp, scenes, translation=0.):
            inv = np.eye(4)
            inv[0, 3] = translation
            return dict(timestamp=torch.tensor([timestamp, timestamp]),
                        img_metas=[dict(T_global=np.eye(4), T_global_inv=inv,
                                        scene_token=scene) for scene in scenes])
        initial = metas(0., ['a', 'b'])
        feature, anchor, history_feature, history_anchor, _ = bank.get(2, initial)
        self.assertEqual(history_feature.shape, (2, 2, 4))
        self.assertFalse(bank.mask.any())
        anchor[:, :, 8] = 2.  # velocity x=2m/s
        bank.cache(torch.ones_like(feature), anchor, torch.ones(2, 4, 3), initial)
        current = metas(.5, ['a', 'c'], translation=-.25)
        _, _, cached_feature, cached_anchor, interval = bank.get(2, current)
        self.assertEqual(bank.mask.tolist(), [True, False])
        torch.testing.assert_close(cached_anchor[0, :, 0], torch.full((2,), .75))
        self.assertEqual(torch.count_nonzero(cached_feature[1]).item(), 0)
        self.assertEqual(torch.count_nonzero(cached_anchor[1]).item(), 0)
        torch.testing.assert_close(interval, torch.tensor([.5, .5]))

    def test_occ_loss_empty_mask_is_finite_and_zero(self):
        branch = FlashOccBranch(
            grid_config=dict(x=[-1, 1, 1], y=[-1, 1, 1], z=[0, 2, 2], depth=[1, 2, 1]))
        logits = torch.randn(1, 2, 2, 16, 18, requires_grad=True)
        labels = torch.full((1, 2, 2, 16), 255)
        loss = branch.loss(logits, labels, torch.zeros_like(labels))['loss_occ']
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == '__main__':
    unittest.main()
