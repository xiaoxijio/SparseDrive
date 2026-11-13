"""SDK backbone initialization and official BEVPool output/gradient checks."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from projects.mmdet3d_plugin.models.ep1.backbone import EfficientNetB0NoSE
from projects.mmdet3d_plugin.models.ep1.flashocc import LSSViewTransform
from tools.prepare_ep1_backbone import extract_backbone


class SDKChecks(unittest.TestCase):
    def test_strict_backbone_transfer_preserves_bn_and_four_pyramid_levels(self):
        source = EfficientNetB0NoSE()
        state = {'module.img_backbone.' + k: v for k, v in source.state_dict().items()}
        extracted = extract_backbone(dict(state_dict=state), source.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'backbone.pth'
            torch.save(dict(state_dict=extracted), filename)
            model = EfficientNetB0NoSE(pretrained=str(filename), norm_eval=True)
            model.init_weights()
        for k, v in source.state_dict().items():
            torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
        model.train()
        self.assertTrue(model.conv_stem.weight.requires_grad)
        self.assertTrue(all(not m.training for m in model.modules()
                            if isinstance(m, nn.BatchNorm2d)))
        self.assertFalse(any(isinstance(m, nn.SiLU) for m in model.modules()))
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 64, 96))
        self.assertEqual([tuple(x.shape) for x in outputs],
                         [(2, 24, 16, 24), (2, 40, 8, 12),
                          (2, 112, 4, 6), (2, 1280, 2, 3)])
        bad = dict(state)
        bad.pop('module.img_backbone.conv_stem.weight')
        with self.assertRaises(ValueError):
            extract_backbone(dict(state_dict=bad), source.state_dict())
        bad = dict(state)
        bad['module.img_backbone.conv_stem.weight'] = torch.zeros(1)
        with self.assertRaises(ValueError):
            extract_backbone(dict(state_dict=bad), source.state_dict())

    def test_export_comparison_preserves_joint_cache_correspondence(self):
        import numpy as np
        from tools.export_ep1_onnx import align_cache_rows
        rng = np.random.RandomState(3)
        expected = [np.zeros(1) for _ in range(4)] + [
            rng.randn(1, 3, 4), rng.randn(1, 3, 11), rng.rand(1, 3)]
        actual = expected[:4] + [x[:, [2, 0, 1]].copy() for x in expected[4:]]
        for a, b in zip(expected, align_cache_rows(expected, actual)):
            np.testing.assert_array_equal(a, b)
        # Features cannot be matched independently from their anchors/confidence.
        actual[4] = actual[4][:, [1, 2, 0]].copy()
        self.assertFalse(np.allclose(expected[4], align_cache_rows(expected, actual)[4]))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA BEVPool comparison')
    def test_official_bev_pool_matches_reference_outputs_and_both_gradients(self):
        torch.manual_seed(7)
        fast = LSSViewTransform(3, 4,
            dict(x=[-2, 2, 1], y=[-2, 2, 1], z=[0, 4, 4], depth=[1, 4, 1]),
            input_size=(32, 32)).cuda()
        reference = copy.deepcopy(fast)
        reference.use_bev_pool = False
        feature = torch.randn(6, 3, 2, 2, device='cuda', requires_grad=True)
        other = feature.detach().clone().requires_grad_()
        transform = torch.eye(4, device='cuda')[None, None].repeat(2, 3, 1, 1)
        transform[..., 0, 0] = transform[..., 1, 1] = .02
        transform[:, 1, 0, 3] = -.5
        transform[1, 2, 1, 3] = -.5
        from projects.mmdet3d_plugin.ops import bev_pool_v2 as kernel
        with patch.object(kernel, 'bev_pool_v2', wraps=kernel.bev_pool_v2) as call:
            actual = fast(feature, transform)
            self.assertEqual(call.call_count, 1)
        expected = reference(other, transform)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        weights = torch.randn_like(expected)
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        torch.testing.assert_close(feature.grad, other.grad, rtol=2e-5, atol=2e-5)
        for (name, a), (_, b) in zip(fast.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-5, msg=name)
        fast.zero_grad()
        empty_transform = transform.clone()
        empty_transform[..., 0, 3] = 100
        empty = fast(feature.detach().requires_grad_(), empty_transform)
        self.assertEqual(empty.count_nonzero().item(), 0)
        empty.sum().backward()
        self.assertTrue(torch.isfinite(fast.depth_net.weight.grad).all())


@unittest.skipUnless(torch.cuda.is_available(), 'Original FlashAttention requires CUDA')
class TemporalSemanticsChecks(unittest.TestCase):
    def test_first_history_and_mixed_batches_match_original_head(self):
        from projects.mmdet3d_plugin.models.ep1.temporal import PaddedSparse4DHead
        from projects.mmdet3d_plugin.models.detection3d.detection3d_head import Sparse4DHead
        from projects.mmdet3d_plugin.models.attention import MultiheadFlashAttention
        torch.manual_seed(11)
        head = PaddedSparse4DHead.__new__(PaddedSparse4DHead)
        nn.Module.__init__(head)
        head.operation_order = ['temp_gnn']
        head.decouple_attn = True
        head.fc_before = nn.Linear(32, 64)
        head.fc_after = nn.Linear(64, 32)
        head.layers = nn.ModuleList([MultiheadFlashAttention(64, 4, dropout=0.)])
        head.instance_bank = nn.Module()
        head.cuda().eval()
        q, qp = [torch.randn(2, 9, 32, device='cuda') for _ in range(2)]
        k, kp = [torch.randn(2, 6, 32, device='cuda') for _ in range(2)]
        with torch.no_grad():
            # Both the ordinary GPU path and single-graph selection must match.
            for export in (False, True):
                head.export_mode = export
                head.layers[0].attn.export_mode = export
                current = Sparse4DHead.graph_model(head, 0, q, query_pos=qp)
                history = Sparse4DHead.graph_model(head, 0, q, k, k,
                                                    query_pos=qp, key_pos=kp)
                for flags in ([False, False], [True, True], [True, False]):
                    head.instance_bank.mask = torch.tensor(flags, device='cuda')
                    actual = head.graph_model(0, q, k, k, query_pos=qp, key_pos=kp)
                    expected = torch.where(head.instance_bank.mask[:, None, None],
                                           history, current)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
