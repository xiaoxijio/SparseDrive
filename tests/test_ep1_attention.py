"""Original FlashAttention and shared-weight ONNX path regression checks."""
import unittest
from unittest.mock import patch
import torch
from projects.mmdet3d_plugin.models.attention import MultiheadFlashAttention
from projects.mmdet3d_plugin.models.blocks import DeformableFeatureAggregation
from projects.mmdet3d_plugin.ops import feature_maps_format
from projects.mmdet3d_plugin.models.ep1.temporal import standard_ops_export


@unittest.skipUnless(torch.cuda.is_available(), 'FlashAttention requires CUDA')
class AttentionChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.module = MultiheadFlashAttention(64, 4, dropout=0.).cuda().eval()

    def test_self_cross_attention_outputs_and_gradients(self):
        for cross in (False, True):
            with self.subTest(cross=cross):
                q = torch.randn(2, 11, 64, device='cuda', requires_grad=True)
                k = torch.randn(2, 7 if cross else 11, 64, device='cuda', requires_grad=True)
                v = torch.randn_like(k, requires_grad=True)
                qp, kp = torch.randn_like(q), torch.randn_like(k)
                with patch.object(self.module.attn.inner_attn, 'forward',
                                  wraps=self.module.attn.inner_attn.forward) as kernel:
                    fused = self.module(q, k, v, query_pos=qp, key_pos=kp)
                    self.assertEqual(kernel.call_count, 1)
                    grad_fused = torch.autograd.grad(fused.square().mean(),
                        [q, k, v, self.module.attn.in_proj_weight])
                    self.module.attn.export_mode = True
                    standard = self.module(q, k, v, query_pos=qp, key_pos=kp)
                    self.assertEqual(kernel.call_count, 1)
                    grad_standard = torch.autograd.grad(standard.square().mean(),
                        [q, k, v, self.module.attn.in_proj_weight])
                torch.testing.assert_close(fused, standard, rtol=3e-3, atol=2e-3)
                for actual, expected in zip(grad_fused, grad_standard):
                    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-5)
                self.module.attn.export_mode = False

    def test_old_ep1_weights_are_mapped_exactly(self):
        original = {k: v.clone() for k, v in self.module.state_dict().items()}
        legacy = {}
        for suffix in ("weight", "bias"):
            for name, tensor in zip(("q", "k", "v"), original["attn.in_proj_" + suffix].chunk(3)):
                legacy[name + "_proj." + suffix] = tensor.clone()
            legacy["out_proj." + suffix] = original["attn.out_proj." + suffix].clone()
        self.module.load_state_dict(legacy, strict=True)
        for key, expected in original.items():
            torch.testing.assert_close(self.module.state_dict()[key], expected, rtol=0, atol=0)

    def make_sampling(self):
        return DeformableFeatureAggregation(
            embed_dims=32, num_groups=4, num_levels=2, num_cams=2,
            attn_drop=0., use_deformable_func=True, match_cuda_sampling=True,
            kps_generator=dict(type="ExportableBoxKeyPoints", num_learnable_pts=2,
                               fix_scale=[[0, 0, 0]])).cuda().eval()

    def test_cuda_sampling_outputs_and_gradients(self):
        module = self.make_sampling()
        features = [torch.randn(4, 2, 32, n, n, device="cuda", requires_grad=True)
                    for n in (8, 4)]
        query = torch.randn(4, 5, 32, device="cuda", requires_grad=True)
        embedding = torch.randn_like(query)
        anchor = torch.zeros(4, 5, 11, device="cuda")
        # Cover image interior, boundaries, and points outside both sides.
        anchor[:, :, 0] = torch.tensor([0., 4., -4., 8., -8.], device="cuda")
        anchor[:, :, 2] = 5
        anchor[:, :, 7] = 1
        projection = torch.eye(4, device="cuda")[None, None].repeat(4, 2, 1, 1)
        projection[:, :, 0, 0] = projection[:, :, 1, 1] = 20
        projection[:, :, 0, 2] = projection[:, :, 1, 2] = 16
        meta = dict(projection_mat=projection,
                    image_wh=torch.full((4, 2, 2), 32., device="cuda"))
        fused = module(query, anchor, embedding, feature_maps_format(features), meta)
        fused_grads = torch.autograd.grad(fused.sum(), [query] + features)
        module.use_deformable_func = False
        reference = module(query, anchor, embedding, features, meta)
        reference_grads = torch.autograd.grad(reference.sum(), [query] + features)
        torch.testing.assert_close(fused, reference, rtol=1e-4, atol=1e-5)
        for actual, expected in zip(fused_grads, reference_grads):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_export_switch_restores_kernel_after_exception(self):
        class Wrapper(torch.nn.Module):
            def __init__(self, module, sampling):
                super().__init__()
                self.attention = module
                self.sampling = sampling

            @standard_ops_export
            def forward(self):
                assert self.attention.attn.export_mode
                assert not self.sampling.use_deformable_func
                raise RuntimeError('intentional')
        wrapper = Wrapper(self.module, self.make_sampling())
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            wrapper()
        self.assertFalse(self.module.attn.export_mode)
        self.assertTrue(wrapper.sampling.use_deformable_func)


if __name__ == '__main__':
    unittest.main()
