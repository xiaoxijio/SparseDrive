"""Original FlashAttention and shared-weight ONNX path regression checks."""
import unittest
from unittest.mock import patch
import torch
from projects.mmdet3d_plugin.models.attention import MultiheadFlashAttention
from projects.mmdet3d_plugin.models.ep1.temporal import standard_attention_export


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

    def test_export_switch_restores_kernel_after_exception(self):
        class Wrapper(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.attention = module

            @standard_attention_export
            def forward(self):
                assert self.attention.attn.export_mode
                raise RuntimeError('intentional')
        wrapper = Wrapper(self.module)
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            wrapper()
        self.assertFalse(self.module.attn.export_mode)


if __name__ == '__main__':
    unittest.main()
