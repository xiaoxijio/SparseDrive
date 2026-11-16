"""Export-only rewrites preserve attention, pooling and border semantics."""
import unittest
import torch
from torch import nn
from projects.mmdet3d_plugin.models.ep1.deployment import (
    single_temporal_attention, numeric_boundary_sampling, rank_six_fusion,
    FixedRigPool, history_is_valid, CachedInitialEncoder)
from projects.mmdet3d_plugin.models.ep1.flashocc import LSSViewTransform
from projects.mmdet3d_plugin.models.blocks import DeformableFeatureAggregation
from projects.mmdet3d_plugin.models.ep1.temporal import PaddedSparse4DHead
from projects.mmdet3d_plugin.models.attention import MultiheadFlashAttention


class DeploymentChecks(unittest.TestCase):
    def test_single_attention_matches_two_path_outputs_and_gradients(self):
        torch.manual_seed(4)
        head = PaddedSparse4DHead.__new__(PaddedSparse4DHead)
        nn.Module.__init__(head)
        head.operation_order = ['temp_gnn']; head.decouple_attn = True
        head.fc_before = nn.Linear(32,64); head.fc_after = nn.Linear(64,32)
        head.layers = nn.ModuleList([MultiheadFlashAttention(64,4,dropout=0.)])
        head.instance_bank = nn.Module(); head.export_mode = True
        head.layers[0].attn.export_mode = True; head.eval()
        q = torch.randn(2,9,32,requires_grad=True); qp = torch.randn_like(q)
        k = torch.randn(2,6,32,requires_grad=True); kp = torch.randn_like(k)
        for valid in ([False,False],[True,True],[True,False]):
            head.instance_bank.mask = torch.tensor(valid)
            expected = head.graph_model(0,q,k,k,query_pos=qp,key_pos=kp)
            actual = single_temporal_attention(head,0,q,k,k,query_pos=qp,key_pos=kp)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
            variables = [q,k,head.layers[0].attn.in_proj_weight]
            a = torch.autograd.grad(actual.square().sum(),variables,retain_graph=True)
            b = torch.autograd.grad(expected.square().sum(),variables,retain_graph=True)
            for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-5)

    def test_fixed_pool_matches_scatter_with_duplicate_rays_and_gradients(self):
        torch.manual_seed(1)
        original = LSSViewTransform(3,4,
            dict(x=[-2,2,1],y=[-2,2,1],z=[0,4,4],depth=[1,4,1]),input_size=(32,32))
        matrix = torch.eye(4)[None,None].repeat(1,3,1,1)
        matrix[...,0,0] = matrix[...,1,1] = .02
        matrix[:,1,0,3] = -.5
        fixed = FixedRigPool(original,matrix)
        feature = torch.randn(3,3,2,2,requires_grad=True)
        expected = original(feature,matrix);actual = fixed(feature,matrix)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        variables = [feature,original.depth_net.weight]
        a = torch.autograd.grad(actual.square().sum(),variables,retain_graph=True)
        b = torch.autograd.grad(expected.square().sum(),variables)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-5)
        self.assertEqual(len(fixed.voxel_indices.unique()),len(fixed.voxel_indices))

    def test_numeric_border_and_rank_six_fusion_match_original(self):
        torch.manual_seed(3)
        feature = [torch.randn(2,2,8,n,n) for n in (8,4)]
        points = torch.tensor([[[[-1.,0.,1.],[0.,0.,1.],[.5,.5,1.],[1.,1.,1.],[2.,2.,1.]]]]).expand(2,-1,-1,-1)
        matrix = torch.eye(4)[None,None].repeat(2,2,1,1)
        wh = torch.ones(2,2,2)
        expected = DeformableFeatureAggregation.feature_sampling(feature,points,matrix,wh,True)
        actual = numeric_boundary_sampling(feature,points,matrix,wh,True)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        class Fusion: num_groups=2;group_dims=4;embed_dims=8;num_pts=5
        weights = torch.randn(2,1,2,2,5,2)
        a = rank_six_fusion(Fusion(),actual,weights)
        b = DeformableFeatureAggregation.multi_view_level_fusion(Fusion(),expected,weights)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_cached_initial_anchor_only_replaces_first_call_each_frame(self):
        original = nn.Sequential(nn.Linear(11,32),nn.LayerNorm(32)).eval()
        anchor = torch.randn(1,9,11)
        cached = CachedInitialEncoder(original,anchor)
        torch.testing.assert_close(cached(anchor),original(anchor),rtol=0,atol=0)
        dynamic = torch.randn(1,6,11)
        torch.testing.assert_close(cached(dynamic),original(dynamic),rtol=0,atol=0)
        cached.first = True
        torch.testing.assert_close(cached(anchor),original(anchor),rtol=0,atol=0)

    def test_host_rejects_scene_switch_negative_and_long_intervals(self):
        self.assertTrue(history_is_valid(.7,'a','a'))
        for dt,scene in [(0.,'a'),(-.5,'a'),(2.1,'a'),(.5,'b'),(float('nan'),'a')]:
            self.assertFalse(history_is_valid(dt,'a',scene))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
