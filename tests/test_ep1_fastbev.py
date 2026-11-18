"""Official-style voxel overwrite, gradients, and dynamic ONNX regression."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from projects.mmdet3d_plugin.models.ep1.fastbev import FastBEVViewTransform, affine_inverse

class FastBEVChecks(unittest.TestCase):
    def test_reference_gradient_and_dynamic_onnx(self):
        torch.set_num_threads(4)
        grid=dict(x=[0,4,1],y=[0,4,1],z=[1,3,1])
        m=FastBEVViewTransform(3,2,grid,(8,8))
        x=torch.randn(2*3,3,4,4,requires_grad=True)
        a=torch.eye(4)[None,None].repeat(2,3,1,1)
        a[:,:,0,0]=0.5; a[:,:,1,1]=0.5
        torch.testing.assert_close(affine_inverse(a), torch.linalg.inv(a)[...,:3,:])
        y=m(x,a)
        # Independent official-style indexed overwrite reference.
        f=m.context_net(x).reshape(2,3,2,4,4)
        projection=torch.linalg.inv(a)[...,:3,:]
        vol=[]
        for zi in range(m.nz):
            p=m.voxel_points[zi].reshape(-1,4).T
            xyz=projection@p
            u=(xyz[:,:,0]/xyz[:,:,2]*0.5).round().long()
            v=(xyz[:,:,1]/xyz[:,:,2]*0.5).round().long()
            valid=(xyz[:,:,2]>0)&(u>=0)&(u<4)&(v>=0)&(v<4)
            out=torch.zeros(2,2,16)
            for bi in range(2):
                for cam in range(3):
                    keep=valid[bi,cam]
                    out[bi,:,keep]=f[bi,cam,:,v[bi,cam,keep],u[bi,cam,keep]]
            vol.append(out.reshape(2,2,4,4))
        ref=torch.stack(vol).mean(0)
        torch.testing.assert_close(y,ref,atol=1e-6,rtol=1e-6)
        y.square().mean().backward()
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
        # Entire view transformer exports without scatter or matrix inverse.
        import onnx, onnxruntime as ort
        m.eval()
        temporary = tempfile.TemporaryDirectory(prefix='s6_fastbev_')
        self.addCleanup(temporary.cleanup)
        name = str(Path(temporary.name) / 'smoke.onnx')
        torch.onnx.export(m,(x.detach(),a),name,input_names=['feature','occ_img2ego'],output_names=['bev'],opset_version=16)
        model=onnx.load(name); onnx.checker.check_model(model)
        ops=sorted(set(n.op_type for n in model.graph.node))
        assert not any('Scatter' in op or 'Inverse' in op for op in ops)
        sess=ort.InferenceSession(name,providers=['CPUExecutionProvider'])
        for matrix in (a, a.clone()):
            if matrix is not a: matrix[:,:,0,3]=0.37
            actual=sess.run(None,dict(feature=x.detach().numpy(),occ_img2ego=matrix.numpy()))[0]
            expected=m(x.detach(),matrix).detach().numpy()
            np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-5)

if __name__ == "__main__":
    unittest.main()
