"""Real-data forward/backward/update and SDK scale persistence smoke check.
Does not run a training schedule. Use --config projects/configs/ep2/q1.py.
"""
import argparse,logging,json,tempfile
from pathlib import Path
import torch
from mmcv import Config
from mmcv.parallel import collate,scatter
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmcv.runner import build_optimizer
import projects.mmdet3d_plugin
from projects.mmdet3d_plugin.apis.ep2_qat import prepare_training,freeze_bn,qat_tool
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args()
logging.disable(logging.INFO);torch.set_num_threads(1);torch.manual_seed(0)
c=Config.fromfile(a.config);c.gpu_ids=[0];c.seed=0
m=build_detector(c.model);m.init_weights();ds=build_dataset(c.data.train)
class ExampleLoader:
 dataset=ds
 collate_fn=staticmethod(lambda rows:collate(rows,samples_per_gpu=1))
m=prepare_training(m,ExampleLoader(),c,logging.getLogger('mmdet'));m.train();freeze_bn(m)
batch=scatter(collate([ds[0]],samples_per_gpu=1),[0])[0]
optimizer=build_optimizer(m,c.optimizer);optimizer.zero_grad()
with torch.cuda.amp.autocast(enabled=False):
 losses=m(**batch);loss,logs=m._parse_losses(losses)
assert torch.isfinite(loss),logs
scaler=torch.cuda.amp.GradScaler(enabled=False);scaler.scale(loss).backward();scaler.unscale_(optimizer)
grads=[p.grad for p in m.parameters() if p.grad is not None]
bad=[n for n,p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
print('LOSS',logs,'NONFINITE_GRADS',bad[:20],flush=True)
assert grads and not bad
assert any(torch.count_nonzero(g).item() for g in grads)
torch.nn.utils.clip_grad_norm_(m.parameters(),25);scaler.step(optimizer);scaler.update();optimizer.zero_grad()
m.eval();qat_tool().disable_observer(m)
state=m.state_dict();scales={k:v for k,v in state.items() if k.endswith(('.scale','.zero_point'))};assert scales
with tempfile.TemporaryDirectory(prefix='ep2_qat_check_') as tmp:
 path=Path(tmp)/'checkpoint.pth';torch.save(dict(state_dict=state),path)
 loaded=torch.load(path,map_location='cpu')['state_dict'];m.load_state_dict(loaded,strict=True);m.cuda()
 for k,v in scales.items():torch.testing.assert_close(m.state_dict()[k].cpu(),v.cpu(),rtol=0,atol=0)
# AMP training retains native CUDA aggregation and FlashAttention.
assert m.use_deformable_func and c.fp16 is None
out=Path(c.work_dir);out.mkdir(parents=True,exist_ok=True)
r=dict(config=a.config,loss=float(loss.detach()),gradient_tensors=len(grads),quantizer_scale_buffers=len(scales),gpu_peak_bytes=torch.cuda.max_memory_allocated(),state_reload_strict=True,qat_sdk='2.1.6',training_run=False)
(out/'qat_smoke.json').write_text(json.dumps(r,indent=2));print('QAT_SMOKE_OK',json.dumps(r),flush=True)
