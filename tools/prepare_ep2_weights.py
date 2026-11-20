"""Prepare strict FP initialization before QAT conversion, composing trained EP1 branches."""
import argparse,json,logging,copy,sys
from pathlib import Path
import torch
from mmcv import Config
from mmdet.models import build_detector
import projects.mmdet3d_plugin
p=argparse.ArgumentParser(description=__doc__);p.add_argument('config');a=p.parse_args()
logging.disable(logging.INFO);torch.set_num_threads(1);torch.manual_seed(0)
c=Config.fromfile(a.config);out=Path(c.get('qat_init_from') or c.load_from)
if out.exists():raise FileExistsError(str(out))
if c.get('qat_init_from'):
    from projects.mmdet3d_plugin.apis.ep2_qat import validate_recipe
    spec=c.qat_initialization
    checkpoint=torch.load(spec.source,map_location='cpu')
    validate_recipe(checkpoint,c.qat)
    state={k[7:] if k.startswith('module.') else k:v for k,v in checkpoint['state_dict'].items()}
    ffn_checkpoint=torch.load(spec.ffn_source,map_location='cpu')
    ffn_state={k[7:] if k.startswith('module.') else k:v for k,v in ffn_checkpoint['state_dict'].items()}
    m=build_detector(c.model)
    prefixes=[n+'.' for n,v in m.named_modules() if type(v).__name__=='AsymmetricFFN']
    assert len(prefixes)==4, prefixes
    replaced=[];reshaped=[]
    for k,v in m.state_dict().items():
        if not any(k.startswith(prefix) for prefix in prefixes):continue
        if k not in ffn_state or ffn_state[k].shape!=v.shape:
            raise ValueError('S3 FFN mismatch: '+k)
        if k not in state:raise ValueError('Missing Q3 FFN key: '+k)
        if state[k].shape!=v.shape:reshaped.append(k)
        state[k]=ffn_state[k].clone();replaced.append(k)
    assert replaced and len(reshaped)==12, reshaped
    # Preserve all learned Q3 quantization state; no optimizer or epoch copied.
    metadata=copy.deepcopy(checkpoint.get('meta',{}))
    metadata={k:v for k,v in metadata.items() if k not in ('epoch','iter','config')}
    metadata['config']=c.pretty_text
    metadata['ep2_qat_initialization']=dict(source=spec.source,ffn_source=spec.ffn_source,
        replaced_ffn_tensors=replaced,reshaped_ffn_tensors=reshaped,
        note='Q3 trained QAT state + trained S3 FFN512; optimizer and iteration reset')
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state_dict=state,meta=metadata),out)
    print('Q4 QAT initialization:',out,'FFN tensors replaced:',len(replaced),
          'reshaped:',len(reshaped),'quantizer scale/zero_point buffers:',
          sum(k.endswith(('.scale','.zero_point')) for k in state),flush=True)
    sys.exit(0)
sources={name:'work_dirs/ep1/'+variant+'/iter_8780.pth' for name,variant in [('s3','s3_det_occ_decoder4_ffn512'),('s4','s4_det_occ_decoder4_occsmall'),('s5','s5_det_occ_decoder4_fpn3')]}
q2=c.model.head.det_head.ffn.feedforward_channels==512
states={name:torch.load(path,map_location='cpu')['state_dict'] for name,path in sources.items() if name!='s3' or q2}
m=build_detector(c.model);m.init_weights();target=m.state_dict();counts={};fresh=[]
for k in target:
 branch='s5'
 if k.startswith('occ_branch.bev_encoder.'):branch='s4'
 # FFN layer indices are unchanged by S3/S5; identify parameters from actual modules.
 if q2:
  for name,module in m.named_modules():
   if type(module).__name__=='AsymmetricFFN' and k.startswith(name+'.'):branch='s3';break
 src=states[branch]
 if k in src and src[k].shape==target[k].shape:target[k]=src[k];counts[branch]=counts.get(branch,0)+1
 else:fresh.append(k)
if fresh:raise RuntimeError('Unexpected uninitialized tensors: '+str(fresh))
m.load_state_dict(target,strict=True)
r=dict(config=a.config,sources={k:v for k,v in sources.items() if k in counts},tensors_by_source=counts,fresh_tensors=fresh,parameters=sum(v.numel() for v in m.parameters()),note='Composed trained branches need joint fine-tuning; optimizer not copied. Not a trained combined model.')
out.parent.mkdir(exist_ok=True);torch.save(dict(state_dict=target,meta=dict(ep2_fp_initialization=r)),out);out.with_suffix('.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
