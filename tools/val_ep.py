"""Evaluate .onnx or .pth detection mAP/NDS and camera-visible Occ3D mIoU."""
from pathlib import Path
import numpy as np
import mmcv
from tqdm import tqdm
import argparse
import json
import time
import torch
from tools.ep1_common import load_model, sample_to_device
from mmdet.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.ep1 import crop_occ

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=lambda x:float(x))+'\n')


class PthRunner:
    def __init__(self, args):
        torch.set_num_threads(args.threads)
        cuda = args.device != 'cpu' and torch.cuda.is_available()
        if args.device == 'cuda' and not cuda:
            raise RuntimeError('CUDA requested but unavailable')
        self.device = 0 if cuda else -1
        checkpoint = torch.load(args.model, map_location='cpu')
        qat_meta = checkpoint.get('meta', {}).get('oritek_qat')
        self.qat_meta = qat_meta
        if qat_meta:
            if not cuda:
                raise RuntimeError('Native QAT validation requires CUDA/FlashAttention')
            from mmcv import Config
            from mmdet.models import build_detector
            from projects.mmdet3d_plugin.apis.ep2_qat import restore_qat_for_eval
            self.cfg = Config.fromfile(args.config)
            import logging
            old_level = logging.root.manager.disable
            logging.disable(logging.INFO)
            try:
                self.model = build_detector(self.cfg.model)
                self.model.init_weights()
            finally:
                logging.disable(old_level)
            self.model = restore_qat_for_eval(self.model, checkpoint, self.cfg.qat).cuda()
            # Moving to CUDA preserves trained buffers; observers stay frozen.
            print('QAT checkpoint restored strictly: SDK', qat_meta['sdk_version'],
                  'trained fake quant enabled; observers frozen', flush=True)
        else:
            if any('quantizer' in k for k in checkpoint.get('state_dict', checkpoint)):
                raise RuntimeError('Quantized checkpoint lacks required QAT metadata')
            self.cfg, self.model = load_model(args.config, args.model,
                device='cuda:0' if cuda else 'cpu', export_checkpoint=True)
        del checkpoint
        cfg = self.cfg.data.val.copy()
        cfg.data_root = args.data_root.rstrip('/') + '/'
        cfg.occ_root = cfg.data_root + 'gts'
        self.dataset = build_dataset(cfg)
        self.model.eval()
        self.model.head.det_head.instance_bank.reset()
        self.frames = 0
        self.seconds = 0.
        print('PTH: native simple_test, per-frame projection_mat/occ_img2ego; device:',
              'cuda:0' if cuda else 'cpu', flush=True)

    def infer(self, index):
        data = sample_to_device(self.dataset, index, self.device)
        started = time.perf_counter()
        scales = {k:v.detach().clone() for k,v in self.model.state_dict().items()
                  if self.qat_meta and k.endswith(('.scale','.zero_point'))} if index==0 else {}
        with torch.no_grad():
            pred = self.model(**data)[0]['img_bbox']
        for k,v in scales.items():
            torch.testing.assert_close(self.model.state_dict()[k], v, rtol=0, atol=0)
        pred = {k:(v.detach().cpu() if torch.is_tensor(v) else v) for k,v in pred.items()}
        self.seconds += time.perf_counter()-started
        self.frames += 1
        return pred

    def provenance(self):
        return dict(scope='Native PTH inference with per-frame dataset calibration; no fixed-rig adapter.',
                    frames=self.frames, device=str(next(self.model.parameters()).device),
                    qat=self.qat_meta, observers_frozen=bool(self.qat_meta),
                    inference_host_seconds=self.seconds)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', help='ONNX model or PyTorch checkpoint; suffix selects backend')
    p.add_argument('--config', required=True, help='Matching architecture config (S1-S5 or Q1/Q2)')
    p.add_argument('--output', dest='out_dir', help='Default: val/<model stem>_<backend>')
    p.add_argument('--data-root', default='data/nuscenes')
    p.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--max-frames', type=int, help='Smoke test only; skips official detection metrics')
    args=p.parse_args()
    suffix=Path(args.model).suffix.lower()
    if suffix not in ('.onnx','.pth'): p.error('Expected .onnx or .pth')
    if not Path(args.model).is_file(): p.error('Model file does not exist')
    if args.threads<1 or (args.max_frames is not None and args.max_frames<1): p.error('threads/max-frames must be positive')
    backend=suffix[1:]
    if args.out_dir is None: args.out_dir='val/'+Path(args.model).stem+'_'+backend
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    if backend=='onnx':
        from tools.onnx_stage1 import Runner
        args.onnx=args.model
        runner=Runner(args)
    else:
        runner=PthRunner(args)
    ds=runner.dataset;results=[];confusion=np.zeros((18,18),np.int64)
    count=min(args.max_frames,len(ds)) if args.max_frames else len(ds)
    if count<1: p.error('Need at least one frame')
    occ_dir=out/'occ';occ_dir.mkdir(exist_ok=True)
    for i in tqdm(range(count),desc=backend.upper()+' validation'):
        pred=runner.infer(i)
        with np.load(ds.get_occ_path(i)) as label:
            target=crop_occ(label['semantics'],ds.occ_source_grid,ds.occ_grid)
            mask=crop_occ(label['mask_camera'],ds.occ_source_grid,ds.occ_grid).astype(bool)&(target!=255)
        confusion+=np.bincount(target[mask].astype(np.int64)*18+pred['occ_pred'][mask],minlength=324).reshape(18,18)
        np.save(occ_dir/(ds.data_infos[i]['token']+'.npy'),pred.pop('occ_pred'))
        results.append(dict(img_bbox=pred))
    mmcv.dump(results,str(out/'detections.pkl'))
    union=confusion.sum(0)+confusion.sum(1)-np.diag(confusion)
    iou=np.divide(np.diag(confusion),union,out=np.full(18,np.nan),where=union>0)
    metrics={'occ/mIoU':float(np.nanmean(iou[:17])),**{'occ/IoU_'+str(i):None if np.isnan(v) else float(v) for i,v in enumerate(iou)}}
    np.save(out/'occ_confusion.npy',confusion)
    if count==len(ds):
        ds.work_dir=str(out)
        mode=dict(runner.cfg.evaluation.eval_mode);mode.update(with_occ=False,with_det=True,with_tracking=False,with_map=False)
        metrics.update(ds.evaluate(results,mode))
    else: metrics['detection_status']='Skipped: partial dataset cannot yield official nuScenes mAP/NDS'
    write_json(out/'metrics.json',metrics)
    report=runner.provenance()
    report.update(model=str(Path(args.model).resolve()),config=str(Path(args.config).resolve()),backend=backend)
    write_json(out/'run.json',report)
    print(metrics)


if __name__=='__main__': main()
