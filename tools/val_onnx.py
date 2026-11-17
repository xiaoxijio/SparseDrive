"""Evaluate ONNX detection mAP/NDS and camera-visible Occ3D mIoU."""
from pathlib import Path
import numpy as np
import mmcv
from tqdm import tqdm
from tools.onnx_stage1 import arguments, Runner, write_json
from projects.mmdet3d_plugin.datasets.ep1 import crop_occ

"""
PYTHONPATH=. python tools/val_onnx.py \
  --onnx onnx/s1.onnx \
  --output val/s1
"""

  
def main():
    p=arguments(__doc__);p.add_argument('--output',dest='out_dir',default='val/onnx');args=p.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    runner=Runner(args);ds=runner.dataset;results=[];confusion=np.zeros((18,18),np.int64)
    count=min(args.max_frames,len(ds)) if args.max_frames else len(ds)
    if count<1: p.error('Need at least one frame')
    occ_dir=out/'occ';occ_dir.mkdir(exist_ok=True)
    for i in tqdm(range(count),desc='ONNX validation'):
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
    write_json(out/'metrics.json',metrics);write_json(out/'run.json',runner.provenance())
    print(metrics)


if __name__=='__main__': main()
