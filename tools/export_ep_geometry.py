"""Export per-frame runtime geometry from the exact validation preprocessing pipeline."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from mmcv import Config
from mmdet.datasets import build_dataset
from tools.ep1_common import sample_to_device, occ_forward_projection
from tqdm import tqdm


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--output',default='val/geometry')
    p.add_argument('--data-root',default='data/nuscenes')
    p.add_argument('--max-frames',type=int)
    args=p.parse_args();torch.set_num_threads(2)
    cfg=Config.fromfile(args.config);dc=cfg.data.val.copy()
    dc.data_root=args.data_root.rstrip('/')+'/';dc.occ_root=dc.data_root+'gts'
    ds=build_dataset(dc);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    count=min(len(ds),args.max_frames) if args.max_frames else len(ds)
    rows=[];previous=None
    for i in tqdm(range(count),desc='Export geometry'):
        data=sample_to_device(ds,i,-1);meta=data['img_metas'][0];info=ds.data_infos[i]
        timestamp=float(meta['timestamp']);scene=meta['scene_token']
        dt=timestamp-previous[1] if previous else .5
        valid=previous is not None and previous[0]==scene and np.isfinite(dt) and 0<dt<=2
        transform=meta['T_global_inv']@previous[2] if valid else np.eye(4)
        file=info['token']+'.npz'
        np.savez(out/file,projection_mat=data['projection_mat'].numpy(),occ_img2ego=data['occ_img2ego'].numpy(),
            occ_ego2img=occ_forward_projection(data['occ_img2ego']).numpy(),
            image_wh=data['image_wh'].numpy(),history_transform=np.asarray(transform,np.float32)[None],
            time_interval=np.array([dt if valid else .5],np.float32),history_valid=np.array([float(valid)],np.float32))
        rows.append(dict(index=i,token=info['token'],scene_token=scene,timestamp=timestamp,file=file))
        previous=(scene,timestamp,np.asarray(meta['T_global']))
    (out/'manifest.json').write_text(json.dumps(dict(config=args.config,frames=rows,
        scope='Geometry only; ORTM also needs processed images and its own sequential history outputs.'),indent=2))


if __name__=='__main__':main()
