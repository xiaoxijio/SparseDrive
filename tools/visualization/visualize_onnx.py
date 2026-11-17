"""Six detection cameras above detection BEV and OCC ground truth/prediction."""
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from tools.onnx_stage1 import arguments, Runner, write_json
from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from projects.mmdet3d_plugin.datasets.ep1 import crop_occ

COLORS=np.array([[0,0,0],[255,120,50],[255,190,0],[255,255,0],[120,180,255],
 [180,0,255],[255,60,80],[255,0,0],[255,220,150],[200,200,200],[160,160,160],
 [180,0,75],[100,100,255],[100,100,100],[0,255,255],[0,175,0],[0,80,0],[245,245,245]],np.uint8)
EDGES=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def title(image,text):
    cv2.rectangle(image,(0,0),(image.shape[1],32),(30,30,30),-1)
    cv2.putText(image,text,(10,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
    return image


def occ_panel(voxels,mask,text,size=600):
    valid=(voxels>=0)&(voxels<17)&mask
    heights=np.where(valid,np.arange(voxels.shape[2])[None,None,:],-1)
    top=heights.argmax(2);labels=np.take_along_axis(voxels,top[...,None],2)[...,0].copy()
    labels[~valid.any(2)]=17
    # Ego x forward is up; ego y left is left. Arrays are indexed x,y,z.
    rgb=COLORS[labels][::-1,::-1]
    panel=cv2.resize(rgb[...,::-1],(size,size),interpolation=cv2.INTER_NEAREST)
    return title(panel,text+' (top occupied voxel, camera mask)')


def combined(ds,index,pred,threshold):
    info=ds.get_data_info(index);raw=ds.data_infos[index]
    boxes=pred['boxes_3d'].numpy();scores=pred['scores_3d'].numpy();labels=pred['labels_3d'].numpy()
    keep=scores>=threshold;boxes=boxes[keep];labels=labels[keep]
    corners=box3d_to_corners(boxes) if len(boxes) else np.empty((0,8,3))
    cameras=[]
    for name in ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']:
        camidx=list(raw['cams']).index(name);image=cv2.imread(info['img_filename'][camidx])
        if image is None:raise FileNotFoundError(info['img_filename'][camidx])
        for pts,label in zip(corners,labels):
            projected=np.c_[pts,np.ones(8)]@np.asarray(info['lidar2img'][camidx]).T
            z=projected[:,2];xy=projected[:,:2]/np.maximum(z[:,None],1e-6)
            for a,b in EDGES:
                if z[a]>.1 and z[b]>.1:
                    q=np.clip(xy[[a,b]],-100000,100000).astype(int)
                    cv2.line(image,tuple(q[0]),tuple(q[1]),tuple(int(v) for v in COLORS[(int(label)%16)+1][::-1]),3)
        cameras.append(title(cv2.resize(image,(600,338)),name+' prediction'))
    top=np.vstack([np.hstack(cameras[:3]),np.hstack(cameras[3:])])
    bev=np.full((600,600,3),245,np.uint8)
    def point(xy):return np.rint(np.stack([300-xy[...,1]*5,300-xy[...,0]*5],-1)).astype(int)
    for radius in [10,20,30,40,50]:
        cv2.circle(bev,(300,300),radius*5,(190,190,190),1)
        cv2.putText(bev,str(radius)+'m',(302,300-radius*5),cv2.FONT_HERSHEY_SIMPLEX,.4,(100,100,100),1)
    gt=info['gt_bboxes_3d'];gt=gt.numpy() if hasattr(gt,'numpy') else np.asarray(gt)
    for source,color in [(gt,(90,150,90)),(boxes,(0,0,220))]:
        if len(source):
            for pts in box3d_to_corners(source):cv2.polylines(bev,[point(pts[[0,3,7,4],:2])],True,color,2)
    cv2.arrowedLine(bev,(300,310),(300,285),(0,0,0),3)
    title(bev,'Detection BEV: green GT / red prediction (lidar)')
    with np.load(ds.get_occ_path(index)) as f:
        target=crop_occ(f['semantics'],ds.occ_source_grid,ds.occ_grid)
        mask=crop_occ(f['mask_camera'],ds.occ_source_grid,ds.occ_grid).astype(bool)
    bottom=np.hstack([bev,occ_panel(target,mask,'OCC ground truth (ego)'),occ_panel(pred['occ_pred'],mask,'OCC prediction (ego)')])
    legend=np.full((80,1800,3),245,np.uint8)
    names=['others','barrier','bicycle','bus','car','construction','motorcycle','pedestrian','traffic cone','trailer','truck','driveable','other flat','sidewalk','terrain','manmade','vegetation','free/unseen']
    for i,name in enumerate(names):
        x=(i%9)*200;y=(i//9)*40
        cv2.rectangle(legend,(x+5,y+10),(x+23,y+28),tuple(int(v) for v in COLORS[i][::-1]),-1)
        cv2.putText(legend,name,(x+28,y+25),cv2.FONT_HERSHEY_SIMPLEX,.43,(30,30,30),1)
    return np.vstack([top,bottom,legend])


def main():
    p=arguments(__doc__);p.add_argument('--output','--out-dir',dest='out_dir',default='vis/onnx')
    p.add_argument('--num-scenes',type=int,default=0);p.add_argument('--interval',type=int,default=1)
    p.add_argument('--score-threshold',type=float,default=.3);args=p.parse_args()
    if args.interval<1 or args.num_scenes<0:p.error('interval >= 1; num-scenes >= 0 (0=all)')
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    runner=Runner(args);scenes={};local_counts={};count=0
    for i,info in enumerate(tqdm(runner.dataset.data_infos,desc='ONNX visualization')):
        token=info['scene_token']
        if token not in scenes:
            if args.num_scenes and len(scenes)>=args.num_scenes:break
            scenes[token]=len(scenes);local_counts[token]=0
        if args.max_frames and count>=args.max_frames:break
        pred=runner.infer(i);local=local_counts[token];local_counts[token]+=1;count+=1
        if local%args.interval:continue
        directory=out/('scene_%03d_%s'%(scenes[token],token))/'combine';directory.mkdir(parents=True,exist_ok=True)
        path=directory/('%04d.jpg'%i)
        if not cv2.imwrite(str(path),combined(runner.dataset,i,pred,args.score_threshold)):raise IOError(str(path))
    write_json(out/'run.json',runner.provenance())


if __name__=='__main__':main()
