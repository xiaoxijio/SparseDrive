"""Sequential ONNX inference shared by validation and combined visualization."""
import argparse
import json
import time
import ctypes
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from mmcv import Config
from mmdet.datasets import build_dataset
from tools.ep1_common import sample_to_device
from projects.mmdet3d_plugin.models.detection3d.decoder import SparseBox3DDecoder
from projects.mmdet3d_plugin.models.ep1.deployment import history_is_valid


def arguments(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--onnx', default='onnx/s1.onnx')
    p.add_argument('--config', default='projects/configs/ep1/s1_det_occ_b0_nose.py')
    p.add_argument('--data-root', default='data/nuscenes')
    p.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--max-frames', type=int, help='Smoke test only; disables official detection evaluation')
    return p


class Runner:
    def __init__(self, args):
        if args.threads < 1 or (args.max_frames is not None and args.max_frames < 1):
            raise ValueError("threads and max-frames must be positive")
        torch.set_num_threads(args.threads)
        self.cfg = Config.fromfile(args.config)
        cfg = self.cfg.data.val.copy()
        cfg.data_root = args.data_root.rstrip('/') + '/'
        cfg.occ_root = cfg.data_root + 'gts'
        self.dataset = build_dataset(cfg)
        options = ort.SessionOptions(); options.intra_op_num_threads = args.threads
        use_cuda = args.device != 'cpu' and 'CUDAExecutionProvider' in ort.get_available_providers()
        if args.device == 'cuda' and not use_cuda:
            raise RuntimeError('CUDA provider unavailable; install onnxruntime-gpu')
        if use_cuda:
            # Use the existing PyTorch CUDA/cuDNN libraries with ORT CUDA 11.
            torch.cuda.init()
            libdir = Path(torch.__file__).parent/'lib'
            for name in ['libcudnn.so.8','libcublas.so.11','libcublasLt.so.11']:
                path = libdir/name
                if path.exists(): ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        providers = [('CUDAExecutionProvider', {'cudnn_conv_algo_search':'HEURISTIC'}) ,'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(args.onnx, options, providers=providers)
        if use_cuda and 'CUDAExecutionProvider' not in self.session.get_providers():
            raise RuntimeError('CUDA initialization failed; refusing silent CPU fallback')
        print('ONNX providers:',self.session.get_providers(),flush=True)
        self.timing = dict(preprocess=0.,inference=0.,decode=0.)
        self.inputs = self.session.get_inputs()
        self.decoder = SparseBox3DDecoder(**{k:v for k,v in self.cfg.model.head.det_head.decoder.items() if k != 'type'})
        self.previous = None; self.cache = None
        calibration = Path(args.onnx).with_suffix('.calibration.npz')
        if not calibration.exists():
            raise FileNotFoundError('Need matching export calibration: ' + str(calibration))
        with np.load(calibration) as f:
            self.rig = {k:f[k] for k in ['projection_mat','image_wh','occ_img2ego']}
        self.rig_errors = []
        print("Evaluating fixed-rig ONNX; sample-varying nuScenes calibration may affect metrics.", flush=True)

    def infer(self, index):
        started=time.perf_counter()
        data = sample_to_device(self.dataset, index, -1)
        meta = data['img_metas'][0]
        scene, timestamp = meta['scene_token'], float(meta['timestamp'])
        valid = self.previous is not None and history_is_valid(timestamp-self.previous[1], self.previous[0], scene)
        feed = {x.name:np.zeros(x.shape, np.float32) for x in self.inputs}
        feed['img'] = data['img'].numpy().astype(np.float32)
        feed['history_transform'] = np.eye(4,dtype=np.float32)[None]
        feed['time_interval'] = np.array([.5],np.float32)
        feed['history_valid'] = np.array([float(valid)],np.float32)
        if valid:
            for name,value in zip(['history_feature','history_anchor','history_confidence'],self.cache):feed[name]=value
            feed['history_transform'] = (meta['T_global_inv'] @ self.previous[2])[None].astype(np.float32)
            feed['time_interval'][:] = timestamp-self.previous[1]
        self.rig_errors.append({k:float(np.max(np.abs(data[k].numpy()-v))) for k,v in self.rig.items()})
        self.timing["preprocess"]+=time.perf_counter()-started
        started=time.perf_counter()
        outputs = dict(zip([v.name for v in self.session.get_outputs()],self.session.run(None,feed)))
        self.timing["inference"]+=time.perf_counter()-started
        started=time.perf_counter()
        self.cache = [outputs[n] for n in ['next_history_feature','next_history_anchor','next_history_confidence']]
        self.previous = (scene,timestamp,np.asarray(meta['T_global']))
        pred = self.decoder.decode([torch.from_numpy(outputs['det_logits'])],
            [torch.from_numpy(outputs['det_boxes'])],quality=[torch.from_numpy(outputs['det_quality'])])[0]
        occ = outputs['occ_classes'][0]
        if occ.shape != (128,128,16) or np.any((occ<0)|(occ>17)):
            raise ValueError('Unexpected occupancy output')
        pred['occ_pred'] = occ.astype(np.uint8)
        self.timing['decode']+=time.perf_counter()-started
        return pred

    def provenance(self):
        return dict(scope='Fixed-rig ONNX on nuScenes; calibration mismatch can affect metrics. Not PTH equivalence.',
            timing_seconds=self.timing, provider=self.session.get_providers(),frames=len(self.rig_errors),
            calibration_max_abs={k:max(x[k] for x in self.rig_errors) for k in self.rig} if self.rig_errors else {})


def write_json(path, data):
    Path(path).write_text(json.dumps(data,indent=2,default=lambda x:float(x))+'\n')
