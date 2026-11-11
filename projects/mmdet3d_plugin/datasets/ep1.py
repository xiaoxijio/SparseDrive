"""Detection-only nuScenes annotations plus aligned Occ3D supervision."""
import copy
import json
import os
import numpy as np
import torch
from pyquaternion import Quaternion
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS, PIPELINES
from .nuscenes_3d_dataset import NuScenes3DDataset


def crop_occ(array, source_grid, target_grid):
    """Crop aligned voxel centers; never interpolate semantic class IDs."""
    slices = []
    for axis, name in enumerate(('x', 'y', 'z')):
        src, dst = source_grid[name], target_grid[name]
        if not np.isclose(src[2], dst[2]):
            raise ValueError('OCC source/target voxel sizes must match')
        start = (dst[0] - src[0]) / src[2]
        count = (dst[1] - dst[0]) / src[2]
        source_count = round((src[1] - src[0]) / src[2])
        if (not np.isclose(start, round(start)) or not np.isclose(count, round(count))
                or round(start) < 0 or round(start + count) > source_count
                or array.shape[axis] != source_count):
            raise ValueError('OCC label shape/grid alignment mismatch on axis ' + name)
        slices.append(slice(round(start), round(start + count)))
    return np.ascontiguousarray(array[tuple(slices)])


@DATASETS.register_module()
class NuScenesDetOccDataset(NuScenes3DDataset):
    def __init__(self, occ_root, occ_source_grid, occ_grid, **kwargs):
        self.occ_root = occ_root
        self.occ_source_grid = occ_source_grid
        self.occ_grid = occ_grid
        super().__init__(**kwargs)
        scene_path = os.path.join(self.data_root, self.version, 'scene.json')
        with open(scene_path) as handle:
            self.scene_names = {scene['token']: scene['name'] for scene in json.load(handle)}

    def get_occ_path(self, index):
        info = self.data_infos[index]
        scene = info.get('scene_name') or self.scene_names[info['scene_token']]
        return os.path.join(self.occ_root, scene, info['token'], 'labels.npz')

    def get_data_info(self, index):
        # Avoid loading/converting map geometry and future planning annotations.
        info = self.data_infos[index]
        lidar2ego, ego2global = np.eye(4), np.eye(4)
        for matrix, prefix in ((lidar2ego, 'lidar2ego'), (ego2global, 'ego2global')):
            matrix[:3, :3] = Quaternion(info[prefix + '_rotation']).rotation_matrix
            matrix[:3, 3] = info[prefix + '_translation']
        result = dict(token=info['token'], scene_token=info['scene_token'],
                      pts_filename=info['lidar_path'], sweeps=info['sweeps'],
                      timestamp=info['timestamp'] / 1e6,
                      lidar2global=ego2global @ lidar2ego,
                      occ_ego2lidar=np.linalg.inv(lidar2ego), occ_path=self.get_occ_path(index))
        image_paths, projections, intrinsics = [], [], []
        for camera in info['cams'].values():
            sensor2lidar = np.eye(4)
            sensor2lidar[:3, :3] = camera['sensor2lidar_rotation']
            sensor2lidar[:3, 3] = camera['sensor2lidar_translation']
            intrinsic = copy.deepcopy(camera['cam_intrinsic'])
            projection = np.eye(4)
            projection[:3, :3] = intrinsic
            image_paths.append(camera['data_path'])
            projections.append(projection @ np.linalg.inv(sensor2lidar))
            intrinsics.append(intrinsic)
        result.update(img_filename=image_paths, lidar2img=projections,
                      cam_intrinsic=intrinsics)
        result.update(self.get_ann_info(index))
        return result

    def get_ann_info(self, index):
        info = self.data_infos[index]
        mask = info['valid_flag'] if self.use_valid_flag else info['num_lidar_pts'] > 0
        boxes = info['gt_boxes'][mask].copy()
        names = info['gt_names'][mask]
        labels = np.array([self.cat2id.get(name, -1) for name in names], dtype=np.int64)
        if self.with_velocity:
            velocity = np.nan_to_num(info['gt_velocity'][mask].copy())
            boxes = np.concatenate([boxes, velocity], -1)
        result = dict(gt_bboxes_3d=boxes, gt_labels_3d=labels, gt_names=names)
        if 'instance_inds' in info:
            result['instance_inds'] = np.asarray(info['instance_inds'], dtype=np.int64)[mask]
        return result

    def evaluate(self, results, eval_mode, **kwargs):
        if len(results) != len(self):
            raise ValueError('Prediction count must match dataset length')
        detection_results = [dict(img_bbox={k: v for k, v in result['img_bbox'].items()
                                             if k != 'occ_pred'}) for result in results]
        if self.work_dir is None:
            self.work_dir = './work_dirs/ep1_eval'
        os.makedirs(self.work_dir, exist_ok=True)
        metrics = super().evaluate(detection_results, eval_mode, **kwargs)
        if eval_mode.get('with_occ', True):
            confusion = np.zeros((18, 18), dtype=np.int64)
            for index, result in enumerate(results):
                prediction = result['img_bbox']['occ_pred']
                with np.load(self.get_occ_path(index)) as label:
                    target = crop_occ(label['semantics'], self.occ_source_grid, self.occ_grid)
                    mask = crop_occ(label['mask_camera'], self.occ_source_grid, self.occ_grid).astype(bool)
                if prediction.shape != target.shape:
                    raise ValueError('OCC prediction and label shapes differ')
                mask &= target != 255
                gt, pred = target[mask].astype(np.int64), prediction[mask].astype(np.int64)
                confusion += np.bincount(gt * 18 + pred, minlength=18 * 18).reshape(18, 18)
            union = confusion.sum(0) + confusion.sum(1) - np.diag(confusion)
            iou = np.divide(np.diag(confusion), union,
                            out=np.full(18, np.nan), where=union > 0)
            # Occ3D class 17 is free space; report occupied semantic classes only.
            metrics['occ/mIoU'] = float(np.nanmean(iou[:17]))
            metrics.update({'occ/IoU_{}'.format(i): float(value) for i, value in enumerate(iou)})
        return metrics


@PIPELINES.register_module()
class PrepareOccEP1:
    def __init__(self, source_grid, target_grid, with_gt=True):
        self.source_grid, self.target_grid, self.with_gt = source_grid, target_grid, with_gt

    def __call__(self, results):
        angle = results.get('aug_config', {}).get('rotate_3d', 0)
        if abs(angle) > 1e-8:
            raise ValueError('EP1 OCC requires rot3d_range=[0, 0]; labels are in ego coordinates')
        # lidar2img already contains image resize/crop/flip/rotation. The inverse
        # maps [u*depth, v*depth, depth, 1] into the keyframe ego frame used by Occ3D.
        ego2img = np.stack(results['lidar2img']) @ results['occ_ego2lidar']
        results['occ_img2ego'] = DC(torch.from_numpy(np.linalg.inv(ego2img).astype(np.float32)),
                                   stack=True, pad_dims=None)
        if self.with_gt:
            with np.load(results['occ_path']) as labels:
                for output, key in (('voxel_semantics', 'semantics'), ('mask_camera', 'mask_camera')):
                    array = crop_occ(labels[key], self.source_grid, self.target_grid)
                    results[output] = DC(torch.from_numpy(array), stack=True, pad_dims=None)
        return results
