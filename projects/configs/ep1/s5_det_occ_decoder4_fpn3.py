"""S5: S2 without the stride-4 FPN output. Biggest static MAC reduction.
Retains dynamic sampling, 450 queries/300 history and all OCC grid/classes.
Small/far objects may lose accuracy; train and evaluate separately.
"""
_base_ = './s2_det_occ_decoder4.py'
work_dir = './work_dirs/ep1/s5_det_occ_decoder4_fpn3'
load_from = 'ckpt/s5_det_occ_decoder4_fpn3_init.pth'
strides = [8, 16, 32]
model = dict(img_neck=dict(start_level=1, num_outs=3),
             head=dict(det_head=dict(deformable_model=dict(num_levels=3))))
train_pipeline = [{'to_float32': True, 'type': 'LoadMultiViewImageFromFiles'},
 {'coord_type': 'LIDAR',
  'file_client_args': {'backend': 'disk'},
  'load_dim': 5,
  'type': 'LoadPointsFromFile',
  'use_dim': 5},
 {'type': 'ResizeCropFlipImage'},
 {'downsample': [8, 16, 32], 'type': 'MultiScaleDepthMapGenerator'},
 {'type': 'BBoxRotation'},
 {'source_grid': {'x': [-40.0, 40.0, 0.4], 'y': [-40.0, 40.0, 0.4], 'z': [-1.0, 5.4, 0.4]},
  'target_grid': {'x': [-25.6, 25.6, 0.4], 'y': [-25.6, 25.6, 0.4], 'z': [-1.0, 5.4, 0.4]},
  'type': 'PrepareOccEP1'},
 {'type': 'PhotoMetricDistortionMultiViewImage'},
 {'mean': [123.675, 116.28, 103.53],
  'std': [58.395, 57.12, 57.375],
  'to_rgb': True,
  'type': 'NormalizeMultiviewImage'},
 {'class_dist_thred': [55, 55, 55, 55, 55, 55, 55, 55, 55, 55], 'type': 'CircleObjectRangeFilter'},
 {'classes': ['car',
              'truck',
              'construction_vehicle',
              'bus',
              'trailer',
              'barrier',
              'motorcycle',
              'bicycle',
              'pedestrian',
              'traffic_cone'],
  'type': 'InstanceNameFilter'},
 {'type': 'NuScenesSparse4DAdaptor'},
 {'keys': ['img',
           'timestamp',
           'projection_mat',
           'image_wh',
           'gt_depth',
           'focal',
           'gt_bboxes_3d',
           'gt_labels_3d',
           'occ_img2ego',
           'voxel_semantics',
           'mask_camera'],
  'meta_keys': ['T_global', 'T_global_inv', 'timestamp', 'instance_id', 'scene_token'],
  'type': 'Collect'}]
data = dict(train=dict(pipeline=train_pipeline))
