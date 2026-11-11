"""EP1 stage 1: shared B0-noSE, sparse detection (150+300), and FlashOCC.

No map, tracking IDs, motion or planning. Configuration is self-contained.
Run tools/prepare_ep1_anchors.py before first use. See docs/ep1_s1.md.
"""
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ep1/s1_det_occ_b0_nose'
load_from = None
resume_from = None
workflow = [('train', 1)]
fp16 = dict(loss_scale=32.0)

class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
               'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
input_shape = (704, 256)
# Detection channels remain 256 to isolate the explicitly requested changes.
# Further 128-channel detection ablation is described in docs/ep1_s1.md.
embed_dims = 256
num_groups = 8
strides = [4, 8, 16, 32]
num_decoder = 6
num_single_frame_decoder = 1
num_anchor = 450
num_temp_instances = 300
grid_config = dict(x=[-25.6, 25.6, 0.4], y=[-25.6, 25.6, 0.4],
                   z=[-1.0, 5.4, 6.4], depth=[1.0, 45.0, 1.0])
occ_source_grid = dict(x=[-40.0, 40.0, 0.4], y=[-40.0, 40.0, 0.4], z=[-1.0, 5.4, 0.4])
occ_grid = dict(x=[-25.6, 25.6, 0.4], y=[-25.6, 25.6, 0.4], z=[-1.0, 5.4, 0.4])
task_config = dict(with_det=True, with_map=False, with_motion_plan=False)

model = dict(
    type='SparseDriveDetOcc',
    use_grid_mask=True,
    use_deformable_func=False,
    img_backbone=dict(type='EfficientNetB0NoSE', pretrained=None, with_cp=True),
    img_neck=dict(type='FPN', in_channels=[24, 40, 112, 1280],
                  out_channels=embed_dims, num_outs=4, start_level=0,
                  upsample_cfg=dict(mode='bilinear', align_corners=False)),
    occ_img_neck=dict(type='OccImageNeck', in_channels=[112, 1280], out_channels=256),
    occ_branch=dict(type='FlashOccBranch', grid_config=grid_config,
                    input_size=input_shape[::-1], in_channels=256, view_channels=64,
                    bev_channels=[128, 256, 512], out_channels=128,
                    Dz=16, num_classes=18, use_mask=True, loss_weight=1.0),
    depth_branch=dict(type='DenseDepthNet', embed_dims=embed_dims,
                      num_depth_layers=3, loss_weight=0.2),
    head=dict(
        type='SparseDriveHead', task_config=task_config,
        det_head=dict(
            type='PaddedSparse4DHead', num_decoder=num_decoder,
            num_single_frame_decoder=num_single_frame_decoder,
            with_instance_id=False, decouple_attn=True, cls_threshold_to_reg=0.05,
            instance_bank=dict(type='PaddedInstanceBank', num_anchor=num_anchor,
                               embed_dims=embed_dims, anchor='data/kmeans/kmeans_det_450.npy',
                               anchor_handler=dict(type='SparseBox3DKeyPointsGenerator'),
                               num_temp_instances=num_temp_instances, confidence_decay=0.6,
                               feat_grad=False),
            anchor_encoder=dict(type='SparseBox3DEncoder', vel_dims=3,
                                embed_dims=[embed_dims // 2, embed_dims // 8,
                                            embed_dims // 8, embed_dims // 4], mode='cat',
                                output_fc=False, in_loops=1, out_loops=4),
            operation_order=(
                ['gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine'] * num_single_frame_decoder
                + ['temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine']
                * (num_decoder - num_single_frame_decoder))[2:],
            temp_graph_model=dict(type='ExportableAttention', embed_dims=embed_dims * 2,
                                  num_heads=num_groups, batch_first=True, dropout=0.1),
            graph_model=dict(type='ExportableAttention', embed_dims=embed_dims * 2,
                             num_heads=num_groups, batch_first=True, dropout=0.1),
            norm_layer=dict(type='LN', normalized_shape=embed_dims), 
            ffn=dict(type='AsymmetricFFN', in_channels=embed_dims * 2,
                     pre_norm=dict(type='LN'), embed_dims=embed_dims,
                     feedforward_channels=embed_dims * 4, num_fcs=2, ffn_drop=0.1,
                     act_cfg=dict(type='ReLU', inplace=True)),
            deformable_model=dict(
                type='DeformableFeatureAggregation', embed_dims=embed_dims,
                num_groups=num_groups, num_levels=4, num_cams=6, attn_drop=0.15,
                use_deformable_func=False, use_camera_embed=True, residual_mode='cat',
                kps_generator=dict(type='ExportableBoxKeyPoints',
                                   num_learnable_pts=6,
                                   fix_scale=[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0],
                                              [0, 0.45, 0], [0, -0.45, 0],
                                              [0, 0, 0.45], [0, 0, -0.45]])),
            refine_layer=dict(type='ExportableBoxRefinement', embed_dims=embed_dims,
                              num_cls=len(class_names), refine_yaw=True,
                              with_quality_estimation=True),
            sampler=dict(type='SparseBox3DTarget', num_dn_groups=0, num_temp_dn_groups=0,
                         dn_noise_scale=[2.0] * 3 + [0.5] * 7, max_dn_gt=32, add_neg_dn=True,
                         cls_weight=2.0, box_weight=0.25,
                         reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
                         cls_wise_reg_weights={class_names.index('traffic_cone'):
                                               [2., 2., 2., 1., 1., 1., 0., 0., 1., 1.]}),
            loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25,
                          loss_weight=2.0),
            loss_reg=dict(type='SparseBox3DLoss',
                          loss_box=dict(type='L1Loss', loss_weight=0.25),
                          loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True),
                          loss_yawness=dict(type='GaussianFocalLoss'),
                          cls_allow_reverse=[class_names.index('barrier')]),
            decoder=dict(type='SparseBox3DDecoder', num_output=300),
            reg_weights=[2.0] * 3 + [1.0] * 7)))

# Training target: 4 x 32-GB GPUs, 4 samples/GPU (global batch = 16).
# The 3090 host is used only for single-sample smoke tests.
num_gpus = 4
batch_size = 4
num_epochs = 100
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
checkpoint_epoch_interval = 10
checkpoint_config = dict(interval=num_iters_per_epoch * checkpoint_epoch_interval,
                         max_keep_ckpts=3)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
data_root = 'data/nuscenes/'
anno_root = 'data/infos/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5,
         file_client_args=dict(backend='disk')),
    dict(type='ResizeCropFlipImage'),
    dict(type='MultiScaleDepthMapGenerator', downsample=strides[:3]),
    dict(type='BBoxRotation'),
    dict(type='PrepareOccEP1', source_grid=occ_source_grid, target_grid=occ_grid),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='CircleObjectRangeFilter', class_dist_thred=[55] * len(class_names)),
    dict(type='InstanceNameFilter', classes=class_names),
    dict(type='NuScenesSparse4DAdaptor'),
    dict(type='Collect',
         keys=['img', 'timestamp', 'projection_mat', 'image_wh', 'gt_depth', 'focal',
               'gt_bboxes_3d', 'gt_labels_3d', 'occ_img2ego', 'voxel_semantics', 'mask_camera'],
         meta_keys=['T_global', 'T_global_inv', 'timestamp', 'instance_id', 'scene_token']),
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage'),
    dict(type='PrepareOccEP1', source_grid=occ_source_grid, target_grid=occ_grid, with_gt=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='NuScenesSparse4DAdaptor'),
    dict(type='Collect', keys=['img', 'timestamp', 'projection_mat', 'image_wh', 'occ_img2ego'],
         meta_keys=['T_global', 'T_global_inv', 'timestamp', 'scene_token']),
]
data_aug_conf = dict(resize_lim=(0.40, 0.47), final_dim=input_shape[::-1],
                     bot_pct_lim=(0., 0.), rot_lim=(-5.4, 5.4), H=900, W=1600,
                     rand_flip=True, rot3d_range=[0, 0])
data_basic_config = dict(
    type='NuScenesDetOccDataset', data_root=data_root, classes=class_names,
    occ_root=data_root + 'gts', occ_source_grid=occ_source_grid, occ_grid=occ_grid,
    modality=dict(use_lidar=False, use_camera=True, use_radar=False,
                  use_map=False, use_external=False), version='v1.0-trainval')
data = dict(
    samples_per_gpu=batch_size, workers_per_gpu=2,
    train=dict(**data_basic_config, ann_file=anno_root + 'nuscenes_infos_train.pkl',
               pipeline=train_pipeline, test_mode=False, data_aug_conf=data_aug_conf,
               with_seq_flag=True, sequences_split_num=2, keep_consistent_seq_aug=True),
    val=dict(**data_basic_config, ann_file=anno_root + 'nuscenes_infos_val.pkl',
             pipeline=test_pipeline, test_mode=True, data_aug_conf=data_aug_conf),
    test=dict(**data_basic_config, ann_file=anno_root + 'nuscenes_infos_val.pkl',
              pipeline=test_pipeline, test_mode=True, data_aug_conf=data_aug_conf))

# Official global batch 64 uses 4e-4; linearly scale to global batch 16.
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.001,
                 paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.5)}))
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
lr_config = dict(policy='CosineAnnealing', warmup='linear', warmup_iters=500,
                 warmup_ratio=1. / 3, min_lr_ratio=1e-3)
runner = dict(type='IterBasedRunner', max_iters=num_iters_per_epoch * num_epochs)
eval_mode = dict(with_det=True, with_tracking=False, with_map=False, with_motion=False,
                 with_planning=False, with_occ=True, tracking_threshold=0.2,
                 motion_threshhold=0.2)
evaluation = dict(interval=num_iters_per_epoch * checkpoint_epoch_interval, eval_mode=eval_mode)
