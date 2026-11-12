"""Stage-1 detection + occupancy sharing a single image backbone."""
import torch
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS, build_head, build_neck
from ..sparsedrive import SparseDrive
from .temporal import standard_attention_export


@DETECTORS.register_module()
class SparseDriveDetOcc(SparseDrive):
    def __init__(self, occ_img_neck, occ_branch, **kwargs):
        super().__init__(**kwargs)
        if self.use_deformable_func:
            raise ValueError('EP1 uses the exportable grid_sample aggregation path')
        if self.head.task_config.get('with_map') or self.head.task_config.get('with_motion_plan'):
            raise ValueError('EP1 is detection + OCC only')
        self.occ_img_neck = build_neck(occ_img_neck)
        self.occ_branch = build_head(occ_branch)

    @auto_fp16(apply_to=('img',), out_fp32=True)
    def extract_shared_features(self, img):
        b, n = img.shape[:2]
        img = img.flatten(0, 1)
        if self.use_grid_mask:
            img = self.grid_mask(img)
        backbone_features = self.img_backbone(img)
        detection = [x.reshape(b, n, *x.shape[1:])
                     for x in self.img_neck(backbone_features)]
        occupancy = self.occ_img_neck(backbone_features[-2:])
        return detection, occupancy

    def forward_train(self, img, **data):
        feature_maps, occ_feature = self.extract_shared_features(img)
        losses = self.head.loss(self.head(feature_maps, data), data)
        occ_logits = self.occ_branch(occ_feature, data['occ_img2ego'])
        losses.update(self.occ_branch.loss(occ_logits, data['voxel_semantics'], data['mask_camera']))
        if self.depth_branch is not None and 'gt_depth' in data:
            depths = self.depth_branch(feature_maps, data.get('focal'))
            losses['loss_dense_depth'] = self.depth_branch.loss(depths, data['gt_depth'])
        return losses

    def simple_test(self, img, **data):
        feature_maps, occ_feature = self.extract_shared_features(img)
        results = self.head.post_process(self.head(feature_maps, data), data)
        occ_logits = self.occ_branch(occ_feature, data['occ_img2ego'])
        predictions = occ_logits.argmax(-1).to(torch.uint8).cpu().numpy()
        for result, prediction in zip(results, predictions):
            result['occ_pred'] = prediction
        return [{'img_bbox': result} for result in results]

    @standard_attention_export
    def forward_export(self, img, projection_mat, image_wh, occ_img2ego,
                       history_feature, history_anchor, history_confidence,
                       history_transform, time_interval, history_valid):
        """One stateless tensor interface for all frames; no Python frame switch.

        ``history_transform`` maps the previous lidar frame into the current
        lidar frame. ``time_interval`` is current minus previous timestamp (s).
        Callers feed the returned state into the next invocation. Pass zeros
        and history_valid=False on scene starts; use identity for the transform.
        """
        metas = dict(projection_mat=projection_mat, image_wh=image_wh,
                     occ_img2ego=occ_img2ego, history_feature=history_feature,
                     history_anchor=history_anchor, history_confidence=history_confidence,
                     history_transform=history_transform, time_interval=time_interval,
                     history_valid=history_valid, img_metas=[{}] * img.shape[0])
        feature_maps, occ_feature = self.extract_shared_features(img)
        detection = self.head.det_head(feature_maps, metas)
        occ_logits = self.occ_branch(occ_feature, occ_img2ego)
        bank = self.head.det_head.instance_bank
        return (detection['classification'][-1], detection['prediction'][-1],
                detection['quality'][-1], occ_logits,
                bank.cached_feature, bank.cached_anchor, bank.confidence)
