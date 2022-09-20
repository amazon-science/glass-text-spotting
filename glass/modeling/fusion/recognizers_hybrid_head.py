# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import inspect
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals as mask_select_foreground_proposals
from detectron2.modeling.roi_heads.rotated_fast_rcnn import RotatedFastRCNNOutputLayers
from detectron2.structures import ImageList
from detectron2.structures import Instances, RotatedBoxes, pairwise_iou_rotated, Boxes
from detectron2.utils.events import get_event_storage

from .fusion_modules import P2P3Fusion, build_hybrid_feature_fusion
from .local_feature_extraction import build_hybrid_feature_extractor
from ..recognition.recognizer_head_v2 import build_recognizer_head
from ..recognition.recognizer_pooler_pad import build_recognizer_pooler_pad
from ..roi_heads.rotated_fast_rcnn import RotatedFastRCNNOutputLayers
from ..roi_heads.rotated_fast_rcnn import overwrite_orientations_on_boxes
from ..roi_heads.rotated_head import add_ground_truth_to_proposals as add_ground_truth_to_proposals_rotated
from ...structures.boxes import box_to_rbox, rbox_to_box


def select_foreground_and_class_proposals(
        proposals: List[Instances], fg_class_ind: int,
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes == fg_class_ind)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


@ROI_HEADS_REGISTRY.register()
class MaskRotatedRecognizerHybridHead(StandardROIHeads):

    @configurable
    def __init__(
            self,
            *,
            img_pooler: ROIPooler,
            recognizer_in_features: list,
            recognizer_pooler: ROIPooler,
            recognizer_head: nn.Module,
            hybrid_net: nn.Module,
            fusion_net: nn.Module,
            recognizer_feature_fusion: nn.Module,
            img_pooler_resolution: int,
            vis_period: int,
            input_format: str,
            pixel_mean: list,
            pixel_std: list,
            mask_pooler_type: str,
            mask_inference: bool,
            apply_orientation_to_proposals: bool,
            **kwargs
    ):
        """
        NOTE: this interface is experimental.

        Args:
            img_pooler (ROIPooler): pooler to extra image region
            hybrid_net (nn.Module): transform image region to make box predictions
        """
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
        # keep self.in_features for backward compatibility
        self.img_pooler = img_pooler
        self.hybrid_net = hybrid_net
        self.img_pooler_resolution = img_pooler_resolution
        self.vis_period = vis_period
        self.input_format = input_format
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.fusion_net = fusion_net
        self.recognizer_feature_fusion = recognizer_feature_fusion
        self.recognizer_in_features = recognizer_in_features
        self.recognizer_pooler = recognizer_pooler
        self.recognizer_head = recognizer_head
        self.mask_pooler_type = mask_pooler_type
        self.mask_inference = mask_inference
        self.apply_orientation_to_proposals = apply_orientation_to_proposals

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        ret["apply_orientation_to_proposals"] = cfg.MODEL.ROI_ORIENTATION_HEAD.APPLY_TO_BOXES_DURING_TRAINING
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_recognizer_head):
            ret.update(cls._init_recognizer_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))
        return ret

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """

        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.

            # We apply orientation following the box prediction, so the recognizer and mask
            # get correctly predicted proposals, based on the gt orientation
            if self.apply_orientation_to_proposals:
                for i in range(len(proposals)):
                    boxes = proposals[i].proposal_boxes.tensor
                    if proposals[i].has('gt_orientation'):
                        gt_orientations = proposals[i].gt_orientation
                    else:
                        gt_orientations = torch.tensor([0], device=proposals[i].proposal_boxes.tensor.device)
                        self.logger.warning('No gt_orientation found')

                    proposals[i].proposal_boxes.tensor = overwrite_orientations_on_boxes(boxes, gt_orientations)

            losses.update(self._forward_recognizer(images, features, proposals))
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))

            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(images, features, pred_instances)
            return pred_instances, {}

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # We override the box predictor with our own custom version
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on
        # we turn off this assert to allow only rotated box heads, without rotated RPN
        # assert pooler_type in ["ROIAlignRotated"], pooler_type
        # assume all channel counts are equal
        assert pooler_type in ["ROIAlignRotated"], pooler_type

        in_channels = [input_shape[f].channels for f in in_features][0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        # This line is the only difference v.s. StandardROIHeads
        box_predictor = RotatedFastRCNNOutputLayers(cfg, box_head.output_shape)
        ret = {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }
        ret['box_predictor'] = RotatedFastRCNNOutputLayers(cfg, ret['box_head'].output_shape)

        return ret

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the RROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`StandardROIHeads.forward`

        Returns:
            list[Instances]: length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the rotated proposal boxes
                - gt_boxes: the ground-truth rotated boxes that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                - gt_classes: the ground-truth classification labels for each proposal
        """
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals_rotated(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou_rotated(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = RotatedBoxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 5))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_proposals = []
        for proposal in proposals:
            if not isinstance(proposal.proposal_boxes, RotatedBoxes):
                box_tensor = proposal.proposal_boxes.tensor
                rbox_tensor = box_to_rbox(box_tensor)
                box_proposal = RotatedBoxes(rbox_tensor)

            else:
                box_proposal = proposal.proposal_boxes
            box_proposals.append(box_proposal)
            proposal.proposal_boxes = box_proposal
        box_features = self.box_pooler(features, box_proposals)
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            assert not torch.jit.is_scripting()
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on
        # assert pooler_type in ["ROIAlign"], pooler_type

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["mask_head"] = build_mask_head(cfg, shape)
        ret["mask_pooler_type"] = pooler_type
        ret["mask_inference"] = cfg.MODEL.ROI_MASK_HEAD.MASK_INFERENCE
        return ret

    def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            # https://github.com/pytorch/pytorch/issues/43942
            if self.training:
                assert not torch.jit.is_scripting()
                return {}
            else:
                return instances

        # https://github.com/pytorch/pytorch/issues/46703
        assert hasattr(self, "mask_head")

        if self.training:
            assert not torch.jit.is_scripting()
            # head is only trained on positive proposals.
            # rotated box instances
            instances, _ = mask_select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]

            if self.training:
                box_proposals = []
                for instance in instances:
                    if self.mask_pooler_type == 'ROIAlign' and isinstance(instance.proposal_boxes, RotatedBoxes):
                        rbox_tensor = instance.proposal_boxes.tensor
                        box_tensor = rbox_to_box(rbox_tensor)
                        box_proposal = Boxes(box_tensor)

                    else:
                        box_proposal = instance.proposal_boxes
                    box_proposals.append(box_proposal)
                    instance.proposal_boxes = box_proposal
                features = self.mask_pooler(features, box_proposals)
            else:
                pred_boxes = []
                for instance in instances:
                    if self.mask_pooler_type == 'ROIAlign' and isinstance(instance.pred_boxes, RotatedBoxes):
                        rbox_tensor = instance.pred_boxes.tensor
                        box_tensor = rbox_to_box(rbox_tensor)
                        pred_box = Boxes(box_tensor)

                    else:
                        pred_box = instance.pred_boxes
                    pred_boxes.append(pred_box)
                    instance.pred_boxes = pred_box
                features = self.mask_pooler(features, pred_boxes)
        else:
            # https://github.com/pytorch/pytorch/issues/41448
            features = dict([(f, features[f]) for f in self.mask_in_features])
        return self.mask_head(features, instances)

    @classmethod
    def _init_recognizer_head(cls, cfg, input_shape):
        if not cfg.MODEL.RECOGNIZER_ON:
            return {}
        # fmt: off
        in_features = cfg.MODEL.ROI_RECOGNIZER_HEAD.IN_FEATURES
        pooler_resolution_w = cfg.MODEL.ROI_RECOGNIZER_HEAD.POOLER_RESOLUTION_WIDTH
        pooler_resolution_h = cfg.MODEL.ROI_RECOGNIZER_HEAD.POOLER_RESOLUTION_HEIGHT
        # pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in
                              [in_features[0]])  # note that it extract only the first features assumed to be p2!
        sampling_ratio = cfg.MODEL.ROI_RECOGNIZER_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_RECOGNIZER_HEAD.POOLER_TYPE
        pooler_pad = cfg.MODEL.ROI_RECOGNIZER_HEAD.RECOGNIZER_HEAD.POOLER_PAD.NAME
        # fmt: on
        assert pooler_type in ["ROIAlignRotated"], pooler_type

        in_channels = [input_shape[f].channels for f in in_features][0]

        if pooler_type:
            recognizer_pooler = ROIPooler(
                output_size=[pooler_resolution_h, pooler_resolution_w],
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_pad:
                recognizer_pooler = build_recognizer_pooler_pad(cfg, orig_pooler=recognizer_pooler)
        else:
            recognizer_pooler = None

        ret = {"recognizer_in_features": in_features}
        ret["recognizer_pooler"] = (recognizer_pooler)

        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution_w, height=pooler_resolution_h
            )
        else:
            shape = {f: input_shape[f] for f in in_features}

        if len(in_features) == 2:
            # Assuming that p2, p3 were given!
            channels = in_channels  # cfg.MODEL.ROI_RECOGNIZER_HEAD.CONV_DIM
            ret["recognizer_feature_fusion"] = P2P3Fusion(channels)

        img_pooler_resolution = cfg.MODEL.ROI_HYBRID_HEAD.POOLER_RESOLUTION
        img_pooler_scales = [1]
        img_sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        img_pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        img_pooler = ROIPooler(
            output_size=[pooler_resolution_h * 16, pooler_resolution_w * 4],
            scales=img_pooler_scales,
            sampling_ratio=img_sampling_ratio,
            pooler_type=img_pooler_type,
        )  # output_size=[pooler_resolution_h * 16, pooler_resolution_w * 4] set for ResNet_FeatureExtractor dimension

        ret["vis_period"] = cfg.VIS_PERIOD
        ret["input_format"] = cfg.INPUT.FORMAT
        ret["pixel_mean"] = cfg.MODEL.PIXEL_MEAN
        ret["pixel_std"] = cfg.MODEL.PIXEL_STD
        ret["img_pooler_resolution"] = img_pooler_resolution
        ret["img_pooler"] = img_pooler
        ret["hybrid_net"] = build_hybrid_feature_extractor(cfg, shape)
        ret["fusion_net"] = build_hybrid_feature_fusion(cfg, shape)
        ret["recognizer_head"] = build_recognizer_head(cfg, shape)
        return ret

    def _forward_recognizer(self, images: ImageList, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the recognizer prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        # if not self.mask_on:
        #     # https://github.com/pytorch/pytorch/issues/43942
        #     if self.training:
        #         assert not torch.jit.is_scripting()
        #         return {}
        #     else:
        #         return instances

        # https://github.com/pytorch/pytorch/issues/46703
        assert hasattr(self, "recognizer_head")

        if self.training:
            assert not torch.jit.is_scripting()
            # head is only trained on positive proposals.
            instances, _ = select_foreground_and_class_proposals(instances,
                                                                 fg_class_ind=self.recognizer_head.class_ind)

        if self.recognizer_pooler is not None:
            features = [features[f] for f in self.recognizer_in_features]
            if len(features) == 2:
                features = [self.recognizer_feature_fusion(features[0], features[1])]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.recognizer_pooler(features, boxes)

        else:
            # https://github.com/pytorch/pytorch/issues/41448
            features = dict([(f, features[f]) for f in self.recognizer_in_features])

        local_features = self.img_pooler([images.tensor], boxes)

        if local_features.shape[0] > 0:
            local_features = self.hybrid_net(local_features)
            features = torch.cat((local_features, features), 1)
            del local_features
        else:
            features = torch.cat((features, features), 1)
        # if local_features.shape[2] == features.shape[2] and local_features.shape[3] == features.shape[3]:

        # else:
        # features = torch.cat((local_features[...,:features.shape[2],:features.shape[2]], features), 1)
        features = self.fusion_net(features)
        return self.recognizer_head(features, instances)

    def forward_with_given_boxes(
            self, images: ImageList, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_recognizer(images, features, instances)
        if self.mask_inference:
            instances_new = self._forward_mask(features, copy.deepcopy(instances))
            for i in range(len(instances)):
                instances[i].pred_masks = instances_new[i].pred_masks
                instances[i].pred_rboxes = instances[i].pred_boxes
                if instances_new[i].pred_boxes.tensor.shape[-1] == 4:
                    instances[i].pred_boxes = instances_new[i].pred_boxes
                #     box_tensor = rbox_to_box(instances_new[i].pred_boxes.tensor)
                #     pred_box = Boxes(box_tensor)
                #     instances[i].pred_boxes = pred_box
                # else:
                #     instances[i].pred_boxes = instances_new[i].pred_boxes
            # instances = self._forward_mask(features, instances)
        # instances = self._forward_keypoint(features, instances)
        return instances
