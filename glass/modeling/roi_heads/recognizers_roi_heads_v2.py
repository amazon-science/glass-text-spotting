# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, List, Tuple

import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads, ROI_HEADS_REGISTRY
from detectron2.structures import Instances
from torch import nn

from ..recognition.recognizer_pooler_pad import build_recognizer_pooler_pad


logger = logging.getLogger(__name__)


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
class RecognizerROIHeadsV2(StandardROIHeads):

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features = cfg.MODEL.ROI_MASK_HEAD.IN_FEATURES
        pooler_resolution_w = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION_WIDTH
        pooler_resolution_h = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION_HEIGHT
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        pooler_pad = cfg.MODEL.ROI_MASK_HEAD.RECOGNIZER_HEAD.POOLER_PAD.NAME
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        if pooler_type:
            mask_pooler = ROIPooler(
                output_size=[pooler_resolution_h, pooler_resolution_w],
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_pad:
                mask_pooler = build_recognizer_pooler_pad(cfg, orig_pooler=mask_pooler)
        else:
            mask_pooler = None

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = (mask_pooler)

        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution_w, height=pooler_resolution_h
            )
        else:
            shape = {f: input_shape[f] for f in in_features}

        ret["mask_head"] = build_mask_head(cfg, shape)
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
            instances, _ = select_foreground_and_class_proposals(instances,
                                                                 fg_class_ind=self.mask_head.class_ind)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)

        else:
            # https://github.com/pytorch/pytorch/issues/41448
            features = dict([(f, features[f]) for f in self.mask_in_features])
        return self.mask_head(features, instances)


class CNN(nn.Module):

    def __init__(self, in_channels):
        """
        Args:
            input_shape (ShapeSpec): shape of the input feature
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.conv2 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.up = nn.Upsample(scale_factor=2)
        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x2 = self.up(x2)
        x1 = x2 + x1
        return x1
