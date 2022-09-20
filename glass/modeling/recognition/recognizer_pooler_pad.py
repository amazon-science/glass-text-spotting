# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from copy import deepcopy

from detectron2.config import configurable
from detectron2.utils.registry import Registry

RECOGNIZER_POOLER_PAD_REGISTRY = Registry("RECOGNIZER_POOLER_PAD")
RECOGNIZER_POOLER_PAD_REGISTRY.__doc__ = """

"""


def build_recognizer_pooler_pad(cfg, orig_pooler):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.RECOGNIZER_HEAD.POOLER_PAD.NAME
    return RECOGNIZER_POOLER_PAD_REGISTRY.get(name)(cfg, orig_pooler)




@RECOGNIZER_POOLER_PAD_REGISTRY.register()
class FeatPadV2(nn.Module):
    """
    padding the features to keep aspect ratio of the input features
    FeatPadV2 - the original features are centralized zero padding is added to the left and right if needed
    """

    @configurable
    def __init__(self, *, pooler):
        super().__init__()
        self.orig_pooler = pooler

    @classmethod
    def from_config(cls, cfg, pooler):
        ret = {}
        ret["pooler"] = pooler
        return ret

    def forward(self, features, boxes):
        augmented_boxes = list()
        update_ind_list = list()
        pad_dst_src_ratio_vec_list = list()
        for boxes_b in boxes:
            tmp_boxes = deepcopy(boxes_b)
            dst_aspect = self.orig_pooler.output_size[1] / self.orig_pooler.output_size[0]
            boxes_tensor = tmp_boxes.tensor
            left_vec = boxes_tensor[:, 0]
            top_vec = boxes_tensor[:, 1]
            right_vec = boxes_tensor[:, 2]
            bottom_vec = boxes_tensor[:, 3]

            width_vec = right_vec - left_vec
            height_vec = bottom_vec - top_vec
            src_aspect_vec = width_vec / height_vec
            dst_src_ratio_vec = dst_aspect / src_aspect_vec
            update_ind = dst_src_ratio_vec > 1

            pad_left_vec = left_vec[update_ind]
            pad_width_vec = width_vec[update_ind]
            pad_dst_src_ratio_vec = dst_src_ratio_vec[update_ind]
            src_padding_vec = pad_width_vec * (pad_dst_src_ratio_vec - 1)

            # padding style -> add padding to the right to keep aspect ratio
            boxes_tensor[update_ind, 2] = pad_left_vec + pad_width_vec + src_padding_vec / 2
            boxes_tensor[update_ind, 0] = pad_left_vec - src_padding_vec / 2

            augmented_boxes.append(tmp_boxes)
            update_ind_list.append(update_ind)
            pad_dst_src_ratio_vec_list.append(pad_dst_src_ratio_vec)

        # run orig pooler
        update_ind = torch.cat(update_ind_list, dim=0)
        pad_dst_src_ratio_vec = torch.cat(pad_dst_src_ratio_vec_list, dim=0)
        features = self.orig_pooler(features, augmented_boxes)

        # mask the output padded features
        if update_ind.sum() > 0:
            dst_padding_vec = (pad_dst_src_ratio_vec - 1) / pad_dst_src_ratio_vec * self.orig_pooler.output_size[1] / 2
            roi_out_pad = features[update_ind]
            dst_padding_vec_right = features.shape[3] - dst_padding_vec

            index_array = torch.arange(roi_out_pad.shape[3], device=features.device)[None, None, None, :]
            mask_right = index_array < dst_padding_vec_right[:, None, None, None]
            mask_left = index_array >= dst_padding_vec[:, None, None, None]
            mask = mask_left * mask_right

            features[update_ind] = features[update_ind] * mask

        return features
