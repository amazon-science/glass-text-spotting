# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from typing import List, Tuple, Union
import torch

from detectron2.structures.boxes import BoxMode as D2BoxMode
from detectron2.structures.boxes import Boxes as D2Boxes
from detectron2.structures.boxes import pairwise_ioa as d2_pairwise_ioa
from detectron2.structures.boxes import pairwise_intersection as d2_pairwise_intersection
from detectron2.structures.rotated_boxes import pairwise_iou_rotated

_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]

# Making sure we have the correct classes loaded from the original boxes model
BoxMode = D2BoxMode
Boxes = D2Boxes
pairwise_ioa = d2_pairwise_ioa
pairwise_intersection = d2_pairwise_intersection


def pairwise_ioa_rotated(boxes1_tensor: torch.Tensor, boxes2_tensor: torch.Tensor):
    """
    Computes the intersection over minimal area for rotated boxes
    :param boxes1_tensor: an M x 5 tensor describing rotated boxes in absolute coordinates
    :param boxes2_tensor: an N x 5 tensor describing rotated boxes in absolute coordinates
    :return: An Intersection-Over-Min-Area tensor (M x N)
    """
    assert (boxes1_tensor.shape[1] == 5) and (boxes2_tensor.shape[1] == 5), "Input tensors don't describe rotated boxes"

    # Using the d2 C based method for fast computation of IoU
    iou = pairwise_iou_rotated(boxes1_tensor, boxes2_tensor)  # M x N

    # We compose matrices of the areas, for mesh computations
    area1 = boxes1_tensor[:, 2] * boxes1_tensor[:, 3]  # M x 1
    area2 = boxes2_tensor[:, 2] * boxes2_tensor[:, 3]  # N x 1
    area1_mesh = area1.repeat(len(boxes2_tensor), 1).T  # M x N area mesh
    area2_mesh = area2.repeat(len(boxes1_tensor), 1)  # M x N area mesh

    # By definition IoU = Intersection / (Area1 + Area2 - Intersection)
    # Therefore we isolate Intersection by "Intersection = IoU * (Area1 + Area2) / (1 + IoU)"
    intersection = (area1_mesh + area2_mesh) * iou / (1 + iou)

    # Now we divide by the minimal area to obtain the intersection over min area metric
    ioa = intersection / torch.min(area1_mesh, area2_mesh)

    return ioa


def box_to_rbox(box_tensor):
    arr = BoxMode.convert(box_tensor, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    original_dtype = arr.dtype
    arr = arr.double()
    arr[:, 0] += arr[:, 2] / 2.0
    arr[:, 1] += arr[:, 3] / 2.0
    angles = torch.zeros((arr.shape[0], 1), dtype=arr.dtype, device=arr.device)
    arr = torch.cat((arr, angles), axis=1).to(dtype=original_dtype)
    return arr


def rbox_to_box(rbox_tensor):
    ret = BoxMode.convert(rbox_tensor, BoxMode.XYWHA_ABS, BoxMode.XYXY_ABS)
    return ret
