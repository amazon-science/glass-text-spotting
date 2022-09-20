# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from typing import List, Tuple, Union
import torch

from detectron2.structures.boxes import BoxMode as D2BoxMode
from detectron2.structures.boxes import Boxes as D2Boxes
from detectron2.structures.boxes import pairwise_ioa as d2_pairwise_ioa
from detectron2.structures.boxes import pairwise_intersection as d2_pairwise_intersection

_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]

# Making sure we have the correct classes loaded from the original boxes model
BoxMode = D2BoxMode
Boxes = D2Boxes
pairwise_ioa = d2_pairwise_ioa
pairwise_intersection = d2_pairwise_intersection


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
