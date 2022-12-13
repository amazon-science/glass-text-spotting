from typing import Tuple

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.layers.nms import nms_rotated
from detectron2.structures.instances import Instances
from detectron2.utils.memory import retry_if_cuda_oom
from . import POST_PROCESSOR_REGISTRY
from .post_processor_rotated_boxes import PostProcessorRotatedBoxes
from ..modeling.recognition.text_encoder import TextEncoder
from ..structures.boxes import Boxes
from ..structures.boxes import pairwise_ioa_rotated
from ..evaluation.text_evaluator import get_instances_text


@POST_PROCESSOR_REGISTRY.register()
class PostProcessorAcademic(PostProcessorRotatedBoxes):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.text_threshold = cfg.POST_PROCESSING.TEXT_THRESHOLD
        self.text_encoder = TextEncoder(cfg)

    def __call__(self, preds, scale_ratio=1, **kwargs):
        # Invoking the regular
        preds = super().__call__(preds)

        # Filtering out predictions based on the text threshold
        texts, text_scores, _ = get_instances_text(preds.pred_text_prob, self.text_encoder)
        preds = preds[torch.tensor(text_scores) >= self.text_threshold]

        return preds

    @staticmethod
    def resize_boxes(preds: Instances, ratio: float, axis='both'):
        print('shlompi')
        if len(preds) == 0:
            return preds

        # Calculate inflation delta for each box
        boxes = preds.pred_boxes.tensor
        # The delta is computed as a proportion of the box height
        if axis == 'both':
            delta_x = ratio * boxes[:, 2]
            delta_y = ratio * boxes[:, 3]
        elif axis == 'vertical':
            delta_x = 0
            delta_y = ratio * boxes[:, 3]
        elif axis == 'horizontal':
            delta_x = ratio * boxes[:, 2]
            delta_y = 0
        else:
            raise Exception('Please provide an axis value of either "both"/"horizontal"/"vertical')

        # The boxes are of X,Y,W,H,A format
        boxes[:, 2] += delta_x
        boxes[:, 3] += delta_y
        preds.pred_boxes.tensor = boxes
        preds.pred_boxes.clip(preds.image_size)
        return preds

    @staticmethod
    def drop_overlapping_boxes(preds: Instances,
                               ioa_threshold: float,
                               valid_score: float,
                               minimal_ioa_thresh=0.01
                               ):
        if len(preds) == 0:
            return preds
        # These overlapping boxes we want to merge
        ioa = pairwise_ioa_rotated(preds.pred_boxes, preds.pred_boxes)  # NxN symmetric IoA matrix
        boxes = preds.pred_boxes.tensor
        scores = preds.scores

        # We compute the areas, assuming these are all rotated boxes
        assert boxes.shape[1] == 5
        areas = boxes[:, 2] * boxes[:, 3]

        # We take the row and column indices only from the upper diagonal to avoid duplicate
        pairs = torch.nonzero(ioa.fill_diagonal_(0).triu() >= minimal_ioa_thresh)  # M x 2 of paired indices

        # If there are no touching predictions we can return preds
        if len(pairs) == 0:
            return preds

        # Computing the pairs where both boxes are above the valid score
        min_pair_score = torch.min(scores[pairs[:, 0]], scores[pairs[:, 1]])  # M x 1
        valid_score_mask = min_pair_score >= valid_score

        # This mask checks which pairs have an IoA higher than the ioa_threshold
        intersection_mask = ioa[pairs[:, 0], pairs[:, 1]] >= ioa_threshold

        # Keeping only the pairs that meet all of the terms
        combined_valid_mask = valid_score_mask & intersection_mask
        # combined_valid_mask = valid_score_mask & intersection_mask

        # STOPPING CONDITION: If the mask is all False, we are done with the procedure
        if (~combined_valid_mask).all():
            return preds

        overlapping_pairs = pairs[combined_valid_mask]
        max_area_valid_boxes = torch.where((areas[overlapping_pairs[:, 0]] > areas[overlapping_pairs[:, 1]])[..., None],
                                           boxes[overlapping_pairs[:, 0]],
                                           boxes[overlapping_pairs[:, 1]])

        # Saving back the merged boxes
        preds.pred_boxes.tensor[overlapping_pairs[:, 0]] = max_area_valid_boxes.clone()
        preds.pred_boxes.tensor[overlapping_pairs[:, 1]] = max_area_valid_boxes.clone()
        # Finally we do NMS to drop the overlapping boxes with the lower confidence (hence the high threshold here)
        inds_to_keep = nms_rotated(preds.pred_boxes.tensor, preds.scores, iou_threshold=0.99)
        # noinspection PyTypeChecker
        preds = preds[inds_to_keep]
        return preds


def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """

    # Converts integer tensors to float temporaries
    #   to ensure true division is performed when
    #   computing scale_x and scale_y.
    if isinstance(output_width, torch.Tensor):
        output_width_tmp = output_width.float()
    else:
        output_width_tmp = output_width

    if isinstance(output_height, torch.Tensor):
        output_height_tmp = output_height.float()
    else:
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        results.pred_masks = retry_if_cuda_oom(paste_masks_in_image)(
            results.pred_masks[:, 0, :, :],  # N, 1, M, M
            results.pred_boxes,
            results.image_size,
            threshold=mask_threshold,
        )
    if results.has("pred_rboxes"):
        results.pred_rboxes.scale(scale_x, scale_y)
        results.pred_rboxes.clip(results.image_size)

    return results


BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 10000000 * 1024 ** 3  # memory limit


def paste_masks_in_image(
        masks: torch.Tensor, boxes: Boxes, image_shape: Tuple[int, int], threshold: float = 0.5
):
    """
    Identical to paste_masks_in_image in mask_ops, only uses modified _do_paste_mask method below
    """

    assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
    N = len(masks)
    if N == 0:
        return masks.new_empty((0,) + image_shape, dtype=torch.uint8)
    if not isinstance(boxes, torch.Tensor):
        boxes = boxes.tensor
    device = boxes.device
    assert len(boxes) == N, boxes.shape

    img_h, img_w = image_shape

    # The actual implementation split the input into chunks,
    # and paste them chunk by chunk.
    if device.type == "cpu" or torch.jit.is_scripting():
        # CPU is most efficient when they are pasted one by one with skip_empty=True
        # so that it performs minimal number of operations.
        num_chunks = N
    else:
        # GPU benefits from parallelism for larger chunks, but may have memory issue
        # int(img_h) because shape may be tensors in tracing
        num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        assert (
                num_chunks <= N
        ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

    img_masks = torch.zeros(
        N, img_h, img_w, device=device, dtype=torch.bool if threshold >= 0 else torch.uint8
    )
    for inds in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(
            masks[inds, None, :, :], boxes[inds], img_h, img_w, False  # skip_empty=device.type == "cpu"
        )

        if threshold >= 0:
            masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
        else:
            # for visualization and debugging
            masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

        if torch.jit.is_scripting():  # Scripting does not use the optimized codepath
            img_masks[inds] = masks_chunk
        else:
            img_masks[(inds,) + spatial_inds] = masks_chunk
    return img_masks


def _do_paste_mask(masks, boxes, img_h: int, img_w: int, skip_empty: bool = True):
    """
    Args:
        masks: N, 1, H, W
        boxes: N, 4 or N, 5
        img_h, img_w (int):
        skip_empty (bool): only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        if skip_empty == False, a mask of shape (N, img_h, img_w)
        if skip_empty == True, a mask of shape (N, h', w'), and the slice
            object for the corresponding region.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device

    # Extract x0, y0, x1, y1 from box or rotated box
    if boxes.shape[-1] == 4:
        x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1
    elif boxes.shape[-1] == 5:
        cx, cy, w, h, a = torch.split(boxes, 1, dim=1)  # each is Nx1
        a = torch.deg2rad(a)
        cos_a = torch.cos(a)
        sin_a = torch.sin(a)
        rot = torch.reshape(torch.stack([cos_a, sin_a, -sin_a, cos_a], 1), (-1, 2, 2))
        sin_t = 0  # torch.sin(0)
        cos_t = 1  # torch.cos(0)
        # Computing X components
        x0 = cx + (h * sin_t - w * cos_t) / 2
        x1 = cx - (h * sin_t - w * cos_t) / 2
        # Computing Y components
        y0 = cy - (h * cos_t + w * sin_t) / 2
        y1 = cy + (h * cos_t + w * sin_t) / 2

    else:
        raise ValueError

    if skip_empty and not torch.jit.is_scripting():
        x0_int = torch.clamp(x0.min().floor() - 1, min=0).to(
            dtype=torch.int32
        )
        y0_int = torch.clamp(y0.min().floor() - 1, min=0).to(
            dtype=torch.int32
        )
        x1_int = torch.clamp(x1.max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(y1.max().ceil() + 1, max=img_h).to(dtype=torch.int32)

    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h

    N = masks.shape[0]

    if boxes.shape[-1] == 4:
        img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
        img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
        img_y = (img_y - y0) / (y1 - y0) * 2 - 1
        img_x = (img_x - x0) / (x1 - x0) * 2 - 1
        # img_x, img_y have shapes (N, w), (N, h)

        gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
        gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
        grid = torch.stack([gx, gy], dim=3)
    elif boxes.shape[-1] == 5:
        # grid = torch.sum(grid[...,None] * rot[:,None,None,...],dim=-2) #NxWxHx2X1 * Nx1x1x2x2 TODO faster implementation
        grid = torch.zeros([N, img_h, img_w, 2], device=device, dtype=torch.float32)
        for i in range(N):
            # center the sampling grid to cx cy
            img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5 - cy[i]
            img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5 - cx[i]
            gx = img_x[None, :].expand(img_y.size(0), img_x.size(0))
            gy = img_y[:, None].expand(img_y.size(0), img_x.size(0))
            # rotate the grid according to rotated box angle and recenter the grid
            igrid = torch.stack([gx, gy], dim=2) @ rot[i]
            igrid[..., 0] += cx[i]
            igrid[..., 1] += cy[i]
            # normalize the grid to -1 1
            igrid[..., 0] = (igrid[..., 0] - x0[i]) / (x1[i] - x0[i]) * 2 - 1
            igrid[..., 1] = (igrid[..., 1] - y0[i]) / (y1[i] - y0[i]) * 2 - 1
            grid[i] = igrid

    if not torch.jit.is_scripting():
        if not masks.dtype.is_floating_point:
            masks = masks.float()
    img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)

    if skip_empty and not torch.jit.is_scripting():
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()
