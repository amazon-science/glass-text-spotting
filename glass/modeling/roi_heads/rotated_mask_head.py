# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import List

import matplotlib
import numpy as np
import torch
from PIL import Image
from detectron2.config import configurable
from detectron2.layers import cat
from detectron2.modeling.roi_heads.mask_head import ROI_MASK_HEAD_REGISTRY, mask_rcnn_inference, \
    MaskRCNNConvUpsampleHead
from detectron2.structures import Instances
from detectron2.structures.masks import polygons_to_bitmask
from detectron2.utils.events import get_event_storage
from detectron2.utils.visualizer import Visualizer as D2Visualizer
from shapely import affinity
from torch.nn import functional as F


def rotate_crop_and_resize(polygons, boxes: torch.Tensor, mask_size: int) -> torch.Tensor:
    """
    Crop each mask by the given rotated box, and resize results to (mask_size, mask_size).

    Args:
        PolygonMask
        Rotatedboxes (Tensor): Nx5 tensor storing the boxes for each mask
        mask_size (int): the size of the rasterized mask.

    Returns:
        Tensor: A bool tensor of shape (N, mask_size, mask_size), where
        N is the number of predicted boxes for this image.
    """
    assert len(boxes) == len(polygons), "{} != {}".format(len(boxes), len(polygons))

    device = boxes.device
    # Put boxes on the CPU, as the polygon representation is not efficient GPU-wise
    # (several small tensors for representing a single instance mask)
    boxes = boxes.to(torch.device("cpu"))

    results = [
        rasterize_polygons_within_rotated_box(poly, box.numpy(), mask_size)
        for poly, box in zip(polygons, boxes)
    ]
    """
    poly: list[list[float]], the polygons for one instance
    box: a tensor of shape (5,)
    """
    if len(results) == 0:
        return torch.empty(0, mask_size, mask_size, dtype=torch.bool, device=device)
    return torch.stack(results, dim=0).to(device=device)

def test(polygon, box):
    def polygon_from_points(points):
        """
        Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
        """
        from shapely.geometry import Polygon
        num_points = len(points)
        # resBoxes=np.empty([1,num_points],dtype='int32')
        resBoxes = np.empty([1, num_points], dtype='float32')
        for inp in range(0, num_points, 2):
            resBoxes[0, int(inp / 2)] = float(points[int(inp)])
            resBoxes[0, int(inp / 2 + num_points / 2)] = float(points[int(inp + 1)])
        pointMat = resBoxes[0].reshape([2, int(num_points / 2)]).T
        return Polygon(pointMat)

    def rotated_boxes_to_polygons(box):

        assert (
                box.shape[-1] == 5
        ), "The last dimension of input shape must be 5 for XYWHA format"
        cx = box[0]
        cy = box[1]
        w = box[2]
        h = box[3]
        a = box[4]
        t = np.deg2rad(-a)
        polygons = np.zeros(8)
        sin_t = np.sin(t)
        cos_t = np.cos(t)
        # Computing X components
        polygons[0] = cx + (h * sin_t - w * cos_t) / 2
        polygons[2] = cx + (h * sin_t + w * cos_t) / 2
        polygons[4] = cx - (h * sin_t - w * cos_t) / 2
        polygons[6] = cx - (h * sin_t + w * cos_t) / 2
        # Computing Y components
        polygons[1] = cy - (h * cos_t + w * sin_t) / 2
        polygons[3] = cy - (h * cos_t - w * sin_t) / 2
        polygons[5] = cy + (h * cos_t + w * sin_t) / 2
        polygons[7] = cy + (h * cos_t - w * sin_t) / 2

        return polygons

    def points_from_polygon(polygon):
        x,y = polygon.exterior.xy
        return np.array((x,y)).T.reshape(-1)

    def visualize_preds_and_gt(img, poly, rotated_poly, rotated_box,aligned_box_poly, cx,cy):

        cmap = matplotlib.cm.get_cmap('gist_ncar')

        class_colors = cmap(np.linspace(0, 1, 4))
        dirn = "/hiero_efs/HieroUsers/roironen/"


        # img = np.transpose(img,[1,2,0])
        v_gt = D2Visualizer(img, None)

        # visualize gt
        poly = points_from_polygon(poly)
        poly[::2] += - cx + 200
        poly[1::2] += - cy + 200

        rotated_poly = points_from_polygon(rotated_poly)
        rotated_poly[::2] += - cx + 400
        rotated_poly[1::2] += - cy + 400

        rotated_box[::2] += - cx + 200
        rotated_box[1::2] += - cy + 200

        aligned_box_poly = points_from_polygon(aligned_box_poly)
        aligned_box_poly[::2] += - cx + 400
        aligned_box_poly[1::2] += - cy + 400
        v_gt = v_gt.overlay_instances(masks=[[poly],
                                             [rotated_poly],
                                             [rotated_box],
                                             [aligned_box_poly]],
                                      labels=[['poly'],['rot_poly'],['box'],['aligned_box']],
                                      alpha=0.5)

        im = Image.fromarray(v_gt.get_image())
        im.save(dirn + "tmp.jpeg")

    cx = box[0]
    cy = box[1]
    w = box[2]
    h = box[3]
    a = box[4]
    poly = polygon_from_points(polygon)
    box_poly = rotated_boxes_to_polygons(box)
    aligned_box_poly = affinity.rotate(polygon_from_points(box_poly), a, (cx, cy))
    rotated_poly = affinity.rotate(poly, a, (cx, cy))
    im = np.ones((800,800,3))*255
    im[0,0,:]=0
    visualize_preds_and_gt(im,poly, rotated_poly, box_poly,aligned_box_poly,cx,cy)

def test_after(original_polygon, rotated_box, rotated_polygon):
    def polygon_from_points(points):
        """
        Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
        """
        from shapely.geometry import Polygon
        num_points = len(points)
        # resBoxes=np.empty([1,num_points],dtype='int32')
        resBoxes = np.empty([1, num_points], dtype='float32')
        for inp in range(0, num_points, 2):
            resBoxes[0, int(inp / 2)] = float(points[int(inp)])
            resBoxes[0, int(inp / 2 + num_points / 2)] = float(points[int(inp + 1)])
        pointMat = resBoxes[0].reshape([2, int(num_points / 2)]).T
        return Polygon(pointMat)

    def rotated_boxes_to_polygons(box):

        assert (
                box.shape[-1] == 5
        ), "The last dimension of input shape must be 5 for XYWHA format"
        cx = box[0]
        cy = box[1]
        w = box[2]
        h = box[3]
        a = box[4]
        t = np.deg2rad(-a)
        polygons = np.zeros(8)
        sin_t = np.sin(t)
        cos_t = np.cos(t)
        # Computing X components
        polygons[0] = cx + (h * sin_t - w * cos_t) / 2
        polygons[2] = cx + (h * sin_t + w * cos_t) / 2
        polygons[4] = cx - (h * sin_t - w * cos_t) / 2
        polygons[6] = cx - (h * sin_t + w * cos_t) / 2
        # Computing Y components
        polygons[1] = cy - (h * cos_t + w * sin_t) / 2
        polygons[3] = cy - (h * cos_t - w * sin_t) / 2
        polygons[5] = cy + (h * cos_t + w * sin_t) / 2
        polygons[7] = cy + (h * cos_t - w * sin_t) / 2

        return polygons

    def points_from_polygon(polygon):
        x,y = polygon.exterior.xy
        return np.array((x,y)).T.reshape(-1)

    def visualize_preds_and_gt(img, poly, rotated_poly, rotated_box,aligned_box_poly, cx,cy):

        cmap = matplotlib.cm.get_cmap('gist_ncar')

        class_colors = cmap(np.linspace(0, 1, 4))
        dirn = "/hiero_efs/HieroUsers/roironen/"


        # img = np.transpose(img,[1,2,0])
        v_gt = D2Visualizer(img, None)

        # visualize gt
        poly = points_from_polygon(poly)
        poly[::2] += - cx + 200
        poly[1::2] += - cy + 200

        rotated_poly = rotated_poly
        rotated_poly[::2] += - cx + 400
        rotated_poly[1::2] += - cy + 400

        rotated_box[::2] += - cx + 200
        rotated_box[1::2] += - cy + 200

        aligned_box_poly = points_from_polygon(aligned_box_poly)
        aligned_box_poly[::2] += - cx + 400
        aligned_box_poly[1::2] += - cy + 400
        v_gt = v_gt.overlay_instances(masks=[[poly],
                                             [rotated_poly],
                                             [rotated_box],
                                             [aligned_box_poly]],
                                      labels=[['poly'],['rot_poly'],['box'],['aligned_box']],
                                      alpha=0.5)

        im = Image.fromarray(v_gt.get_image())
        im.save(dirn + "tmp_after.jpeg")

    cx = rotated_box[0]
    cy = rotated_box[1]
    w = rotated_box[2]
    h = rotated_box[3]
    a = rotated_box[4]
    poly = polygon_from_points(original_polygon)
    box_poly = rotated_boxes_to_polygons(rotated_box)
    aligned_box_poly = affinity.rotate(polygon_from_points(box_poly), a, (cx, cy))
    # rotated_poly = affinity.rotate(poly, a, (cx, cy))
    im = np.ones((800,800,3))*255
    im[0,0,:]=0
    visualize_preds_and_gt(im,poly, rotated_polygon, box_poly,aligned_box_poly,cx,cy)

def rasterize_polygons_within_rotated_box(
    polygons: List[np.ndarray], box: np.ndarray, mask_size: int
) -> torch.Tensor:
    """
    Rasterize the polygons into a mask image and
    crop the mask content in the given box.
    The cropped mask is resized to (mask_size, mask_size).

    This function is used when generating training targets for mask head in Mask R-CNN.
    Given original ground-truth masks for an image, new ground-truth mask
    training targets in the size of `mask_size x mask_size`
    must be provided for each predicted box. This function will be called to
    produce such targets.

    Args:
        polygons (list[ndarray[float]]): a list of polygons, which represents an instance.
        box: 5-element numpy array
        mask_size (int):

    Returns:
        Tensor: BoolTensor of shape (mask_size, mask_size)
    """
    # test(copy.deepcopy(polygons[0]),copy.deepcopy(box))
    # 1. Recenter, rotate return to original coordinates
    rotated_polygon = copy.deepcopy(polygons)

    cx = box[0]
    cy = box[1]
    w = box[2]
    h = box[3]
    a = box[4]
    a = np.deg2rad(-a)

    for i, p in enumerate(rotated_polygon):
        p[0::2] = p[0::2] - cx
        p[1::2] = p[1::2] - cy
        p = p.reshape((-1, 2)).T

        rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        p = rot @ p
        p = p.T.reshape(-1)

        p[0::2] = p[0::2] + cx
        p[1::2] = p[1::2] + cy
        rotated_polygon[i] = p

    # test_after(copy.deepcopy(polygons[0]), copy.deepcopy(box), copy.deepcopy(rotated_polygon[0]))
    # 2. Shift the polygons w.r.t the boxes
    # The relative angle in rotated box coordinate system is 0
    sin_t = np.sin(0)
    cos_t = np.cos(0)
    # Computing X components
    x0 = cx + (h * sin_t - w * cos_t) / 2

    # Computing Y components
    y0 = cy - (h * cos_t + w * sin_t) / 2

    for p in rotated_polygon:
        p[0::2] = p[0::2] - x0
        p[1::2] = p[1::2] - y0

    # 2. Rescale the polygons to the new box size
    # max() to avoid division by small number
    ratio_h = mask_size / max(h, 0.1)
    ratio_w = mask_size / max(w, 0.1)

    if ratio_h == ratio_w:
        for p in rotated_polygon:
            p *= ratio_h
    else:
        for p in rotated_polygon:
            p[0::2] *= ratio_w
            p[1::2] *= ratio_h

    # 3. Rasterize the polygons with coco api
    mask = polygons_to_bitmask(rotated_polygon, mask_size, mask_size)
    mask = torch.from_numpy(mask)
    return mask

@torch.jit.unused
def rotated_mask_rcnn_loss(pred_mask_logits: torch.Tensor, instances: List[Instances], vis_period: int = 0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)
        #

        #
        gt_masks_per_image = rotate_crop_and_resize(instances_per_image.gt_masks,
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    return mask_loss


@ROI_MASK_HEAD_REGISTRY.register()
class RotatedMaskRCNNConvUpsampleHead(MaskRCNNConvUpsampleHead):
    @configurable
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["loss_weight"] = cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT
        pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        assert pooler_type in ["ROIAlignRotated"], pooler_type
        return ret

    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        x = self.layers(x)
        if self.training:
            assert not torch.jit.is_scripting()
            return {"loss_mask": rotated_mask_rcnn_loss(x, instances, self.vis_period) * self.loss_weight}
        else:
            mask_rcnn_inference(x, instances)
            return instances
