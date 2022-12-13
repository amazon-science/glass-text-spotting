import time
import logging

import cv2
import numpy as np
import torch
from detectron2.layers.nms import nms_rotated
from detectron2.structures.instances import Instances
from detectron2.utils.registry import Registry

from ..structures.boxes import pairwise_ioa_rotated

POST_PROCESSOR_REGISTRY = Registry("POST_PROCESSOR")
POST_PROCESSOR_REGISTRY.__doc__ = """
Registry for the post processor algorithms used following the detection process by the model
Usually accustomed to the output type, axis-aligned-boxes/rotated-boxes/etc...

Each returned object should have the same `__init__` signature, as well as a `process()` method called from the 
d2 runner module.
"""


def build_post_processor(cfg, *args, **kwargs):
    """
    Build a post-processing component defined by `cfg.POST_PROCESSING.NAME`.
    """
    name = cfg.POST_PROCESSING.NAME

    return POST_PROCESSOR_REGISTRY.get(name)(cfg, *args, **kwargs)


@POST_PROCESSOR_REGISTRY.register()
class PostProcessorRotatedBoxes:

    def __init__(self, cfg):
        self.logger = logging.getLogger(__name__)
        self.skip_all = cfg.POST_PROCESSING.SKIP_ALL

        # Below this threshold we will NEVER consider merging two boxes, has to be bigger than 0 for numerical reasons
        self.minimal_ioa_thresh = 0.01

        self.class_names = list(cfg.MODEL.ROI_HEADS.CLASS_NAMES)
        self.word_ind = self.class_names.index('word')

        # Score thresholds for each of our classes printed accordingly
        self.detect_threshold = cfg.POST_PROCESSING.DETECT_THRESHOLD

        # We drop boxes that have less than this number of pixels in one of their dimensions
        self.min_box_dim = cfg.POST_PROCESSING.MIN_BOX_DIMENSION

        # Above this IoU threshold we merge two boxes
        self.merge_ioa_thresh = cfg.POST_PROCESSING.MERGE_IOA_THRESH
        self.pairs_height_ratio_thresh = cfg.POST_PROCESSING.PAIRS_HEIGHT_RATIO_THRESH

        # How many pixels to add (or subtract) to each box [top, right, bottom, left]
        self.box_px_padding = cfg.POST_PROCESSING.BOX_PX_PADDING

        self.max_input_size = cfg.INPUT.MAX_SIZE_TEST
        # Boxes above this threshold are considered valid
        self.valid_score = cfg.POST_PROCESSING.VALID_CONFIDENCE
        assert self.valid_score <= self.detect_threshold, \
            "Valid score threshold must be smaller than the other class thresholds, to prevent word-in-word  cases"

        self.max_angle_diff = cfg.POST_PROCESSING.MAX_ANGLE_DIFF

    def __call__(self, preds: Instances):
        """
        :param preds: The word predictions as given from the D2 inference pipeline
        :return:
        """
        if self.skip_all:
            self.logger.warning('SKIPPING POST PROCESSING - "SKIP_ALL" is "True" in config file')
            return preds
        word_start_time = time.perf_counter()

        # Filtering out boxes that are smaller than self.min_box_dim
        preds = self.filter_small_boxes(preds)

        preds = self.post_process_word_preds(preds)

        self.logger.info(f'Merged and removed {len(preds)} Words')
        self.logger.info(f'Post-Process Word Time: {(time.perf_counter() - word_start_time) * 1e3:.1f} ms')

        # Updating the polygons
        preds.pred_polygons = self.boxes_to_polygons(preds.pred_boxes.tensor)

        return preds

    def filter_small_boxes(self, preds: Instances):
        if len(preds) == 0:
            return preds
        boxes = preds.pred_boxes.tensor
        min_box_dim = torch.min(boxes[:, 2], boxes[:, 3])
        return preds[min_box_dim >= self.min_box_dim]

    def post_process_word_preds(self, preds: Instances):

        # another round of merging for all the word boxes if the terms for printed are met
        # this time we take the stricter parameter for each merge type
        preds = preds[preds.scores >= self.valid_score]
        preds = self.merge_intersecting_boxes(
            preds,
            ioa_threshold=self.merge_ioa_thresh,
            pairs_height_ratio_thresh=self.pairs_height_ratio_thresh)
        preds = preds[preds.scores >= self.detect_threshold]
        return preds

    def merge_intersecting_boxes(self, preds: Instances,
                                 ioa_threshold: float,
                                 pairs_height_ratio_thresh: float,
                                 ):
        if len(preds) == 0:
            return preds

        # We continue merging boxes until no boxes are merged in each iteration, we break inside
        while True:

            # These overlapping boxes we want to merge
            boxes = preds.pred_boxes.tensor
            ioa = pairwise_ioa_rotated(boxes, boxes)  # NxN symmetric IoA matrix
            scores = preds.scores

            # We take the row and column indices only from the upper diagonal to avoid duplicate
            pairs = torch.nonzero(ioa.fill_diagonal_(0).triu() >= self.minimal_ioa_thresh)  # M x 2 of paired indices

            # If there are no touching predictions we can break off and continue
            if len(pairs) == 0:
                break

            # Intrinsic box parameters used for defining the valid pair masks
            box_heights = boxes[:, 3]  # M x 1
            box_angles = boxes[:, 4]

            # The angle difference between two boxes
            box_angles_diff = box_angles[pairs[:, 1]] - box_angles[pairs[:, 0]]
            box_angles_diff = torch.abs((box_angles_diff + 180) % 360 - 180)  # Always in [0, 180) range

            # We only merge if the angle diff is smaller than 15 degrees
            similar_angle_mask = ((box_angles_diff < self.max_angle_diff) |
                                  (box_angles_diff > (180 - self.max_angle_diff)))

            # We only merge if the boxes are of similar height
            pairs_height_ratio = box_heights[pairs[:, 1]] / box_heights[pairs[:, 0]]  # M x 1
            similar_height_mask = (pairs_height_ratio_thresh < pairs_height_ratio) & (
                    pairs_height_ratio < (1 / (pairs_height_ratio_thresh + 1e-6)))  # M x 1

            # Computing the pairs where both boxes are above the valid score
            min_pair_score = torch.min(scores[pairs[:, 0]], scores[pairs[:, 1]])  # M x 1
            valid_score_mask = min_pair_score >= self.valid_score

            # This mask checks which pairs have an IoA higher than the ioa_threshold
            ioa_mask = ioa[pairs[:, 0], pairs[:, 1]] >= ioa_threshold
            intersection_mask = ioa_mask

            # Keeping only the pairs that meet all the terms
            combined_valid_mask = valid_score_mask & similar_height_mask & intersection_mask & similar_angle_mask

            # STOPPING CONDITION: If the mask is all False, we are done with the merge procedure
            if (~combined_valid_mask).all():
                break

            valid_pairs = pairs[combined_valid_mask]
            valid_boxes1 = boxes[valid_pairs[:, 0]]
            valid_boxes2 = boxes[valid_pairs[:, 1]]
            scores1 = preds.scores[valid_pairs[:, 0]]
            scores2 = preds.scores[valid_pairs[:, 1]]

            # # Merging the boxes that met all the terms
            start_time = time.perf_counter()
            valid_boxes1 = self._merge_rotated_boxes(valid_boxes1, valid_boxes2, scores1, scores2)
            merge_time = time.perf_counter() - start_time
            self.logger.info(f'Merge time: {1000 * merge_time:.1f} ms')

            # The computation is symmetric to boxes1 and boxes2, because items can appear multiple times in valid_pairs
            valid_boxes2 = valid_boxes1.clone()
            # Saving back the merged boxes
            preds.pred_boxes.tensor[valid_pairs[:, 0]] = valid_boxes1
            preds.pred_boxes.tensor[valid_pairs[:, 1]] = valid_boxes2

            # Finally we do NMS to drop the overlapping boxes with the lower confidence (hence the high threshold here)
            inds_to_keep = nms_rotated(preds.pred_boxes.tensor, preds.scores, iou_threshold=0.99)
            # noinspection PyTypeChecker
            preds = preds[inds_to_keep]
        return preds

    @classmethod
    def _merge_rotated_boxes(cls, boxes1: torch.Tensor, boxes2: torch.Tensor,
                             scores1: torch.Tensor = None, scores2: torch.Tensor = None) -> torch.Tensor:
        """
        Merges detectron2 rotated boxes
        :param boxes1: An Nx5 tensor with the rotated boxes in XYWHA format
        :param boxes2: Tensor of boxes to merge with the boxes in boxes1, also in dimension Nx5
        :param scores1: The scores of boxes 1 in dimension Nx1
        :param scores2: The scores of boxes 2 in dimension Nx1
        :return: An Nx5 tensor, each correspondingly containing the merger of box1 and box2
        """
        assert len(boxes1) == len(boxes2), 'We only combine pairs of boxes, please insert same boxes lengths'

        polygons1 = cls.boxes_to_polygons(boxes1)  # (N, 4, 2)
        polygons2 = cls.boxes_to_polygons(boxes2)  # (N, 4 ,2)

        angles1 = boxes1[:, 4] * np.pi / 180
        angles2 = boxes2[:, 4] * np.pi / 180
        if scores1 is not None and scores2 is not None:
            merged_angle = torch.where(scores1 >= scores2, angles1, angles2)
        else:
            merged_angle = torch.atan2(torch.sin(angles1) + torch.sin(angles2),
                                       torch.cos(angles1) + torch.cos(angles2)) * 180 / np.pi

        # Concatenating the vertices to a single polygon tensor
        polygons = torch.hstack((polygons1, polygons2))  # (N, 8, 2)

        # Going back from the polygons to the rotated boxes approximation
        merged_rotated_boxes = cls.polygons_to_rotated_boxes(polygons, orientations=merged_angle)

        return merged_rotated_boxes

    @staticmethod
    def boxes_to_polygons(boxes: torch.Tensor) -> torch.Tensor:
        """
        Returns the polygons from a rotated boxes tensor representation.
        The first vertex is the top left
        :param boxes: A torch tensor depicting the rotated boxes in XYWHA format, (N, 5)
        :return: The polygon representation tensor of the rotated boxes in dimension (N, 4, 2)
        """
        # The number of boxes
        n = len(boxes)
        if n == 0:
            return torch.tensor([]).reshape((0, 4, 2)).to(dtype=boxes.dtype, device=boxes.device)

        # Extracting the vectors out of rotated box
        cx, cy, w, h, a = boxes.T  # Each tensor of size N
        t = (-a / 180) * np.pi  # Computing in radians

        # Instantiating the output polygons
        polygons = torch.zeros((n, 4, 2)).to(dtype=boxes.dtype, device=boxes.device)
        # Computing X components
        sin_t = torch.sin(t)
        cos_t = torch.cos(t)
        polygons[:, 0, 0] = cx + (h * sin_t - w * cos_t) / 2
        polygons[:, 1, 0] = cx + (h * sin_t + w * cos_t) / 2
        polygons[:, 2, 0] = cx - (h * sin_t - w * cos_t) / 2
        polygons[:, 3, 0] = cx - (h * sin_t + w * cos_t) / 2
        # Computing Y components
        polygons[:, 0, 1] = cy - (h * cos_t + w * sin_t) / 2
        polygons[:, 1, 1] = cy - (h * cos_t - w * sin_t) / 2
        polygons[:, 2, 1] = cy + (h * cos_t + w * sin_t) / 2
        polygons[:, 3, 1] = cy + (h * cos_t - w * sin_t) / 2

        return polygons

    @staticmethod
    def polygons_to_rotated_boxes(polygons: torch.Tensor, orientations: torch.Tensor = None) -> torch.Tensor:
        """
        Transforms the input polygon tensor to a rotated box tensor

        :param polygons:
        :param orientations: The orientation angle for each of the output rotated boxes Nx5
        :return:
        """
        np_polygons = polygons.cpu().numpy()
        rotated_boxes = torch.zeros((len(polygons), 5))
        for i, polygon in enumerate(np_polygons):
            center, shape, angle = cv2.minAreaRect(np.array(polygon))
            angle = 90 - angle  # Flipping to adhere to our angle conventions
            # angle = (angle + 180) % 360 - 180  # making sure we are in (-180, 180]

            # Computing the diff angle between desired orientation and computed rectangle angles
            diff_angle = (orientations[i] - angle) if orientations is not None else 0.
            diff_angle = (diff_angle + 180) % 360 - 180

            # Correcting the rectangle according to the input orientation
            if -45 < diff_angle <= 45:
                width, height = shape[1], shape[0]
            elif 45 < diff_angle <= 135:
                width, height = shape[0], shape[1]
                angle += 90
            elif -135 < diff_angle <= -45:
                width, height = shape[0], shape[1]
                angle -= 90
            else:  # angle is > 135 or < -135
                width, height = shape[1], shape[0]
                angle += 180
            angle = (angle + 180) % 360 - 180
            rotated_boxes[i] = torch.tensor([center[0], center[1], width, height, angle])
        return rotated_boxes.to(device=polygons.device, dtype=polygons.dtype)
