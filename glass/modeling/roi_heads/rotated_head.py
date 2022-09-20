import numpy as np
import torch
import logging
from typing import Dict, List
import math

from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.roi_heads.rotated_fast_rcnn import RROIHeads, ROI_HEADS_REGISTRY, \
    RotatedFastRCNNOutputLayers, StandardROIHeads
from detectron2.structures import RotatedBoxes, pairwise_iou_rotated, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.poolers import ROIPooler

from glass.modeling.roi_heads.rotated_fast_rcnn import RotatedFastRCNNOutputLayers
from glass.structures.boxes import box_to_rbox

logger = logging.getLogger(__name__)


@ROI_HEADS_REGISTRY.register()
class RotatedROIHeads(RROIHeads):

    def __init__(self, *args, **kwargs):
        # We initialize with the parent of RROIHeads, to avoid fail when working with E2E/Mask head
        StandardROIHeads.__init__(self, *args, **kwargs)

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
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou_rotated(
                targets_per_image.gt_boxes.tensor, proposals_per_image.proposal_boxes.tensor
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


def add_ground_truth_to_proposals(gt_boxes, proposals):
    """
    Call `add_ground_truth_to_proposals_single_image` for all images.

    Args:
        gt_boxes(list[Boxes]): list of N elements. Element i is a Boxes
            representing the gound-truth for image i.
        proposals (list[Instances]): list of N elements. Element i is a Instances
            representing the proposals for image i.

    Returns:
        list[Instances]: list of N Instances. Each is the proposals for the image,
            with field "proposal_boxes" and "objectness_logits".
    """
    assert gt_boxes is not None

    assert len(proposals) == len(gt_boxes)
    if len(proposals) == 0:
        return proposals

    return [
        add_ground_truth_to_proposals_single_image(gt_boxes_i, proposals_i)
        for gt_boxes_i, proposals_i in zip(gt_boxes, proposals)
    ]

def add_ground_truth_to_proposals_single_image(gt_boxes, proposals):
    """
    Augment `proposals` with ground-truth boxes from `gt_boxes`.

    Args:
        Same as `add_ground_truth_to_proposals`, but with gt_boxes and proposals
        per image.

    Returns:
        Same as `add_ground_truth_to_proposals`, but for only one image.
    """
    device = proposals.objectness_logits.device
    # Assign all ground-truth boxes an objectness logit corresponding to
    # P(object) = sigmoid(logit) =~ 1.
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
    gt_logits = gt_logit_value * torch.ones(len(gt_boxes), device=device)

    # support box proposals with rotated head
    if proposals.proposal_boxes.tensor.shape[1] == 4:
        proposal_boxes_tensor = box_to_rbox(proposals.proposal_boxes.tensor)
        proposals.proposal_boxes = RotatedBoxes(proposal_boxes_tensor)

    # Concatenating gt_boxes with proposals requires them to have the same fields
    gt_proposal = Instances(proposals.image_size)
    gt_proposal.proposal_boxes = gt_boxes
    gt_proposal.objectness_logits = gt_logits
    new_proposals = Instances.cat([proposals, gt_proposal])

    return new_proposals