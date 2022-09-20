# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Callable, Any
import torch
import torch.nn.functional as F
from fvcore.nn import smooth_l1_loss

from detectron2.layers import cat
from detectron2.structures import Boxes
from detectron2.utils.events import get_event_storage
from detectron2.modeling.proposal_generator.rrpn import RRPN
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY


@PROPOSAL_GENERATOR_REGISTRY.register()
class RotatedRPN(RRPN):

    @torch.jit.unused
    def losses(
            self,
            anchors: List[Boxes],
            pred_objectness_logits: List[torch.Tensor],
            gt_labels: List[torch.Tensor],
            pred_anchor_deltas: List[torch.Tensor],
            gt_boxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

        # Preparing the inputs for the loss computation
        anchors = type(anchors[0]).cat(anchors).tensor  # Ax(4 or 5)
        gt_anchor_deltas = [self.box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, sum(Hi*Wi*Ai), 4 or 5)

        if self.box_reg_loss_type == "smooth_l1":
            localization_loss = smooth_l1_loss(
                cat(pred_anchor_deltas, dim=1)[pos_mask],
                gt_anchor_deltas[pos_mask],
                self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "sine_square_loss":
            localization_loss = sine_square_loss(cat(pred_anchor_deltas, dim=1)[pos_mask], gt_anchor_deltas[pos_mask],
                                                 self.smooth_l1_beta, reduction="sum",
                                                 weights=self.box2box_transform.weights)

        else:
            raise ValueError(f"Invalid rpn box reg loss type '{self.box_reg_loss_type}'")

        valid_mask = gt_labels >= 0
        objectness_loss = F.binary_cross_entropy_with_logits(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            reduction="sum",
        )
        normalizer = self.batch_size_per_image * num_images
        losses = {
            "loss_rpn_cls": objectness_loss / normalizer,
            "loss_rpn_loc": localization_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses


def _angle_loss(
        angle_loss_func,
        pred_box_deltas: torch.Tensor,
        gt_box_deltas: torch.Tensor,
        smooth_l1_beta: float = 0.,
        reduction: str = "none",
        weights: torch.Tensor = None
) -> torch.Tensor:
    # Extracting the spatial coordinates and the angular coordinate
    x, y, w, h, angle = pred_box_deltas.unbind(dim=-1)
    x_gt, y_gt, w_gt, h_gt, angle_gt = gt_box_deltas.unbind(dim=-1)

    # Building only the box deltas
    pred_box_proposal_deltas = torch.stack([x, y, w, h], dim=-1)
    gt_box_proposal_deltas = torch.stack([x_gt, y_gt, w_gt, h_gt], dim=-1)

    # Computing traditional L1 loss
    loss_box_reg = smooth_l1_loss(
        pred_box_proposal_deltas,
        gt_box_proposal_deltas,
        beta=smooth_l1_beta,
        reduction='none',
    )

    # Normalizing the angle, because the delta angle is already multiplied by angle
    angle_weight = weights[4]
    angle_diff = (angle - angle_gt) / angle_weight

    # This is the angle loss computation
    loss_angle_reg = angle_weight * angle_loss_func(angle_diff)

    # Combining to a single loss function and applying the reduction of our choice
    loss = torch.cat([loss_box_reg, loss_angle_reg[:, None]], dim=-1)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def sine_square_loss(pred_box_deltas: torch.Tensor, gt_box_deltas: torch.Tensor,
                     smooth_l1_beta: float = 0, reduction: str = "none", weights: torch.Tensor = None):
    angle_loss_func: Callable[[Any], torch.Tensor] = lambda x: torch.pow(torch.sin(x), 2)
    return _angle_loss(angle_loss_func, pred_box_deltas, gt_box_deltas, smooth_l1_beta, reduction, weights)
