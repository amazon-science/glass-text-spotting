# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable

import torch
from fvcore.nn import smooth_l1_loss


def arc_loss():
    raise NotImplementedError


def cosine_loss(pred_box_deltas: torch.Tensor, gt_box_deltas: torch.Tensor,
                smooth_l1_beta: float = 0, reduction: str = "none", weights: torch.Tensor = None):
    angle_loss_func: Callable[[Any], torch.Tensor] = lambda x: torch.abs(1 - torch.cos(x))
    return _angle_loss(angle_loss_func, pred_box_deltas, gt_box_deltas, smooth_l1_beta, reduction, weights)


def sine_loss(pred_box_deltas: torch.Tensor, gt_box_deltas: torch.Tensor,
              smooth_l1_beta: float = 0, reduction: str = "none", weights: torch.Tensor = None):
    angle_loss_func: Callable[[Any], torch.Tensor] = lambda x: torch.abs(torch.sin(x))
    return _angle_loss(angle_loss_func, pred_box_deltas, gt_box_deltas, smooth_l1_beta, reduction, weights)


def sine_square_loss(pred_box_deltas: torch.Tensor, gt_box_deltas: torch.Tensor,
                     smooth_l1_beta: float = 0, reduction: str = "none", weights: torch.Tensor = None):
    angle_loss_func: Callable[[Any], torch.Tensor] = lambda x: torch.pow(torch.sin(x), 2)
    return _angle_loss(angle_loss_func, pred_box_deltas, gt_box_deltas, smooth_l1_beta, reduction, weights)


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
