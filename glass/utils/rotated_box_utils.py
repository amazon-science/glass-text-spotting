import torch


def overwrite_orientations_on_boxes(boxes: torch.tensor, orientations):
    """
    We take the predicted orientation (cardinal direction of the instance), and apply it to the predicted
    rotated box, namely, modifying the angle by multiples of 90 degrees, and possibly switching between
    height and width (if orientation angle is 90 or 270)
    :param boxes: Rotated boxes tensor with N x 5 shape
    :param orientations: A vector of integer absolute orientations in the range [0, 3]
    :return:
    """
    if len(orientations) == 0:
        logger.warning('apply orientations to boxes got an empty tensor for orientations')
        return boxes

    if orientations[0] is None:
        raise Exception('Cannot apply orientation to boxes, no orientation head')

    # Extracting the individual coordinates
    cx, cy, width, height, angle = boxes.T
    box_orientations = (torch.round(angle / 90) % 4).to(torch.long)
    orientation_deltas = (box_orientations - orientations) % 4

    # Replacing width with height if orientation is 90 or 270
    mask_orientation_is_90_270 = (orientation_deltas == 1) | (orientation_deltas == 3)
    new_width = torch.where(mask_orientation_is_90_270, height, width)
    new_height = torch.where(mask_orientation_is_90_270, width, height)

    # Modifying the angle of the box according to the orientation, and bringing back to [-180, 180) range
    new_angle = ((angle + (90 * orientation_deltas) + 180) % 360) - 180

    # Building back the modified box tensor for the cardinal rotated boxes
    new_boxes = torch.stack([cx, cy, new_width, new_height, new_angle]).T
    return new_boxes
