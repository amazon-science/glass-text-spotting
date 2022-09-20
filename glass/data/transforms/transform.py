# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import cv2
from detectron2.data.transforms import RotationTransform


class FastResizeTransform:
    """
    Overrides the costly ResizeTransform in the original detectron2 package
    """

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        # Using nearest we save valuable time in training (~10%)
        ret = cv2.resize(img, dsize=(self.new_w, self.new_h), interpolation=cv2.INTER_NEAREST)
        return ret


# Patching up the RotationTransform for supporting rotated boxes
def rotate_rotated_box(transform, rotated_boxes):
    """
    Apply the rotation transform on rotated boxes. Rotated boxes are Nx5, with X,Y,W,H,A
    """
    # Shifting the center points center_x and center_y
    rotated_boxes[:, :2] = transform.apply_coords(rotated_boxes[:, :2])
    # Shifting the angle by the rotation amount
    rotated_boxes[:, 4] += transform.angle
    return rotated_boxes


# Registering the above method to the RotationTransform class from detectron2
RotationTransform.register_type("rotated_box", rotate_rotated_box)
