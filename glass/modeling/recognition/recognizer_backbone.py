# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from torch import nn
import fvcore.nn.weight_init as weight_init

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.registry import Registry

RECOGNIZER_BACKBONE_REGISTRY = Registry("RECOGNIZER_BACKBONE")
RECOGNIZER_BACKBONE_REGISTRY.__doc__ = """

"""


def build_recognizer_backbone(cfg, input_shape):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.RECOGNIZER_HEAD.BACKBONE.NAME
    return RECOGNIZER_BACKBONE_REGISTRY.get(name)(cfg, input_shape)


def build_recognizer_backbonev2(cfg, input_shape):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.ROI_RECOGNIZER_HEAD.RECOGNIZER_HEAD.BACKBONE.NAME
    return RECOGNIZER_BACKBONE_REGISTRY.get(name)(cfg, input_shape)


@RECOGNIZER_BACKBONE_REGISTRY.register()
class CNN_V1_1(nn.Module):
    """
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, conv_norm):
        """
        Args:
            input_shape (ShapeSpec): shape of the input feature
        """
        super().__init__()
        self.conv1 = Conv2d(
            input_shape.channels,
            input_shape.channels,
            kernel_size=[2, 1],
            stride=[2, 1],
            padding=0,
            bias=not conv_norm,
            norm=get_norm(conv_norm, input_shape.channels),
            activation=nn.ReLU(),
        )

        self.conv2 = Conv2d(
            input_shape.channels,
            input_shape.channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, input_shape.channels),
            activation=nn.ReLU(),
        )

        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {}
        ret["input_shape"] = input_shape
        ret["conv_norm"] = cfg.MODEL.ROI_RECOGNIZER_HEAD.NORM
        return ret

    def forward(self, x):
        x1 = self.conv1(x)
        conv2 = self.conv2(x1)
        out = conv2 + x1
        return out


@RECOGNIZER_BACKBONE_REGISTRY.register()
class CNN_V2_1(nn.Module):
    """
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, conv_norm):
        """
        Args:
            input_shape (ShapeSpec): shape of the input feature
        """
        super().__init__()
        self.conv1 = Conv2d(
            input_shape.channels,
            input_shape.channels,
            kernel_size=[2, 1],
            stride=[2, 1],
            padding=0,
            bias=not conv_norm,
            norm=get_norm(conv_norm, input_shape.channels),
            activation=nn.ReLU(),
        )

        self.conv2 = Conv2d(
            input_shape.channels,
            input_shape.channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, input_shape.channels),
            activation=nn.ReLU(),
        )

        self.conv3 = Conv2d(
            input_shape.channels,
            input_shape.channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, input_shape.channels),
            activation=nn.ReLU(),
        )

        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)
        weight_init.c2_msra_fill(self.conv3)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {}
        ret["input_shape"] = input_shape
        ret["conv_norm"] = cfg.MODEL.ROI_RECOGNIZER_HEAD.NORM
        return ret

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x12 = x2 + x1
        x3 = self.conv3(x12)
        out = x12 + x3
        return out


@RECOGNIZER_BACKBONE_REGISTRY.register()
class Identity(nn.Module):
    @configurable
    def __init__(self):
        super().__init__()

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {}
        return ret

    def forward(self, x):
        return x


@RECOGNIZER_BACKBONE_REGISTRY.register()
class CNN_V1(nn.Module):
    """
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, conv_norm):
        """
        Args:
            input_shape (ShapeSpec): shape of the input feature
        """
        super().__init__()
        self.conv1 = Conv2d(
            input_shape.channels,
            input_shape.channels,
            kernel_size=[2, 1],
            stride=[2, 1],
            padding=0,
            bias=not conv_norm,
            norm=get_norm(conv_norm, input_shape.channels),
            activation=nn.ReLU(),
        )

        self.conv2 = Conv2d(
            input_shape.channels,
            input_shape.channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, input_shape.channels),
            activation=nn.ReLU(),
        )

        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {}
        ret["input_shape"] = input_shape
        ret["conv_norm"] = cfg.MODEL.ROI_MASK_HEAD.NORM
        return ret

    def forward(self, x):
        x1 = self.conv1(x)
        conv2 = self.conv2(x1)
        out = conv2 + x1
        return out


@RECOGNIZER_BACKBONE_REGISTRY.register()
class CNN_V1_RECT(nn.Module):
    """
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, conv_norm):
        """
        Args:
            input_shape (ShapeSpec): shape of the input feature
        """
        super().__init__()
        self.conv1 = Conv2d(
            input_shape.channels,
            input_shape.channels,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=0,
            bias=not conv_norm,
            norm=get_norm(conv_norm, input_shape.channels),
            activation=nn.ReLU(),
        )

        self.conv2 = Conv2d(
            input_shape.channels,
            input_shape.channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, input_shape.channels),
            activation=nn.ReLU(),
        )

        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {}
        ret["input_shape"] = input_shape
        ret["conv_norm"] = cfg.MODEL.ROI_MASK_HEAD.NORM
        return ret

    def forward(self, x):
        x1 = self.conv1(x)
        conv2 = self.conv2(x1)
        out = conv2 + x1
        return out


@RECOGNIZER_BACKBONE_REGISTRY.register()
class CNN_V2(nn.Module):
    """
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, conv_norm):
        """
        Args:
            input_shape (ShapeSpec): shape of the input feature
        """
        super().__init__()
        self.conv1 = Conv2d(
            input_shape.channels,
            input_shape.channels,
            kernel_size=[2, 1],
            stride=[2, 1],
            padding=0,
            bias=not conv_norm,
            norm=get_norm(conv_norm, input_shape.channels),
            activation=nn.ReLU(),
        )

        self.conv2 = Conv2d(
            input_shape.channels,
            input_shape.channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, input_shape.channels),
            activation=nn.ReLU(),
        )

        self.conv3 = Conv2d(
            input_shape.channels,
            input_shape.channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, input_shape.channels),
            activation=nn.ReLU(),
        )

        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)
        weight_init.c2_msra_fill(self.conv3)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {}
        ret["input_shape"] = input_shape
        ret["conv_norm"] = cfg.MODEL.ROI_MASK_HEAD.NORM
        return ret

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x12 = x2 + x1
        x3 = self.conv3(x12)
        out = x12 + x3
        return out
