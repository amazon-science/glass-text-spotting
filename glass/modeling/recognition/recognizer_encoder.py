# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from detectron2.config import configurable
from detectron2.utils.registry import Registry
from torch import nn
from torch.nn import init

RECOGNIZER_ENCODER_REGISTRY = Registry("RECOGNIZER_ENCODER")
RECOGNIZER_ENCODER_REGISTRY.__doc__ = """

"""


def build_recognizer_encoder(cfg, input_shape):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.RECOGNIZER_HEAD.ENCODER.NAME
    return RECOGNIZER_ENCODER_REGISTRY.get(name)(cfg, input_shape)


def build_recognizer_encoderv2(cfg, input_shape):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.ROI_RECOGNIZER_HEAD.RECOGNIZER_HEAD.ENCODER.NAME
    return RECOGNIZER_ENCODER_REGISTRY.get(name)(cfg, input_shape)


@RECOGNIZER_ENCODER_REGISTRY.register()
class Identity(nn.Module):
    @configurable
    def __init__(self, height_reduction):
        super().__init__()
        self.height_reduction = height_reduction

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {}
        ret["height_reduction"] = cfg.MODEL.ROI_MASK_HEAD.RECOGNIZER_HEAD.ENCODER.HEIGHT_REDUCTION
        return ret

    def forward(self, x):
        if self.height_reduction == 'mean':
            x = x.mean(dim=2)  # NxCxHxW -> NxCxHW
        elif self.height_reduction == 'flatten':
            x = x.flatten(2)  # NxCxHxW -> NxCxHW
        else:
            raise NotImplementedError
        return x.permute(0, 2, 1)  # NxCxHW -> NxHWxC


@RECOGNIZER_ENCODER_REGISTRY.register()
class BiLSTMBlock(nn.Module):
    @configurable
    def __init__(self, input_size, hidden_size, output_size, num_of_layers=2):
        super().__init__()
        self.bilsm_stack = nn.Sequential(
            *[BiLSTM(input_size, hidden_size, output_size) for _ in range(num_of_layers)]
        )

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {}
        ret["input_size"] = input_shape.channels
        ret["hidden_size"] = input_shape.channels
        ret["output_size"] = input_shape.channels
        ret["num_of_layers"] = cfg.MODEL.ROI_MASK_HEAD.RECOGNIZER_HEAD.ENCODER.NUM_OF_LAYERS
        return ret

    def forward(self, features):
        x = features.mean(dim=2).transpose(1, 2).contiguous()
        return self.bilsm_stack(x)


@RECOGNIZER_ENCODER_REGISTRY.register()
class IdentityV2(nn.Module):
    @configurable
    def __init__(self, height_reduction):
        super().__init__()
        self.height_reduction = height_reduction

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {}
        ret["height_reduction"] = cfg.MODEL.ROI_RECOGNIZER_HEAD.RECOGNIZER_HEAD.ENCODER.HEIGHT_REDUCTION
        return ret

    def forward(self, x):
        if self.height_reduction == 'mean':
            x = x.mean(dim=2)  # NxCxHxW -> NxCxHW
        elif self.height_reduction == 'flatten':
            x = x.flatten(2)  # NxCxHxW -> NxCxHW
        else:
            raise NotImplementedError
        return x.permute(0, 2, 1)  # NxCxHW -> NxHWxC


@RECOGNIZER_ENCODER_REGISTRY.register()
class BiLSTMBlockV2(nn.Module):
    @configurable
    def __init__(self, input_size, hidden_size, output_size, num_of_layers=2):
        super().__init__()
        self.bilsm_stack = nn.Sequential(
            *[BiLSTM(input_size, hidden_size, output_size) for _ in range(num_of_layers)]
        )

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {}
        ret["input_size"] = input_shape.channels
        ret["hidden_size"] = input_shape.channels
        ret["output_size"] = input_shape.channels
        ret["num_of_layers"] = cfg.MODEL.ROI_RECOGNIZER_HEAD.RECOGNIZER_HEAD.ENCODER.NUM_OF_LAYERS
        return ret

    def forward(self, features):
        x = features.mean(dim=2).transpose(1, 2).contiguous()
        return self.bilsm_stack(x)


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        init.normal_(self.linear.weight, std=0.01)
        init.constant_(self.linear.bias, 0)
        for param in self.rnn.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

    def forward(self, features):
        """
        :param features: features in [N, C, H, W] [Batch, Channels, Height, Width]
        :return:
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(features)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output
