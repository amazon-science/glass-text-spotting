# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from detectron2.config import configurable
from detectron2.utils.registry import Registry
from torch import nn

from .prediction_aster import AttentionRecognitionHead

RECOGNIZER_DECODER_REGISTRY = Registry("RECOGNIZER_DECODER")
RECOGNIZER_DECODER_REGISTRY.__doc__ = """

"""


def build_recognizer_decoder(cfg, input_shape):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.RECOGNIZER_HEAD.DECODER.NAME
    return RECOGNIZER_DECODER_REGISTRY.get(name)(cfg, input_shape)


def build_recognizer_decoderv2(cfg, input_shape):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.ROI_RECOGNIZER_HEAD.RECOGNIZER_HEAD.DECODER.NAME
    return RECOGNIZER_DECODER_REGISTRY.get(name)(cfg, input_shape)


@RECOGNIZER_DECODER_REGISTRY.register()
class ASTER(nn.Module):
    @configurable
    def __init__(self, num_classes, max_word_len, in_channels):
        super().__init__()
        self.max_word_len = max_word_len
        self.recognizer = AttentionRecognitionHead(num_classes=num_classes,
                                                   in_planes=in_channels,
                                                   sDim=in_channels,
                                                   attDim=in_channels,
                                                   max_len_labels=max_word_len)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {}
        ret["num_classes"] = len(
            cfg.MODEL.ROI_MASK_HEAD.CHARACTER_SET) + 2  # ['[GO]', '[s]']) TODO(oronans): handle the case of UNK
        ret["max_word_len"] = int(cfg.MODEL.ROI_MASK_HEAD.MAX_WORD_LENGTH) + 1  # +1 for stop
        ret["in_channels"] = input_shape.channels
        return ret

    def forward(self, features, labels=None):
        if self.training:
            return self.recognizer([features.contiguous(), labels, self.max_word_len])
        else:
            stop_symbol_index = 0
            preds, _ = self.recognizer.sample(features.contiguous(), None,
                                              self.max_word_len,
                                              stop_symbol_index)
            return preds


@RECOGNIZER_DECODER_REGISTRY.register()
class ASTER_V2(nn.Module):
    @configurable
    def __init__(self, num_classes, max_word_len, in_channels):
        super().__init__()
        self.max_word_len = max_word_len
        self.recognizer = AttentionRecognitionHead(num_classes=num_classes,
                                                   in_planes=in_channels,
                                                   sDim=in_channels,
                                                   attDim=in_channels,
                                                   max_len_labels=max_word_len)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {}
        ret["num_classes"] = len(
            cfg.MODEL.ROI_RECOGNIZER_HEAD.CHARACTER_SET) + 2  # ['[GO]', '[s]']) TODO(oronans): handle the case of UNK
        ret["max_word_len"] = int(cfg.MODEL.ROI_RECOGNIZER_HEAD.MAX_WORD_LENGTH) + 1  # +1 for stop
        ret["in_channels"] = input_shape.channels
        return ret

    def forward(self, features, labels=None):
        if self.training:
            return self.recognizer([features.contiguous(), labels, self.max_word_len])
        else:
            stop_symbol_index = 0
            preds, _ = self.recognizer.sample(features.contiguous(), None,
                                              self.max_word_len,
                                              stop_symbol_index)
            return preds
