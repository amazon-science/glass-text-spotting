# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from detectron2.config import configurable
from detectron2.utils.registry import Registry
from torch import nn

HYBRID_FEATURE_FUSION_REGISTRY = Registry("HYBRID_FEATURE_FUSION")


def build_hybrid_feature_fusion(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.HYBRID_FUSION.NAME
    return HYBRID_FEATURE_FUSION_REGISTRY.get(name)(cfg, input_shape)


@HYBRID_FEATURE_FUSION_REGISTRY.register()
class MultiAspectGCAttention(nn.Module):

    @configurable
    def __init__(
            self,
            *,
            inplanes,
            ratio,
            headers,
            pooling_type='att',
            outplane,
            fusion_type,
    ):
        super().__init__()
        assert pooling_type in ['avg', 'att']

        assert fusion_type in ['channel_add', 'channel_mul', 'channel_concat']
        assert inplanes % headers == 0 and inplanes >= 8  # inplanes must be divided by headers evenly
        assert inplanes % 2 == 0
        self.headers = headers
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_type = fusion_type
        self.att_scale = False
        self.out = nn.Conv2d(inplanes, outplane, kernel_size=3, padding=1)
        self.single_header_inplanes = int(inplanes / headers)
        self.order = torch.zeros(inplanes)
        self.order[0::2] = torch.arange(inplanes)[:int(inplanes / 2)]
        self.order[1::2] = torch.arange(inplanes)[int(inplanes / 2):]
        self.order = self.order.long()
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(self.single_header_inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if fusion_type == 'channel_add':
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        elif fusion_type == 'channel_concat':
            self.channel_concat_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
            # for concat
            self.cat_conv = nn.Conv2d(2 * self.inplanes, self.inplanes, kernel_size=1)
        elif fusion_type == 'channel_mul':
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))

    @classmethod
    def from_config(cls, cfg, input_shape):
        global_in_channels = input_shape.channels
        local_in_channels = cfg.MODEL.LOCAL_FEATURE_EXTRACTOR.NUM_FEATURES
        return {"inplanes": local_in_channels + global_in_channels,
                "ratio": cfg.MODEL.HYBRID_FUSION.RATIO,
                "headers": cfg.MODEL.HYBRID_FUSION.HEADERS,
                "outplane": cfg.MODEL.HYBRID_FUSION.NUM_FEATURES,
                "fusion_type": cfg.MODEL.HYBRID_FUSION.FUSION_TYPE}

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            # [N*headers, C', H , W] C = headers * C'
            x = x.view(batch * self.headers, self.single_header_inplanes, height, width)
            input_x = x

            # [N*headers, C', H * W] C = headers * C'
            # input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.view(batch * self.headers, self.single_header_inplanes, height * width)

            # [N*headers, 1, C', H * W]
            input_x = input_x.unsqueeze(1)
            # [N*headers, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N*headers, 1, H * W]
            context_mask = context_mask.view(batch * self.headers, 1, height * width)

            # scale variance
            if self.att_scale and self.headers > 1:
                context_mask = context_mask / torch.sqrt(self.single_header_inplanes)

            # [N*headers, 1, H * W]
            context_mask = self.softmax(context_mask)

            # [N*headers, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N*headers, 1, C', 1] = [N*headers, 1, C', H * W] * [N*headers, 1, H * W, 1]
            context = torch.matmul(input_x, context_mask)

            # [N, headers * C', 1, 1]
            context = context.view(batch, self.headers * self.single_header_inplanes, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        x = x[:, self.order, ...]  # reorder local and global features
        context = self.spatial_pool(x)

        out = x

        if self.fusion_type == 'channel_mul':
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        elif self.fusion_type == 'channel_add':
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        else:
            # [N, C, 1, 1]
            channel_concat_term = self.channel_concat_conv(context)

            # use concat
            _, C1, _, _ = channel_concat_term.shape
            N, C2, H, W = out.shape

            out = torch.cat([out, channel_concat_term.expand(-1, -1, H, W)], dim=1)
            out = self.cat_conv(out)
            out = nn.functional.layer_norm(out, [self.inplanes, H, W])
            out = nn.functional.relu(out)
        out = self.out(out)
        return out


@HYBRID_FEATURE_FUSION_REGISTRY.register()
class SimpleAttention(nn.Module):
    @configurable
    def __init__(
            self,
            *,
            in_channels,
            out_channels
    ):
        super().__init__()
        self.linear = nn.Linear(in_channels, in_channels, bias=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    @classmethod
    def from_config(cls, cfg, input_shape):
        global_in_channels = input_shape.channels
        local_in_channels = cfg.MODEL.LOCAL_FEATURE_EXTRACTOR.NUM_FEATURES
        return {"in_channels": local_in_channels + global_in_channels,
                "out_channels": cfg.MODEL.HYBRID_FUSION.NUM_FEATURES
                }

    def forward(self, x):
        x = x.transpose(1, -1)
        x = self.linear(x) * x
        x = x.transpose(1, -1)
        x = self.conv(x)
        return x


@HYBRID_FEATURE_FUSION_REGISTRY.register()
class LocalOnly(nn.Module):
    '''
    1x1Conv after global local feature concatenation channel-wise
    '''

    @configurable
    def __init__(
            self,
            *,
            global_in_channels,
            local_in_channels,
    ):
        super().__init__()
        self.global_in_channels = global_in_channels
        self.local_in_channels = local_in_channels

    @classmethod
    def from_config(cls, cfg, input_shape):
        global_in_channels = input_shape.channels
        local_in_channels = cfg.MODEL.LOCAL_FEATURE_EXTRACTOR.NUM_FEATURES
        assert cfg.MODEL.HYBRID_FUSION.NUM_FEATURES == local_in_channels
        return {"global_in_channels": global_in_channels,
                "local_in_channels": local_in_channels
                }

    def forward(self, x):
        assert x.shape[1] == self.global_in_channels + self.local_in_channels
        x = x[:, :self.local_in_channels, ...]
        return x


@HYBRID_FEATURE_FUSION_REGISTRY.register()
class Conv1x1(nn.Module):
    '''
    1x1Conv after global local feature concatenation channel-wise
    '''

    @configurable
    def __init__(
            self,
            *,
            in_channels,
            out_channels,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    @classmethod
    def from_config(cls, cfg, input_shape):
        global_in_channels = input_shape.channels
        local_in_channels = cfg.MODEL.LOCAL_FEATURE_EXTRACTOR.NUM_FEATURES
        return {"in_channels": local_in_channels + global_in_channels,
                "out_channels": cfg.MODEL.HYBRID_FUSION.NUM_FEATURES
                }

    def forward(self, x):
        x = self.conv(x)
        return x

