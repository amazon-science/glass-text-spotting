# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List
import torch
from torch import nn
from torch.nn import functional as F
from tabulate import tabulate
import detectron2.utils.comm as comm

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.mask_head import ROI_MASK_HEAD_REGISTRY
from detectron2.utils.registry import Registry

from .text_encoder import TextEncoder
from .recognizer_backbone import build_recognizer_backbone, build_recognizer_backbonev2
from .recognizer_encoder import build_recognizer_encoder, build_recognizer_encoderv2
from .recognizer_decoder import build_recognizer_decoder, build_recognizer_decoderv2


def print_text_pred_examples(text_encoder, encoded_text_gt, text_preds):
    if not comm.is_main_process():
        return
    labels_text = [x['text'] for x in
                   text_encoder.decode_prod_v2(encoded_text_gt.cpu().numpy()[:, 1:])]
    pred_probs, preds_indices = text_preds.detach().cpu().max(dim=2)
    pred_text = text_encoder.decode_prod_v2(pred_probs=pred_probs.numpy(),
                                            pred_indices=preds_indices.numpy())
    max_examples = 7
    pred_dis_text = [x['text'] for x in pred_text[:max_examples]]
    pred_dis_label = ['** 100% **' if x == y['text'] else x for x, y in
                      zip(labels_text[:max_examples], pred_text[:max_examples])]
    dict_to_print = {
        'Pred      ': pred_dis_text,
        'Label     ': pred_dis_label,
    }
    print(tabulate(dict_to_print, tablefmt='psql', headers='keys', showindex=False))


@torch.jit.unused
def decoder_loss(preds: torch.Tensor, targets: torch.Tensor):
    target = targets[:, 1:]
    loss = F.cross_entropy(input=preds.view(-1, preds.shape[-1]),
                           target=target.contiguous().view(-1),
                           ignore_index=0)

    return loss


def _sample_words(labels, strategy, sample_words_strategy_prob, max_batch_size):
    if torch.rand(1) > sample_words_strategy_prob:
        strategy = 'random'

    if strategy == 'long_first':
        word_len = (labels > 0).sum(dim=1)
        ind = word_len.argsort(descending=True)
        return ind[:max_batch_size]

    elif strategy == 'random':
        batch_ind = torch.randint(len(labels), (max_batch_size,))

    else:
        raise NotImplementedError

    return batch_ind


class BaseRecognizerRCNNHead(nn.Module):
    """
    Implement the basic Mask R-CNN losses and inference logic described in :paper:`Mask R-CNN`
    """

    @configurable
    def __init__(self, *, vis_period=0):
        """
        NOTE: this interface is experimental.

        Args:
            vis_period (int): visualization period
        """
        super().__init__()
        self.vis_period = vis_period

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {"vis_period": cfg.VIS_PERIOD}

    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """

        if self.training:
            if x.shape[0] == 0:
                return {"loss_decoder": torch.tensor(0)}
            features = self.layers(x)

            labels = []
            for x in instances:
                if x.has('gt_text_labels'):
                    labels.append(x.gt_text_labels)
            if len(labels) == 0:
                return {"loss_decoder": torch.tensor(0)}
            labels = torch.cat(labels, dim=0)
            # labels = torch.cat([x.gt_text_labels if x.has('gt_text_labels') else torch.zeros([1,27],device=features.device)  for x in instances], dim=0)

            loss_lambda = self.loss_weight
            if self.ignore_empty_text:
                no_empty_ind = labels.sum(dim=1) > 1
                if no_empty_ind.sum() > 0:
                    features = features[no_empty_ind, :]
                    labels = labels[no_empty_ind, :]
                else:
                    loss_lambda = 0

            if len(labels) > self.max_batch_size:
                batch_ind = _sample_words(labels=labels,
                                          strategy=self.sample_words_strategy,
                                          sample_words_strategy_prob=self.sample_words_strategy_prob,
                                          max_batch_size=self.max_batch_size
                                          )

                features = features[batch_ind]
                labels = labels[batch_ind]

            encoded_features = self.encoder(features)
            preds = self.decoder(features=encoded_features, labels=labels)

            vis_itr = 500
            if vis_itr > 0 and get_event_storage().iter % vis_itr == 0:
                print_text_pred_examples(text_encoder=self.text_encoder, encoded_text_gt=labels, text_preds=preds)

            loss = decoder_loss(preds=preds, targets=labels)

            return {"loss_decoder": loss_lambda * loss}

        else:
            if x.shape[0] == 0:
                return instances
            features = self.layers(x)
            encoded_features = self.encoder(features)
            preds = self.decoder(encoded_features)

            num_boxes_per_image = [len(i) for i in instances]
            preds = preds.split(num_boxes_per_image, dim=0)

            for tmp_pred, tmp_instance in zip(preds, instances):
                tmp_instance.pred_text_prob = tmp_pred

            return instances

    def layers(self, x):
        """
        Neural network layers that makes predictions from input features.
        """
        raise NotImplementedError


# To get torchscript support, we make the head a subclass of `nn.Sequential`.
# Therefore, to add new layers in this head class, please make sure they are
# added in the order they will be used in forward().
@ROI_MASK_HEAD_REGISTRY.register()
class RecognizerRCNNHeadV2(BaseRecognizerRCNNHead, nn.Sequential):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    def __init__(self, cfg, input_shape):
        """
        Args:
            input_shape (ShapeSpec): shape of the input feature
        """
        super().from_config(cfg, input_shape)
        super().__init__()

        backbone = build_recognizer_backbone(cfg, input_shape)

        self.max_word_length = cfg.MODEL.ROI_MASK_HEAD.MAX_WORD_LENGTH
        self.ignore_empty_text = cfg.MODEL.ROI_MASK_HEAD.IGNORE_EMPTY_TEXT
        self.class_ind = cfg.MODEL.ROI_MASK_HEAD.CLASS_IND
        self.max_batch_size = cfg.MODEL.ROI_MASK_HEAD.MAX_BATCH_SIZE
        self.loss_weight = cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT
        self.sample_words_strategy = cfg.MODEL.ROI_MASK_HEAD.SAMPLE_WORDS_STRATEGY
        self.sample_words_strategy_prob = cfg.MODEL.ROI_MASK_HEAD.SAMPLE_WORDS_STRATEGY_PROB
        self.inner_layers = list()

        self.add_module("backbone", backbone)
        self.inner_layers.append(backbone)

        self.encoder = build_recognizer_encoder(cfg, input_shape)
        self.decoder = build_recognizer_decoder(cfg, input_shape)

        self.text_encoder = TextEncoder(cfg)

    # @configurable
    # def __init__(self, input_shape: ShapeSpec, *,
    #              backbone,
    #              encoder,
    #              decoder,
    #              sample_words_strategy,
    #              sample_words_strategy_prob,
    #              max_word_length,
    #              character_set, unk_symbol, ignore_empty_text, class_ind,
    #              loss_weight, max_batch_size, **kwargs):
    #     """
    #     Args:
    #         input_shape (ShapeSpec): shape of the input feature
    #     """
    #     super().__init__(**kwargs)
    #     self.class_ind = class_ind
    #     self.inner_layers = list()
    #     self.max_word_length = max_word_length
    #     self.ignore_empty_text = ignore_empty_text
    #     self.loss_weight = loss_weight
    #     self.max_batch_size = max_batch_size
    #     self.sample_words_strategy = sample_words_strategy
    #     self.sample_words_strategy_prob = sample_words_strategy_prob
    #
    #     self.add_module("backbone", backbone)
    #     self.inner_layers.append(backbone)
    #
    #     self.encoder = encoder
    #
    #     self.decoder = decoder
    #
    #     self.text_encoder = TextEncoder(character=character_set,
    #                                     max_word_length=max_word_length,
    #                                     symbol_char=unk_symbol)
    #
    # @classmethod
    # def from_config(cls, cfg, input_shape):
    #     ret = super().from_config(cfg, input_shape)
    #     ret["input_shape"] = input_shape
    #     ret["backbone"] = build_recognizer_backbone(cfg, input_shape)
    #     ret["encoder"] = build_recognizer_encoder(cfg, input_shape)
    #     ret["decoder"] = build_recognizer_decoder(cfg, input_shape)
    #     ret["max_word_length"] = cfg.MODEL.ROI_MASK_HEAD.MAX_WORD_LENGTH
    #     ret["character_set"] = cfg.MODEL.ROI_MASK_HEAD.CHARACTER_SET
    #     ret["unk_symbol"] = cfg.MODEL.ROI_MASK_HEAD.UNK_SYMBOL_PRED
    #     ret["ignore_empty_text"] = cfg.MODEL.ROI_MASK_HEAD.IGNORE_EMPTY_TEXT
    #     ret["class_ind"] = cfg.MODEL.ROI_MASK_HEAD.CLASS_IND
    #     ret["max_batch_size"] = cfg.MODEL.ROI_MASK_HEAD.MAX_BATCH_SIZE
    #     ret["loss_weight"] = cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT
    #     ret["loss_weight"] = cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT
    #     ret["sample_words_strategy"] = cfg.MODEL.ROI_MASK_HEAD.SAMPLE_WORDS_STRATEGY
    #     ret["sample_words_strategy_prob"] = cfg.MODEL.ROI_MASK_HEAD.SAMPLE_WORDS_STRATEGY_PROB
    #     return ret

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["input_shape"] = input_shape
        ret["backbone"] = build_recognizer_backbone(cfg, input_shape)
        ret["encoder"] = build_recognizer_encoder(cfg, input_shape)
        ret["decoder"] = build_recognizer_decoder(cfg, input_shape)
        ret["max_word_length"] = cfg.MODEL.ROI_MASK_HEAD.MAX_WORD_LENGTH
        ret["character_set"] = cfg.MODEL.ROI_MASK_HEAD.CHARACTER_SET
        ret["unk_symbol"] = cfg.MODEL.ROI_MASK_HEAD.UNK_SYMBOL_PRED
        ret["ignore_empty_text"] = cfg.MODEL.ROI_MASK_HEAD.IGNORE_EMPTY_TEXT
        ret["class_ind"] = cfg.MODEL.ROI_MASK_HEAD.CLASS_IND
        ret["max_batch_size"] = cfg.MODEL.ROI_MASK_HEAD.MAX_BATCH_SIZE
        ret["loss_weight"] = cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT
        ret["sample_words_strategy"] = cfg.MODEL.ROI_MASK_HEAD.SAMPLE_WORDS_STRATEGY
        ret["sample_words_strategy_prob"] = cfg.MODEL.ROI_MASK_HEAD.SAMPLE_WORDS_STRATEGY_PROB
        return ret

    def layers(self, x):
        for layer in self.inner_layers:
            x = layer(x)
        return x


ROI_RECOGNIZER_HEAD_REGISTRY = Registry("ROI_RECOGNIZER_HEAD")


@ROI_RECOGNIZER_HEAD_REGISTRY.register()
class RecognizerRCNNHeadV3(BaseRecognizerRCNNHead, nn.Sequential):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    def __init__(self, cfg, input_shape):
        """
        Args:
            input_shape (ShapeSpec): shape of the input feature
        """
        super().from_config(cfg, input_shape)
        super().__init__()

        backbone = build_recognizer_backbonev2(cfg, input_shape)

        self.max_word_length = cfg.MODEL.ROI_RECOGNIZER_HEAD.MAX_WORD_LENGTH
        self.ignore_empty_text = cfg.MODEL.ROI_RECOGNIZER_HEAD.IGNORE_EMPTY_TEXT
        self.class_ind = cfg.MODEL.ROI_RECOGNIZER_HEAD.CLASS_IND
        self.max_batch_size = cfg.MODEL.ROI_RECOGNIZER_HEAD.MAX_BATCH_SIZE
        self.loss_weight = cfg.MODEL.ROI_RECOGNIZER_HEAD.LOSS_WEIGHT
        self.sample_words_strategy = cfg.MODEL.ROI_RECOGNIZER_HEAD.SAMPLE_WORDS_STRATEGY
        self.sample_words_strategy_prob = cfg.MODEL.ROI_RECOGNIZER_HEAD.SAMPLE_WORDS_STRATEGY_PROB
        self.inner_layers = list()

        self.add_module("backbone", backbone)
        self.inner_layers.append(backbone)

        self.encoder = build_recognizer_encoderv2(cfg, input_shape)
        self.decoder = build_recognizer_decoderv2(cfg, input_shape)

        self.text_encoder = TextEncoder(cfg)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["input_shape"] = input_shape
        ret["backbone"] = build_recognizer_backbone(cfg, input_shape)
        ret["encoder"] = build_recognizer_encoderv2(cfg, input_shape)
        ret["decoder"] = build_recognizer_decoderv2(cfg, input_shape)
        ret["max_word_length"] = cfg.MODEL.ROI_RECOGNIZER_HEAD.MAX_WORD_LENGTH
        ret["character_set"] = cfg.MODEL.ROI_RECOGNIZER_HEAD.CHARACTER_SET
        ret["unk_symbol"] = cfg.MODEL.ROI_RECOGNIZER_HEAD.UNK_SYMBOL_PRED
        ret["ignore_empty_text"] = cfg.MODEL.ROI_RECOGNIZER_HEAD.IGNORE_EMPTY_TEXT
        ret["class_ind"] = cfg.MODEL.ROI_RECOGNIZER_HEAD.CLASS_IND
        ret["max_batch_size"] = cfg.MODEL.ROI_RECOGNIZER_HEAD.MAX_BATCH_SIZE
        ret["loss_weight"] = cfg.MODEL.ROI_RECOGNIZER_HEAD.LOSS_WEIGHT
        ret["sample_words_strategy"] = cfg.MODEL.ROI_RECOGNIZER_HEAD.SAMPLE_WORDS_STRATEGY
        ret["sample_words_strategy_prob"] = cfg.MODEL.ROI_RECOGNIZER_HEAD.SAMPLE_WORDS_STRATEGY_PROB
        return ret

    def layers(self, x):
        for layer in self.inner_layers:
            x = layer(x)
        return x


def build_recognizer_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_RECOGNIZER_HEAD.NAME
    return ROI_RECOGNIZER_HEAD_REGISTRY.get(name)(cfg, input_shape)
