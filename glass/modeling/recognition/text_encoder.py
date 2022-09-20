# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np


class TextEncoder:
    """ Convert between text-label and text-index """

    def __init__(self, cfg):
        """
        Encodes and decodes text into tensors which can be GPU ingested
        :param cfg: A D2 config instance
        """
        if cfg.MODEL.ROI_RECOGNIZER_HEAD.NAME == "RecognizerRCNNHeadV3":
            self.max_word_length = cfg.MODEL.ROI_RECOGNIZER_HEAD.MAX_WORD_LENGTH
            character_set = cfg.MODEL.ROI_RECOGNIZER_HEAD.CHARACTER_SET
            self.symbol_char = cfg.MODEL.ROI_RECOGNIZER_HEAD.UNK_SYMBOL_PRED
            self.mode = cfg.MODEL.ROI_RECOGNIZER_HEAD.LABELS_TYPE
            self.ignore_text = cfg.MODEL.ROI_RECOGNIZER_HEAD.IGNORE_TEXT
            self.ignore_empty_text = cfg.MODEL.ROI_RECOGNIZER_HEAD.IGNORE_EMPTY_TEXT
        else: #TODO legacy code for backward compatibility
            self.max_word_length = cfg.MODEL.ROI_MASK_HEAD.MAX_WORD_LENGTH
            character_set = cfg.MODEL.ROI_MASK_HEAD.CHARACTER_SET
            self.symbol_char = cfg.MODEL.ROI_MASK_HEAD.UNK_SYMBOL_PRED
            self.mode = cfg.MODEL.ROI_MASK_HEAD.LABELS_TYPE
            self.ignore_text = cfg.MODEL.ROI_MASK_HEAD.IGNORE_TEXT
            self.ignore_empty_text = cfg.MODEL.ROI_MASK_HEAD.IGNORE_EMPTY_TEXT
        # [GO] for the start token of the attention decoder.
        # [s] for end-of-sentence token.
        # [UNK] explicit prediction of characters out of the character set as UNK

        self.extra_tokens = list()
        if self.mode == 'attention':
            self.extra_tokens += ['[GO]', '[s]']
        elif self.mode == 'ctc':
            self.extra_tokens += ['[blank]']
        else:
            raise NotImplementedError

        if self.symbol_char:
            self.extra_tokens += ['[UNK]']

        self.character = self.extra_tokens + list(character_set)

        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i


    def encode(self, text_list):
        text_list = [x if (x is not None) and (len(x) < self.max_word_length)
                     else '' for x in text_list]
        if len(self.ignore_text)>0:
            assert self.ignore_empty_text, 'IGNORE_EMPTY_TEXT has to be True for non empty IGNORE_TEXT'
            text_list = [w if w not in self.ignore_text else '' for w in text_list]

        max_encoding_length = self.max_word_length
        if self.mode == 'attention':
            max_encoding_length += 2
        encoded_text = torch.zeros([len(text_list), max_encoding_length], dtype=torch.long)
        for i in range(len(text_list)):
            text = list(text_list[i])
            if self.mode == 'attention':
                text.append('[s]')
            if self.symbol_char:
                enc_text = [self.dict[char] if char in self.dict.keys() else self.dict['[UNK]'] for char in text]
            else:
                enc_text = [self.dict[char] for char in text if char in self.dict.keys()]

            shift = 1 if self.mode == 'attention' else 0
            encoded_text[i][shift:shift + len(enc_text)] = torch.tensor(enc_text, dtype=torch.long)

        return encoded_text

    def char_encode(self, char):
        assert len(char) == 1
        return self.dict[char] if char in self.dict.keys() else self.dict['[UNK]']

    def _get_pred_indices_mask_attention(self, pred_indices, include_stop_symbol_conf=True):
        """
        :param pred_indices:  np.array of predicted character indices
        :param include_stop_symbol_conf: True/False if to include the stop symbol in mask

        :return boolean np.array with the shape of pred_indices True before 'stop' ('[s]') symbol,
         and False after or ar at the 'UNK' symbol
        """

        # mask pred
        stop_symbol_index = self.character.index('[s]')
        pred_indices_mask = (pred_indices == stop_symbol_index).cumsum(axis=1) < 1
        if include_stop_symbol_conf:
            word_length = np.minimum(pred_indices_mask.sum(axis=1), pred_indices_mask.shape[1] - 1)
            pred_indices_mask[np.arange(len(word_length)), word_length] = True

        # mask the 'UNK' symbol
        if self.symbol_char:
            unknown_symbol = self.character.index('[UNK]')
            pred_indices_mask[pred_indices == unknown_symbol] = False

        return pred_indices_mask

    def decode_prod_v2(self, pred_indices, pred_probs=None, include_stop_symbol_conf=True):
        if self.mode == 'attention':
            return self.decode_attention(pred_indices=pred_indices, pred_probs=pred_probs,
                                         include_stop_symbol_conf=include_stop_symbol_conf)
        elif self.mode == 'ctc':
            return self.decode_ctc(pred_indices=pred_indices, pred_probs=pred_probs,
                                   include_stop_symbol_conf=include_stop_symbol_conf)
    decode = decode_prod_v2

    def decode_attention(self, pred_indices, pred_probs=None, include_stop_symbol_conf=True):
        """
        :param pred_indices:  np.array of predicted character indices
        :param pred_probs: np.array of corresponding probabilities for the pred_indices
        :param include_stop_symbol_conf: True/False if to include the stop symbol confidence in the word score
        :param scores_rounding: rounding for the per-character and word confidence scores (slows computation by x3)

        Example:
        # preds = model(image, text_for_pred, is_train=False)
        # pred_indices = np.array(preds[inference_block]['JoinBeamSearch'][0].data.cpu())
        # pred_probs = np.array(preds[inference_block]['JoinBeamSearch'][1].data.cpu())

        """

        stop_symbol_index = self.character.index('[s]')
        pred_indices_mask = self._get_pred_indices_mask_attention(pred_indices=pred_indices,
                                                                  include_stop_symbol_conf=include_stop_symbol_conf)

        # calculate word level scores
        if pred_probs is not None:
            pred_probs[~pred_indices_mask] = 1  # for vectorized product calc
            word_probabilities = pred_probs.prod(axis=1)

        texts = []
        for i, pred_ind in enumerate(pred_indices):
            text_indices = pred_ind[pred_indices_mask[i]]
            if stop_symbol_index and include_stop_symbol_conf and (text_indices[-1] == stop_symbol_index):
                text = ''.join([self.character[i] for i in text_indices[:-1]])
            else:
                text = ''.join([self.character[i] for i in text_indices])
            if pred_probs is not None:
                char_conf = pred_probs[i, pred_indices_mask[i]]
                word_conf = word_probabilities[i]
            else:  # usually gt labels
                char_conf = [1] * len(text)
                word_conf = 1

            texts.append({"text": text, 'score': word_conf, 'character_scores': char_conf})
        return texts

    def decode_ctc(self, pred_indices, pred_probs=None, include_stop_symbol_conf=True):
        """ convert text-index into text-label. """
        texts = list()
        if pred_probs is None:  # backward-compatibility
            pred_probs = np.ones_like(pred_indices)
        for t, prob in zip(pred_indices, pred_probs):
            char_list = []
            score = []
            for i in range(self.max_word_length):
                if t[i] != 0:
                    if i > 0 and t[i - 1] == t[i]:  # repeated characters, if higher score replace
                        if score[-1] < prob[i]:
                            score[-1] = prob[i]
                    else:
                        score.append(prob[i])  # score for this character
                        char_list.append(self.character[t[i]] if t[i] < len(self.character) else '')
            text = ''.join(char_list).replace('[UNK]', '')
            if len(score):
                word_conf = np.array(score).cumprod()[-1]
                char_conf = score
            else:
                word_conf = 1.0
                char_conf = [1.0]

            texts.append({"text": text, 'score': word_conf, 'character_scores': char_conf})

        return texts
