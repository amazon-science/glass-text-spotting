# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from detectron2.engine.defaults import DefaultTrainer
from detectron2.config import CfgNode

from ..data.build import build_detection_train_loader, build_detection_test_loader


class Trainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        return build_detection_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name=None):
        return build_detection_test_loader(cfg)

