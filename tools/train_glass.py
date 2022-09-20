# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import torch


from detectron2.utils import comm
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.utils.logger import setup_logger

from glass.config import add_e2e_config, add_glass_config, add_dataset_config, merge_from_dataset_config
from glass.data.dataset_manager import DatasetManager
from glass.engine.trainer import Trainer


def setup(args):
    """
    Create config and perform basic setups.
    """
    cfg = get_cfg()

    add_e2e_config(cfg)
    add_glass_config(cfg)
    add_dataset_config(cfg)
    merge_from_dataset_config(cfg, args.datasets)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Getting the output dir from the args
    cfg.OUTPUT_DIR = args.output
    rank = comm.get_local_rank()
    logger = setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=rank, name='GLASS')

    if args.debug:
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.DATALOADER.PREFETCH_FACTOR = 2
        args.num_gpus = 1

    # We need to make sure all normalization is empty
    if args.num_gpus == 1:
        logger.warning('Overriding NORM values to regular batch norm - "BN", because only one GPU is working')
        cfg.MODEL.FPN.NORM = 'BN' if cfg.MODEL.FPN.NORM == 'SyncBN' else cfg.MODEL.FPN.NORM
        cfg.MODEL.RESNETS.NORM = 'BN' if cfg.MODEL.RESNETS.NORM == 'SyncBN' else cfg.MODEL.RESNETS.NORM
        cfg.MODEL.ROI_BOX_HEAD.NORM = 'BN' if cfg.MODEL.ROI_BOX_HEAD.NORM == 'SyncBN' else cfg.MODEL.ROI_BOX_HEAD.NORM
        cfg.MODEL.ROI_MASK_HEAD.NORM = 'BN' if cfg.MODEL.ROI_MASK_HEAD.NORM == 'SyncBN' else cfg.MODEL.ROI_MASK_HEAD.NORM
        cfg.MODEL.ROI_RECOGNIZER_HEAD.NORM = 'BN' if cfg.MODEL.ROI_MASK_HEAD.NORM == 'SyncBN' else cfg.MODEL.ROI_MASK_HEAD.NORM

    cfg.freeze()
    # Registering the datasets we provided in the datasets config
    DatasetManager(cfg).register(rotated_boxes=True)

    default_setup(cfg, args)

    return cfg


def main(args):
    cfg = setup(args)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    # Setting up the default detectron2 command line arguments:
    # The ones that we use are:
    #   * config-file: path to the config file
    #   * resume: toggles whether to resume a run or start a new one (default=False)
    #   * opts: pairs of arguments that override the cfg file, i.e. --opts MODEL.BACKBONE.FREEZE_AT 2

    parser: argparse.ArgumentParser = default_argument_parser()
    # Adding our own custom arguments
    parser.add_argument('--datasets', help='Path to the dataset config', required=True)
    parser.add_argument('--output', help='Path to the output dir', required=True)
    parser.add_argument('--debug', help='Activates debug mode, for PyCharm debugging', action='store_true')

    # The default is changed to 0 so we choose automatically the number of GPUs to run if no argument is specified
    parser.set_defaults(num_gpus=0)
    arguments = parser.parse_args()

    # Updating number of GPUs according to user input
    arguments.num_gpus = arguments.num_gpus if arguments.num_gpus > 0 else torch.cuda.device_count()

    print("Command Line Args:", arguments)
    launch(
        main,
        arguments.num_gpus,
        num_machines=arguments.num_machines,
        machine_rank=arguments.machine_rank,
        dist_url=arguments.dist_url,
        args=(arguments,),
    )
