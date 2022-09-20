# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import logging
import operator
from detectron2.data.common import AspectRatioGroupedDataset, MapDataset
from detectron2.data import samplers
# noinspection PyProtectedMember
from detectron2.data.build import get_world_size, trivial_batch_collator, worker_init_reset_seed
from glass.data.dataset_mapper import DatasetMapper
from glass.data.dataset_manager import DatasetManager


def build_detection_train_loader(cfg, mapper=None):
    """
    A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Start workers to work on the dicts. Each worker will:

       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will return.

    Args:
        cfg (CfgNode): the config
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            By default it will be `DatasetMapper(cfg, True)`.

    Returns:
        an infinite iterator of training data
    """
    logger = logging.getLogger(__name__)

    # We attempt to load the datasets from file
    dataset_manager = DatasetManager(cfg, is_train=True)
    logger.info('Serializing train datasets and loading them into memory...')
    dataset = dataset_manager.build()
    if dataset is None:
        raise FileNotFoundError('Please run serialize_datasets to generate the dataset pickle files')

    if mapper is None:
        mapper = DatasetMapper(cfg, is_train=True)
    dataset = MapDataset(dataset, mapper)
    sampler = samplers.TrainingSampler(len(dataset))

    images_per_worker = num_of_images_per_worker(cfg)
    if cfg.DATALOADER.ASPECT_RATIO_GROUPING:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=None,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
            prefetch_factor=2,
        )  # yield individual mapped dict
        data_loader = AspectRatioGroupedDataset(data_loader, images_per_worker)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_worker, drop_last=True
        )
        # drop_last so the batch always have the same size
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
            prefetch_factor=2,
        )
    return data_loader


def build_detection_test_loader(cfg):
    """
    Similar to `build_detection_train_loader`. But this time it returns the data loader
    for the evaluation (running on each image exactly once, out of the validation/test datasets)

    Args:
        cfg: a detectron2 CfgNode
    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    # Building a mapped datasets for evaluation
    logger = logging.getLogger(__name__)
    dataset_manager = DatasetManager(cfg, is_train=False)
    logger.info('Serializing test datasets and loading them into memory...')
    dataset = dataset_manager.build()
    mapper = DatasetMapper(cfg, is_train=False)
    dataset = MapDataset(dataset, mapper)

    # Building the inference sampler
    images_per_worker = num_of_images_per_worker(cfg)
    sampler = samplers.InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, images_per_worker, drop_last=False)

    # Finally we define a data loader with our mapped dataset and sampler
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        prefetch_factor=2,
    )
    return data_loader


def num_of_images_per_worker(cfg):
    num_workers = get_world_size()
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    assert (
            images_per_batch % num_workers == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    assert (
            images_per_batch >= num_workers
    ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    images_per_worker = images_per_batch // num_workers
    return images_per_worker
