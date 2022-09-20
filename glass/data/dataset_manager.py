# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import hashlib
import io
import logging
import os

import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.structures import BoxMode
from fvcore.common.timer import Timer
from iopath.common.file_io import PathManager
from pycocotools.coco import COCO


class DatasetManager:

    def __init__(self, cfg, is_train=True, max_train_assets=None, max_test_assets=None):
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg
        self.is_train = is_train
        self.dataset_config_name = cfg.DATASETS.CONFIG
        self.dataset_names = cfg.DATASETS.TRAIN
        self.test_dataset_names = cfg.DATASETS.TEST
        self.dataset_root = cfg.DATASETS.ROOT
        self.meta = None  # Will be populated after we load or build

        # We define the paths for the dataset file and additional small file containing meta data
        suffix = '_train' if is_train else '_test'
        dataset_file_name = os.path.splitext(self.dataset_config_name)[0]
        self.dataset_pkl_path = os.path.join(self.dataset_root, dataset_file_name + suffix + '.pkl')
        # TODO(tsiper): Add a hashing mechanism instead of just dataset names, which is risky
        self.dataset_meta_pkl_path = os.path.join(self.dataset_root, dataset_file_name + suffix + '_meta.pkl')
        self.max_train_assets = max_train_assets
        self.max_test_assets = max_test_assets

    def build(self):
        self.logger.info('Loading and building the dataset annotation dictionaries')
        dataset_names = self.dataset_names if self.is_train else self.test_dataset_names
        if dataset_names:
            dataset_dicts = get_detection_dataset_dicts(
                dataset_names,
                filter_empty=self.cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                min_keypoints=self.cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
                if self.cfg.MODEL.KEYPOINT_ON
                else 0,
                proposal_files=self.cfg.DATASETS.PROPOSAL_FILES_TRAIN if self.cfg.MODEL.LOAD_PROPOSALS else None,
            )
            self.meta = MetadataCatalog.get(dataset_names[0])
            return dataset_dicts
        else:
            self.logger.warning('No dataset names are available to build')

    def register(self, rotated_boxes=False):
        """
        Registers the datasets to the dataset catalog
        :return: An updated config file, after registering the datasets within the global DatasetCatalog
        """
        # We define the paths for the dataset file and additional small file containing meta data
        # We register all the unique datasets while preserving order
        registered_datasets_set = set(DatasetCatalog.list())
        train_datasets = list(set(self.cfg.DATASETS.TRAIN) - registered_datasets_set)
        test_datasets = list(set(self.cfg.DATASETS.TEST) - registered_datasets_set)

        is_train = [True] * len(train_datasets) + [False] * len(test_datasets)
        all_datasets = train_datasets + test_datasets
        for i, dataset_name in enumerate(all_datasets):
            metadata = dict()
            dataset_max_num_of_images = self.max_train_assets if is_train[i] else self.max_test_assets
            self.logger.info(f'Processing {i}/{len(all_datasets)} - {dataset_name}')
            image_root = os.path.join(self.dataset_root, dataset_name)
            json_file_path = os.path.join(image_root, 'annotations.json')
            self._register_coco_instances(name=dataset_name, metadata=metadata, json_file=json_file_path,
                                          image_root=image_root, rotated_boxes=rotated_boxes,
                                          max_num_of_images=dataset_max_num_of_images,
                                          )

            self.logger.info(f'Registered and added {dataset_name} - {i + 1}/{len(all_datasets)}')
        self.cfg.freeze()

    @staticmethod
    def _register_coco_instances(name, metadata, json_file, image_root, rotated_boxes=False, max_num_of_images=None):
        """
        Register a dataset in COCO's json annotation format for
        instance detection, instance segmentation and keypoint detection.

        Args:
            name (str): the name that identifies a dataset, e.g. "coco_2014_train".
            metadata (dict): extra metadata associated with this dataset.  You can
                leave it as an empty dict.
            json_file (str): path to the json instance annotation file.
            image_root (str or path-like): directory which contains all the images.
            rotated_boxes (bool): Whether to add support for rotated boxes
        """
        # 1. register a function which returns dicts
        DatasetCatalog.register(name, lambda: _load_coco_json(json_file, image_root, name, rotated_boxes=rotated_boxes,
                                                              extra_annotation_keys=['word_length',
                                                                                     'angle',
                                                                                     'orientation',
                                                                                     'rotated_box',
                                                                                     'text',
                                                                                     'id'],
                                                              max_num_of_images=max_num_of_images)
                                )
        # 2. Optionally, add metadata about this dataset,
        # since they might be useful in evaluation, visualization or logging
        MetadataCatalog.get(name).set(
            json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
        )


def _load_coco_json(json_file, image_root, dataset_name=None, rotated_boxes=False, extra_annotation_keys=None,
                    max_num_of_images=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        rotated_boxes (bool): If True we load as rotated boxes, based on the polygon object in the coco record
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.


    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """

    logger = logging.getLogger(__name__)
    timer = Timer()
    json_file = PathManager().get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # Extracting additional meta data
        meta.info = coco_api.dataset.get('info')

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning("Category ids in annotations are not in [1, #categories]! "
                               "We'll apply a mapping for you. """)
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    if max_num_of_images is not None:
        img_ids = img_ids[:max_num_of_images]

    imgs = coco_api.loadImgs(img_ids)

    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))
    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])
    num_instances_without_valid_segmentation = 0

    for img_ann in imgs_anns:
        sing_dataset_dict = _single_image_process(img_ann, image_root, dataset_name, ann_keys,
                                                  rotated_boxes, id_map)
        dataset_dicts.append(sing_dataset_dict)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )
    return dataset_dicts


def rotated_box_anno_to_xywha(rotated_box):
    np_box = np.array(rotated_box)  # An array with 4x2 vertices, the first is top left, going clockwise
    center_x, center_y = np.mean(np_box, axis=0)
    # width is measured on the first side of np_box
    width = np.linalg.norm(np_box[1] - np_box[0])
    # width is measured on the second side of np_box
    height = np.linalg.norm(np_box[2] - np_box[1])
    # Angle is measures as the tangens of the upper side vs its x,y values
    angle = np.rad2deg(np.arctan2(np_box[0, 1] - np_box[1, 1], np_box[1, 0] - np_box[0, 0]))
    return [center_x, center_y, width, height, angle]


def _single_image_process(imgs_anns, image_root, dataset_name, ann_keys, rotated_boxes, id_map):
    img_dict, anno_dict_list = imgs_anns
    record = dict()
    record["file_name"] = os.path.join(image_root, img_dict["file_name"])
    record["dataset_name"] = dataset_name.replace('_coco', '')
    if 'height' in img_dict and 'width' in img_dict:
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
    else:
        image = cv2.imread(record['file_name'])
        record['height'] = image.shape[0]
        record['width'] = image.shape[1]
    image_id = record["image_id"] = img_dict["id"]

    objs = []
    for anno_dict in anno_dict_list:
        obj = _object_from_annotation(anno_dict, image_id, dataset_name, ann_keys, rotated_boxes, id_map)
        objs.append(obj)

    record["annotations"] = objs
    return record


def _object_from_annotation(anno, image_id, dataset_name, ann_keys, rotated_boxes, id_map):
    assert anno["image_id"] == image_id
    assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

    obj = {key: anno[key] for key in ann_keys if key in anno}
    segm = anno.get("segmentation", None)
    if segm:  # either list[list[float]] or dict(RLE)
        if not isinstance(segm, dict):
            # filter out invalid polygons (< 3 points)
            segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
            # if len(segm) == 0:
            #     num_instances_without_valid_segmentation += 1
            #     continue  # ignore this instance
        obj["segmentation"] = segm

    keypts = anno.get("keypoints", None)
    if keypts:  # list[int]
        for idx, v in enumerate(keypts):
            if idx % 3 != 2:
                # COCO's segmentation coordinates are floating points in [0, H or W],
                # but keypoint coordinates are integers in [0, H-1 or W-1]
                # Therefore we assume the coordinates are "pixel indices" and
                # add 0.5 to convert to floating point coordinates.
                keypts[idx] = v + 0.5
        obj["keypoints"] = keypts

    text = anno.get("rec", "")
    if text:
        obj["text"] = text

    # Deciding on box mode, based on rotated_boxes
    if rotated_boxes:
        # If we have the rotated box, we work with it
        if anno.get('rotated_box'):
            obj['bbox'] = rotated_box_anno_to_xywha(anno['rotated_box'])
        else:
            obj['bbox'] = BoxMode.convert(obj['bbox'], from_mode=BoxMode.XYWH_ABS, to_mode=BoxMode.XYWHA_ABS)
        obj['bbox_mode'] = BoxMode.XYWHA_ABS
    else:
        obj['bbox_mode'] = BoxMode.XYWH_ABS

    # Getting orientation related fields
    obj['angle'] = anno.get('angle')
    obj['orientation'] = anno.get('orientation')

    # Getting language specific fields

    if id_map:
        obj['category_id'] = id_map[obj['category_id']]

    # Setting a unique id for each annotation box (CocoDatasetCreator sets unique ids only within each dataset)
    unique_id_string = '{}_{}'.format(dataset_name, obj.get('id'))
    obj['id'] = int(hashlib.md5(unique_id_string.encode()).hexdigest()[:10], 16)
    return obj
