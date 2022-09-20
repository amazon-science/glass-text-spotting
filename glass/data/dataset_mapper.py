# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
import pycocotools.mask as mask_util
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import PolygonMasks, BoxMode

from .text_encoder import TextEncoder
from ..utils.common_utils import rgb2grey


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.
    """

    def __init__(self, cfg, is_train=True):
        logger = logging.getLogger(__name__)
        self.img_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.orientation_on = cfg.MODEL.ORIENTATION_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        self.load_gt_text = cfg.MODEL.RECOGNIZER_ON or cfg.TEST.USE_FILTERED_METRICS
        self.rotated_boxes_on = cfg.MODEL.ROTATED_BOXES_ON

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE,
                                         cfg.INPUT.CROP.SIZE)  # TODO (amir): add RandomCropWithInstace here
            logger.info("CropGen used in training: " + str(self.crop_gen))
            self.crop_probability = cfg.INPUT.CROP.PROBABILITY
            self.crop_size = cfg.INPUT.CROP.SIZE[0]
        else:
            self.crop_gen = None
            self.crop_probability = 0
            self.crop_size = 1.

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        # Adding random rotation
        if cfg.INPUT.ROTATION.ENABLED and is_train:
            self.tfm_gens = [T.RandomRotation(angle=cfg.INPUT.ROTATION.ANGLES, sample_style='choice')] \
                            + self.tfm_gens
            logger.info("Added RotationTransform for Orientation: " + str(self.tfm_gens))

        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

        if self.load_gt_text:
            self.converter = TextEncoder(cfg)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        dataset_dict = _clone_dict(dataset_dict)  # it will be modified by code below
        image = self._read_and_verify_image(dataset_dict)

        crop_flag = (self.crop_gen is not None) and (self.crop_probability > np.random.random())
        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if crop_flag else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            if crop_flag:
                h, w = image.shape[:2]
                # Aspect ratio preserving crop
                c = self.crop_size + np.random.rand() * (1 - self.crop_size)
                abs_crop_size = int(h * c + 0.5), int(w * c + 0.5)
                crop_tfm = utils.gen_crop_transform_with_instance(
                    abs_crop_size,
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if crop_flag:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # PyTorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [self._transform_annotation(anno, transforms, image_shape)
                     for anno in dataset_dict.pop("annotations")
                     if anno.get("iscrowd", 0) == 0]
            if self.rotated_boxes_on and len(annos):
                instances = utils.annotations_to_instances_rotated(annos, image_shape)

                # Adding mask support explicitly for the rotated boxes instance creation
                if self.mask_on and "segmentation" in annos[0]:
                    segms = [obj["segmentation"] for obj in annos]
                    instances.gt_masks = PolygonMasks(segms)

            elif len(annos):
                instances = utils.annotations_to_instances(
                    annos, image_shape, mask_format=self.mask_format
                )
                # Create a tight bounding box from masks, useful when image is cropped
                if self.crop_gen and instances.has("gt_masks"):
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            # Adding the multi-label annotations
            if self.orientation_on and len(annos) and 'orientation' in annos[0]:
                instances.gt_orientation = self._build_gt_orientation(annos)

            # Adding the word-length
            if len(annos) and 'word_length' in annos[0]:
                word_lengths = [anno.get('word_length', 0) for anno in annos]
                instances.gt_word_lengths = torch.tensor(word_lengths)

            if self.load_gt_text:
                text_list = [x['text'] for x in annos]
                text_tensor = self.converter.encode(text_list=text_list)
                instances.gt_text_labels = text_tensor

            if 'id' in annos[0]:
                anno_id_list = [x['id'] for x in annos]
                instances.anno_ids = torch.tensor(anno_id_list, dtype=torch.long)

            dataset_dict['instances'] = utils.filter_empty_instances(instances)
        return dataset_dict

    def _read_and_verify_image(self, dataset_dict):
        if self.img_format == 'GREY':
            image = utils.read_image(dataset_dict["file_name"], format="RGB")
            image = rgb2grey(image, three_channels=True)
        else:
            image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        return image

    @staticmethod
    def _load_seg_image(seg_image_path, transforms):
        seg_gt = utils.read_image(seg_image_path, "L").squeeze(2)
        seg_gt = transforms.apply_segmentation(
            seg_gt)  # Todo: verify that apply seg is implemented, cardinal rotation, etc.
        seg_gt = torch.as_tensor(seg_gt.astype("long"))
        return seg_gt

    @staticmethod
    def _build_gt_orientation(annos):
        gt_orientation = torch.zeros(len(annos))
        for i, anno in enumerate(annos):
            gt_orientation[i] = int(np.round(anno['orientation'] / 90) % 4)
        return gt_orientation

    def _build_gt_script(self, annos):
        gt_script = torch.zeros(len(annos))
        for i, anno in enumerate(annos):
            try:
                gt_script[i] = self.script_names.index(anno['script'])
            except ValueError as e:
                raise ValueError(f'{e}: Script {anno["script"]} was not found in script names - ill defined script')
        return gt_script

    def _transform_annotation(self, anno, transforms, image_shape):
        """
        Applies the transformations to the annotations,
        and also updates the orientation field accordingly if the image was rotated by 90/180/270
        :param anno: An annotation dictionary
        :param transforms: A TransformList object with transforms
        :param image_shape: The image shape vector
        :return: An updated anno dictionary
        """
        # First we update the regular way for a detectron2 annotation

        # If we deal with rotated boxes we only transform the rotated box
        if anno['bbox_mode'] == BoxMode.XYWHA_ABS:
            rotated_box = np.array(anno['bbox']).reshape(1, 5)
            anno['bbox'] = transforms.apply_rotated_box(rotated_box)[0]
            if self.mask_on and "segmentation" in anno:
                # each instance contains 1 or more polygons
                segm = anno["segmentation"]
                if isinstance(segm, list):
                    # polygons
                    polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
                    anno["segmentation"] = [
                        p.reshape(-1) for p in transforms.apply_polygons(polygons)
                    ]
                elif isinstance(segm, dict):
                    # RLE
                    mask = mask_util.decode(segm)
                    mask = transforms.apply_segmentation(mask)
                    assert tuple(mask.shape[:2]) == image_shape
                    anno["segmentation"] = mask
                else:
                    raise ValueError(
                        "Cannot transform segmentation of type '{}'!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict.".format(type(segm))
                    )
        else:
            anno = utils.transform_instance_annotations(
                anno, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )

        # Updating the angle, if a rotation transform is found
        angle = anno.get('angle')
        angle = angle if angle else 0.0
        for tfm in transforms.transforms:
            if isinstance(tfm, T.RotationTransform):
                angle += tfm.angle
            break
        # Updating the orientation value
        anno['orientation'] = int((90 * np.round(angle / 90)) % 360 if angle else 0)
        return anno

def _clone_dict(d):
    return {k: v for k, v in d.items()}