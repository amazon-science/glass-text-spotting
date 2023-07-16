import logging
from timeit import default_timer as timer
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.structures import Instances

from glass.config import add_e2e_config, add_glass_config, add_dataset_config, add_post_process_config
from glass.postprocess import build_post_processor
from glass.utils.common_utils import rgb2grey
from glass.modeling.recognition.text_encoder import TextEncoder


class GlassRunner:

    def __init__(self, model_path: str, config_path: str, opts: List[str] = None, post_process=True):
        """
        Initializes a runner that can run inference using a detectron2 based model
        :param model_path: Path to the detectron2 model
        :param config_path: Path to the configuration file of the detectron2 model
        :param opts: Additional option pairs to override settings in the configuration
        :param post_process: Whether to run post-processing or not
        """
        # Loading and initializing the config
        self.logger = logging.getLogger(__name__)
        cfg = get_cfg()

        add_e2e_config(cfg)
        add_glass_config(cfg)
        add_dataset_config(cfg)
        add_post_process_config(cfg)
        cfg.merge_from_file(config_path)
        cfg.merge_from_list(opts or list())

        self.model_path = model_path
        self.config_path = config_path
        self.post_process_flag = post_process

        # Logging
        self.logger.info('Building GLASS Text Spotting Model')
        self.logger.info(f'Model path: {model_path}')
        self.logger.info(f'Config path: {config_path}')
        self.logger.info(f'Post-Process: {post_process}')

        # Initializing the architecture
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.cpu_device = torch.device("cpu")
        self.device = self.model.device

        # Loading the weights
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(model_path)

        # Defining the pre-processing transforms (resize, etc...)
        self.min_target_size = self.cfg.INPUT.MIN_SIZE_TEST
        self.max_target_size = self.cfg.INPUT.MAX_SIZE_TEST
        self.max_upscale_ratio = self.cfg.INPUT.MAX_UPSCALE_RATIO
        self.input_format = self.cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR", "GREY"], self.input_format

        self.text_encoder = TextEncoder(self.cfg)
        self.post_processor = build_post_processor(self.cfg)

    def __call__(self, image_tensor: np.ndarray) -> Instances:
        """
        Args:
            original_image (np.ndarray):

        Returns:
            preds (Instances)
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
            original_image_tensor (torch.Tensor):
        """
        # if self.input_format == "RGB":
        #     # whether the model expects BGR inputs or RGB
        #     original_image = original_image[:, :, ::-1]
        # if self.input_format == "GREY":
        #     original_image = rgb2grey(original_image, three_channels=True)
        # image_height, image_width = original_image.shape[1:]

        #image_tensor, scale_ratio = self._image_to_tensor(original_image, self.model.device)
        image_tensor = image_tensor.to(device).to(torch.float32)
        height = image_tensor.shape[1]
        width = image_tensor.shape[2]
        inputs = {'image': image_tensor, 'height': height, 'width': width}

        with torch.no_grad():
            raw_predictions = self.model([inputs])[0]
        preds = raw_predictions['instances']

        # Scaling the predictions to the image domain
        # if scale_ratio != 1:
        #     preds.pred_boxes.scale(1 / scale_ratio, 1 / scale_ratio)
        # preds._image_size = (image_height, image_width)

        self.logger.info(f'Detected {len(preds)} raw word instances')

        preds = self.post_processor(preds)

        self.logger.info(f'Post-processing output is {len(preds)} word instances')
        return preds

    def get_inference_scale_ratio(self, image_shape):
        height, width = image_shape[:2]
        max_image_dim = max(height, width)

        if max_image_dim > self.max_target_size:
            scale_ratio = self.max_target_size / max_image_dim
        elif max_image_dim < self.min_target_size:
            scale_ratio = min(self.max_upscale_ratio, self.min_target_size / max_image_dim)
        else:
            scale_ratio = 1
        return scale_ratio

    def _image_to_tensor(self, original_image, device, interpolation='bilinear'):
        """
        Transfers the image to the GPU as a resized tensor
        :param np.ndarray original_image: The original image as a numpy array (H, W, C)
        :param device: The cuda device (or CPU) to which we send the tensor
        :param str interpolation: Either 'nearest' or 'bilinear' are supported for interpolation algorithms
        :return: Both the resized image tensor, and the original image tensor
        """
        height, width = original_image.shape[1:]

        image_tensor = torch.as_tensor(original_image.transpose((2, 0, 1)))
        image_tensor = image_tensor.to(device).to(torch.float32)

        # Computing the necessary scale ratio (> 1 for enlarging image)
        scale_ratio = self.get_inference_scale_ratio(original_image.shape)

        # Resizing if necessary, if not we just clone the image
        if scale_ratio != 1:
            new_height, new_width = int(np.round(scale_ratio * height)), int(np.round(scale_ratio * width))
            image_tensor_resized = torch.nn.functional.interpolate(image_tensor.unsqueeze(dim=0),
                                                                   size=(new_height, new_width),
                                                                   mode=interpolation,
                                                                   align_corners=False).squeeze(dim=0)
        else:
            image_tensor_resized = image_tensor.clone()
        return image_tensor_resized, scale_ratio

    def preds_boxes_to_polygons(self, pred_boxes):
        box_tensor = pred_boxes.tensor
        polygons = self.post_processor.boxes_to_polygons(boxes=box_tensor)
        return polygons
