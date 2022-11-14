from typing import List, Dict, Optional

import torch

from detectron2.config import configurable
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.structures import Instances
from ...postprocess.post_processor import build_post_processor
from ...postprocess.post_processor_academic import detector_postprocess


@META_ARCH_REGISTRY.register()
class GlassRCNN(GeneralizedRCNN):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(self, *, post_processor, inflate_ratio, filter_small_boxes, transcript_filtering,
                 drop_overlapping_boxes,
                 ioa_threshold, valid_score, **kwargs):
        super().__init__(**kwargs)
        self.post_processor = post_processor
        self.inflate_ratio = inflate_ratio
        self.transcript_filtering = transcript_filtering
        self.filter_small_boxes = filter_small_boxes
        self.ioa_threshold = ioa_threshold
        self.valid_score = valid_score
        self.drop_overlapping_boxes = drop_overlapping_boxes

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['post_processor'] = build_post_processor(cfg)

        ret['inflate_ratio'] = cfg.POST_PROCESSING.INFLATE_RATIO if hasattr(cfg.POST_PROCESSING, 'INFLATE_RATIO') \
            else None
        ret['transcript_filtering'] = cfg.POST_PROCESSING.TRANSCRIPT_FILTERING if hasattr(cfg.POST_PROCESSING,
                                                                                          'TRANSCRIPT_FILTERING') \
            else None
        ret['filter_small_boxes'] = cfg.POST_PROCESSING.MIN_BOX_DIMENSION if hasattr(cfg.POST_PROCESSING,
                                                                                     'MIN_BOX_DIMENSION') \
            else None
        ret['drop_overlapping_boxes'] = cfg.POST_PROCESSING.DROP_OVERLAPPING if hasattr(cfg.POST_PROCESSING,
                                                                                        'DROP_OVERLAPPING') \
            else None
        ret['ioa_threshold'] = cfg.POST_PROCESSING.IOA_THRESHOLD if hasattr(cfg.POST_PROCESSING, 'IOA_THRESHOLD') \
            else None
        ret['valid_score'] = cfg.INFERENCE_TH_TEST if hasattr(cfg, 'INFERENCE_TH_TEST') else 0

        return ret

    def inference(
            self,
            batched_inputs: List[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]] = None,
            do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return self._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def _postprocess(self, instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []

        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):

            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            if self.filter_small_boxes:
                results_per_image = self.post_processor.filter_small_boxes(results_per_image)
            if self.inflate_ratio:
                results_per_image = self.post_processor.resize_boxes(results_per_image, self.inflate_ratio)

            if self.drop_overlapping_boxes:
                results_per_image = self.post_processor.drop_overlapping_boxes(results_per_image,
                                                                               self.ioa_threshold,
                                                                               self.valid_score)
            r = detector_postprocess(results_per_image, height, width)

            processed_results.append({"instances": r})
        return processed_results
