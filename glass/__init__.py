import detectron2 # noqa

# These imports are used for d2 registry updates
from glass.modeling.meta_arch.glass_rcnn import META_ARCH_REGISTRY  # noqa
from glass.modeling.proposal_generator.rotated_rpn import PROPOSAL_GENERATOR_REGISTRY # noqa
from glass.modeling.fusion.local_feature_extraction import LOCAL_FEATURE_EXTRACTOR_REGISTRY # noqa
from glass.modeling.fusion.recognizers_hybrid_head import ROI_HEADS_REGISTRY # noqa
from glass.modeling.recognition.recognizer_head_v2 import ROI_RECOGNIZER_HEAD_REGISTRY # noqa
from glass.modeling.roi_heads.rotated_mask_head import ROI_MASK_HEAD_REGISTRY # noqa


# Replacing the resize transform with a fast variant
from glass.data.transforms.transform import FastResizeTransform
from detectron2.data.transforms import ResizeTransform

ResizeTransform.apply_image = FastResizeTransform.apply_image

# Patching the matcher class with our own matcher with a few bug fixes and optimizations
from glass.modeling.matcher import Matcher as GlassMatcher  # noqa
from detectron2.modeling.matcher import Matcher  # noqa

Matcher.set_low_quality_matches_ = GlassMatcher.set_low_quality_matches_
Matcher.__call__ = GlassMatcher.__call__
