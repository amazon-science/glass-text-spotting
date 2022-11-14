import logging
import os
from collections import OrderedDict
from datetime import datetime

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    DatasetEvaluators,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger

from glass.data.dataset_manager import DatasetManager
from glass.config import add_e2e_config, add_dataset_config, add_glass_config, \
    add_post_process_config, merge_from_dataset_config
from glass.evaluation.text_evaluator import TextEvaluator


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_list.append(TextEvaluator(dataset_name, cfg, True, output_folder))

        if len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]

        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create config and perform basic setups.
    """
    model_path = args.model
    config_file = args.config_file or os.path.join(os.path.dirname(model_path), 'config.yaml')
    cfg = get_cfg()
    add_e2e_config(cfg)
    add_glass_config(cfg)
    add_post_process_config(cfg)
    add_dataset_config(cfg)
    cfg.merge_from_file(config_file)
    merge_from_dataset_config(cfg, args.datasets)

    cfg.VIS_PERIOD = 0
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.35
    cfg.INFERENCE_TH_TEST = 0.3
    cfg.INFERENCE_DETECTION_TH_TEST = 0.65

    # cfg.MODEL.RPN.PRE_NMS_TOPK_TEST  = 4000
    # cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    # cfg.TEST.DETECTIONS_PER_IMAGE = 500

    # cfg.EDIT_DISTANCE_THR = 2.0
    cfg.DATALOADER.STEP_DATA_LOADER_SHUFFLE = False
    cfg.MODEL.ROI_MASK_HEAD.IGNORE_TEXT = [""]
    cfg.MODEL.ROI_MASK_HEAD.IGNORE_EMPTY_TEXT = False
    cfg.MODEL.ROI_RECOGNIZER_HEAD.IGNORE_TEXT = [""]
    cfg.MODEL.ROI_RECOGNIZER_HEAD.IGNORE_EMPTY_TEXT = False
    cfg.TEST.MIN_SIZE_TEST = 1000
    cfg.INPUT.MIN_SIZE_TEST = 1000
    cfg.INPUT.MAX_SIZE_TEST = 1000000

    cfg.IS_WORD_SPOTTING = False
    cfg.onlyRemoveFirstLastCharacter = True
    cfg.TEST.LEXICON_TYPE = None  # 0 (None), 1 (generic), 2 (weak), 3 (strong)
    cfg.TEST.LEXICON_WEIGHTED = False

    cfg.MODEL.ROI_MASK_HEAD.MASK_INFERENCE = True
    cfg.MODEL.ORIENTATION_ON = False
    cfg.MODEL.ROI_ORIENTATION_HEAD.APPLY_TO_BOXES = False
    # cfg.MODEL.ROI_HEADS.NAME = 'MaskRotatedRecognizerHybridHeadV2'

    cfg.POST_PROCESSING.IOA_THRESHOLD = 1.0
    cfg.POST_PROCESSING.DROP_OVERLAPPING = False
    # cfg.POST_PROCESSING.INFLATE_RATIO = -0.1
    # cfg.POST_PROCESSING.MIN_BOX_DIMENSION = 2

    cfg.merge_from_list(args.opts)

    # Getting the output dir from the args
    rank = comm.get_local_rank()
    date_string = datetime.now().strftime("%Y_%m/%d_%H%M")
    save_dir = args.output + '/{}'.format(date_string)
    cfg.OUTPUT_DIR = save_dir
    setup_logger(output=save_dir, distributed_rank=rank, name='RekognitionHieroDetectron2',
                 abbrev_name='RekogD2')

    if args.debug:
        cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATALOADER.PREFETCH_FACTOR = 2
    args.num_gpus = 1

    # We always get the optimal batch for evaluation (it's faster this way, and doesn't affect training)
    cfg.freeze()

    # Adding our custom datasets to the catalog
    DatasetManager(cfg).register()
    default_setup(cfg=cfg, args=args)

    return cfg


def main(args):
    if comm.is_main_process():
        pass
    cfg = setup(args)
    model = Trainer.build_model(cfg)

    def count_parameters(model):
        # table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        backbone = 0
        hybrid_net = 0
        fusion_net = 0
        mask_head = 0
        box_head = 0
        recognizer_head = 0
        rpn_head = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            if 'backbone' in name:
                backbone += param
            elif 'hybrid_net' in name:
                hybrid_net += param
            elif 'fusion_net' in name:
                fusion_net += param
            elif 'mask_head' in name:
                mask_head += param
            elif 'box_head' in name:
                box_head += param
            elif 'recognizer_head' in name:
                recognizer_head += param
            elif 'rpn_head' in name:
                rpn_head += param
            print(f"{name} Total Trainable Params: {param}")

            total_params += param
        print(f"Total Trainable Params: {total_params}")
        print(
            f"Total Trainable Params Backcone: {backbone / 1e6} RES34 {hybrid_net / 1e6} rpn_head {rpn_head / 1e6} fusion_net {fusion_net / 1e6} mask_head {mask_head / 1e6} box_head {box_head / 1e6} recognizer_head {recognizer_head / 1e6}")

    count_parameters(model)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(args.model)
    res = Trainer.test(cfg, model)
    if comm.is_main_process():
        verify_results(cfg, res)
    return res


if __name__ == "__main__":
    parser = default_argument_parser()
    # Adding our own custom arguments
    parser.add_argument('--datasets', help='Path to the dataset config')
    parser.add_argument('--model', help='Path to the evaluated model')
    parser.add_argument('--output', help='Path to output directory')
    parser.add_argument('--debug', help='Activates debug mode, for PyCharm debugging', action='store_true')

    arguments = parser.parse_args()

    print("Command Line Args:", arguments)
    launch(
        main,
        arguments.num_gpus,
        num_machines=arguments.num_machines,
        machine_rank=arguments.machine_rank,
        dist_url=arguments.dist_url,
        args=(arguments,),
    )
