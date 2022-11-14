import copy
import glob
import itertools
import json
import logging
import os
import re
import shutil
import zipfile
from collections import OrderedDict

import numpy as np
import rasterio
import shapely
import torch
from fvcore.common.file_io import PathManager
from rasterio import features
from shapely.geometry import Polygon
from shapely.geometry import Polygon, LinearRing

from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils import comm
from . import text_eval_script
from .lexicon_utils import find_match_word, get_lexicon
from ..modeling.recognition.text_encoder import TextEncoder


class TextEvaluator(DatasetEvaluator):
    """
    Evaluate text proposals and recognition.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self._tasks = ("polygon", "recognition")
        self._distributed = distributed
        self._output_dir = output_dir
        # self.save_analysis = cfg.SAVE_ANALYSIS
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self.text_encoder = TextEncoder(cfg)
        self._metadata = MetadataCatalog.get(dataset_name)
        self.edit_distance_thr = cfg.EDIT_DISTANCE_THR if hasattr(cfg, 'EDIT_DISTANCE_THR') else 1.5

        if not hasattr(self._metadata, "json_file"):
            raise AttributeError(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'."
            )

        self._results = OrderedDict()
        self._word_spotting = cfg.IS_WORD_SPOTTING
        self.onlyRemoveFirstLastCharacter = cfg.onlyRemoveFirstLastCharacter
        self._text_eval_confidence = cfg.INFERENCE_TH_TEST
        self._detection_eval_confidence = cfg.INFERENCE_DETECTION_TH_TEST if hasattr(cfg,
                                                                                     'INFERENCE_DETECTION_TH_TEST') else 0.0

        if 'totaltext' in dataset_name.lower():
            self.dataset = 'totaltext'
            self._text_eval_gt_path = '/hiero_efs/HieroUsers/roironen/experiments/E2E_EVAL_TotalText/gt_totaltext.zip'
        elif 'icdar15' in dataset_name.lower():
            self.dataset = 'icdar15'
            self._text_eval_gt_path = '/hiero_efs/HieroUsers/roironen/experiments/E2E_EVAL_ICDAR15/gt_icdar15_masktextspotterv3.zip'
        elif 'icdar2013_r45' in dataset_name.lower():
            self.dataset = 'icdar13_45'
            self._text_eval_gt_path = '/hiero_efs/HieroUsers/yairk/outputs/WeaklySupervisedOCR/latest/tests/ic13_r45/full_ft_high_abc_alp/2021-10-06_12-09/eval/test_gts_fixed.zip'
        elif 'icdar2013_r60' in dataset_name.lower():
            self.dataset = 'icdar13_60'
            self._text_eval_gt_path = '/hiero_efs/HieroUsers/yairk/outputs/WeaklySupervisedOCR/latest/tests/ic13_r60/full_ft_high_abc_alp/2021-10-06_12-14/eval/test_gts_fixed.zip'
        elif 'icdar2013' in dataset_name.lower():
            self.dataset = 'icdar13_0'
            self._text_eval_gt_path = '/hiero_efs/HieroUsers/yairk/outputs/WeaklySupervisedOCR/latest/tests/ic13_r60/full_ft_high_abc_alp/2021-10-06_12-14/eval/test_gts_fixed.zip'
        else:
            self.dataset = 'textocr'
            self._text_eval_gt_path = '/hiero_efs/HieroUsers/roironen/experiments/E2E_EVAL_TEXTOCR/gt_textocr.zip'  # '/hiero_efs/HieroUsers/roironen/Deploy/gt_textocr_debug.zip' '/hiero_efs/HieroUsers/roironen/experiments/E2E_EVAL_TEXTOCR/gt_textocr_.zip'

        # get lexicon
        self.weighted_ed = cfg.TEST.LEXICON_WEIGHTED if hasattr(cfg.TEST, 'LEXICON_WEIGHTED') else False
        self.lexicon_type = cfg.TEST.LEXICON_TYPE if hasattr(cfg.TEST, 'LEXICON_TYPE') else None
        self.lexicon, self.pairs = None, None
        if self.lexicon_type:
            self.lexicon, self.pairs = get_lexicon(self.dataset, self.lexicon_type)

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):

        for input, output in zip(inputs, outputs):
            prediction = {"file_name": input["file_name"]}
            instances = output["instances"].to(self._cpu_device)
            prediction["instances"] = instances_to_coco_json(instances, input["file_name"], self.text_encoder,
                                                             self.onlyRemoveFirstLastCharacter)

            self._predictions.append(prediction)

    def sort_detection(self, temp_dir, text_cf_th=0.5, detection_cf_th=0.0):
        origin_file = temp_dir
        output_file = "final_" + temp_dir

        if not os.path.isdir(output_file):
            os.mkdir(output_file)

        files = glob.glob(origin_file + '*.txt')
        files.sort()

        for i in files:
            out = i.replace(origin_file, output_file)
            fin = open(i, 'r').readlines()
            fout = open(out, 'w')
            for iline, line in enumerate(fin):
                ptr = line.strip().split(',####')
                rec = ptr[1]
                cors = ptr[0].split(',')
                assert (len(cors) % 2 == 0), 'cors invalid.'
                pts = [(int(cors[j]), int(cors[j + 1])) for j in range(0, len(cors), 2)]
                try:
                    pgt = Polygon(pts)
                except Exception as e:
                    print(e)
                    print('An invalid detection in {} line {} is removed ... '.format(i, iline))
                    continue

                if not pgt.is_valid:
                    print('An invalid detection in {} line {} is removed ... '.format(i, iline))
                    continue

                pRing = LinearRing(pts)
                if pRing.is_ccw:
                    pts.reverse()
                outstr = ''
                for ipt in pts[:-1]:
                    outstr += (str(int(ipt[0])) + ',' + str(int(ipt[1])) + ',')
                outstr += (str(int(pts[-1][0])) + ',' + str(int(pts[-1][1])))
                outstr = outstr + ',####' + rec
                fout.writelines(outstr + '\n')
            fout.close()
        os.chdir(output_file)

        def zipdir(path, ziph):
            # ziph is zipfile handle
            for root, dirs, files in os.walk(path):
                for file in files:
                    ziph.write(os.path.join(root, file))

        if not os.path.isdir('../{}_{}'.format(text_cf_th, detection_cf_th)):
            os.mkdir('../{}_{}'.format(text_cf_th, detection_cf_th))
        zipf = zipfile.ZipFile('../{}_{}/det.zip'.format(text_cf_th, detection_cf_th), 'w', zipfile.ZIP_DEFLATED)
        zipdir('./', zipf)
        zipf.close()
        os.chdir("../")
        # clean temp files
        shutil.rmtree(origin_file)
        shutil.rmtree(output_file)
        return '{}_{}/det.zip'.format(text_cf_th, detection_cf_th)

    def to_eval_format(self, file_path, temp_dir="temp_results", text_cf_th=0.5, detection_cf_th=0.0):
        def fis_ascii(s):
            a = (ord(c) < 128 for c in s)
            return all(a)

        def de_ascii(s):
            a = [c for c in s if ord(c) < 128]
            outa = ''
            for i in a:
                outa += i
            return outa

        with open(file_path, 'r') as f:
            data = json.load(f)
            with open('temp_all_det_cors_{}_{}.txt'.format(str(text_cf_th), str(detection_cf_th)), 'w') as f2:
                for ix in range(len(data)):
                    if data[ix]['score_text'] > 0.001:
                        outstr = '{}: '.format(data[ix]['image_id'])
                        xmin = 1000000
                        ymin = 1000000
                        xmax = 0
                        ymax = 0
                        if 'polys' in data[ix]:
                            for i in range(len(data[ix]['polys'])):
                                outstr = outstr + str(int(data[ix]['polys'][i][0])) + ',' + str(
                                    int(data[ix]['polys'][i][1])) + ','
                        ass = de_ascii(data[ix]['rec'])
                        ### Add lexicon
                        if self.lexicon:
                            scores_numpy = data[ix]['character_probs']
                            if self.lexicon_type == 3 and self.dataset.startswith('icdar'):
                                # Type 3 lexicon is per image
                                lexicon = self.lexicon[data[ix]['image_id']]
                                pairs = self.pairs[data[ix]['image_id']]
                            else:
                                lexicon = self.lexicon
                                pairs = self.pairs
                            match_word, match_dist = find_match_word(ass, lexicon=lexicon, pairs=pairs,
                                                                     scores_numpy=scores_numpy,
                                                                     weighted_ed=self.weighted_ed,
                                                                     text_encoder=self.text_encoder)
                            if match_dist < self.edit_distance_thr or self.lexicon_type == 1:
                                # use line
                                ass = match_word
                            else:
                                continue  # do not write words that don't match the lexicon
                        ###
                        if self.lexicon or self._word_spotting:
                            ass = self.match_transcript(ass)
                        if len(ass) >= 0:  #
                            outstr = outstr + str(round(data[ix]['score_text'], 3)) + '|' + str(
                                round(data[ix]['score_detection'], 3)) + ',####' + ass + '\n'
                            f2.writelines(outstr)
                f2.close()
        dirn = temp_dir
        lsc_text = [text_cf_th]
        lsc_detection = [detection_cf_th]
        fres = open('temp_all_det_cors_{}_{}.txt'.format(str(text_cf_th), str(detection_cf_th)), 'r').readlines()
        for isc_text, isc_detection in zip(lsc_text, lsc_detection):
            if not os.path.isdir(dirn):
                os.mkdir(dirn)

            for line in fres:
                line = line.strip()
                s = line.split(': ')
                if self.dataset == 'totaltext':
                    filename = '{:07d}.txt'.format(int(s[0]))
                elif self.dataset.startswith('icdar'):
                    filename = '{}.txt'.format(int(s[0]))
                elif self.dataset == 'textocr':
                    filename = '{:07d}.txt'.format(int(s[0]))
                else:
                    raise ValueError
                outName = os.path.join(dirn, filename)
                with open(outName, 'a') as fout:
                    ptr = s[1].strip().split(',####')
                    scores = ptr[0].split(',')[-1]
                    score_text = scores.split('|')[0]
                    score_detection = scores.split('|')[1]
                    if float(score_text) < isc_text or float(score_detection) < isc_detection:
                        continue
                    cors = ','.join(e for e in ptr[0].split(',')[:-1])
                    fout.writelines(cors + ',####' + ptr[1] + '\n')
        os.remove('temp_all_det_cors_{}_{}.txt'.format(str(text_cf_th), str(detection_cf_th)))

    def evaluate_with_official_code(self, result_path, gt_path):
        return text_eval_script.text_eval_main(det_file=result_path, gt_file=gt_path,
                                               is_word_spotting=self._word_spotting)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        # sort prediction
        if self.dataset == 'totaltext':
            predictions = sorted(predictions, key=lambda k: k['file_name'])
        elif self.dataset.startswith('icdar'):
            predictions = sorted(predictions, key=lambda k: float(re.split('([-+]?[0-9]*\.]*)', k['file_name'])[1]))

        for i, pred in enumerate(predictions):
            id = i if self.dataset == 'totaltext' \
                else i + 1  # 'icdar15'
            pred['image_id'] = id
            for x in pred['instances']:
                x['image_id'] = id
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        PathManager.mkdirs(self._output_dir)

        file_path = os.path.join(self._output_dir, "text_results.json")
        self._logger.info("Saving results to {}".format(file_path))
        with PathManager.open(file_path, "w") as f:
            f.write(json.dumps(coco_results))
            f.flush()

        self._results = OrderedDict()

        # eval text
        temp_dir = "temp_det_results/"
        self.to_eval_format(file_path, temp_dir, self._text_eval_confidence, self._detection_eval_confidence)
        result_path = self.sort_detection(temp_dir, self._text_eval_confidence, self._detection_eval_confidence)
        text_result = self.evaluate_with_official_code(result_path, self._text_eval_gt_path)
        os.remove(result_path)
        # parse
        template = "(\S+): (\S+): (\S+), (\S+): (\S+), (\S+): (\S+)"
        for task in ("e2e_method", "det_only_method"):
            result = text_result[task]
            groups = re.match(template, result).groups()
            self._results[groups[0]] = {groups[i * 2 + 1]: float(groups[(i + 1) * 2]) for i in range(3)}
        return copy.deepcopy(self._results)

    def match_transcript(self, transcription):
        specialCharacters = str("'!?.:,*+\"()·[]/\#$%;<=>@^_`{|}~")

        if self._word_spotting:
            # special case 's at final
            if transcription[len(transcription) - 2:] == "'s" or transcription[len(transcription) - 2:] == "'S":
                transcription = transcription[0:len(transcription) - 2]

            # hypens at init or final of the word
            transcription = transcription.strip('-')

            for character in specialCharacters:
                transcription = transcription.replace(character, ' ')

            transcription = transcription.strip()

        else:
            if len(transcription) > 0 and specialCharacters.find(transcription[0]) > -1:
                transcription = transcription[1:]

            if len(transcription) > 0 and specialCharacters.find(transcription[-1]) > -1:
                transcription = transcription[:-1]

        return transcription


def get_instances_text(text_probs, text_encoder, onlyRemoveFirstLastCharacter=True):
    if len(text_probs):
        text_probs = text_probs.detach().cpu()
        pred_probs, preds_indices = text_probs.max(dim=2)
        text_probs = text_probs.numpy()
        pred_text_objects = text_encoder.decode_prod_v2(pred_probs=pred_probs.numpy(),
                                                        pred_indices=preds_indices.numpy())
        pred_text = [x['text'] for x in pred_text_objects]
        pred_text_scores = [x['score'] for x in pred_text_objects]
        # specialCharacters = str(r'!?.:,*"()·[]/\'')
        specialCharacters = str("'!?.:,*+\"()·[]/\#$%;<=>@^_`{|}~")

        if onlyRemoveFirstLastCharacter:
            for i in range(len(pred_text)):
                if len(pred_text[i]) > 0 and specialCharacters.find(pred_text[i][0]) > -1:
                    pred_text[i] = pred_text[i][1:]

                if len(pred_text[i]) > 0 and specialCharacters.find(pred_text[i][-1]) > -1:
                    pred_text[i] = pred_text[i][:-1]

    else:
        pred_text = []
        pred_text_scores = []
        text_probs = []

    return pred_text, pred_text_scores, text_probs


def instances_to_coco_json(instances, file_name, text_encoder, onlyRemoveFirstLastCharacter):
    num_instances = len(instances)
    if num_instances == 0:
        return []

    if instances.has('pred_masks'):
        pred_masks = instances.pred_masks.numpy()
        polygons = masks_to_polygons(pred_masks)
    else:
        boxes = instances.pred_boxes.tensor.numpy()
        if boxes.shape[1] == 4:
            polygons = boxes_to_polygons(boxes).tolist()
        elif boxes.shape[1] == 5:
            # Deflate boxes
            polygons = rotated_boxes_to_polygons(boxes).tolist()
        else:
            assert ''
    if instances.has('pred_rboxes'):
        rboxes = rotated_boxes_to_polygons(instances.pred_rboxes.tensor.numpy()).tolist()
    else:
        rboxes = [[]] * len(polygons)
    if instances.has('pred_boxes'):
        boxes = boxes_to_polygons(instances.pred_boxes.tensor.numpy()).tolist()
    else:
        boxes = [[]] * len(polygons)
    text_probs = instances.pred_text_prob
    pred_text, scores_text, text_probs = get_instances_text(text_probs, text_encoder, onlyRemoveFirstLastCharacter)

    scores_detection = instances.scores.tolist()

    results = []

    for poly, rec, score_text, character_probs, box, rbox, score_detection in zip(polygons, pred_text, scores_text,
                                                                                  text_probs, boxes, rboxes,
                                                                                  scores_detection):

        if len(rec) > 0:
            if len(poly) >= 3:
                result = {
                    "image_id": file_name,
                    "category_id": 1,
                    "polys": poly,
                    "boxes": box,
                    "rboxes": rbox,
                    "rec": rec,
                    "score_text": np.float64(score_text).tolist(),
                    "character_probs": np.float64(character_probs).tolist(),
                    "score_detection": np.float64(score_detection).tolist()
                }
                results.append(result)
            # else:
            #     poly = rbox
            #     result = {
            #         "image_id": file_name,
            #         "category_id": 1,
            #         "polys": poly,
            #         "boxes": box,
            #         "rboxes": rbox,
            #         "rec": rec,
            #         "score": np.float64(score).tolist(),
            #     }
            #     results.append(result)
            #     print('polygon size is 0, replacing it by rotated box')

    return results


def boxes_to_polygons(boxes):
    n = len(boxes)
    if n == 0:
        return np.array([]).reshape((0, 4, 2))
    polygons = np.zeros((n, 4, 2))
    polygons[:, 0, 0] = boxes[:, 0]  # x0
    polygons[:, 0, 1] = boxes[:, 1]  # y0
    polygons[:, 1, 0] = boxes[:, 2]  # x1
    polygons[:, 1, 1] = boxes[:, 1]  # y0
    polygons[:, 2, 0] = boxes[:, 2]  # x1
    polygons[:, 2, 1] = boxes[:, 3]  # y1
    polygons[:, 3, 0] = boxes[:, 0]  # x0
    polygons[:, 3, 1] = boxes[:, 3]  # y1
    return polygons


def rotated_boxes_to_polygons(boxes):
    n = len(boxes)
    if n == 0:
        return np.array([]).reshape((0, 4, 2))
    assert (
            boxes.shape[-1] == 5
    ), "The last dimension of input shape must be 5 for XYWHA format"
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    a = boxes[:, 4]
    t = np.deg2rad(-a)
    polygons = np.zeros((n, 4, 2))
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    # Computing X components
    polygons[:, 0, 0] = cx + (h * sin_t - w * cos_t) / 2
    polygons[:, 1, 0] = cx + (h * sin_t + w * cos_t) / 2
    polygons[:, 2, 0] = cx - (h * sin_t - w * cos_t) / 2
    polygons[:, 3, 0] = cx - (h * sin_t + w * cos_t) / 2
    # Computing Y components
    polygons[:, 0, 1] = cy - (h * cos_t + w * sin_t) / 2
    polygons[:, 1, 1] = cy - (h * cos_t - w * sin_t) / 2
    polygons[:, 2, 1] = cy + (h * cos_t + w * sin_t) / 2
    polygons[:, 3, 1] = cy + (h * cos_t - w * sin_t) / 2

    return polygons


def masks_to_polygons(masks):
    n = len(masks)
    if n == 0:
        return np.array([]).reshape((0, 1, 2))
    all_polygons = []
    for mask in masks:
        polygon = None
        # polygon = []
        # keep the largest polygon
        for shape, value in features.shapes(mask.astype(np.int16), mask=(mask),
                                            transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
            # polygon.append(shapely.geometry.shape(shape))
            if polygon:
                new_polygon = shapely.geometry.shape(shape)
                if polygon.area < new_polygon.area:
                    polygon = new_polygon
            else:
                polygon = shapely.geometry.shape(shape)
        # polygon = shapely.geometry.MultiPolygon(polygon)
        # if not polygon.is_valid:
        #     polygon = polygon.buffer(0)
        #     if polygon.type == 'Polygon':
        #         polygon = shapely.geometry.MultiPolygon([polygon])
        if polygon:
            x, y = polygon.exterior.coords.xy
            all_polygons.append(np.array([x, y]).T.tolist())
        else:
            all_polygons.append([])
    return all_polygons


CTLABELS = [' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4',
            '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^',
            '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
            't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']
