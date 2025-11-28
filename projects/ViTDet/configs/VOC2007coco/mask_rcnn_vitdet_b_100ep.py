from detectron2.config import LazyCall as L
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.evaluation.pascal_voc_evaluation import PascalVOCDetectionEvaluator
from detectron2.evaluation.cityscapes_evaluation import CityscapesEvaluator, CityscapesInstanceEvaluator, CityscapesSemSegEvaluator
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.data.detection_utils import get_fed_loss_cls_weights

from ..COCO.mask_rcnn_vitdet_b_100ep import (
    dataloader,
    model,
    train,
    lr_multiplier,
    optimizer,
)

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
import os

DATASET_ROOT = './datasets/VOC2007coco'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'train')
VAL_PATH = os.path.join(DATASET_ROOT, 'val')
TRAIN_JSON = os.path.join(ANN_ROOT, 'train.json')
VAL_JSON = os.path.join(ANN_ROOT, 'val.json')


# 数据集类别元数据
DATASET_CATEGORIES_TRAIN = [
    {"color": [220, 20, 60], "isthing": 1, "id": 0, "name": "car"},
    {"color": [119, 11, 32], "isthing": 1, "id": 1, "name": "person"},
    {"color": [0, 0, 142], "isthing": 1, "id": 2, "name": "horse"},
    {"color": [0, 0, 230], "isthing": 1, "id": 3, "name": "bicycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 4, "name": "aeroplane"},
    {"color": [120, 166, 157], "isthing": 1, "id": 5, "name": "train"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "diningtable"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "dog"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "chair"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "cat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "bird"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "boat"},
    {"color": [220, 220, 0], "isthing": 1, "id": 12, "name": "pottedplant"},
    {"color": [175, 116, 175], "isthing": 1, "id": 13, "name": "tvmonitor"},
    {"color": [250, 0, 30], "isthing": 1, "id": 14, "name": "sofa"},
    {"color": [165, 42, 42], "isthing": 1, "id": 15, "name": "motorbike"},
    {"color": [255, 77, 255], "isthing": 1, "id": 16, "name": "bottle"},
    {"color": [0, 226, 252], "isthing": 1, "id": 17, "name": "bus"},
    {"color": [182, 182, 255], "isthing": 1, "id": 18, "name": "sheep"},
    {"color": [0, 82, 0], "isthing": 1, "id": 19, "name": "cow"},

]

def get_dataset_instances_meta_train():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k["id"] for k in DATASET_CATEGORIES_TRAIN if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES_TRAIN if k["isthing"] == 1]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES_TRAIN if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

DatasetCatalog.register("voc2007coco_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH, "voc2007coco_train"))
MetadataCatalog.get("voc2007coco_train").set(json_file=TRAIN_JSON, image_root=TRAIN_PATH, evaluator_type="coco", **get_dataset_instances_meta_train())


DATASET_CATEGORIES_VAL = [
    {"color": [220, 20, 60], "isthing": 1, "id": 0, "name": "car"},
    {"color": [119, 11, 32], "isthing": 1, "id": 1, "name": "person"},
    {"color": [0, 0, 142], "isthing": 1, "id": 2, "name": "horse"},
    {"color": [0, 0, 230], "isthing": 1, "id": 3, "name": "bicycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 4, "name": "aeroplane"},
    {"color": [120, 166, 157], "isthing": 1, "id": 5, "name": "train"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "diningtable"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "dog"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "chair"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "cat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "bird"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "boat"},
    {"color": [220, 220, 0], "isthing": 1, "id": 12, "name": "pottedplant"},
    {"color": [175, 116, 175], "isthing": 1, "id": 13, "name": "tvmonitor"},
    {"color": [250, 0, 30], "isthing": 1, "id": 14, "name": "sofa"},
    {"color": [165, 42, 42], "isthing": 1, "id": 15, "name": "motorbike"},
    {"color": [255, 77, 255], "isthing": 1, "id": 16, "name": "bottle"},
    {"color": [0, 226, 252], "isthing": 1, "id": 17, "name": "bus"},
    {"color": [182, 182, 255], "isthing": 1, "id": 18, "name": "sheep"},
    {"color": [0, 82, 0], "isthing": 1, "id": 19, "name": "cow"},

]

def get_dataset_instances_meta_val():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k["id"] for k in DATASET_CATEGORIES_VAL if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES_VAL if k["isthing"] == 1]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES_VAL if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

DatasetCatalog.register("voc2007coco_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "voc2007coco_val"))
MetadataCatalog.get("voc2007coco_val").set(json_file=VAL_JSON, image_root=VAL_PATH, evaluator_type="coco", **get_dataset_instances_meta_val())

dataloader.train.dataset.names = "voc2007coco_train"
dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
        dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.001
    )
)

dataloader.test.dataset.names = "voc2007coco_val"
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    output_dir = "output"
)

# # Schedule
# # 100 ep = 156250 iters * 64 images/iter / 100000 images/ep
# train.max_iter = 156250
# train.eval_period = 30000

# lr_multiplier.scheduler.milestones = [138889, 150463]
# lr_multiplier.scheduler.num_updates = train.max_iter
# lr_multiplier.warmup_length = 250 / train.max_iter

# optimizer.lr = 2e-4
