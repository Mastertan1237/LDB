from detectron2.config import LazyCall as L
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.evaluation.pascal_voc_evaluation import PascalVOCDetectionEvaluator
from detectron2.evaluation.cityscapes_evaluation import CityscapesEvaluator, CityscapesInstanceEvaluator, CityscapesSemSegEvaluator
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.data.detection_utils import get_fed_loss_cls_weights

from ..COCO.mask_rcnn_vitdet_b_100ep_dbias import (
    dataloader,
    model,
    train,
    lr_multiplier,
    optimizer,
)

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
import os


DATASET_ROOT = './datasets/Watercolor'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'train')
TEST_PATH = os.path.join(DATASET_ROOT, 'test') 
TRAIN_JSON = os.path.join(ANN_ROOT, 'train.json')
TEST_JSON = os.path.join(ANN_ROOT, 'test.json') 

DATASET_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 0, "name": "bicycle"},
    {"color": [119, 11, 32], "isthing": 1, "id": 1, "name": "bird"},
    {"color": [0, 0, 142], "isthing": 1, "id": 2, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 3, "name": "cat"},
    {"color": [106, 0, 228], "isthing": 1, "id": 4, "name": "dog"},
    {"color": [120, 166, 157], "isthing": 1, "id": 5, "name": "person"},
]

def get_dataset_instances_meta_train():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k["id"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

DatasetCatalog.register("watercolorcoco_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH, "watercolorcoco_train"))
MetadataCatalog.get("watercolorcoco_train").set(json_file=TRAIN_JSON, image_root=TRAIN_PATH, evaluator_type="coco", **get_dataset_instances_meta_train())

def get_dataset_instances_meta_test():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k["id"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

DatasetCatalog.register("watercolorcoco_test", lambda: load_coco_json(TEST_JSON, TEST_PATH, "watercolorcoco_test"))
MetadataCatalog.get("watercolorcoco_test").set(json_file=TEST_JSON, image_root=TEST_PATH, evaluator_type="coco", **get_dataset_instances_meta_test())

# dataloader.train.dataset.names = "cityscapes_fine_instance_seg_train"
dataloader.train.dataset.names = "watercolorcoco_train"
dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
        dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.001
    )
)
# # dataloader.test.dataset.names = "cityscapes_fine_instance_seg_val"
dataloader.test.dataset.names = "watercolorcoco_test"
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    output_dir = "output/watercolor"
)

model.roi_heads.num_classes = 6

# # Schedule
# # 100 ep = 156250 iters * 64 images/iter / 100000 images/ep
train.max_iter = 35000

lr_multiplier.scheduler.milestones = [20000, 34000]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 250 / train.max_iter

optimizer.lr = 2e-4
