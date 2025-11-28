# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class Coco6ClassDataset(CocoDataset):
    """Dataset for COCO format with 6 classes for domain incremental learning.

    The 6 shared classes across VOC, Clipart, Watercolor, and Comic domains:
    bicycle, bird, car, cat, dog, person
    """

    METAINFO = {
        'classes': ('bicycle', 'bird', 'car', 'cat', 'dog', 'person'),
        'palette': [
            (119, 11, 32),   # bicycle
            (165, 42, 42),   # bird
            (0, 0, 142),     # car
            (255, 77, 255),  # cat
            (0, 226, 252),   # dog
            (220, 20, 60)    # person
        ]
    }
