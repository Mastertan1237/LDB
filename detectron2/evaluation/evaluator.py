# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds

import xml.dom.minidom
import os
import numpy as np
import timm
from torchvision.transforms import Compose, ToTensor, Normalize,Resize,CenterCrop
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans, k_means

from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
import cv2


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def combine_model(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], original_model = None
):
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    # logger.info("Start inference on {} batches".format(len(data_loader)))

    # total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    results = {}
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        avg_feature = nn.Parameter(torch.zeros(1, 768)).cuda()

        for idx, inputs in enumerate(data_loader):


            start_compute_time = time.perf_counter()
            if original_model is not None:
                if isinstance(original_model, nn.Module):
                    stack.enter_context(inference_context(original_model))
                    stack.enter_context(torch.no_grad())
                outputs, res = original_model([inputs, None, True])
                cls_features = res['bbfeatures']
                fea_batch = res['fea_batch']
            else:
                cls_features = None

            if idx == 0:
                avg_feature = avg_feature
            else:
                avg_feature = torch.cat((avg_feature, fea_batch), 0)

            # outputs = model(inputs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

    if results is None:
        results = {}

    from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
    import numpy as np
    avg_feature = np.array(avg_feature.cpu())
    c_1, rss_1 = kmeans(avg_feature, 1)
    return results, c_1


def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], original_model = None
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0

    all_box_feature = []
    all_label = []

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()

            if original_model is not None:
                if isinstance(original_model, nn.Module):
                    stack.enter_context(inference_context(original_model))
                    stack.enter_context(torch.no_grad())
                outputs, res = original_model([inputs, None, True])
                # cls_features = res['bbfeatures']
                cls_features = None
            else:
                cls_features = None
            # print(inputs)
            outputs, res = model([inputs, cls_features, False])


            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()
    # tsne(all_box_feature, all_label)

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


# for final NMC evaluation
def inference_on_dataset_eval(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], original_model = None, model_base = None
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    model_base = torch.load("model_base3.pth")  

    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
            stack.enter_context(inference_context(original_model))
        stack.enter_context(torch.no_grad())
        start_data_time = time.perf_counter()

        import sklearn
        from sklearn.neighbors import KNeighborsClassifier
        from collections import OrderedDict
        knn = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto')
        KNN_train = model_base["d0.key"]  # .unsqueeze(0).cpu()
        KNN_label = np.zeros((model_base["d0.key"].shape[0], 1), dtype = int)  
        for i in range(1, 4):     # for session
            KNN_train = np.concatenate((KNN_train, model_base["d"+str(i)+".key"]), axis=0) 
            li = np.zeros((model_base["d0.key"].shape[0], 1), dtype = int)
            for j in range(li.shape[0]):
                li[j][0] = i
            KNN_label = np.concatenate((KNN_label, li), axis=0)
        knn.fit(KNN_train, KNN_label)

        # key = model_base["d0.key"].unsqueeze(0)
        # for i in range(1, 4):
        #     key = torch.cat([key, model_base["d"+str(i)+".key"].unsqueeze(0)], dim=0)
        fea_batch = 0
        fea_batch_last = 0
        cal_zero = 0
        total_img = 0
        correct_pred = 0

        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            if original_model is not None:
                # if isinstance(original_model, nn.Module):
                #     stack.enter_context(inference_context(original_model))
                #     stack.enter_context(torch.no_grad())
                outputs, res = original_model([inputs, None, True])
                # outputs, res = model([inputs, None, True])
                cls_features = res['bbfeatures']
                fea_batch = res['fea_batch']
            else:
                cls_features = None

           
            y_pred = knn.predict(np.array(fea_batch.cpu())) # batch=1

            y_pred = y_pred.tolist()
            y_pred = max(set(y_pred), key = y_pred.count)
            if y_pred == 0:
                correct_pred = correct_pred + 1
            total_img = total_img + 1

            print("correct_pred: ", correct_pred, total_img, correct_pred/total_img)
            if y_pred == 0:
                cal_zero = cal_zero + 1

            if y_pred != 0:
                new_state_dict = OrderedDict()
                for k in list(model.state_dict().keys()):
                    if k.startswith(tuple(['roi_heads.box_predictor', 'roi_heads.mask_head.predictor'])): 
                        new_state_dict[k] = model_base["d"+str(y_pred)+"."+k]
                    elif "dbias" in k:
                        new_state_dict[k] = model_base["d"+str(y_pred)+"."+k]
                
                model.load_state_dict(new_state_dict, strict=False)
            else:
                new_state_dict = OrderedDict()
                for k in list(model.state_dict().keys()):
                    if k.startswith(tuple(['roi_heads.box_predictor', 'roi_heads.mask_head.predictor'])):
                        new_state_dict[k] = model_base[k]
                model.load_state_dict(new_state_dict, strict=False)

            outputs, res = model([inputs, cls_features, False])

            # outputs, res = model([inputs, None, True])
            #reduce_sim = res['reduce_sim']

            #outputs = model(inputs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()
        
        print("cal_zero: ", cal_zero)

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

