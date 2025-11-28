# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class DomainIncrementalHook(Hook):
    """Hook for domain incremental training.

    This hook freezes specified parameters at the beginning of training
    and only allows LDB parameters (dbias_*) and classification/regression
    heads to be updated.

    Args:
        freeze_backbone (bool): Whether to freeze backbone. Defaults to True.
        freeze_neck (bool): Whether to freeze neck. Defaults to True.
        freeze_encoder (bool): Whether to freeze encoder base params
            (excluding LDB). Defaults to True.
        freeze_decoder (bool): Whether to freeze decoder base params
            (excluding LDB). Defaults to True.
        print_trainable_params (bool): Whether to print trainable parameters
            info. Defaults to True.
    """

    priority = 'NORMAL'

    def __init__(self,
                 freeze_backbone: bool = True,
                 freeze_neck: bool = True,
                 freeze_encoder: bool = True,
                 freeze_decoder: bool = True,
                 print_trainable_params: bool = True) -> None:
        self.freeze_backbone = freeze_backbone
        self.freeze_neck = freeze_neck
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.print_trainable_params = print_trainable_params

    def before_train(self, runner: Runner) -> None:
        """Freeze specified parameters before training starts."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        # Freeze specified components
        self._freeze_parameters(model)

        # Print trainable parameters info
        if self.print_trainable_params:
            self._print_trainable_info(model, runner)

    def _freeze_parameters(self, model) -> None:
        """Freeze parameters based on configuration.

        Only LDB parameters (dbias_*) and bbox_head parameters remain
        trainable.
        """
        for name, param in model.named_parameters():
            # Always keep LDB parameters trainable
            if 'dbias_' in name:
                param.requires_grad = True
                continue

            # Always keep bbox_head parameters trainable
            if 'bbox_head' in name:
                param.requires_grad = True
                continue

            # Freeze backbone
            if self.freeze_backbone and 'backbone' in name:
                param.requires_grad = False
                continue

            # Freeze neck
            if self.freeze_neck and 'neck' in name:
                param.requires_grad = False
                continue

            # Freeze encoder (excluding LDB)
            if self.freeze_encoder and 'encoder' in name:
                param.requires_grad = False
                continue

            # Freeze decoder (excluding LDB)
            if self.freeze_decoder and 'decoder' in name:
                param.requires_grad = False
                continue

    def _print_trainable_info(self, model, runner: Runner) -> None:
        """Print information about trainable parameters."""
        total_params = 0
        trainable_params = 0
        ldb_params = 0
        head_params = 0
        frozen_params = 0

        trainable_names = []
        frozen_names = []

        for name, param in model.named_parameters():
            num_params = param.numel()
            total_params += num_params

            if param.requires_grad:
                trainable_params += num_params
                trainable_names.append(name)
                if 'dbias_' in name:
                    ldb_params += num_params
                if 'bbox_head' in name:
                    head_params += num_params
            else:
                frozen_params += num_params
                frozen_names.append(name)

        print_log('=' * 60, logger='current')
        print_log('Domain Incremental Training - Parameter Statistics',
                  logger='current')
        print_log('=' * 60, logger='current')
        print_log(f'Total parameters: {total_params:,}', logger='current')
        print_log(f'Trainable parameters: {trainable_params:,} '
                  f'({100*trainable_params/total_params:.2f}%)',
                  logger='current')
        print_log(f'  - LDB parameters: {ldb_params:,}', logger='current')
        print_log(f'  - Head parameters: {head_params:,}', logger='current')
        print_log(f'Frozen parameters: {frozen_params:,} '
                  f'({100*frozen_params/total_params:.2f}%)',
                  logger='current')
        print_log('=' * 60, logger='current')

        # Print sample of trainable parameter names
        print_log('Trainable parameter groups:', logger='current')
        ldb_count = 0
        head_count = 0
        other_count = 0
        for name in trainable_names:
            if 'dbias_' in name:
                ldb_count += 1
            elif 'bbox_head' in name:
                head_count += 1
            else:
                other_count += 1

        print_log(f'  - LDB params: {ldb_count} tensors', logger='current')
        print_log(f'  - Head params: {head_count} tensors', logger='current')
        if other_count > 0:
            print_log(f'  - Other params: {other_count} tensors',
                      logger='current')
        print_log('=' * 60, logger='current')
