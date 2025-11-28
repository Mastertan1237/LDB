# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from ..layers import (LDBDeformableDetrTransformerDecoder,
                      LDBDeformableDetrTransformerEncoder,
                      SinePositionalEncoding)
from .deformable_detr import DeformableDETR


@MODELS.register_module()
class LDBDeformableDETR(DeformableDETR):
    """LDB version of Deformable DETR.

    Implements Learnable Domain Bias (LDB) for domain incremental learning.
    Supports freezing backbone and neck for incremental training.

    Args:
        freeze_backbone (bool): Whether to freeze backbone parameters.
            Defaults to False.
        freeze_neck (bool): Whether to freeze neck parameters.
            Defaults to False.
    """

    def __init__(self,
                 *args,
                 freeze_backbone: bool = False,
                 freeze_neck: bool = False,
                 **kwargs) -> None:
        self.freeze_backbone = freeze_backbone
        self.freeze_neck = freeze_neck
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers with LDB encoder and decoder."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        # Use LDB versions of encoder and decoder
        self.encoder = LDBDeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = LDBDeformableDetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_queries,
                                                self.embed_dims * 2)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
            self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
            self.pos_trans_fc = nn.Linear(self.embed_dims * 2,
                                          self.embed_dims * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        else:
            self.reference_points_fc = nn.Linear(self.embed_dims, 2)

        # Freeze backbone and neck if specified
        if self.freeze_backbone:
            self._freeze_module(self.backbone)
        if self.freeze_neck:
            self._freeze_module(self.neck)

    def _freeze_module(self, module: nn.Module) -> None:
        """Freeze all parameters in a module.

        Args:
            module (nn.Module): Module to freeze.
        """
        for param in module.parameters():
            param.requires_grad = False

    def get_ldb_parameters(self) -> List[nn.Parameter]:
        """Get all LDB parameters (dbias_*).

        Returns:
            List[nn.Parameter]: List of LDB parameters.
        """
        ldb_params = []
        for name, param in self.named_parameters():
            if 'dbias_' in name:
                ldb_params.append(param)
        return ldb_params

    def get_head_parameters(self) -> List[nn.Parameter]:
        """Get all bbox_head parameters.

        Returns:
            List[nn.Parameter]: List of bbox_head parameters.
        """
        return list(self.bbox_head.parameters())

    def get_trainable_parameters_info(self) -> Dict[str, int]:
        """Get information about trainable parameters.

        Returns:
            Dict[str, int]: Dictionary containing parameter counts.
        """
        total_params = 0
        trainable_params = 0
        ldb_params = 0
        head_params = 0
        backbone_params = 0
        neck_params = 0
        encoder_params = 0
        decoder_params = 0

        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if 'dbias_' in name:
                    ldb_params += param.numel()
                if 'bbox_head' in name:
                    head_params += param.numel()
            if 'backbone' in name:
                backbone_params += param.numel()
            if 'neck' in name:
                neck_params += param.numel()
            if 'encoder' in name:
                encoder_params += param.numel()
            if 'decoder' in name:
                decoder_params += param.numel()

        return {
            'total': total_params,
            'trainable': trainable_params,
            'ldb': ldb_params,
            'head': head_params,
            'backbone': backbone_params,
            'neck': neck_params,
            'encoder': encoder_params,
            'decoder': decoder_params
        }
