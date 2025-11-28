# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.model import ModuleList
from torch import Tensor

from mmdet.registry import MODELS
from .deformable_detr_layers import (
    DeformableDetrTransformerDecoder,
    DeformableDetrTransformerDecoderLayer,
    DeformableDetrTransformerEncoder,
    DeformableDetrTransformerEncoderLayer,
)

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


@MODELS.register_module()
class LDBDeformableDetrTransformerEncoderLayer(
        DeformableDetrTransformerEncoderLayer):
    """LDB version of Deformable DETR Transformer Encoder Layer.

    Adds learnable domain bias after attention and FFN.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # LDB parameters for attention
        self.dbias_vector_attn = nn.Parameter(torch.zeros(self.embed_dims))
        self.dbias_alpha_attn = nn.Linear(self.embed_dims, 1)
        # LDB parameters for FFN
        self.dbias_vector_ffn = nn.Parameter(torch.zeros(self.embed_dims))
        self.dbias_alpha_ffn = nn.Linear(self.embed_dims, 1)

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, **kwargs) -> Tensor:
        """Forward function with LDB injection.

        Args:
            query (Tensor): Input query, shape (bs, num_queries, dim).
            query_pos (Tensor): Positional encoding, shape (bs, num_queries,
                dim).
            key_padding_mask (Tensor): Mask, shape (bs, num_queries).

        Returns:
            Tensor: Output with LDB applied, shape (bs, num_queries, dim).
        """
        identity = query
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )
        query = self.norms[0](query)
        # Apply LDB after attention
        attn_alpha = self.dbias_alpha_attn(identity)
        query = query + attn_alpha * self.dbias_vector_attn

        identity = query
        query = self.ffn(query)
        query = self.norms[1](query)
        # Apply LDB after FFN
        ffn_alpha = self.dbias_alpha_ffn(identity)
        query = query + ffn_alpha * self.dbias_vector_ffn

        return query


@MODELS.register_module()
class LDBDeformableDetrTransformerDecoderLayer(
        DeformableDetrTransformerDecoderLayer):
    """LDB version of Deformable DETR Transformer Decoder Layer.

    Adds learnable domain bias after self-attention, cross-attention, and FFN.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # LDB parameters for self-attention
        self.dbias_vector_self = nn.Parameter(torch.zeros(self.embed_dims))
        self.dbias_alpha_self = nn.Linear(self.embed_dims, 1)
        # LDB parameters for cross-attention
        self.dbias_vector_cross = nn.Parameter(torch.zeros(self.embed_dims))
        self.dbias_alpha_cross = nn.Linear(self.embed_dims, 1)
        # LDB parameters for FFN
        self.dbias_vector_ffn = nn.Parameter(torch.zeros(self.embed_dims))
        self.dbias_alpha_ffn = nn.Linear(self.embed_dims, 1)

    def forward(
        self,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        query_pos: Tensor = None,
        key_pos: Tensor = None,
        self_attn_mask: Tensor = None,
        cross_attn_mask: Tensor = None,
        key_padding_mask: Tensor = None,
        **kwargs,
    ) -> Tensor:
        """Forward function with LDB injection.

        Args:
            query (Tensor): Input query, shape (bs, num_queries, dim).
            key (Tensor, optional): Key tensor.
            value (Tensor, optional): Value tensor.
            query_pos (Tensor, optional): Query positional encoding.
            key_pos (Tensor, optional): Key positional encoding.
            self_attn_mask (Tensor, optional): Self-attention mask.
            cross_attn_mask (Tensor, optional): Cross-attention mask.
            key_padding_mask (Tensor, optional): Key padding mask.

        Returns:
            Tensor: Output with LDB applied, shape (bs, num_queries, dim).
        """
        # Self-attention
        identity = query
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs,
        )
        query = self.norms[0](query)
        # Apply LDB after self-attention
        self_alpha = self.dbias_alpha_self(identity)
        query = query + self_alpha * self.dbias_vector_self

        # Cross-attention
        identity = query
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )
        query = self.norms[1](query)
        # Apply LDB after cross-attention
        cross_alpha = self.dbias_alpha_cross(identity)
        query = query + cross_alpha * self.dbias_vector_cross

        # FFN
        identity = query
        query = self.ffn(query)
        query = self.norms[2](query)
        # Apply LDB after FFN
        ffn_alpha = self.dbias_alpha_ffn(identity)
        query = query + ffn_alpha * self.dbias_vector_ffn

        return query


@MODELS.register_module()
class LDBDeformableDetrTransformerEncoder(DeformableDetrTransformerEncoder):
    """LDB version of Deformable DETR Transformer Encoder.

    Uses LDBDeformableDetrTransformerEncoderLayer instead of the base layer.
    """

    def _init_layers(self) -> None:
        """Initialize encoder layers with LDB version."""
        self.layers = ModuleList([
            LDBDeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])

        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, '
                    'please install fairscale by executing the '
                    'following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])

        self.embed_dims = self.layers[0].embed_dims


@MODELS.register_module()
class LDBDeformableDetrTransformerDecoder(DeformableDetrTransformerDecoder):
    """LDB version of Deformable DETR Transformer Decoder.

    Uses LDBDeformableDetrTransformerDecoderLayer instead of the base layer.
    """

    def _init_layers(self) -> None:
        """Initialize decoder layers with LDB version."""
        self.layers = ModuleList([
            LDBDeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
