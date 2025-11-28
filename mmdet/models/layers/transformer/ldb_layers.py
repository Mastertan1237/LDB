import torch
import torch.nn as nn
from torch import Tensor
from mmdet.models.layers.transformer.deformable_detr_layers import (
    DeformableDetrTransformerEncoderLayer,
    DeformableDetrTransformerDecoderLayer,
)


class LDBDeformableDetrTransformerEncoderLayer(DeformableDetrTransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dbias_vector_attn = nn.Parameter(torch.zeros(self.embed_dims))
        self.dbias_alpha_attn = nn.Linear(self.embed_dims, 1)
        self.dbias_vector_ffn = nn.Parameter(torch.zeros(self.embed_dims))
        self.dbias_alpha_ffn = nn.Linear(self.embed_dims, 1)

    def forward(self, query: Tensor, query_pos: Tensor, key_padding_mask: Tensor, **kwargs) -> Tensor:
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
        attn_alpha = self.dbias_alpha_attn(identity)
        query = query + attn_alpha * self.dbias_vector_attn
        identity = query
        query = self.ffn(query)
        query = self.norms[1](query)
        ffn_alpha = self.dbias_alpha_ffn(identity)
        query = query + ffn_alpha * self.dbias_vector_ffn
        return query


class LDBDeformableDetrTransformerDecoderLayer(DeformableDetrTransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dbias_vector_self = nn.Parameter(torch.zeros(self.embed_dims))
        self.dbias_alpha_self = nn.Linear(self.embed_dims, 1)
        self.dbias_vector_cross = nn.Parameter(torch.zeros(self.embed_dims))
        self.dbias_alpha_cross = nn.Linear(self.embed_dims, 1)
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
        self_alpha = self.dbias_alpha_self(identity)
        query = query + self_alpha * self.dbias_vector_self
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
        cross_alpha = self.dbias_alpha_cross(identity)
        query = query + cross_alpha * self.dbias_vector_cross
        identity = query
        query = self.ffn(query)
        query = self.norms[2](query)
        ffn_alpha = self.dbias_alpha_ffn(identity)
        query = query + ffn_alpha * self.dbias_vector_ffn
        return query

