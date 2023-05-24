from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def get_position_encoding_table(max_sequence_size: int, d_hidden: int) -> Tensor:
    def get_angle(position: int, i: int) -> float:
        return position / np.power(10000, 2 * (i // 2) / d_hidden)

    def get_angle_vector(position: int) -> List[float]:
        return [get_angle(position, hid_j) for hid_j in range(d_hidden)]

    pe_table = Tensor([get_angle_vector(pos_i) for pos_i in range(max_sequence_size)])
    pe_table[:, 0::2] = np.sin(pe_table[:, 0::2])  # dim 2i
    pe_table[:, 1::2] = np.cos(pe_table[:, 1::2])  # dim 2i +1
    return pe_table


class ScaledDotProductAttention(nn.Module):
    def __init__(self, head_dim: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = head_dim**0.5

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, attion_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        scores = torch.matmul(query.unsqueeze(1), key.transpose(-1, -2)) / self.scale
        # Todo: scores 에 mask 적용
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        context = torch.matmul(attn_prob, value).squeeze()
        return context, attn_prob


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_hidden: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        ff_dim: int,
        max_sequence_size: int,
        padding_id: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.src_emb = nn.Embedding(input_dim, d_hidden)
        pe_table = get_position_encoding_table(max_sequence_size, d_hidden)
        self.pos_emb = nn.Embedding.from_pretrained(pe_table, freeze=True)
        self.padding_id = padding_id

    def forward(self, enc_inputs: Tensor):
        """Encoder Layer

        Args:
            enc_inputs (Tensor): (bs, max_seq_size)
        """
        position = self.get_position(enc_inputs=enc_inputs)
        conb_emb = self.src_emb(enc_inputs) + self.pos_emb(
            position
        )  # Embedding + pos_enbeding : [batch_size, max_seq_size, d_hidden]

    def get_position(self, enc_inputs: Tensor) -> Tensor:
        position = (
            torch.arange(
                enc_inputs.size(1), device=enc_inputs.device, dtype=enc_inputs.dtype
            )
            .expand(enc_inputs.size(0), enc_inputs.size(1))
            .contiguous()
            + 1
        )  # -> [bs, max_seq_size]
        pos_mask = enc_inputs.eq(
            self.padding_id
        )  # padding은 True, padding 아닌 것 False 로 벡터를 만들어줌
        position.masked_fill_(pos_mask, 0)
        return position


class Transformer(nn.Module):
    def __init__(
        self,
        enc_d_input: int,  # source language vocab size
        enc_layers: int,
        enc_heads: int,
        enc_head_dim: int,
        enc_ff_dim: int,
        dec_d_input: int,  # target language vocab size
        dec_layers: int,
        dec_heads: int,
        dec_head_dim: int,
        dec_ff_dim: int,
        d_hidden: int,
        max_sequence_size: int,
        dropout_rate: float = 0.0,
        padding_id: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            input_dim=enc_d_input,
            d_hidden=d_hidden,
            n_layers=enc_layers,
            n_heads=enc_heads,
            head_dim=enc_head_dim,
            ff_dim=enc_ff_dim,
            max_sequence_size=max_sequence_size,
            padding_id=padding_id,
            dropout=dropout_rate,
        )

    def forward(self, enc_inputs: Tensor, dec_input: Tensor) -> Tensor:
        """Transformer

        Args:
            enc_inputs (Tensor): source input tensor (batch_size, max_seq_len)
            dec_input (Tensor): target input tensor (batch_size, max_seq_len)

        Returns:
            Tensor: _description_
        """
        enc_outputs = self.encoder(enc_inputs)
