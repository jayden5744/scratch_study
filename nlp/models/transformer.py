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


def get_attn_pad_mask(seq_q: Tensor, seq_k: Tensor, padding_id: int) -> Tensor:
    # seq_q, seq_k => [batch_size, seq_len] 입력문장
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(padding_id).unsqueeze(
        1
    )  # => [batch_size, 1, len_k]  True / False
    return pad_attn_mask.expand(
        seq_k.size(0), seq_q.size(1), seq_k.size(1)
    ).contiguous()  # => [batch_size, len_q, len_k]


def get_attn_decoder_mask(dec_input: Tensor) -> Tensor:
    subsequent_mask = (
        torch.ones_like(dec_input)
        .unsqueeze(-1)
        .expand(dec_input.size(0), dec_input.size(1), dec_input.size(1))
    )  # => [batch_size, seq_len, seq_len]
    subsequent_mask = subsequent_mask.triu(
        diagonal=1
    )  # upper triangular part of a matrix(2-D) => [batch_size, seq_len, seq_len]
    return subsequent_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, head_dim: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = head_dim**0.5

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, attion_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """_summary_

        Args:
            query (Tensor): (bs, n_heads, max_seq, d_hidden)
            key (Tensor): (bs, n_heads, max_seq, d_hidden)
            value (Tensor): (bs, n_heads, max_seq, d_hidden)
            attion_mask (Tensor): (bs, n_heads, max_seq, max_seq)

        Returns:
            Tuple[Tensor, Tensor]: _description_
        """
        scores = (
            torch.matmul(query, key.transpose(-1, -2)) / self.scale
        )  # => [bs, n_heads, len_q(=max_seq), len_k(=max_seq)]
        # Todo : score masking되는지 확인
        scores.masked_fill_(attion_mask, -1e9)
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        context = torch.matmul(attn_prob, value).squeeze()
        return context, attn_prob


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_hidden: int, n_heads: int, head_dim: int, dropout: float = 0
    ) -> None:
        super().__init__()
        self.weight_q = nn.Linear(d_hidden, n_heads * head_dim)
        self.weight_k = nn.Linear(d_hidden, n_heads * head_dim)
        self.weight_v = nn.Linear(d_hidden, n_heads * head_dim)

        self.self_attention = ScaledDotProductAttention(
            head_dim=head_dim, dropout_rate=dropout
        )
        self.linear = nn.Linear(n_heads * head_dim, d_hidden)

        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.head_dim = head_dim

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor):
        """MultiHeadAttention

        Args:
            query (Tensor):  input word vector (bs, max_seq_len, d_hidden)
            key (Tensor): input word vector (bs, max_seq_len, d_hidden)
            value (Tensor): input word vector (bs, max_seq_len, d_hidden)
            attn_mask (Tensor): attn_mask (bs, max_seq_len, max_seq_len)

        Returns:
            _type_: _description_
        """
        batch_size = query.size(0)

        q_s = (
            self.weight_q(query)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )  # => [batch_size, n_heads, len_q, head_dim]

        k_s = (
            self.weight_k(key)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )  # => [batch_size, n_heads, len_q, head_dim]

        v_s = (
            self.weight_v(value)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )  # => [batch_size, n_heads, len_q, head_dim]

        attn_mask = attn_mask.unsqueeze(1).repeat(
            1, self.n_heads, 1, 1
        )  # => [batch_size, n_heads, len_q, len_k]

        context, _ = self.self_attention(
            q_s, k_s, v_s, attn_mask
        )  # => [bs, n_heads, max_seq_size, d_hidden]

        context = (
            context.transpose(1, 2)  # => [bs, max_seq_size, n_heads, d_hidden]
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.head_dim)
        )  # => [batch_size, len_q, n_heads * head_dim]

        output = self.linear(context)  # => [batch_size, len_q, d_hidden]
        output = self.dropout(output)
        return output


class PoswiseFeedForwardNet(nn.Module):
    def __init__(
        self,
        d_hidden: int,
        ff_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer1 = nn.Linear(d_hidden, ff_dim)
        self.layer2 = nn.Linear(ff_dim, d_hidden)

        self.active = F.relu  # gelu
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.dropout(self.active(self.layer1(inputs)))
        output = self.dropout(self.layer2(output))
        return output


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_hidden: int,
        n_heads: int,
        head_dim: int,
        ff_dim: int,
        dropout: float = 0.0,
        layer_norm_epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        self.mh_attention = MultiHeadAttention(
            d_hidden=d_hidden, n_heads=n_heads, head_dim=head_dim, dropout=dropout
        )
        self.layer_norm_1 = nn.LayerNorm(d_hidden, eps=layer_norm_epsilon)
        self.ffnn = PoswiseFeedForwardNet(
            d_hidden=d_hidden, ff_dim=ff_dim, dropout=dropout
        )
        self.layer_norm_2 = nn.LayerNorm(d_hidden, eps=layer_norm_epsilon)

    def forward(self, enc_inputs: Tensor, enc_self_attn_mask: Tensor) -> Tensor:
        mh_output = self.mh_attention(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask
        )
        mh_output = self.layer_norm_1(enc_inputs + mh_output)
        ffnn_output = self.ffnn(mh_output)
        ffnn_output = self.layer_norm_2(ffnn_output)
        return ffnn_output


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_hidden: int,
        n_heads: int,
        head_dim: int,
        ff_dim: int,
        dropout: float = 0.0,
        layer_norm_epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        self.mh_attention = MultiHeadAttention(
            d_hidden=d_hidden, n_heads=n_heads, head_dim=head_dim, dropout=dropout
        )
        self.layer_norm_1 = nn.LayerNorm(d_hidden, eps=layer_norm_epsilon)
        self.ffnn = PoswiseFeedForwardNet(
            d_hidden=d_hidden, ff_dim=ff_dim, dropout=dropout
        )
        self.layer_norm_2 = nn.LayerNorm(d_hidden, eps=layer_norm_epsilon)

    # def forward(self, enc_inputs: Tensor, enc_self_attn_mask: Tensor) -> Tensor:
    # mh_output = self.mh_attention(
    #     enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask
    # )
    # mh_output = self.layer_norm_1(enc_inputs + mh_output)
    # ffnn_output = self.ffnn(mh_output)
    # ffnn_output = self.layer_norm_2(ffnn_output)
    # return ffnn_output


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
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_hidden=d_hidden,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.padding_id = padding_id

    def forward(self, enc_inputs: Tensor):
        """Encoder

        Args:
            enc_inputs (Tensor): (bs, max_seq_size)
        """
        position = self.get_position(enc_inputs=enc_inputs)
        conb_emb = self.src_emb(enc_inputs) + self.pos_emb(
            position
        )  # Embedding + pos_enbeding : [batch_size, max_seq_size, d_hidden]
        enc_self_attn_mask = get_attn_pad_mask(
            enc_inputs, enc_inputs, self.padding_id
        )  # =>[batch_size, max_seq_size, max_seq_size]

        enc_outputs = conb_emb
        for layer in self.layers:
            enc_outputs = layer(enc_outputs, enc_self_attn_mask)
            # enc_outputs => [batch_size, len_q, d_model]
        return enc_outputs

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


class Decoder(nn.Module):
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
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_hidden=d_hidden,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.padding_id = padding_id

    def forward(self, dec_inputs: Tensor, enc_output: Tensor):
        position = self.get_position(dec_inputs=dec_inputs)
        conb_emb = self.src_emb(dec_inputs) + self.pos_emb(
            position
        )  # Embedding + pos_enbeding : [batch_size, max_seq_size, d_hidden]
        dec_self_attn_mask = get_attn_pad_mask(
            dec_inputs, dec_inputs, self.padding_id
        )  # =>[batch_size, max_seq_size, max_seq_size]

        dec_mask = get_attn_decoder_mask(dec_inputs)

        # dec_self_attn_mask + dec_mask : torch.gt => 1번 레이어

        # dec_inputs, enc_output -> get_attn_pad_mask => mask 2번 레이어

    def get_position(self, dec_inputs: Tensor) -> Tensor:
        position = (
            torch.arange(
                dec_inputs.size(1), device=dec_inputs.device, dtype=dec_inputs.dtype
            )
            .expand(dec_inputs.size(0), dec_inputs.size(1))
            .contiguous()
            + 1
        )  # -> [bs, max_seq_size]
        pos_mask = dec_inputs.eq(
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
        self.decoder = Decoder(
            input_dim=dec_d_input,
            d_hidden=d_hidden,
            n_layers=dec_layers,
            n_heads=dec_heads,
            head_dim=dec_head_dim,
            ff_dim=dec_ff_dim,
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
        dec_output = self.decoder(dec_input, enc_outputs)
        return dec_output
