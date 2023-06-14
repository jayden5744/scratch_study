import random
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .seq2seq import Encoder


class DotProductAttention(nn.Module):
    def __init__(self, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """dot product attention

        Args:
            query (Tensor): t 시점의 decoder cell에서의 hidden state
            key (Tensor): 모든 시점의 encoder cell에서의 hidden state
            value (Tensor): 모든 시점의 encoder cell에서의 hidden state

        Returns:
            _type_: _description_
        """
        scores = torch.matmul(query.unsqueeze(1), key.transpose(-1, -2))
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        context = torch.matmul(attn_prob, value).squeeze()
        return context, attn_prob


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        output_size: int,
        hidden_size: int,
        n_layers: int,
        dropout: float,
        mode: str = "lstm",
        batch_first: bool = True,
        bias: bool = True,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.layers = self.select_mode(
            mode=mode,
            hidden_size=hidden_size,
            n_layers=n_layers,
            bidirectional=bidirectional,
            bias=bias,
            dropout=dropout,
            batch_first=batch_first,
        )

        self.attention = DotProductAttention(dropout_rate=dropout)
        self.linear = nn.Linear(
            hidden_size * 2, output_size
        )  # [hidden_size, output_size]
        self.softmax = nn.LogSoftmax(dim=1)

        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, hidden_size)

        if bidirectional:
            hidden_size *= 2

        self.attention = DotProductAttention(dropout_rate=dropout)
        self.linear = nn.Linear(hidden_size, output_size)  # [hidden_size, output_size]
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(
        self,
        dec_input: Tensor,
        hidden: Optional[Union[Tensor, Tuple[Tensor, Tensor]]],
        encoder_outputs: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        embeded = self.embedding(dec_input)
        relu_embeded = F.relu(embeded)  # 논문에는 없는 것
        output, hidden = self.layers(relu_embeded, hidden)
        if self.bidirectional:
            output = self.fc(output)

        context, attn_prob = self.attention(
            query=output, key=encoder_outputs, value=encoder_outputs
        )
        output = torch.cat(
            (output, context), dim=1
        )  # output : [batch_size, hidden_size * 2]
        # linear output 을 tanh를 적용
        output = self.softmax(
            self.linear(output)
        )  # [batch_size, hidden_size * 2] * [hidden_size * 2, vocab_size] = [batch_size, vocab_size]
        return output, hidden, attn_prob

    def select_mode(
        self,
        mode: str,
        hidden_size: int,
        n_layers: int,
        dropout: float,
        batch_first: bool = True,
        bias: bool = True,
        bidirectional: bool = False,
    ):
        if mode == "lstm":
            return nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers=n_layers,
                bidirectional=bidirectional,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first,
            )

        elif mode == "rnn":
            return nn.RNN(
                hidden_size,
                hidden_size,
                num_layers=n_layers,
                bidirectional=bidirectional,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first,
            )

        elif mode == "gru":
            return nn.GRU(
                hidden_size,
                hidden_size,
                num_layers=n_layers,
                bidirectional=bidirectional,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first,
            )

        else:
            raise ValueError("param `mode` must be one of [rnn, lstm, gru]")


class Seq2SeqWithAttention(nn.Module):
    def __init__(
        self,
        enc_d_input: int,  # source language vocab size
        dec_d_input: int,  # target language vocab size
        d_hidden: int,
        n_layers: int,
        max_sequence_size: int,
        mode: str = "lstm",
        dropout_rate: float = 0.0,
        bidirectional: bool = True,
        bias: bool = True,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            input_size=enc_d_input,
            hidden_size=d_hidden,
            n_layers=n_layers,
            dropout=dropout_rate,
            mode=mode,
            bidirectional=bidirectional,
            bias=bias,
            batch_first=batch_first,
        )
        self.decoder = AttentionDecoder(
            output_size=dec_d_input,
            hidden_size=d_hidden,
            n_layers=n_layers,
            dropout=dropout_rate,
            mode=mode,
            bidirectional=bidirectional,
            bias=bias,
            batch_first=batch_first,
        )
        self.max_sequence_size = max_sequence_size
        self.vocab_size = dec_d_input
        self.d_hidden = d_hidden

    def forward(
        self,
        enc_input: Tensor,
        dec_input: Tensor,
        teacher_forcing_rate: Optional[float] = 1.0,
    ) -> Tensor:
        enc_hidden = None
        encoder_output = torch.zeros(
            enc_input.size(0), self.max_sequence_size, self.d_hidden
        )  # -> [batch_size, max_sequence_size, d_hidden]
        for i in range(self.max_sequence_size):
            enc_input_i = enc_input[:, i]
            enc_output, enc_hidden = self.encoder(
                enc_input=enc_input_i, enc_hidden=enc_hidden
            )
            encoder_output[:, i, :] = enc_output

        decoder_output = torch.zeros(
            dec_input.size(0), self.max_sequence_size, self.vocab_size
        )
        dec_hidden = enc_hidden
        for i in range(self.max_sequence_size):
            if i == 0 or random.random() <= teacher_forcing_rate:
                dec_input_i = dec_input[:, i]
            else:
                dec_input_i = dec_output_i.topk(1)[1].squeeze().detach()

            dec_output_i, dec_hidden, attn_prob = self.decoder(
                dec_input_i, dec_hidden, encoder_output
            )
            decoder_output[:, i, :] = dec_output_i
        return decoder_output
