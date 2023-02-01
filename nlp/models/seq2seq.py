import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int,
        dropout: float,
        mode: str = "lstm",
        batch_first: bool = True,
        bias: bool = True,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.layers = self.select_mode(
            mode=mode,
            hidden_size=hidden_size,
            n_layers=n_layers,
            bidirectional=bidirectional,
            bias=bias,
            dropout=dropout,
            batch_first=batch_first,
        )

    def forward(self, enc_input: Tensor):
        embeded = self.embedding(enc_input)
        output, hidden = self.layers(embeded)
        return output, hidden

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


class Decoder(nn.Module):
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
        self.embedding = nn.Embedding(hidden_size, output_size)
        self.layers = self.select_mode(
            mode=mode,
            hidden_size=hidden_size,
            n_layers=n_layers,
            bidirectional=bidirectional,
            bias=bias,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, enc_input: Tensor, hidden):
        embeded = self.embedding(enc_input)
        relu_embeded = F.relu(embeded)  # 논문에는 없는 것
        output, hidden = self.layers(relu_embeded)
        output = self.softmax(self.linear(output[0]))
        return output, hidden

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
