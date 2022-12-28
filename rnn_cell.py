import math
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn


class RNNCellBase(nn.Module):
    __constants__ = ["input_size", "hidden_size", "bias"]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        num_chunks: int,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(RNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.ih = nn.Linear(
            in_features=input_size,
            out_features=num_chunks * hidden_size,
            bias=bias,
            **factory_kwargs
        )
        self.hh = nn.Linear(
            in_features=hidden_size,
            out_features=num_chunks * hidden_size,
            bias=bias,
            **factory_kwargs
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)



class RNNCell(RNNCellBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(RNNCell, self).__init__(
            input_size, hidden_size, bias, num_chunks=1, **factory_kwargs
        )
        self.nonlinearity = nonlinearity
        if self.nonlinearity not in ["tanh", "relu"]:
            raise ValueError("Invalid nonlinearity selected for RNN.")

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )

        hy = self.ih(input) + self.hh(hx)

        if self.nonlinearity == "tanh":
            ret = torch.tanh(hy)
        else:
            ret = torch.relu(hy)

        return ret
