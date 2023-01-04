from typing import List, Optional

import torch
from torch import Tensor

from .base import RNNBase, RNNCellBase


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
        """An Elman RNN cell with tanh or ReLU non-linearity.

        .. math::

            h' = \tanh(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})

        If :attr:`nonlinearity` is `'relu'`, then ReLU is used in place of tanh.

        Args:
            input_size: The number of expected features in the input `x`
            hidden_size: The number of features in the hidden state `h`
            bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
                Default: ``True``
            nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super(RNNCell, self).__init__(
            input_size, hidden_size, bias, num_chunks=1, **factory_kwargs
        )

        self.nonlinearity = nonlinearity
        if self.nonlinearity not in ["tanh", "relu"]:
            raise ValueError("Invalid nonlinearity selected for RNN.")

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            input (Tensor): tensor containing input features
                - shape : (batch_size, input_size) or (input_size)
            hx (Optional[Tensor], optional): tensor containing the initial hidden state.
                - Defaults to None.
                - shape : (batch_size, hidden_size) or (hidden_size)

        Returns:
            Tensor: tensor containing the next hidden state for each element in the batch
                - shape : (batch_size, hidden_size)
        """
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


class RNN(RNNBase):
    def __init__(self, *args, **kwargs) -> None:
        self.nonlinearity = kwargs.pop("nonlinearity", "tanh")
        if self.nonlinearity == "tanh":
            mode = "RNN_TANH"

        elif self.nonlinearity == "relu":
            mode = "RNN_RELU"

        else:
            raise ValueError("Unknown nonlinearity '{}'".format(self.nonlinearity))
        super(RNN, self).__init__(mode, *args, **kwargs)
        self.forward_rnn = self.init_layers()

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        """

        N: batch_size
        L: sequence length
        D: 2 if bidirectional=True otherwise 1
        H_in: input_size
        H_out: hidden_size

        Args:
            input (Tensor): The input can also be a packed variable length sequence.
                - (L, H_in) for unbatched input
                - (L, N, H_in) when ``batch_first`` = False
                - (N, L, H_in) when ``batch_first`` = True
            hx (Optional[Tensor], optional):
                - (num_layers, H_out) for unbatched input
                - (num_layers, batch_size, H_out) : for batched input

        Returns:
            Tensor: _description_
        """
        batch_dim = 0 if self.batch_first else 1
        is_batch = input.dim() == 3
        if not is_batch:
            input = input.unsqueeze(batch_dim)  # -> [1, L, H_in] or [L, 1, H_in]
            if hx is not None:  # hidden state를 넣었다면
                if hx.dim() != 2:  # unbatch hidden state 이므로 반드시 dimension이 2가 되어야 한다
                    raise RuntimeError(
                        f"For unbatched 2-D input, hx should also be 2-D but got {hx.dim()}-D tensor"
                    )
                hx = hx.unsqueeze(1)  # -> [num_layers, 1, H_out]

        else:
            if (
                hx is not None and hx.dim() != 3
            ):  # hidden state를 넣었는데, dimension이 3개가 아닐 때
                raise RuntimeError(
                    f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor"
                )

        # input -> [N, L, H_in] or [L, N, H_in]
        # hx -> [num_layers, batch_size, H_out] or None

        batch_size = input.size(0) if self.batch_first else input.size(1)
        sequence_size = input.size(1) if self.batch_first else input.size(0)

        if hx is None:
            hx = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )

        # hx -> [num_layers, batch_size, H_out]

        hidden_state = []
        next_hidden = []
        for layer_idx, rnn_cell in enumerate(self.forward_rnn):
            print(layer_idx)
            input_state = input  # 여기가 if else 바꿔야지 layer 2부터 동작 가능

            h_i = hx[
                layer_idx, :, :
            ]  # -> [1, batch_size, H_out] : layer_idx 번째의 previous hidden state

            for i in range(sequence_size):
                input_i = (
                    input_state[:, i, :] if self.batch_first else input_state[i, :, :]
                )  # -> [N, 1, H_in] or [1, N, H_in]
                h_i = rnn_cell(input_i, h_i)
                # Todo: dropout
                next_hidden.append(h_i)
            hidden_state.append(torch.stack(next_hidden, dim=batch_dim))  # ->

    def init_layers(self) -> List[RNNCellBase]:
        layers = []
        for layer_idx in range(self.num_layers):
            input_size = self.input_size if layer_idx == 0 else self.hidden_size
            layers.append(
                RNNCell(
                    input_size=input_size,
                    hidden_size=self.hidden_size,
                    bias=self.bias,
                    nonlinearity=self.nonlinearity,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        return layers
