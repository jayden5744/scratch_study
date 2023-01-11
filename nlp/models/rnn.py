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
        if self.bidirectional:
            self.backward_rnn = self.init_layers()

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
        sequence_dim = 1 if self.batch_first else 0
        # print(f"batch_dim: {batch_dim}")
        # print(f"sequence_dim: {sequence_dim}")
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

        batch_size = input.size(batch_dim)
        sequence_size = input.size(sequence_dim)
        print(f"batch_size: {batch_size}")
        print(f"sequence_size: {sequence_size}")

        if hx is None:
            hx = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )

        # hx -> [num_layers * D, batch_size, H_out]

        hidden_state = []
        if self.bidirectional:
            next_hidden_f, next_hidden_b = [], []
            for layer_idx, (forward_cell, backward_cell) in enumerate(
                zip(self.forward_rnn, self.backward_rnn)
            ):
                if layer_idx == 0:
                    input_f_state = input
                    input_b_state = input

                else:
                    input_f_state = torch.stack(next_hidden_f, dim=sequence_dim)
                    input_b_state = torch.stack(next_hidden_b, dim=sequence_dim)
                    next_hidden_f, next_hidden_b = [], []

                h_f_i = hx[2 * layer_idx, :, :]
                h_b_i = hx[2 * layer_idx + 1, :, :]

                for i in range(sequence_size):
                    input_f_i = (
                        input_f_state[:, i, :]
                        if self.batch_first
                        else input_f_state[i, :, :]
                    )
                    input_b_i = (
                        input_b_state[:, -(i + 1), :]
                        if self.batch_first
                        else input_b_state[-(i + 1), :, :]
                    )

                    h_f_i = forward_cell(input_f_i, h_f_i)
                    h_b_i = backward_cell(input_b_i, h_b_i)
                    if self.dropout:
                        h_f_i = self.dropout(h_f_i)
                        h_b_i = self.dropout(h_b_i)

                    next_hidden_f.append(h_f_i)
                    next_hidden_b.append(h_b_i)
                hidden_state.append(torch.stack(next_hidden_f, dim=sequence_dim))
                hidden_state.append(torch.stack(next_hidden_b[::-1], dim=sequence_dim))
            hidden_states = torch.stack(
                hidden_state, dim=0
            )  # -> [num_layers * D, N, L, H_out]

            output_f_state = hidden_states[-2, :, :, :]
            output_b_state = hidden_states[-1, :, :, :]
            output = torch.cat([output_f_state, output_b_state], dim=2)

        else:
            next_hidden = []
            for layer_idx, rnn_cell in enumerate(self.forward_rnn):
                # print(rnn_cell)
                if layer_idx == 0:
                    input_state = input  # -> [L, N, H_in] or [N, L, H_in]

                else:
                    input_state = torch.stack(
                        next_hidden, dim=sequence_dim
                    )  # -> [L, N, H_out]
                    next_hidden = []
                # print(f"{layer_idx} layer input_state : {input_state.size()}")

                h_i = hx[
                    layer_idx, :, :
                ]  # -> [N, H_out] : layer_idx 번째의 previous hidden state
                # print(f"{layer_idx}th h_i : {h_i.size()}")

                for i in range(sequence_size):
                    input_i = (
                        input_state[:, i, :]
                        if self.batch_first
                        else input_state[i, :, :]
                    )  # -> [N, H_in]
                    # print(f"input_{i} : {input_i.size()}")
                    h_i = rnn_cell(input_i, h_i)  # -> [N, H_out]
                    # print(f"h_{i} : {h_i.size()}")
                    if self.dropout:
                        h_i = self.dropout(h_i)
                    next_hidden.append(h_i)  # -> 각 층의 hidden_state 들
                hidden_state.append(
                    torch.stack(next_hidden, dim=sequence_dim)  # -> [L, N, H_out]
                )
                print(
                    f"hidden_state_{layer_idx} : {hidden_state[layer_idx].size()}"
                )  # -> [L, N, H_out] or [N, L, H_out]
            # print(len(hidden_state))
            # print(hidden_state[0].size())
            hidden_states = torch.stack(
                hidden_state, dim=0
            )  # -> [num_layers, L, N, H_out]
            print(f"hidden_states : {hidden_states.size()}")

            output = hidden_states[-1, :, :, :]  # -> [L, N, H_out]

        hn = (
            hidden_states[:, :, -1, :]
            if self.batch_first
            else hidden_states[:, -1, :, :]
        )  # -> [num_layers * D, N, H_out]
        return output, hn

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
