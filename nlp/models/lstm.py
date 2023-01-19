from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .base import RNNBase, RNNCellBase


class LSTMCell(RNNCellBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LSTMCell, self).__init__(
            input_size, hidden_size, bias, num_chunks=4, **factory_kwargs
        )
        """A long short-term memory (LSTM) cell.

        .. math::

            \begin{array}{ll}
            i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
            f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
            g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
            o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
            c' = f * c + i * g \\
            h' = o * \tanh(c') \\
            \end{array}

        where :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.

        Args:
            input_size (int): The number of expected features in the input `x`
            hidden_size (int): The number of features in the hidden state `h`
            bias (bool, optional): If ``False``, then the layer does not use bias weights `b_ih` and
            `b_hh`. 
                - Defaults to True.
            
        """

    def forward(
        self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            input (Tensor): tensor containing input features
                - shape : (batch_size, input_size) or (input_size)
            hx (Optional[Tuple[Tensor, Tensor]], optional): tensor containing the initial hidden state.
                - Defaults to None.
                - shape : (h_0, c_0)
                    - h_0 : (batch_size, hidden_size) or (hidden_size)
                    - c_0 : (batch_size, hidden_size) or (hidden_size)

        Returns:
            Tuple[Tensor, Tensor]: tensor containing the next hidden state for each element in the batch
                - shape : (h_1, c_1)
                    - h_1 : (batch_size, hidden_size) or (hidden_size)
                    - c_1 : (batch_size, hidden_size) or (hidden_size)
        """
        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
            hx = (hx, hx)

        hx, cx = hx

        gates = self.ih(input) + self.hh(hx)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        c_t = cx * f_t + i_t * g_t

        hy = o_t * torch.tanh(c_t)

        return (hy, c_t)


class LSTM(RNNBase):
    def __init__(self, *args, **kwargs) -> None:
        super(LSTM, self).__init__("LSTM", *args, **kwargs)
        self.forward_lstm = self.init_layers()
        if self.bidirectional:
            self.backward_lstm = self.init_layers()

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        batch_dim = 0 if self.batch_first else 1
        sequence_dim = 1 if self.batch_first else 0
        batch_size = input.size(batch_dim)
        sequence_size = input.size(sequence_dim)
        is_batch = input.dim() == 3
        if not is_batch:
            input = input.unsqueeze(batch_dim)  # -> [1, L, H_in] or [L, 1, H_in]

        if hx is None:
            h_zeros = torch.zeros(  # hidden state 초기화
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )

            c_zeros = torch.zeros(  # cell state 초기화
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )
            hx = (h_zeros, c_zeros)

        elif is_batch:
            if hx[0].dim() != 3 or hx[1].dim() != 3:
                msg = (
                    "For batched 3-D input, hx and cx should "
                    f"also be 3-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors"
                )
                raise RuntimeError(msg)

        else:
            if hx[0].dim() != 2 or hx[1].dim() != 2:
                msg = (
                    "For unbatched 2-D input, hx and cx should "
                    f"also be 2-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors"
                )
                raise RuntimeError(msg)
            hx = (hx[0].unsqueeze(1), hx[1].unsqueeze(1))

        hidden_state = []
        cell_state = []
        if self.bidirectional:
            next_hidden_f, next_hidden_b = [], []
            next_cell_f, next_cell_b = [], []
            for layer_idx, (forward_cell, backward_cell) in enumerate(
                zip(self.forward_lstm, self.backward_lstm)
            ):
                if layer_idx == 0:
                    input_f_state = input
                    input_b_state = input

                else:
                    input_f_state = torch.stack(next_hidden_f, dim=sequence_dim)
                    input_b_state = torch.stack(next_hidden_b, dim=sequence_dim)
                    next_hidden_f, next_hidden_b = [], []
                    next_cell_f, next_cell_b = [], []

                h_f_i = hx[0][2 * layer_idx, :, :]
                h_b_i = hx[0][2 * layer_idx + 1, :, :]
                c_f_i = hx[1][2 * layer_idx, :, :]
                c_b_i = hx[1][2 * layer_idx + 1, :, :]

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

                    h_f_i, c_f_i = forward_cell(input_f_i, (h_f_i, c_f_i))
                    h_b_i, c_b_i = backward_cell(input_b_i, (h_b_i, c_b_i))
                    if self.dropout:
                        h_f_i = self.dropout(h_f_i)
                        h_b_i = self.dropout(h_b_i)
                        c_f_i = self.dropout(c_f_i)
                        c_b_i = self.dropout(c_b_i)

                    next_hidden_f.append(h_f_i)
                    next_hidden_b.append(h_b_i)
                    next_cell_f.append(c_f_i)
                    next_cell_b.append(c_b_i)
                hidden_state.append(torch.stack(next_hidden_f, dim=sequence_dim))
                hidden_state.append(torch.stack(next_hidden_b[::-1], dim=sequence_dim))
                cell_state.append(torch.stack(next_cell_f, dim=sequence_dim))
                cell_state.append(torch.stack(next_cell_b[::-1], dim=sequence_dim))

            hidden_states = torch.stack(hidden_state, dim=0)
            cell_states = torch.stack(cell_state, dim=0)

            output_f_state = hidden_states[-2, :, :, :]
            output_b_state = hidden_states[-1, :, :, :]
            output = torch.cat([output_f_state, output_b_state], dim=2)

        else:
            next_hidden, next_cell = [], []
            for layer_idx, lstm_cell in enumerate(self.forward_lstm):
                if layer_idx == 0:
                    input_state = input

                else:
                    input_state = torch.stack(next_hidden, dim=sequence_dim)
                    next_hidden = []
                    next_cell = []

                h_i = hx[0][layer_idx, :, :]
                c_i = hx[1][layer_idx, :, :]

                for i in range(sequence_size):
                    input_i = (
                        input_state[:, i, :]
                        if self.batch_first
                        else input_state[i, :, :]
                    )

                    h_i, c_i = lstm_cell(input_i, (h_i, c_i))
                    if self.dropout:
                        h_i = self.dropout(h_i)
                        c_i = self.dropout(c_i)

                    next_hidden.append(h_i)
                    next_cell.append(c_i)

                hidden_state.append(torch.stack(next_hidden, dim=sequence_dim))
                cell_state.append(torch.stack(next_cell, dim=sequence_dim))

            hidden_states = torch.stack(hidden_state, dim=0)
            cell_states = torch.stack(cell_state, dim=0)

            output = hidden_states[-1, :, :, :]  # -> [L, N, H_out]

        hn = (
            hidden_states[:, :, -1, :]
            if self.batch_first
            else hidden_states[:, -1, :, :]
        )
        cn = cell_states[:, :, -1, :] if self.batch_first else cell_states[:, -1, :, :]
        return output, (hn, cn)

    def init_layers(self) -> List[LSTMCell]:
        layers = []
        for layer_idx in range(self.num_layers):
            input_size = self.input_size if layer_idx == 0 else self.hidden_size
            layers.append(
                LSTMCell(
                    input_size=input_size,
                    hidden_size=self.hidden_size,
                    bias=self.bias,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        return layers
