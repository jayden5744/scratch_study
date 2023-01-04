from typing import Optional, Tuple

import torch
from torch import Tensor

from .base import RNNCellBase


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
