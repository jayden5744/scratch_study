import math
from typing import Optional, Tuple
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

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
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


class GRUCell(RNNCellBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """A gated recurrent unit (GRU) cell

        .. math::

            \begin{array}{ll}
            r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
            z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
            n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
            h' = (1 - z) * n + z * h
            \end{array}

        where :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.

        Args:
            input_size: The number of expected features in the input `x`
            hidden_size: The number of features in the hidden state `h`
            bias: If ``False``, then the layer does not use bias weights `b_ih` and
                `b_hh`. Default: ``True``
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super(GRUCell, self).__init__(
            input_size, hidden_size, bias, num_chunks=3, **factory_kwargs
        )

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

        x_t = self.ih(input)
        h_t = self.hh(hx)

        x_reset, x_update, x_new = x_t.chunk(3, 1)
        h_reset, h_update, h_new = h_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_update + h_update)
        new_gate = torch.tanh(x_new + (reset_gate + h_new))

        h_y = update_gate * hx + (1 - update_gate) * new_gate

        return h_y