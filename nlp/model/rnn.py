from typing import Optional

import torch
from torch import Tensor

from .base import RNNCellBase


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
