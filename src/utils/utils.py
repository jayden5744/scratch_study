import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    if torch.cuda.is_available():
        device = "cuda"

    elif torch.backends.mps.is_available():
        device = "mps"

    else:
        device = "cpu"
    return torch.device(device)
