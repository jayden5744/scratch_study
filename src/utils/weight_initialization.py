import torch.nn as nn


def xavier_uniform_initialize(submodule):
    nn.init.xavier_uniform_(submodule.weight)


def xavier_normal_initialize(submodule):
    nn.init.xavier_normal_(submodule.weight)


def he_uniform_initialize(submodule):
    nn.init.kaiming_uniform_(submodule.weight, nonlinearity="relu")


def he_normal_initialize(submodule):
    nn.init.kaiming_normal_(submodule.weight, nonlinearity="relu")


def select_weight_initialize_method(
    method: str, distribution: str, model: nn.Module
) -> None:
    """
    Initialize weight method
        - weight initialization of choice [he, xavier]
        - weight distribution of choice [uniform, normal]
    Args:
        method(str): weight initialization method
        distribution: weight distribution
        model: Transformer Model
    """
    if method == "xavier" and distribution == "uniform":
        if hasattr(model, "weight") and model.weight.dim() > 1:
            model.apply(xavier_uniform_initialize)

    elif method == "xavier" and distribution == "normal":
        if hasattr(model, "weight") and model.weight.dim() > 1:
            model.apply(xavier_normal_initialize)

    elif method == "he" and distribution == "uniform":
        if hasattr(model, "weight") and model.weight.dim() > 1:
            model.apply(he_uniform_initialize)

    elif method == "he" and distribution == "normal":
        if hasattr(model, "weight") and model.weight.dim() > 1:
            model.apply(he_normal_initialize)

    else:
        raise ValueError(
            "weight initialization of choice [he, xavier] and "
            "Weight distribution of choice [uniform, normal]"
        )
