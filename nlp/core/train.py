import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..utils.utils import count_parameters
from ..utils.weight_initialization import select_weight_initialize_method
from .base import AbstracTools


class Trainer(AbstracTools):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.model = self.get_model()
        self.model.train()
        self.optimizer = self.init_optimizer()

        select_weight_initialize_method(
            method=self.arg.model.weight_init,
            distribution=self.arg.model.weight_distribution,
            model=self.model,
        )

    def train(self):
        print(f"The model {count_parameters(self.model)} trainerble parameters.")

    def init_optimizer(self) -> None:
        optimizer_type = self.arg.trainer.optimizer
        if optimizer_type == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.arg.trainer.learning_rate,
                betas=(self.arg.trainer.optimizer_b1, self.arg.trainer.optimizer_b2),
                eps=self.arg.trainer.optimizer_e,
                weight_decay=self.arg.trainer.weight_decay,
            )

        elif optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.arg.trainer.learning_rate,
                betas=(self.arg.trainer.optimizer_b1, self.arg.trainer.optimizer_b2),
                eps=self.arg.trainer.optimizer_e,
                weight_decay=self.arg.trainer.weight_decay,
            )

        else:
            raise ValueError("trainer param `optimizer` must be one of [Adam, AdamW].")
        return optimizer
