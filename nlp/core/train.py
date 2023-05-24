import os

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

import wandb

from ..utils.utils import count_parameters
from ..utils.weight_initialization import select_weight_initialize_method
from .base import AbstractTools


class Trainer(AbstractTools):
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

        self.train_loader, self.valid_loader = self.get_loader()
        self.loss_function = nn.CrossEntropyLoss(
            ignore_index=self.arg.data.pad_id,
            label_smoothing=self.arg.trainer.label_smoothing_value,
        )

        # wandb.init(config=self.arg)

    def train(self):
        print(f"The model {count_parameters(self.model)} trainerble parameters.")
        # wandb.watch(self.model)
        epoch_step = len(self.train_loader) + 1  # 한 epoch의 스텝 수
        total_step = self.arg.trainer.epochs * epoch_step  # 전체 학습 step 수
        step = 0
        for epoch in range(self.arg.trainer.epochs):
            for idx, data in enumerate(self.train_loader, 1):
                try:
                    self.optimizer.zero_grad()
                    src_input, trg_input, trg_output = data
                    output = self.model(src_input, trg_input)

                    loss = self.calculate_loss(output, trg_output)

                    if step % self.arg.trainer.print_train_step == 0:
                        # wandb.log({"Train Loss": loss.item()})
                        print(
                            "[Train] epoch: {0:2d}  iter: {1:4d}/{2:4d}  step: {3:6d}/{4:6d}  => loss: {5:10f}".format(
                                epoch, idx, epoch_step, step, total_step, loss.item()
                            )
                        )

                    if step % self.arg.trainer.print_valid_step == 0:
                        val_loss = self.valid()
                        # wandb.log({"Valid Loss": val_loss})
                        print(
                            "[Valid] epoch: {0:2d}  iter: {1:4d}/{2:4d}  step: {3:6d}/{4:6d}  => loss: {5:10f}".format(
                                epoch, idx, epoch_step, step, total_step, val_loss
                            )
                        )

                    if step % self.arg.trainer.save_step == 0:
                        self.save_model(epoch, step)

                    loss.backward()
                    self.optimizer.step()
                    step += 1

                except Exception as e:
                    self.save_model(epoch, step)
                    raise e

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

    def calculate_loss(self, predict: Tensor, target: Tensor) -> Tensor:
        """_summary_

        Args:
            predict (Tensor): [batch_size, max_seq_size, vocab_size]
            target (Tensor): [batch_size, max_seq_size]

        Returns:
            Tensor: _description_
        """
        # predict -> (batch_size, n_classes, ~)
        # target -> (batch_size, ~)
        predict = predict.transpose(1, 2)  # [batch_size, vocab_size, max_seq_size]
        return self.loss_function(predict, target)

    def save_model(self, epoch: int, step: int) -> None:
        model_name = f"{str(step).zfill(6)}_{self.arg.model.model_type}.pth"  # 000000_{seq2seq}.pth
        model_path = os.path.join(self.arg.data.model_path, model_name)
        os.makedirs(self.arg.data.model_path, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "step": step,
                "data": self.arg.data,
                "model": self.arg.model,
                "trainer": self.arg.trainer,
                "model_state_dict": self.model.state_dict(),
            },
            model_path,
        )

    def valid(self) -> float:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in self.valid_loader:
                src_input, trg_input, trg_output = data
                output = self.model(src_input, trg_input)
                loss = self.calculate_loss(output, trg_output)
                total_loss += loss.item()

        # validation sample 확인
        input_sentence = self.tensor2sentence(src_input[0].tolist(), self.src_vocab)
        predict_sentence = self.tensor2sentence(
            output.topk(1)[1].squeeze()[0, :].tolist(), self.trg_vocab
        )
        target_sentence = self.tensor2sentence(trg_input[0].tolist(), self.trg_vocab)
        self.print_result(input_sentence, predict_sentence, target_sentence)
        self.model.train()
        return total_loss / len(self.valid_loader)
