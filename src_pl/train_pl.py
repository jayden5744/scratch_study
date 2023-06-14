from typing import Any, Dict, Tuple

import lightning.pytorch as pl
import sentencepiece as spm
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from src.datasets.data_helper import create_or_load_tokenizer
from src.models import Seq2Seq, Seq2SeqWithAttention, Transformer


class TranslationModel(pl.LightningModule):
    def __init__(self, arg: DictConfig) -> None:
        super().__init__()
        self.arg = arg
        self.src_vocab, self.trg_vocab = self.get_vocab()
        self.model = self.get_model()
        self.loss_function = nn.CrossEntropyLoss(
            ignore_index=self.arg.data.pad_id,
            label_smoothing=self.arg.trainer.label_smoothing_value,
        )
        self.validation_step_outputs = []

    def forward(self, enc_inputs: Tensor) -> Any:
        # Todo: 확인 필요
        dec_input = torch.Tensor([self.src_vocab["<s>"]])
        return self.model(enc_inputs, dec_input)

    def _shared_eval_step(self, batch, batch_idx: int) -> Tensor:
        # validation step과 test step의 공통으로 사용되는 부분
        src_input, trg_input, trg_output = batch
        output = self.model(src_input, trg_input)
        loss = self.calculate_loss(output, trg_output)
        return loss

    def training_step(self, batch, batch_idx: int) -> Dict[str, Tensor]:
        src_input, trg_input, trg_output = batch
        output = self.model(src_input, trg_input)
        loss = self.calculate_loss(output, trg_output)
        metrics = {"loss": loss}
        self.log_dict(metrics)
        return metrics

    def validation_step(self, batch, batch_idx: int) -> Dict[str, Tensor]:
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self) -> Any:
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

    def on_validation_epoch_end(self):
        # validation 1 epoch 끝나고 나서 수행하게 될 로직
        pass

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
        if self.device.type == "mps":
            # mps float64를 처리할 수 없음
            # TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.
            predict = predict.to(device="cpu")
            target = target.to(device="cpu")

        return self.loss_function(predict, target)

    def get_params(self) -> Dict[str, Any]:
        model_type = self.arg.model.model_type
        if model_type == "seq2seq":
            params = {
                "enc_d_input": self.arg.data.src_vocab_size,
                "dec_d_input": self.arg.data.trg_vocab_size,
                "d_hidden": self.arg.model.d_hidden,
                "n_layers": self.arg.model.n_layers,
                "mode": self.arg.model.mode,
                "dropout_rate": self.arg.model.dropout_rate,
                "bidirectional": self.arg.model.bidirectional,
                "bias": self.arg.model.bias,
                "batch_first": self.arg.model.batch_first,
                "max_sequence_size": self.arg.model.max_sequence_size,
            }

        elif model_type == "attention":
            params = {
                "enc_d_input": self.arg.data.src_vocab_size,
                "dec_d_input": self.arg.data.trg_vocab_size,
                "d_hidden": self.arg.model.d_hidden,
                "n_layers": self.arg.model.n_layers,
                "mode": self.arg.model.mode,
                "dropout_rate": self.arg.model.dropout_rate,
                "bidirectional": self.arg.model.bidirectional,
                "bias": self.arg.model.bias,
                "batch_first": self.arg.model.batch_first,
                "max_sequence_size": self.arg.model.max_sequence_size,
            }

        elif model_type == "transformer":
            params = {
                "max_sequence_size": self.arg.model.max_sequence_size,
                "d_hidden": self.arg.model.d_hidden,
                "dropout_rate": self.arg.model.dropout_rate,
                "enc_d_input": self.arg.data.src_vocab_size,
                "enc_layers": self.arg.model.enc_layers,
                "enc_heads": self.arg.model.enc_heads,
                "enc_head_dim": self.arg.model.enc_head_dim,
                "enc_ff_dim": self.arg.model.enc_ff_dim,
                "dec_d_input": self.arg.data.trg_vocab_size,
                "dec_layers": self.arg.model.dec_layers,
                "dec_heads": self.arg.model.dec_heads,
                "dec_head_dim": self.arg.model.dec_head_dim,
                "dec_ff_dim": self.arg.model.dec_ff_dim,
                "padding_id": self.arg.data.pad_id,
            }

        else:
            raise ValueError(
                "param `model_type` must be one of [seq2seq, attention, transformer]"
            )
        return params

    def get_model(self) -> nn.Module:
        model_type = self.arg.model.model_type
        params = self.get_params()
        if model_type == "seq2seq":
            model = Seq2Seq(**params)

        elif model_type == "attention":
            model = Seq2SeqWithAttention(**params)

        elif model_type == "transformer":
            model = Transformer(**params)

        else:
            raise ValueError(
                "param `model_type` must be one of [seq2seq, attention, transformer]"
            )
        return model

    def get_vocab(
        self,
    ) -> Tuple[spm.SentencePieceProcessor, spm.SentencePieceProcessor]:
        src_vocab = create_or_load_tokenizer(
            file_path=self.arg.data.src_train_path,
            save_path=self.arg.data.dictionary_path,
            language=self.arg.data.src_language,
            vocab_size=self.arg.data.src_vocab_size,
            tokenizer_type=self.arg.data.tokenizer,
            bos_id=self.arg.data.bos_id,
            eos_id=self.arg.data.eos_id,
            unk_id=self.arg.data.unk_id,
            pad_id=self.arg.data.pad_id,
        )

        trg_vocab = create_or_load_tokenizer(
            file_path=self.arg.data.trg_train_path,
            save_path=self.arg.data.dictionary_path,
            language=self.arg.data.trg_language,
            vocab_size=self.arg.data.trg_vocab_size,
            tokenizer_type=self.arg.data.tokenizer,
            bos_id=self.arg.data.bos_id,
            eos_id=self.arg.data.eos_id,
            unk_id=self.arg.data.unk_id,
            pad_id=self.arg.data.pad_id,
        )
        return src_vocab, trg_vocab
