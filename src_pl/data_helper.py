import lightning.pytorch as pl
import sentencepiece as spm
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from omegaconf import DictConfig
from torch.utils.data import DataLoader, RandomSampler

from src.datasets.data_helper import TrainDataset


class TranslationData(pl.LightningDataModule):
    def __init__(
        self,
        arg_data: DictConfig,
        src_vocab: spm.SentencePieceProcessor,
        trg_vocab: spm.SentencePieceProcessor,
        max_seq_size: int,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.arg_data = arg_data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_seq_size = max_seq_size
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        # 데이터를 다운로드, split 하거나 기타 등등
        # only called on 1 GPU/TPU in distributed
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        # make assignments here (train/val/test split)
        # called on every process in DDP
        self.train_dataset = TrainDataset(
            x_path=self.arg_data.src_train_path,
            src_vocab=self.src_vocab,
            y_path=self.arg_data.trg_train_path,
            trg_vocab=self.trg_vocab,
            max_sequence_size=self.max_seq_size,
        )

        self.valid_dataset = TrainDataset(
            x_path=self.arg_data.src_valid_path,
            src_vocab=self.src_vocab,
            y_path=self.arg_data.trg_valid_path,
            trg_vocab=self.trg_vocab,
            max_sequence_size=self.max_seq_size,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_sampler = RandomSampler(self.train_dataset)
        return DataLoader(
            dataset=self.train_dataset,
            sampler=train_sampler,
            batch_size=self.batch_size,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        valid_sampler = RandomSampler(self.valid_dataset)
        return DataLoader(
            dataset=self.valid_dataset,
            sampler=valid_sampler,
            batch_size=self.batch_size,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return super().test_dataloader()

    def teardown(self, stage: str) -> None:
        # clean up after fit or test
        # called on every process in DDP
        # setup 정반대
        return super().teardown(stage)
