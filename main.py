import hydra
from omegaconf import DictConfig

from nlp.core.train import Trainer


@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig) -> None:
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    train()
