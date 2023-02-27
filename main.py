import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig) -> None:
    print(cfg)


if __name__ == "__main__":
    train()