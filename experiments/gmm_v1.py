import hydra
from omegaconf import DictConfig
import jax

from fabjax.train.generic_training_loop import train

from experiments.setup_training import setup_fab_config, setup_general_train_config
from fabjax.targets.gmm_v1 import GaussianMixture2D


@hydra.main(config_path="./config", config_name="gmm_v1.yaml")
def run(cfg: DictConfig):
    local = False
    if local:
        if "logger" in cfg.keys():
            del cfg.logger

    if cfg.training.use_64_bit:
        jax.config.update("jax_enable_x64", True)

    target = GaussianMixture2D()

    fab_config = setup_fab_config(cfg, target)
    experiment_config = setup_general_train_config(fab_config)
    train(experiment_config)


if __name__ == '__main__':
    run()
