import hydra
from omegaconf import DictConfig
import jax

from fabjax.train.generic_training_loop import train

from experiments.setup_training import setup_fab_config, setup_general_train_config
from fabjax.targets.many_well import ManyWellEnergy

@hydra.main(config_path="./config", config_name="many_well.yaml")
def run(cfg: DictConfig):
    if cfg.training.use_64_bit:
        jax.config.update("jax_enable_x64", True)

    target = ManyWellEnergy(dim=cfg.target.dim)

    fab_config = setup_fab_config(cfg, target)
    experiment_config = setup_general_train_config(fab_config)
    train(experiment_config)


if __name__ == '__main__':
    run()
