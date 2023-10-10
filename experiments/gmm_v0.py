import hydra
from omegaconf import DictConfig

from fabjax.train.generic_training_loop import train

from experiments.setup_training import setup_fab_config, setup_general_train_config
from fabjax.targets.gmm_v0 import GMM

@hydra.main(config_path="./config", config_name="gmm_v0.yaml")
def run(cfg: DictConfig):
    target = GMM()

    fab_config = setup_fab_config(cfg, target)
    experiment_config = setup_general_train_config(fab_config)
    train(experiment_config)


if __name__ == '__main__':
    run()
