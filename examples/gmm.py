from typing import NamedTuple

import optax
from molboil.train.train import TrainConfig, Logger, ListLogger
from molboil.train.train import train

from fabjax.train.fab import build_fab_no_buffer_init_step_fns, LogProbFn, TrainStateNoBuffer
from fabjax.flow import build_flow, Flow, FlowDistConfig
from fabjax.sampling import build_ais, build_blackjax_hmc, AnnealedImportanceSampler
from fabjax.targets.gmm import GMM

class FABTrainConfig(NamedTuple):
    dim: int
    n_iteration: int
    batch_size: int
    flow: Flow
    ais: AnnealedImportanceSampler
    log_p_fn: LogProbFn
    optimizer: optax.GradientTransformation
    n_checkpoints: int = 0
    logger: Logger = ListLogger()
    seed: int = 0
    n_eval: int = 0
    save: bool = False


def plot(state: TrainStateNoBuffer, flow: Flow, ais: AnnealedImportanceSampler)


def setup_fab_config():
    # Setup params
    # Train
    alpha = 2.  # alpha-divergence param
    dim = 2
    n_iterations = 1000
    batch_size = 64
    lr = 3e-4

    # Flow
    n_layers = 4
    conditioner_mlp_units = (32, 32)

    # AIS.
    hmc_n_outer_steps = 1
    init_step_size = 1.
    target_p_accept = 0.65
    tune_step_size = True
    n_intermediate_distributions = 2
    spacing_type = 'linear'


    # Setup flow and target.
    flow_config = FlowDistConfig(dim=dim, n_layers=n_layers, conditioner_mlp_units=conditioner_mlp_units)
    flow = build_flow(flow_config)
    gmm = GMM(dim, n_mixes=20, loc_scaling=20)

    # Setup AIS.
    transition_operator = build_blackjax_hmc(dim=2, n_outer_steps=hmc_n_outer_steps,
                                                 init_step_size=init_step_size, target_p_accept=target_p_accept,
                                                 adapt_step_size=tune_step_size)
    ais = build_ais(transition_operator=transition_operator,
                    n_intermediate_distributions=n_intermediate_distributions, spacing_type=spacing_type,
                    alpha=alpha)

    # Optimizer
    optimizer = optax.adam(lr)


    config = FABTrainConfig(dim=dim, n_iteration=n_iterations, batch_size=batch_size, flow=flow,
                            log_p_fn=gmm.log_prob, ais=ais, optimizer=optimizer)
    return config


def setup_molboil_train_config(fab_config: FABTrainConfig) -> TrainConfig:
    """Convert fab_config into what we need for running the molboil training loop."""

    init, step = build_fab_no_buffer_init_step_fns(
        fab_config.flow, log_p_fn=fab_config.log_p_fn,
        ais = fab_config.ais, optimizer = fab_config.optimizer,
        batch_size = fab_config.batch_size)

    def eval_and_plot_fn(state, subkey, iteration, save, plots_dir) -> None:
        pass  # dummy for now

    train_config = TrainConfig(n_iteration=fab_config.n_iteration,
                n_checkpoints=fab_config.n_checkpoints,
                logger = fab_config.logger,
                seed = fab_config.seed,
                n_eval = fab_config.n_eval,
                init_state = init,
                update_state = step,
                eval_and_plot_fn = eval_and_plot_fn,
                save = fab_config.save
                )
    return train_config


if __name__ == '__main__':
    fab_config = setup_fab_config()
    train_config = setup_molboil_train_config(fab_config)
    train(train_config)
