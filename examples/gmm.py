from typing import NamedTuple

import chex
import optax
import jax
import matplotlib.pyplot as plt
from molboil.train.train import TrainConfig, Logger, ListLogger
from molboil.train.train import train

from fabjax.train.fab import build_fab_no_buffer_init_step_fns, LogProbFn, TrainStateNoBuffer
from fabjax.flow import build_flow, Flow, FlowDistConfig
from fabjax.sampling import build_ais, build_blackjax_hmc, AnnealedImportanceSampler, simple_resampling
from fabjax.targets.gmm import GMM
from fabjax.utils.plot import plot_marginal_pair, plot_contours_2D

class FABTrainConfig(NamedTuple):
    dim: int
    n_iteration: int
    batch_size: int
    plot_batch_size: int
    n_eval: int
    flow: Flow
    ais: AnnealedImportanceSampler
    log_p_fn: LogProbFn
    optimizer: optax.GradientTransformation
    n_checkpoints: int = 0
    logger: Logger = ListLogger()
    seed: int = 0
    save: bool = False




def setup_plotter(flow_config: FABTrainConfig, plot_bound=30):

    @jax.jit
    @chex.assert_max_traces(3)
    def get_data_for_plotting(state: TrainStateNoBuffer, key: chex.PRNGKey):
        x0 = flow_config.flow.sample_apply(state.flow_params, key, (flow_config.plot_batch_size,))

        def log_q_fn(x: chex.Array) -> chex.Array:
            return flow_config.flow.log_prob_apply(state.flow_params, x)

        point, log_w, ais_state, ais_info = flow_config.ais.step(x0, state.ais_state, log_q_fn, flow_config.log_p_fn)
        x_ais = point.x
        _, x_ais_resampled = simple_resampling(key, log_w, x_ais)

        return x0, x_ais, x_ais_resampled

    def plot(state: TrainStateNoBuffer, key: chex.PRNGKey):
        x0, x_ais, x_ais_resampled = get_data_for_plotting(state, key)

        fig, axs = plt.subplots(3, figsize=(5, 15))
        plot_marginal_pair(x0, axs[0], bounds=(-plot_bound, plot_bound))
        plot_marginal_pair(x_ais, axs[1], bounds=(-plot_bound, plot_bound))
        plot_marginal_pair(x_ais_resampled, axs[2], bounds=(-plot_bound, plot_bound))
        plot_contours_2D(flow_config.log_p_fn, axs[0], bound=plot_bound, levels=50)
        plot_contours_2D(flow_config.log_p_fn, axs[1], bound=plot_bound, levels=50)
        plot_contours_2D(flow_config.log_p_fn, axs[2], bound=plot_bound, levels=50)
        axs[0].set_title("flow samples")
        axs[1].set_title("ais samples")
        axs[2].set_title("resampled ais samples")
        plt.tight_layout()
        plt.show()
    return plot


def setup_fab_config():
    # Setup params
    # Train
    alpha = 2.  # alpha-divergence param
    dim = 2
    n_iterations = int(5e4)
    n_eval = 10
    batch_size = 128
    plot_batch_size = batch_size
    lr = 1e-4
    max_global_norm = 100.

    # Flow
    n_layers = 4
    conditioner_mlp_units = (32, 32)

    # AIS.
    hmc_n_outer_steps = 1
    init_step_size = 1e-3
    target_p_accept = 0.65
    tune_step_size = False
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
    optimizer = optax.chain(optax.zero_nans(), optax.clip_by_global_norm(max_global_norm), optax.adam(lr))


    config = FABTrainConfig(dim=dim, n_iteration=n_iterations, batch_size=batch_size, flow=flow,
                            log_p_fn=gmm.log_prob, ais=ais, optimizer=optimizer, plot_batch_size=plot_batch_size,
                            n_eval=n_eval)
    return config


def setup_molboil_train_config(fab_config: FABTrainConfig) -> TrainConfig:
    """Convert fab_config into what we need for running the molboil training loop."""

    init, step = build_fab_no_buffer_init_step_fns(
        fab_config.flow, log_p_fn=fab_config.log_p_fn,
        ais = fab_config.ais, optimizer = fab_config.optimizer,
        batch_size = fab_config.batch_size)

    plotter = setup_plotter(fab_config)

    def eval_and_plot_fn(state, subkey, iteration, save, plots_dir) -> dict:
        plotter(state, subkey)
        return {}

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
