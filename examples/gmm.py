from typing import NamedTuple, Callable, Optional, Union

import chex
import optax
import jax
import matplotlib.pyplot as plt

from fabjax.train.generic_training_loop import TrainConfig, train, ListLogger, Logger
from fabjax.train.evaluate import setup_fab_eval_function
from fabjax.train import build_fab_no_buffer_init_step_fns, LogProbFn, \
    TrainStateNoBuffer, build_fab_with_buffer_init_step_fns, TrainStateWithBuffer
from fabjax.buffer.prioritised_buffer import build_prioritised_buffer, PrioritisedBuffer
from fabjax.flow import build_flow, Flow, FlowDistConfig
from fabjax.sampling import build_smc, build_blackjax_hmc, SequentialMonteCarloSampler, simple_resampling, \
    build_metropolis
from fabjax.targets.gmm import GMM
from fabjax.utils.plot import plot_marginal_pair, plot_contours_2D
from fabjax.utils.optimize import get_optimizer, OptimizerConfig


class FABTrainConfig(NamedTuple):
    dim: int
    n_iteration: int
    batch_size: int
    plot_batch_size: int
    n_eval: int
    flow: Flow
    smc: SequentialMonteCarloSampler
    plotter: Callable
    eval_fn: Callable
    log_p_fn: LogProbFn
    optimizer: optax.GradientTransformation
    use_buffer: bool
    buffer: Optional[PrioritisedBuffer] = None
    n_updates_per_smc_forward_pass: Optional[int] = None
    use_reverse_kl_loss: bool = False
    w_adjust_clip: float = 10.
    n_checkpoints: int = 0
    logger: Logger = ListLogger()
    seed: int = 0
    save: bool = False
    use_64_bit: bool = False




def setup_plotter(flow, smc, log_p_fn, plot_batch_size, plot_bound: float,
                  buffer: Optional[PrioritisedBuffer] = None):
    @jax.jit
    @chex.assert_max_traces(3)
    def get_data_for_plotting(state: Union[TrainStateNoBuffer, TrainStateWithBuffer], key: chex.PRNGKey):
        x0 = flow.sample_apply(state.flow_params, key, (plot_batch_size,))

        def log_q_fn(x: chex.Array) -> chex.Array:
            return flow.log_prob_apply(state.flow_params, x)

        point, log_w, smc_state, smc_info = smc.step(x0, state.smc_state, log_q_fn, log_p_fn)
        x_smc = point.x
        _, x_smc_resampled = simple_resampling(key, log_w, x_smc)

        if buffer is not None:
            x_buffer = buffer.sample(key, state.buffer_state, plot_batch_size)[0]
        else:
            x_buffer = None

        return x0, x_smc, x_smc_resampled, x_buffer

    def plot(state: Union[TrainStateNoBuffer, TrainStateWithBuffer], key: chex.PRNGKey):
        x0, x_smc, x_smc_resampled, x_buffer = get_data_for_plotting(state, key)

        if buffer:
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            axs = axs.flatten()
            plot_marginal_pair(x_buffer, axs[3], bounds=(-plot_bound, plot_bound))
            plot_contours_2D(log_p_fn, axs[3], bound=plot_bound, levels=50)
            axs[3].set_title("buffer samples")
        else:
            fig, axs = plt.subplots(3, figsize=(5, 15))
        plot_marginal_pair(x0, axs[0], bounds=(-plot_bound, plot_bound))
        plot_marginal_pair(x_smc, axs[1], bounds=(-plot_bound, plot_bound))
        plot_marginal_pair(x_smc_resampled, axs[2], bounds=(-plot_bound, plot_bound))
        plot_contours_2D(log_p_fn, axs[0], bound=plot_bound, levels=50)
        plot_contours_2D(log_p_fn, axs[1], bound=plot_bound, levels=50)
        plot_contours_2D(log_p_fn, axs[2], bound=plot_bound, levels=50)
        axs[0].set_title("flow samples")
        axs[1].set_title("smc samples")
        axs[2].set_title("resampled smc samples")
        plt.tight_layout()
        plt.show()
    return plot


def setup_fab_config():
    # Setup params

    # Train
    easy_mode = False
    train_long = False
    use_64_bit = False  # Can help improve stability.
    alpha = 2.  # alpha-divergence param
    use_kl_loss = False  # Include additional reverse KL loss.
    dim = 2
    n_eval = 10
    batch_size = 128
    eval_batch_size = 256
    plot_batch_size = 1000

    # Setup buffer.
    with_buffer = True
    buffer_max_length = batch_size*100
    buffer_min_length = batch_size*10
    n_updates_per_smc_forward_pass = 4
    w_adjust_clip = 10.

    # Flow.
    n_layers = 8
    conditioner_mlp_units = (80, 80)
    act_norm = False

    # SMC.
    use_resampling = False
    use_hmc = True
    hmc_n_outer_steps = 1
    hmc_init_step_size = 1e-3
    metro_n_outer_steps = 1
    hmc_n_inner_steps = 3
    metro_init_step_size = 5.  # Needs to be big enough to jump between modes

    target_p_accept = 0.65
    n_intermediate_distributions = 2
    spacing_type = 'linear'



    # Setup flow and target.
    flow_config = FlowDistConfig(dim=dim, n_layers=n_layers, conditioner_mlp_units=conditioner_mlp_units,
                                 act_norm=act_norm)
    flow = build_flow(flow_config)

    if easy_mode:
        target_loc_scaling = 10
        n_mixes = 4
        n_iterations = int(2e3)
    else:
        target_loc_scaling = 40
        n_mixes = 40
        if train_long:
            n_iterations = int(1e4)
        else:
            n_iterations = int(2e3)

    optimizer_config = OptimizerConfig(
        init_lr=3e-4,
        dynamic_grad_ignore_and_clip=True  # Ignore massive gradients.
    )


    gmm = GMM(dim, n_mixes=n_mixes, loc_scaling=target_loc_scaling, log_var_scaling=2., seed=0)
    log_prob_target = gmm.log_prob

    # Setup smc.
    if use_hmc:
        tune_step_size = True
        transition_operator = build_blackjax_hmc(dim=2, n_outer_steps=hmc_n_outer_steps,
                                                     init_step_size=hmc_init_step_size,
                                                 target_p_accept=target_p_accept,
                                                 adapt_step_size=tune_step_size,
                                                 n_inner_steps=hmc_n_inner_steps)
    else:
        tune_step_size = False
        transition_operator = build_metropolis(dim, metro_n_outer_steps, metro_init_step_size,
                                               target_p_accept=target_p_accept, tune_step_size=tune_step_size)

    smc = build_smc(transition_operator=transition_operator,
                    n_intermediate_distributions=n_intermediate_distributions, spacing_type=spacing_type,
                    alpha=alpha, use_resampling=use_resampling)

    # Optimizer
    optimizer, lr = get_optimizer(optimizer_config)

    # Prioritized buffer
    if with_buffer:
        buffer = build_prioritised_buffer(dim=dim, max_length=buffer_max_length, min_length_to_sample=buffer_min_length)
    else:
        buffer = None
        n_updates_per_smc_forward_pass = None

    # Plotter
    plotter = setup_plotter(flow=flow, smc=smc, log_p_fn=gmm.log_prob, plot_batch_size=plot_batch_size,
                            plot_bound=target_loc_scaling * 1.5, buffer=buffer)
    
    # Eval function
    # Eval uses AIS, and sets alpha=1 which is equivalent to targetting p.
    ais_eval = build_smc(transition_operator=transition_operator,
                    n_intermediate_distributions=n_intermediate_distributions, spacing_type=spacing_type,
                    alpha=1., use_resampling=False)
    eval_fn = setup_fab_eval_function(flow=flow, ais=ais_eval, log_p_x=log_prob_target,
                                      batch_size=eval_batch_size,
                                      inner_batch_size=batch_size)

    config = FABTrainConfig(dim=dim, n_iteration=n_iterations, batch_size=batch_size, flow=flow,
                            log_p_fn=log_prob_target, smc=smc, optimizer=optimizer, plot_batch_size=plot_batch_size,
                            n_eval=n_eval, plotter=plotter, buffer=buffer, use_buffer=with_buffer,
                            n_updates_per_smc_forward_pass=n_updates_per_smc_forward_pass,
                            use_64_bit=use_64_bit, w_adjust_clip=w_adjust_clip,
                            eval_fn=eval_fn, use_reverse_kl_loss=use_kl_loss)

    return config


def setup_molboil_train_config(fab_config: FABTrainConfig) -> TrainConfig:
    """Convert fab_config into what we need for running the molboil training loop."""

    if fab_config.use_buffer:
        assert fab_config.buffer is not None and fab_config.n_updates_per_smc_forward_pass is not None
        init, step = build_fab_with_buffer_init_step_fns(
            flow=fab_config.flow, log_p_fn=fab_config.log_p_fn,
            smc=fab_config.smc, optimizer=fab_config.optimizer,
            batch_size=fab_config.batch_size,
            buffer=fab_config.buffer, n_updates_per_smc_forward_pass=fab_config.n_updates_per_smc_forward_pass,
            w_adjust_clip=fab_config.w_adjust_clip,
            use_reverse_kl_loss=fab_config.use_reverse_kl_loss
        )
    else:
        init, step = build_fab_no_buffer_init_step_fns(
            fab_config.flow, log_p_fn=fab_config.log_p_fn,
            smc=fab_config.smc, optimizer=fab_config.optimizer,
            batch_size=fab_config.batch_size)
        

    def eval_and_plot_fn(state, subkey, iteration, save, plots_dir) -> dict:
        fab_config.plotter(state, subkey)
        info = fab_config.eval_fn(state, subkey)
        return info

    train_config = TrainConfig(n_iteration=fab_config.n_iteration,
                n_checkpoints=fab_config.n_checkpoints,
                logger = fab_config.logger,
                seed = fab_config.seed,
                n_eval = fab_config.n_eval,
                init_state = init,
                update_state = step,
                eval_and_plot_fn = eval_and_plot_fn,
                save = fab_config.save,
                use_64_bit=fab_config.use_64_bit
                )
    return train_config


if __name__ == '__main__':
    fab_config = setup_fab_config()
    train_config = setup_molboil_train_config(fab_config)
    train(train_config)
