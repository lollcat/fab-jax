from typing import NamedTuple, Callable, Optional, Union

import os
import pathlib

import chex
import optax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import wandb

from fabjax.train.generic_training_loop import TrainConfig
from fabjax.train.evaluate import setup_fab_eval_function
from fabjax.train import build_fab_no_buffer_init_step_fns, LogProbFn, \
    TrainStateNoBuffer, build_fab_with_buffer_init_step_fns, TrainStateWithBuffer
from fabjax.buffer.prioritised_buffer import build_prioritised_buffer, PrioritisedBuffer
from fabjax.flow import build_flow, Flow, FlowDistConfig
from fabjax.sampling import build_smc, build_blackjax_hmc, SequentialMonteCarloSampler, simple_resampling, \
    build_metropolis

from fabjax.utils.optimize import get_optimizer, OptimizerConfig
from fabjax.utils.loggers import Logger, ListLogger, WandbLogger, PandasLogger
from fabjax.targets.base import Target


def setup_logger(cfg: DictConfig) -> Logger:
    if hasattr(cfg.logger, "wandb"):
        logger = WandbLogger(**cfg.logger.wandb, config=dict(cfg))
    elif hasattr(cfg.logger, "list_logger"):
        logger = ListLogger()
    elif hasattr(cfg.logger, 'pandas_logger'):
        logger = PandasLogger(save_path=cfg.training.save_dir,
                              save_period=cfg.logger.pandas_logger.save_period,
                              save=cfg.training.save)
    else:
        raise Exception("No logger specified, try adding the wandb or "
                        "pandas logger to the config file.")
    return logger


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
    logger: Logger
    buffer: Optional[PrioritisedBuffer] = None
    n_updates_per_smc_forward_pass: Optional[int] = None
    use_reverse_kl_loss: bool = False
    w_adjust_clip: float = 10.
    n_checkpoints: int = 0
    seed: int = 0
    save: bool = False
    use_64_bit: bool = False




def setup_plotter(flow, smc, target: Target, plot_batch_size,
                  buffer: Optional[PrioritisedBuffer] = None):
    @jax.jit
    @chex.assert_max_traces(3)
    def get_data_for_plotting(state: Union[TrainStateNoBuffer, TrainStateWithBuffer], key: chex.PRNGKey):
        x0 = flow.sample_apply(state.flow_params, key, (plot_batch_size,))

        def log_q_fn(x: chex.Array) -> chex.Array:
            return flow.log_prob_apply(state.flow_params, x)

        point, log_w, smc_state, smc_info = smc.step(x0, state.smc_state, log_q_fn, target.log_prob)
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
            figs_and_axes = [plt.subplots(2, 2, figsize=(10, 10)) for _ in range(target.n_plots)]
            figs = [fig for fig, axs in figs_and_axes]
            axs = [axs.flatten() for fig, axs in figs_and_axes]
            target.visualise(x_buffer, [ax[3] for ax in axs])
            for ax in axs:
                ax[3].set_title("buffer samples")
        else:
            figs_and_axes = [plt.subplots(3, figsize=(5, 15)) for _ in range(target.n_plots)]
            figs = [fig for fig, axs in figs_and_axes]
            axs = [axs for fig, axs in figs_and_axes]
        target.visualise(x0, [ax[0] for ax in axs])
        target.visualise(x_smc, [ax[1] for ax in axs])
        target.visualise(x_smc_resampled, [ax[2] for ax in axs])

        for ax in axs:
            ax[0].set_title("flow samples")
            ax[1].set_title("smc samples")
            ax[2].set_title("resampled smc samples")

        plt.tight_layout()
        plt.show()
    return plot


def setup_fab_config(cfg: DictConfig, target: Target) -> FABTrainConfig:

    if "logger" in cfg.keys():
        logger = setup_logger(cfg)

        if isinstance(logger, WandbLogger) and cfg.training.save_in_wandb_dir:
            save_path = os.path.join(wandb.run.dir, cfg.training.save_dir)
        else:
            save_path = cfg.training.save_dir

        pathlib.Path(save_path).mkdir(exist_ok=True, parents=True)
    else:
        logger = ListLogger()


    dim = target.dim

    # Train
    use_64_bit = cfg.training.use_64_bit  # Can help improve stability.
    alpha = cfg.fab.alpha  # alpha-divergence param
    use_kl_loss = cfg.fab.use_kl_loss  # Include additional reverse KL loss.
    n_eval = cfg.training.n_eval
    batch_size = cfg.training.batch_size
    eval_batch_size = cfg.training.eval_batch_size
    plot_batch_size = cfg.training.plot_batch_size

    # Setup buffer.
    with_buffer = cfg.fab.buffer.with_buffer
    buffer_max_length = batch_size*cfg.fab.buffer.buffer_max_length_in_batches
    buffer_min_length = batch_size*cfg.fab.buffer.buffer_min_length_in_batches
    n_updates_per_smc_forward_pass = cfg.fab.buffer.n_updates_per_smc_forward_pass
    w_adjust_clip = jnp.inf if cfg.fab.w_adjust_clip is None else cfg.fab.w_adjust_clip

    # Flow.
    n_layers = cfg.flow.n_layers
    conditioner_mlp_units = cfg.flow.conditioner_mlp_units
    act_norm = cfg.flow.act_norm

    # SMC.
    use_resampling = cfg.fab.smc.use_resampling
    n_intermediate_distributions = cfg.fab.smc.n_intermediate_distributions
    spacing_type = cfg.fab.smc.spacing_type



    # Setup flow and target.
    flow_config = FlowDistConfig(dim=dim, **cfg.flow)
    flow = build_flow(flow_config)

    opt_cfg = dict(cfg.training.optimizer)
    n_iter_warmup = opt_cfg.pop('warmup_n_epoch') * cfg.fab.buffer.n_updates_per_smc_forward_pass
    n_iter_total = cfg.training.n_epoch * cfg.fab.buffer.n_updates_per_smc_forward_pass
    optimizer_config = OptimizerConfig(**opt_cfg,
                                       n_iter_total=n_iter_total,
                                       n_iter_warmup=n_iter_warmup)

    log_prob_target = target.log_prob

    # Setup smc.
    if cfg.fab.smc.transition_operator == 'hmc':
        transition_operator = build_blackjax_hmc(
            dim=target.dim,
            n_outer_steps=cfg.fab.smc.hmc.n_outer_steps,
            init_step_size=cfg.fab.smc.hmc.init_step_size,
            target_p_accept=cfg.fab.smc.hmc.target_p_accept,
            adapt_step_size=cfg.fab.smc.hmc.tune_step_size,
            n_inner_steps=cfg.fab.smc.hmc.n_inner_steps)
    elif cfg.fab.smc.transition_operator == "metropolis":
        transition_operator = build_metropolis(target.dim, cfg.fab.smc.metropolis.n_outer_steps,
                                               cfg.fab.smc.metropolis.init_step_size,
                                               target_p_accept=cfg.fab.smc.metropolis.target_p_accept,
                                               tune_step_size=cfg.fab.smc.metropolis.tune_step_size)
    else:
        raise NotImplementedError

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
    plotter = setup_plotter(flow=flow, smc=smc, target=target, plot_batch_size=plot_batch_size, buffer=buffer)
    
    # Eval function
    # Eval uses AIS, and sets alpha=1 which is equivalent to targetting p.
    ais_eval = build_smc(transition_operator=transition_operator,
                    n_intermediate_distributions=n_intermediate_distributions, spacing_type=spacing_type,
                    alpha=1., use_resampling=False)
    _eval_fn = setup_fab_eval_function(
          flow=flow, ais=ais_eval, log_p_x=log_prob_target,
          eval_n_samples=eval_batch_size,
          inner_batch_size=batch_size,
          log_Z_n_samples=cfg.training.n_samples_log_Z,
          log_Z_true=target.log_Z
         )

    @jax.jit
    def eval_fn(state: Union[TrainStateNoBuffer, TrainStateWithBuffer], key: chex.PRNGKey) -> dict:
        key1, key2 = jax.random.split(key)
        info = _eval_fn(state, key1)
        target_info = target.evaluate(
            model_log_prob_fn=lambda x: flow.log_prob_apply(state.flow_params, x),
            model_sample_and_log_prob_fn=lambda key, shape: flow.sample_and_log_prob_apply(state.flow_params, key, shape),
            key=key2
        )
        info.update(target_info)
        return info


    config = FABTrainConfig(dim=dim, n_iteration=cfg.training.n_epoch, batch_size=batch_size, flow=flow,
                            log_p_fn=log_prob_target, smc=smc, optimizer=optimizer, plot_batch_size=plot_batch_size,
                            n_eval=n_eval, plotter=plotter, buffer=buffer, use_buffer=with_buffer,
                            n_updates_per_smc_forward_pass=n_updates_per_smc_forward_pass,
                            use_64_bit=use_64_bit, w_adjust_clip=w_adjust_clip,
                            eval_fn=eval_fn, use_reverse_kl_loss=use_kl_loss,
                            logger=logger)

    return config


def setup_general_train_config(fab_config: FABTrainConfig) -> TrainConfig:
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

    train_config = TrainConfig(n_iteration=int(fab_config.n_iteration),
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