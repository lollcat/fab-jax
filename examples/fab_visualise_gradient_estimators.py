from typing import Tuple

import chex
import jax.numpy as jnp
import jax
import distrax
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from fabjax.sampling.mcmc.hmc import build_blackjax_hmc
from fabjax.sampling.mcmc.metropolis import build_metropolis
from fabjax.sampling.smc import build_smc, SequentialMonteCarloSampler
from fabjax.sampling.resampling import simple_resampling
from fabjax.sampling.base import create_point
from fabjax.utils.plot import plot_contours_2D, plot_marginal_pair, plot_history
from fabjax.utils.logging import ListLogger


# TODO: Create nice visualisation of the signal to noise ratio of the gradient estimator.


def analytic_alpha_2_div(mean_q: chex.Array, mean_p: chex.Array) -> chex.Array:
    """Calculate \int p(x)^2/q dx where p and q are unit-variance Gaussians."""
    return jnp.exp(jnp.sum(mean_p**2 + mean_q**2 - 2*mean_p*mean_q))


def plot_1d(samples: chex.Array):
    chex.assert_rank(samples, 2)
    chex.assert_axis_dimension(samples, 1, 1)
    dist_q, dist_p, true_Z = problem_setup(1)

    x_linspace = jnp.linspace(-5., 5., 50)[:, None]
    log_q = dist_q.log_prob(x_linspace)
    log_p = dist_p.log_prob(x_linspace)

    fig, axs = plt.subplots()
    axs.plot(x_linspace, jnp.exp(log_q), label="q")
    axs.plot(x_linspace, jnp.exp(log_p), label="p")
    axs.plot(x_linspace, jnp.exp(2*log_p - log_q)/true_Z, label="p^2/q")
    axs.hist(jnp.squeeze(samples), label="samples", density=True, bins=100, alpha=0.3)
    plt.legend()
    plt.show()



def problem_setup(dim: int) -> Tuple[distrax.Distribution, distrax.Distribution, chex.Array]:

    loc_q = jnp.zeros((dim,)) + 0.5
    scale_q = jnp.ones((dim,))
    dist_q = distrax.MultivariateNormalDiag(loc_q, scale_q)

    loc_p = loc_q - 1
    scale_p = jnp.ones((dim,))
    dist_p = distrax.MultivariateNormalDiag(loc_p, scale_p)
    true_Z = analytic_alpha_2_div(loc_q, loc_p)

    return dist_q, dist_p, true_Z


def setup_smc(
        dim: int,
        n_intermediate_distributions: int,
        tune_step_size: bool = False,
        transition_n_outer_steps: int = 1,
        target_p_accept: float = 0.65,
        spacing_type = 'linear',
        resampling: bool = False,
        alpha = 2.,
        use_hmc: bool = False,
        init_step_size: float = 1.

) -> SequentialMonteCarloSampler:
    if use_hmc:
        transition_operator = build_blackjax_hmc(dim=dim, n_outer_steps=transition_n_outer_steps,
                                                 init_step_size=init_step_size, target_p_accept=target_p_accept,
                                                 adapt_step_size=tune_step_size)
    else:
        transition_operator = build_metropolis(dim=dim, n_steps=transition_n_outer_steps, init_step_size=init_step_size,
                                               target_p_accept=target_p_accept, tune_step_size=tune_step_size)

    smc = build_smc(transition_operator=transition_operator,
                    n_intermediate_distributions=n_intermediate_distributions, spacing_type=spacing_type,
                    alpha=alpha, use_resampling=resampling)
    return smc

def run_and_plot_1d():
    dim = 1
    batch_size = 10000
    alpha = 2.  # sets target to just be p.
    n_smc_distributions = 10
    key = jax.random.PRNGKey(0)

    smc = setup_smc(n_intermediate_distributions=n_smc_distributions,
                    tune_step_size=False,
                    alpha=alpha,
                    spacing_type='linear',
                    use_hmc=True,
                    dim=dim,
                    transition_n_outer_steps=1,
                    init_step_size=1.)
    smc_state = smc.init(key)

    key, subkey = jax.random.split(key)
    dist_q, dist_p, true_Z = problem_setup(dim)
    positions = dist_q.sample(seed=subkey, sample_shape=(batch_size,))
    point, log_w, smc_state, info = smc.step(x0=positions, smc_state=smc_state, log_q_fn=dist_q.log_prob,
                                             log_p_fn=dist_p.log_prob)

    plot_1d(point.x)

    # Compare to 1 step of mcmc.
    start_point = jax.vmap(create_point, in_axes=(0, None, None, None))(positions, dist_q.log_prob, dist_p.log_prob,
                                                                        smc.transition_operator.uses_grad)
    final_trans_op_state = jax.tree_map(lambda x: x[-1], smc_state.transition_operator_state)
    mcmc_point, _, info = smc.transition_operator.step(
        point=start_point, transition_operator_state=final_trans_op_state,
        beta=smc.betas[-2], alpha=alpha, log_q_fn=dist_q.log_prob, log_p_fn=dist_p.log_prob)
    plot_1d(mcmc_point.x)




def run_ess_with_more_dists():
    dim = 1
    batch_size = 10000
    alpha = 2.  # 1. sets target to just be p.
    key = jax.random.PRNGKey(0)

    logger = ListLogger()
    dist_q, dist_p, true_Z = problem_setup(dim)
    if alpha == 1.:
        true_Z = 1.

    n_distributions_list = [1, 1, 1, 8, 8, 8, 32, 32, 32, 64, 64, 64, 128, 128, 128]
    for n_smc_distributions in tqdm(n_distributions_list):
        smc = setup_smc(n_intermediate_distributions=n_smc_distributions,
                        tune_step_size=False,
                        alpha=alpha,
                        spacing_type='linear',
                        use_hmc=False,
                        dim=dim,
                        transition_n_outer_steps=1,
                        init_step_size=2.)
        key, subkey = jax.random.split(key)
        smc_state = smc.init(key)

        key, subkey = jax.random.split(key)
        positions, log_q = dist_q.sample_and_log_prob(seed=subkey, sample_shape=(batch_size,))

        log_w_check = dist_p.log_prob(positions) - dist_q.log_prob(positions)
        ess_q_p_check = 1 / jnp.mean(jnp.exp(log_w_check)**2)
        point, log_w, _, info = smc.step(x0=positions, smc_state=smc_state, log_q_fn=dist_q.log_prob,
                                                 log_p_fn=dist_p.log_prob)
        log_z_estimate = jax.nn.logsumexp(log_w) - jnp.log(log_w.shape[0])
        ess =  1 / jnp.mean(jnp.exp(log_w)**2)
        info.update(log_z_estimate=log_z_estimate)
        info.update(variance=jnp.var(log_w))

        logger.write(info)

        plot_1d(point.x)

    plt.title("ESS")
    plt.plot(n_distributions_list, logger.history['ess_smc_final'])
    plt.show()
    plt.plot(n_distributions_list, np.log(logger.history['log_z_estimate']), "o", label='log_z_estimate')
    plt.plot(n_distributions_list, np.log(jnp.ones(len(n_distributions_list))*true_Z), "--", label='log_z_true')
    plt.title("Z estimate")
    plt.legend()
    plt.show()
    plt.title("variance")
    plt.plot(n_distributions_list, logger.history["variance"])
    plt.show()




if __name__ == '__main__':
    from jax import config
    config.update("jax_enable_x64", True)
    # run_and_plot_1d()
    run_ess_with_more_dists()
