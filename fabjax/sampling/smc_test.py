"""smc seems to be working, but is very sensitive to the tuning of the step size of HMC."""
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


def analytic_alpha_2_div(mean_q: chex.Array, mean_p: chex.Array):
    """Calculate \int p(x)^2/q dx where p and q are unit-variance Gaussians."""
    return jnp.exp(jnp.sum(mean_p**2 + mean_q**2 - 2*mean_p*mean_q))


def plot_samples(x_smc, x_p, x_q, dist_q, dist_p):
    # Visualise samples.
    bound = 10
    fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharex=True, sharey=True)
    # smc samples
    plot_contours_2D(dist_q.log_prob, axs[0, 0], bound=bound)
    plot_marginal_pair(x_smc, axs[0, 0], bounds=(-bound, bound), alpha=0.2)
    plot_contours_2D(lambda x: 2*dist_p.log_prob(x) - dist_q.log_prob(x), axs[0, 1], bound=bound)
    plot_marginal_pair(x_smc, axs[0, 1], bounds=(-bound, bound), alpha=0.2)

    # Samples from the distributions themselves
    plot_contours_2D(dist_p.log_prob, axs[1, 0], bound=bound)
    plot_marginal_pair(x_q, axs[1, 0], bounds=(-bound, bound), alpha=0.2)
    plot_contours_2D(dist_p.log_prob, axs[1, 1], bound=bound)
    plot_marginal_pair(x_p, axs[1, 1], bounds=(-bound, bound), alpha=0.2)

    axs[0, 0].set_title("samples smc vs log prob q contours")
    axs[0, 1].set_title("samples smc vs log prob p^2/q contours")
    axs[1, 0].set_title("samples q vs log prob q contours")
    axs[1, 1].set_title("samples p vs log prob p contours")
    plt.tight_layout()
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
                                                 adapt_step_size=tune_step_size, step_size_multiplier=1.5)
    else:
        transition_operator = build_metropolis(dim=dim, n_steps=transition_n_outer_steps, init_step_size=init_step_size,
                                               target_p_accept=target_p_accept, tune_step_size=tune_step_size,
                                               step_size_multiplier=1.5)

    smc = build_smc(transition_operator=transition_operator,
                    n_intermediate_distributions=n_intermediate_distributions, spacing_type=spacing_type,
                    alpha=alpha, use_resampling=resampling)
    return smc



def test_smc_visualise_as_step_size_tuned():
    """Visualise smc as we tune the hmc params"""

    dim = 2
    key = jax.random.PRNGKey(0)
    batch_size = 1000
    n_smc_distributions = 20

    dist_q, dist_p, true_Z = problem_setup(dim)
    smc = setup_smc(dim, n_smc_distributions,
                    init_step_size=0.1,
                    tune_step_size=True) # not very good initial step size
    x_p = dist_p.sample(seed=key, sample_shape=(batch_size,))
    x_q = dist_q.sample(seed=key, sample_shape=(batch_size,))


    smc_state = smc.init(key)

    logger = ListLogger()



    n_iters = 20
    n_plots = 5
    plot_iter = np.linspace(0, n_iters-1, n_plots, dtype=int)

    # Run MCMC chain.
    for i in range(n_iters):
        # Loop so transition operator step size can be tuned.
        # This makes a massive difference.
        key, subkey = jax.random.split(key)
        positions = dist_q.sample(seed=subkey, sample_shape=(batch_size,))
        point, log_w, smc_state, info = smc.step(positions, smc_state, dist_q.log_prob, dist_p.log_prob)

        # Plot and log info.
        if i in plot_iter:
            plot_samples(point.x, x_p, x_q, dist_q, dist_p)
        Z_estimate = jnp.mean(jnp.exp(log_w))
        log_abs_error = jnp.log(jnp.abs(Z_estimate - true_Z))  # log makes scale more reasonable in plot.
        info.update(log_abs_error_z_estimate=log_abs_error)
        logger.write(info)

    plt.plot(logger.history['log_abs_error_z_estimate'])
    plt.show()


def test_reasonable_resampling():
    """One way to check the log weights are reasonable is to resample in proportion to them, and see if it makes the
    samples more similar to the target."""

    dim = 2
    key = jax.random.PRNGKey(0)
    batch_size = 1000
    n_smc_distributions = 1  # Make smc low quality.

    # Setup
    dist_q, dist_p, true_Z = problem_setup(dim)
    smc = setup_smc(dim, n_smc_distributions, tune_step_size=False)
    x_p = dist_p.sample(seed=key, sample_shape=(batch_size,))
    x_q = dist_q.sample(seed=key, sample_shape=(batch_size,))
    smc_state = smc.init(key)

    # Generate samples.
    key, subkey = jax.random.split(key)
    positions = dist_q.sample(seed=subkey, sample_shape=(batch_size,))
    point, log_w, smc_state, info = smc.step(positions, smc_state, dist_q.log_prob, dist_p.log_prob)

    _, resamples = simple_resampling(key, log_w, point)

    # Visualise samples before and after resampling.
    plot_samples(point.x, x_p, x_q, dist_q, dist_p)
    plot_samples(resamples.x, x_p, x_q, dist_q, dist_p)


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


def test_ess_with_more_dists():
    dim = 1
    batch_size = 10000
    alpha = 2.  # 1. to set target to just be p.
    key = jax.random.PRNGKey(0)

    logger = ListLogger()
    dist_q, dist_p, true_Z = problem_setup(dim)
    if alpha == 1.:
        true_Z = 1.

    n_distributions_list = [1, 1, 1, 8, 8, 8, 32, 32, 32, 64, 64, 64, 128, 128, 128]
    for n_smc_distributions in tqdm(n_distributions_list):
        smc = setup_smc(dim=dim,
                        n_intermediate_distributions=n_smc_distributions,
                        tune_step_size=False,
                        alpha=alpha,
                        spacing_type='linear',
                        use_hmc=False,
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