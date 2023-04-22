# from typing import Tuple
#
# import chex
# import jax.numpy as jnp
# import jax
# import distrax
# import matplotlib.pyplot as plt
# import numpy as np
# from tqdm import tqdm
#
# from fabjax.sampling.mcmc.hmc import build_blackjax_hmc
# from fabjax.sampling.mcmc.metropolis import build_metropolis
# from fabjax.sampling.smc import build_smc, SequentialMonteCarloSampler
# from fabjax.sampling.resampling import simple_resampling
# from fabjax.sampling.base import create_point
# from fabjax.utils.plot import plot_contours_2D, plot_marginal_pair, plot_history
# from fabjax.utils.logging import ListLogger
#
#
# # TODO: Create nice visualisation of the signal to noise ratio of the gradient estimator.
#
#
# def analytic_alpha_2_div(mean_q: chex.Array, mean_p: chex.Array) -> chex.Array:
#     """Calculate \int p(x)^2/q dx where p and q are unit-variance Gaussians."""
#     return jnp.exp(jnp.sum(mean_p**2 + mean_q**2 - 2*mean_p*mean_q))
#
#
# def plot_1d(samples: chex.Array):
#     chex.assert_rank(samples, 2)
#     chex.assert_axis_dimension(samples, 1, 1)
#     dist_q, dist_p, true_Z = problem_setup(1)
#
#     x_linspace = jnp.linspace(-5., 5., 50)[:, None]
#     log_q = dist_q.log_prob(x_linspace)
#     log_p = dist_p.log_prob(x_linspace)
#
#     fig, axs = plt.subplots()
#     axs.plot(x_linspace, jnp.exp(log_q), label="q")
#     axs.plot(x_linspace, jnp.exp(log_p), label="p")
#     axs.plot(x_linspace, jnp.exp(2*log_p - log_q)/true_Z, label="p^2/q")
#     axs.hist(jnp.squeeze(samples), label="samples", density=True, bins=100)
#     plt.legend()
#     plt.show()
#
#
#
# def problem_setup(dim: int) -> Tuple[distrax.Distribution, distrax.Distribution, chex.Array]:
#
#     loc_q = jnp.zeros((dim,)) + 0.5
#     scale_q = jnp.ones((dim,))
#     dist_q = distrax.MultivariateNormalDiag(loc_q, scale_q)
#
#     loc_p = jnp.zeros((dim,)) - 1
#     scale_p = scale_q
#     dist_p = distrax.MultivariateNormalDiag(loc_p, scale_p)
#     true_Z = analytic_alpha_2_div(loc_q, loc_p)
#
#     return dist_q, dist_p, true_Z
#
#
# def setup_smc(
#         dim: int,
#         n_intermediate_distributions: int,
#         tune_step_size: bool = False,
#         transition_n_outer_steps: int = 1,
#         target_p_accept: float = 0.65,
#         spacing_type = 'linear',
#         resampling: bool = False,
#         alpha = 2.,
#         use_hmc: bool = False,
#         init_step_size: float = 1.
#
# ) -> SequentialMonteCarloSampler:
#     if use_hmc:
#         transition_operator = build_blackjax_hmc(dim=dim, n_outer_steps=transition_n_outer_steps,
#                                                  init_step_size=init_step_size, target_p_accept=target_p_accept,
#                                                  adapt_step_size=tune_step_size)
#     else:
#         transition_operator = build_metropolis(dim=dim, n_steps=transition_n_outer_steps, init_step_size=init_step_size,
#                                                target_p_accept=target_p_accept, tune_step_size=tune_step_size)
#
#     smc = build_smc(transition_operator=transition_operator,
#                     n_intermediate_distributions=n_intermediate_distributions, spacing_type=spacing_type,
#                     alpha=alpha, use_resampling=resampling)
#     return smc
#
# def run_and_plot_1d():
#     dim = 1
#     batch_size = 10000
#     alpha = 2.  # sets target to just be p.
#     n_smc_distributions = 10
#     key = jax.random.PRNGKey(0)
#
#     smc = setup_smc(n_intermediate_distributions=n_smc_distributions,
#                     tune_step_size=False,
#                     alpha=alpha,
#                     spacing_type='linear',
#                     use_hmc=False,
#                     dim=dim,
#                     transition_n_outer_steps=10,
#                     init_step_size=2.)
#     smc_state = smc.init(key)
#
#     key, subkey = jax.random.split(key)
#     dist_q, dist_p, true_Z = problem_setup(dim)
#     positions = dist_q.sample(seed=subkey, sample_shape=(batch_size,))
#     point, log_w, smc_state, info = smc.step(positions, smc_state, dist_q.log_prob, dist_p.log_prob)
#
#     plot_1d(point.x)
#     pass
#
#
#
#
# # def smc_visualise_with_num_dists_1d():
# #     """Visualise quality of smc estimates with increasing number of intermediate distributions."""
# #     dim = 1
# #     key = jax.random.PRNGKey(0)
# #     batch_size = 128
# #     alpha = 2.  # sets target to just be p.
# #
# #     logger = ListLogger()
# #
# #     dist_q, dist_p, true_Z = problem_setup(dim)
# #     true_Z = 1. if alpha == 1. else true_Z
# #     x_p = dist_p.sample(seed=key, sample_shape=(batch_size,))
# #     x_q = dist_q.sample(seed=key, sample_shape=(batch_size,))
# #
# #
# #
# #     for n_smc_distributions in tqdm([1, 4, 16, 64, 128]):
# #         # Setup
# #         smc = setup_smc(n_smc_distributions, tune_step_size=True, alpha=alpha, spacing_type='geometric')
# #         smc_state = smc.init(key)
# #
# #
# #
# #         for i in range(20):  # Allow for some step size tuning - ends up being SUPER IMPORTANT.
# #             # Generate samples.
# #             key, subkey = jax.random.split(key)
# #             positions = dist_q.sample(seed=subkey, sample_shape=(batch_size,))
# #             point, log_w, smc_state, info = smc.step(positions, smc_state, dist_q.log_prob, dist_p.log_prob)
# #
# #         Z_estimate = jnp.mean(jnp.exp(log_w))
# #         log_abs_error = jnp.log(jnp.abs(Z_estimate - true_Z))  # log makes scale more reasonable in plot.
# #         info.update(log_abs_error_z_estimate=log_abs_error)
# #         info.update(z_estimate=Z_estimate)
# #         logger.write(info)
# #
# #         plot_samples(point.x, x_p, x_q, dist_q, dist_p)
# #         _, resamples = simple_resampling(key, log_w, point)
# #         plot_samples(resamples.x, x_p, x_q, dist_q, dist_p)
# #
# #     keys = ['log_abs_error_z_estimate', 'ess_smc_final', 'dist1_mean_acceptance_rate', 'z_estimate', 'dist1_step_size']
# #     history = {key: logger.history[key] for key in keys}
# #     plot_history(history)
# #     plt.show()
#
#
# if __name__ == '__main__':
#     run_and_plot_1d()
#
