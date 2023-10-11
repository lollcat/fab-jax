"""Taken from https://github.com/google-deepmind/annealed_flow_transport/blob/master/annealed_flow_transport/densities.py."""

from typing import List

import os.path as osp
import pathlib

import chex
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import jax

from fabjax.targets.base import Target
from fabjax.targets import cox_utils as cp_utils


Array = chex.Array


class LogGaussianCoxPines(Target):
  """Log Gaussian Cox process posterior in 2D for pine saplings data.

  This follows Heng et al 2020 https://arxiv.org/abs/1708.08396 .

  config.file_path should point to a csv file of num_points columns
  and 2 rows containg the Finnish pines data.

  config.use_whitened is a boolean specifying whether or not to use a
  reparameterization in terms of the Cholesky decomposition of the prior.
  See Section G.4 of https://arxiv.org/abs/2102.07501 for more detail.
  The experiments in the paper have this set to False.

  num_dim should be the square of the lattice sites per dimension.
  So for a 40 x 40 grid num_dim should be 1600.
  """

  def __init__(self,
               num_dim: int = 32*32,
               use_whitened: bool = False):

    # Discretization is as in Controlled Sequential Monte Carlo
    # by Heng et al 2017 https://arxiv.org/abs/1708.08396
    self._num_latents = num_dim
    self._num_grid_per_dim = int(np.sqrt(num_dim))

    file_path = osp.join(pathlib.Path(__file__).parent.resolve(), "df_pines.csv")
    pines_array = self.get_pines_points(file_path)[1:, 1:]
    bin_counts = jnp.array(
        cp_utils.get_bin_counts(pines_array,
                                self._num_grid_per_dim))

    self._flat_bin_counts = jnp.reshape(bin_counts, (self._num_latents))

    # This normalizes by the number of elements in the grid
    self._poisson_a = 1./self._num_latents
    # Parameters for LGCP are as estimated in Moller et al, 1998
    # "Log Gaussian Cox processes" and are also used in Heng et al.

    self._signal_variance = 1.91
    self._beta = 1./33

    self._bin_vals = cp_utils.get_bin_vals(self._num_grid_per_dim)

    def short_kernel_func(x, y):
      return cp_utils.kernel_func(x, y, self._signal_variance,
                                  self._num_grid_per_dim, self._beta)

    self._gram_matrix = cp_utils.gram(short_kernel_func, self._bin_vals)
    self._cholesky_gram = jnp.linalg.cholesky(self._gram_matrix)
    self._white_gaussian_log_normalizer = -0.5 * self._num_latents * jnp.log(
        2. * jnp.pi)

    half_log_det_gram = jnp.sum(jnp.log(jnp.abs(jnp.diag(self._cholesky_gram))))
    self._unwhitened_gaussian_log_normalizer = -0.5 * self._num_latents * jnp.log(
        2. * jnp.pi) - half_log_det_gram
    # The mean function is a constant with value mu_zero.
    self._mu_zero = jnp.log(126.) - 0.5*self._signal_variance

    if use_whitened:
      self._posterior_log_density = self.whitened_posterior_log_density
    else:
      self._posterior_log_density = self.unwhitened_posterior_log_density

    super().__init__(dim=num_dim, log_Z=self.gt_logz(), can_sample=False, n_plots=0,
                   n_model_samples_eval=1000, n_target_samples_eval=2000)

  def get_pines_points(self, file_path):
    """Get the pines data points."""
    with open(file_path, "rt") as input_file:
      b = np.genfromtxt(input_file, delimiter=",")
    return b

  def whitened_posterior_log_density(self, white: Array) -> Array:
    quadratic_term = -0.5 * jnp.sum(white**2)
    prior_log_density = self._white_gaussian_log_normalizer + quadratic_term
    latent_function = cp_utils.get_latents_from_white(white, self._mu_zero,
                                                      self._cholesky_gram)
    log_likelihood = cp_utils.poisson_process_log_likelihood(
        latent_function, self._poisson_a, self._flat_bin_counts)
    return prior_log_density + log_likelihood

  def unwhitened_posterior_log_density(self, latents: Array) -> Array:
    white = cp_utils.get_white_from_latents(latents, self._mu_zero,
                                            self._cholesky_gram)
    prior_log_density = -0.5 * jnp.sum(
        white * white) + self._unwhitened_gaussian_log_normalizer
    log_likelihood = cp_utils.poisson_process_log_likelihood(
        latents, self._poisson_a, self._flat_bin_counts)
    return prior_log_density + log_likelihood

  def log_prob(self, x: Array) -> Array:
    if x.ndim == 1:
        return self._posterior_log_density(x)
    else:
        assert x.ndim == 2
        return jax.vmap(self._posterior_log_density)(x)

  def gt_logz(self):
    if self._num_grid_per_dim == 40:
        return 501.806  # from long run SMC
    elif self._num_grid_per_dim == 32:
        return 503.3939
    else:
        return None

  def visualise(self,
                  samples: chex.Array,
                  axes: List[plt.Axes],
                  ) -> None:
    return None

CoxDist = LogGaussianCoxPines


# class CoxDist(Target):
# Using prev code
#     def __init__(self, num_bins_per_dim: int = 40):
#         fcsv = osp.join(pathlib.Path(__file__).parent.resolve(), "df_pines.csv")
#         self.num_bins_per_dim = num_bins_per_dim
#         self.cox = Cox(fcsv, num_bins_per_dim, use_whitened=False)
#
#         dim = int(num_bins_per_dim ** 2)
#
#         super().__init__(dim=dim, log_Z=self.gt_logz(), can_sample=False, n_plots=0,
#                          n_model_samples_eval=1000, n_target_samples_eval=2000)
#
#     def gt_logz(self):
#         if self.num_bins_per_dim == 40:
#             return 501.806  # from long run SMC
#         elif self.num_bins_per_dim == 32:
#             return 503.3939
#         else:
#             return None
#
#     def log_prob(self, x: chex.Array):
#         batched = x.ndim == 2
#         if not batched:
#             x = x[None, ]
#         log_prob = self.cox.evaluate_log_density(x)
#         if not batched:
#             log_prob = jnp.squeeze(log_prob, axis=0)
#         return log_prob
#
#     def visualise(self,
#                   samples: chex.Array,
#                   axes: List[plt.Axes],
#                   ) -> None:
#         return None
#
