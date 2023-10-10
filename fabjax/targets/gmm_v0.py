from typing import Callable, Tuple, Optional, List

import chex
import jax
import jax.numpy as jnp
import distrax
import matplotlib.pyplot as plt

from fabjax.targets.base import Target, LogProbFn
from fabjax.utils.plot import plot_marginal_pair, plot_contours_2D


class GMM(Target):
    def __init__(
            self,
            dim: int, n_mixes: int, loc_scaling: float,
            var_scaling: float = 1.0, seed: int = 0,
                 ) -> None:
        super().__init__(dim=dim, log_Z=0.0, can_sample=True, n_plots=1,
                         n_model_samples_eval=1000, n_target_samples_eval=1000)

        self.seed = seed
        self.n_mixes = n_mixes

        key = jax.random.PRNGKey(seed)
        logits = jnp.ones(n_mixes)
        mean = jax.random.uniform(shape=(n_mixes, dim), key=key, minval=-1.0, maxval=1.0) * loc_scaling
        var = jnp.ones(shape=(n_mixes, dim)) * var_scaling

        mixture_dist = distrax.Categorical(logits=logits)
        components_dist = distrax.Independent(
            distrax.Normal(loc=mean, scale=var), reinterpreted_batch_ndims=1
        )
        self.distribution = distrax.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            components_distribution=components_dist,
        )

        self._plot_bound = loc_scaling * 1.5


    def log_prob(self, x: chex.Array) -> chex.Array:
        log_prob = self.distribution.log_prob(x)

        # Can have numerical instabilities once log prob is very small. Manually override to prevent this.
        # This will cause the flow will ignore regions with less than 1e-4 probability under the target.
        valid_log_prob = log_prob > -1e4
        log_prob = jnp.where(valid_log_prob, log_prob, -jnp.inf*jnp.ones_like(log_prob))
        return log_prob

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> chex.Array:
        return self.distribution.sample(seed=seed, sample_shape=sample_shape)


    def visualise(
            self,
            samples: chex.Array,
            axes: List[plt.Axes],
             ) -> None:
        """Visualise samples from the model."""
        assert len(axes) == self.n_plots

        ax = axes[0]
        plot_marginal_pair(samples, ax, bounds=(-self._plot_bound, self._plot_bound))
