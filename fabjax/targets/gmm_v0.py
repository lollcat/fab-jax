from typing import List

import chex
import jax
import jax.numpy as jnp
import distrax
import matplotlib.pyplot as plt

from fabjax.targets.base import Target
from fabjax.utils.plot import plot_marginal_pair, plot_contours_2D


class GMM(Target):
    def __init__(
            self,
            dim: int = 2, n_mixes: int = 40, loc_scaling: float = 40,
            scale_scaling: float = 1.0, seed: int = 0,
                 ) -> None:
        super().__init__(dim=dim, log_Z=0.0, can_sample=True, n_plots=1,
                         n_model_samples_eval=1000, n_target_samples_eval=1000)

        self.seed = seed
        self.n_mixes = n_mixes

        key = jax.random.PRNGKey(seed)
        logits = jnp.ones(n_mixes)
        mean = jax.random.uniform(shape=(n_mixes, dim), key=key, minval=-1.0, maxval=1.0) * loc_scaling
        scale = jnp.ones(shape=(n_mixes, dim)) * scale_scaling

        mixture_dist = distrax.Categorical(logits=logits)
        components_dist = distrax.Independent(
            distrax.Normal(loc=mean, scale=scale), reinterpreted_batch_ndims=1
        )
        self.distribution = distrax.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            components_distribution=components_dist,
        )

        self._plot_bound = loc_scaling * 1.5


    def log_prob(self, x: chex.Array) -> chex.Array:
        log_prob = self.distribution.log_prob(x)
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
        plot_contours_2D(self.log_prob, ax, bound=self._plot_bound, levels=50)
