from typing import List

import chex
import jax.numpy as jnp
import distrax
import matplotlib.pyplot as plt

from fabjax.targets.base import Target
from fabjax.utils.plot import plot_marginal_pair, plot_contours_2D


class GaussianMixture2D(Target):
    """
    2-D Guassian mixture.
    https://arxiv.org/abs/2310.02679
    https://github.com/zdhNarsil/Diffusion-Generative-Flow-Samplers/blob/main/target/distribution/gm.py
    """
    def __init__(self, scale=0.5477222):
        dim = 2
        super().__init__(dim=dim, log_Z=0.0, can_sample=True, n_plots=1,
                         n_model_samples_eval=1000, n_target_samples_eval=1000)
        mean_ls = [
            [-5., -5.], [-5., 0.], [-5., 5.],
            [0., -5.], [0., 0.], [0., 5.],
            [5., -5.], [5., 0.], [5., 5.],
        ]
        nmode = len(mean_ls)
        mean = jnp.stack([jnp.array(xy) for xy in mean_ls])
        comp = distrax.Independent(
            distrax.Normal(loc=mean, scale=jnp.ones_like(mean)*scale),
            reinterpreted_batch_ndims=1
        )
        mix = distrax.Categorical(logits=jnp.ones(nmode))
        self.gmm = distrax.MixtureSameFamily(mixture_distribution=mix,
                                             components_distribution=comp)
        self._plot_bound = 8

    def log_prob(self, x):
        log_prob = self.gmm.log_prob(x)
        return log_prob


    def sample(self, seed, sample_shape):
        return self.gmm.sample(seed=seed, sample_shape=sample_shape)

    def visualise(
            self,
            samples: chex.Array,
            axes: List[plt.Axes],
             ) -> None:
        """Visualise samples from the model."""
        assert len(axes) == self.n_plots

        ax = axes[0]
        plot_contours_2D(self.log_prob, ax, bound=self._plot_bound, levels=20)
        plot_marginal_pair(samples, ax, bounds=(-self._plot_bound, self._plot_bound))

