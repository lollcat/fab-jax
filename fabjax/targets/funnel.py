from typing import List

import jax.numpy as jnp
import distrax
import chex
import jax.random
import matplotlib.pyplot as plt

from fabjax.targets.base import Target

class FunnelSet(Target):
    """
    x0 ~ N(0, 3^2), xi | x0 ~ N(0, exp(x0)), i = 1, ..., 9
        https://arxiv.org/abs/2310.02679
    https://github.com/zdhNarsil/Diffusion-Generative-Flow-Samplers/blob/main/target/distribution
    """
    def __init__(self, dim: int = 10) -> None:
        super().__init__(dim=dim, log_Z=0., can_sample=True, n_plots=0,
                         n_model_samples_eval=1000, n_target_samples_eval=1000)
        self.data_ndim = dim
        self.dist_dominant = distrax.Normal(jnp.array([0.0]), jnp.array([3.0]))
        self.mean_other = jnp.zeros(dim - 1, dtype=float)
        self.cov_eye = jnp.eye(dim - 1).reshape((1, dim - 1, dim - 1))


    def log_prob(self, x: chex.Array):
        batched = x.ndim == 2
        if not batched:
            x = x[None, ]

        dominant_x = x[:, 0]
        log_density_dominant = self.dist_dominant.log_prob(dominant_x)  # (B, )
        # log_density_other = self._dist_other(dominant_x).log_prob(x[:, 1:])  # (B, )

        log_sigma = 0.5 * x[:, 0:1]
        sigma2 = jnp.exp(x[:, 0:1])
        neglog_density_other = 0.5*jnp.log(2*jnp.pi) + log_sigma + 0.5 * x[:, 1:] ** 2 / sigma2
        log_density_other = jnp.sum(-neglog_density_other, axis=-1)

        log_prob = log_density_dominant + log_density_other
        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)
        return log_prob

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> chex.Array:
        key1, key2 = jax.random.split(seed)
        dominant_x = self.dist_dominant.sample(seed=key1, sample_shape=sample_shape)  # (B,1)
        x_others = self._dist_other(dominant_x).sample(seed=key2)  # (B, dim-1)
        return jnp.hstack([dominant_x, x_others])

    def _dist_other(self, dominant_x):
        variance_other = jnp.exp(dominant_x)
        cov_other = variance_other.reshape(-1, 1, 1) * self.cov_eye
        # use covariance matrix, not std
        return distrax.MultivariateNormalFullCovariance(self.mean_other, cov_other)

    def visualise(self,
                  samples: chex.Array,
                  axes: List[plt.Axes],
                  ) -> None:
        return None
