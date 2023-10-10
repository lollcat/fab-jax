from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import distrax

from fabjax.targets.base import Target, LogProbFn


class GMM(Target):
    def __init__(self, dim: int, n_mixes: int, loc_scaling: float,
                 var_scaling: float = 1.0, seed: int = 0,
                 eval_n_samples: int = 1000,
                 ) -> None:
        self.seed = seed
        self.n_mixes = n_mixes
        self.dim = dim
        self.eval_n_samples = eval_n_samples

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

    def log_prob(self, x: chex.Array) -> chex.Array:
        log_prob = self.distribution.log_prob(x)

        # Can have numerical instabilities once log prob is very small. Manually override to prevent this.
        # This will cause the flow will ignore regions with less than 1e-4 probability under the target.
        valid_log_prob = log_prob > -1e4
        log_prob = jnp.where(valid_log_prob, log_prob, -jnp.inf*jnp.ones_like(log_prob))
        return log_prob

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> chex.Array:
        return self.distribution.sample(seed=seed, sample_shape=sample_shape)



    def visualise(self,
                 model_log_prob_fn: LogProbFn,
                 model_sample_fn: Callable[[chex.PRNGKey, chex.Shape], chex.Array],
                 key: chex.PRNGKey,
                 ) -> dict:
        """Visualise samples from the model."""
        raise NotImplemented
