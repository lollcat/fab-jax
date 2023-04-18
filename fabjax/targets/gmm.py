import jax
import jax.numpy as jnp
import distrax

class GMM:
    def __init__(self, dim, n_mixes, loc_scaling, log_var_scaling=0.1, seed=0):
        self.seed = seed
        self.n_mixes = n_mixes
        self.dim = dim
        key = jax.random.PRNGKey(seed)
        logits = jnp.ones(n_mixes)
        mean = jax.random.uniform(shape=(n_mixes, dim), key=key, minval=-1.0, maxval=1.0) * loc_scaling
        log_var = jnp.ones(shape=(n_mixes, dim)) * log_var_scaling

        mixture_dist = distrax.Categorical(logits=logits)
        var = jax.nn.softplus(log_var)
        components_dist = distrax.Independent(
            distrax.Normal(loc=mean, scale=var), reinterpreted_batch_ndims=1
        )
        self.distribution = distrax.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            components_distribution=components_dist,
        )

    def log_prob(self, x):
        log_prob = self.distribution.log_prob(x)

        # Can have numerical instabilities once log prob is very small. Manually override to prevent this.
        # This will cause the flow will ignore regions with less than 1e-4 probability under the target.
        valid_log_prob = log_prob > -1e4
        log_prob = jnp.where(valid_log_prob, log_prob, -jnp.inf*jnp.ones_like(log_prob))
        return log_prob

    def sample(self, seed, sample_shape):
        return self.distribution.sample(seed=seed, sample_shape=sample_shape)
