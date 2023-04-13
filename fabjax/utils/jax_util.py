import jax.numpy as jnp
import chex

def broadcasted_where(valid_samples: chex.Array, a1: chex.Array, a2: chex.Array):
    chex.assert_equal_shape((a1, a2))
    # broadcast over shape suffix of a1 and a2.
    for i in range(a1.ndim - valid_samples.ndim):
        valid_samples = jnp.expand_dims(valid_samples, axis=-1)
    return jnp.where(valid_samples, a1, a2)

inverse_softplus = lambda x: jnp.log(jnp.exp(x) - 1.)

