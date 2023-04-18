from typing import Tuple

import distrax
import haiku as hk
import jax.numpy as jnp
import jax
import chex

from fabjax.utils.jax_util import inverse_softplus

class ActNorm(distrax.Bijector):
    """Unconditioned Affine transform."""
    def __init__(self, dim):
        super().__init__(event_ndims_in = 1,
               is_constant_jacobian = True,
               is_constant_log_det = True)

        scale_logit = hk.get_parameter(name="scale_logit", shape=(dim,), init=jnp.zeros, dtype=float)
        self.scale = jax.nn.softplus(scale_logit + inverse_softplus(1.))
        self.shift = hk.get_parameter(name="scale_logit", shape=(dim,), init=jnp.zeros, dtype=float)


    def forward_and_log_det(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        if x.ndim == 2:
            scale = jnp.ones_like(x) * self.scale[None, :]
            shift = jnp.ones_like(x) * self.shift[None, :]
        else:
            assert x.ndim == 1
            scale = self.scale
            shift = self.shift
        bijector = distrax.Block(distrax.ScalarAffine(shift=shift, scale=scale), ndims=1)
        return bijector.forward_and_log_det(x)

    def inverse_and_log_det(self, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        if y.ndim == 2:
            scale = jnp.ones_like(y) * self.scale[None, :]
            shift = jnp.ones_like(y) * self.shift[None, :]
        else:
            assert y.ndim == 1
            scale = self.scale
            shift = self.shift
        bijector = distrax.Block(distrax.ScalarAffine(shift=shift, scale=scale), ndims=1)
        return bijector.inverse_and_log_det(y)


def build_act_norm_layer(
    dim: int,
    identity_init: bool):
    assert identity_init == True, "Only this option currently configured."
    return ActNorm(dim)
