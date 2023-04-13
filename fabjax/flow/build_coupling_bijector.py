from typing import Sequence

import chex
import distrax
import jax.nn
import jax.numpy as jnp

from fabjax.flow.distrax_with_extra import SplitCouplingWithExtra, ChainWithExtra, BijectorWithExtra
from fabjax.utils.nets import ConditionerMLP
from fabjax.utils.jax_util import inverse_softplus


def make_conditioner(name, n_output_params, mlp_units, identity_init):
    def conditioner(x: chex.Array) -> chex.Array:
        mlp = ConditionerMLP(name=name, mlp_units=mlp_units,
                             n_output_params=n_output_params,
                             zero_init=identity_init)
        if x.ndim == 1:
            params = mlp(x[None, :])
            params = jnp.squeeze(params, axis=0)
        else:
            params = mlp(x)
        return params
    return conditioner


def build_split_coupling_bijector(
        dim: int,
        identity_init: bool,
        mlp_units: Sequence[int],
        transform_type: str = 'real_nvp',
) -> BijectorWithExtra:

    if transform_type != 'real_nvp':
        raise NotImplementedError
        # TODO: implement spline transform

    split_index = dim // 2

    bijectors = []
    for swap in (True, False):
        params_after_split = dim - split_index
        params_transformed = params_after_split if swap else split_index
        conditioner_n_params_out = params_transformed*2

        conditioner = make_conditioner(f'splitcoupling_conditioner_swap{swap}', conditioner_n_params_out,
                                       mlp_units, identity_init)

        def bijector_fn(params: chex.Array) -> distrax.Bijector:
            scale_logit, shift = jnp.split(params, 2, axis=-1)
            scale = jax.nn.softplus(scale_logit + inverse_softplus(1.))
            return distrax.ScalarAffine(shift=shift, scale=scale)


        bijector = SplitCouplingWithExtra(
            split_index=split_index,
            event_ndims=1,
            conditioner=conditioner,
            bijector=bijector_fn,
            swap=swap,
            split_axis=-1
        )
        bijectors.append(bijector)

    return ChainWithExtra(bijectors)




