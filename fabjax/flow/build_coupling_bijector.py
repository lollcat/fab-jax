from typing import Sequence

import chex
import distrax
import jax.numpy as jnp

from fabjax.flow.distrax_with_extra import SplitCouplingWithExtra, ChainWithExtra, BijectorWithExtra
from fabjax.utils.nets import ConditionerMLP


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
        def conditioner(x: chex.Array) -> chex.Array:
            params = ConditionerMLP(name=f'splitcoupling_conditioner_swap{swap}', mlp_units=mlp_units,
                                    n_output_params=dim*2,
                                    zero_init=identity_init)(x)
            return params

        def bijector_fn(params: chex.Array) -> distrax.Bijector:
            log_scale, shift = jnp.split(params, 2, axis=-1)
            return distrax.ScalarAffine(shift=shift, log_scale=log_scale)


        bijector = SplitCouplingWithExtra(
            split_index=split_index,
            event_ndims=1,
            conditioner=conditioner,
            bijector=bijector_fn,
            swap=swap,
        )
        bijectors.append(bijector)

    return ChainWithExtra(bijectors)




