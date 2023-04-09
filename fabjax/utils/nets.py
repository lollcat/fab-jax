from typing import Sequence

import haiku as hk
import jax.numpy as jnp

class ConditionerMLP(hk.Module):
    """Used for coupling flow."""
    def __init__(self, name: str, mlp_units: Sequence[int], n_output_params: int, zero_init: bool):
        super().__init__(name=name)
        mlp_components = [
                hk.nets.MLP(mlp_units, activate_final=True),
                hk.Linear(n_output_params, b_init=jnp.zeros, w_init=jnp.zeros) if zero_init else
                hk.Linear(n_output_params,
                          b_init=hk.initializers.VarianceScaling(0.01),
                          w_init=hk.initializers.VarianceScaling(0.01))
            ]
        self.mlp_function = hk.Sequential(mlp_components)

    def __call__(self, params):
        return self.mlp_function(params)
