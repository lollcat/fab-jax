"""Translate bijector from
https://github.com/google-deepmind/annealed_flow_transport to our interface."""
from typing import Callable

import chex
import distrax
import jax.numpy as jnp

from fabjax.flow.conv_flow.flow import ConfigurableFlow

Array = chex.Array

class FlowAFT(distrax.Bijector):
    def __init__(self, build_aft_flow: Callable[[], ConfigurableFlow]):
        super().__init__(event_ndims_in=1, event_ndims_out=1)
        self._build_aft_flow = build_aft_flow

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        if x.ndim == 2:
            return self._build_aft_flow().__call__(x)
        else:
            assert x.ndim == 1
            y, log_det = self._build_aft_flow().__call__(x[None])
            return jnp.squeeze(y), jnp.squeeze(log_det)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        if y.ndim == 2:
            return self._build_aft_flow().inverse(y)
        else:
            assert y.ndim == 1
            x, log_det = self._build_aft_flow().inverse(y[None])
            return jnp.squeeze(x), jnp.squeeze(log_det)
