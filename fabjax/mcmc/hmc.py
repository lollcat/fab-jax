from typing import Tuple, Callable, NamedTuple
from functools import partial

import chex
import jax.numpy as jnp
import jax.random


from fabjax.base import TransitionOperator, Point, LogProbFn
from fabjax.mcmc.blackjax_hmc_rewrite import kernel as hmc_kernel, init as hmc_init


class HMCState(NamedTuple):
    key: chex.PRNGKey
    inverse_mass_maxtric: chex.Array


def build_blackjax_hmc(
                 dim: int,
                 n_outer_steps: int = 1,
                 n_inner_steps: int = 5,
                 init_step_size: float = 1e-4,
                 alpha: float = 2.0,
) -> TransitionOperator:
    # TODO: Use alpha as an argument.

    one_step = hmc_kernel(divergence_threshold=1000)


    def init(key: chex.PRNGKey) -> HMCState:
        inverse_mass_matrix = jnp.ones(dim)
        return HMCState(key, inverse_mass_matrix)

    def step(point: Point,
             transition_operator_state: HMCState,
             beta: chex.Array,
             log_q_fn: LogProbFn,
             log_p_fn: LogProbFn,
             ) -> \
            Tuple[Point, chex.ArrayTree, dict]:

        hmc_state = hmc_init(point, beta)

        def scan_fn(body, xs):
            key = xs
            hmc_state = body
            hmc_state, info = one_step(
                key,
                hmc_state,
                log_q_fn=log_q_fn,
                log_p_fn=log_p_fn,
                step_size=init_step_size,
                inverse_mass_matrix=transition_operator_state.inverse_mass_maxtric,
                num_integration_steps=n_inner_steps)
            return hmc_state, info

        key, subkey = jax.random.split(transition_operator_state.key)
        hmc_state, infos = jax.lax.scan(scan_fn, hmc_state, jax.random.split(subkey, n_outer_steps))

        info = {f"mean_acceptance_rate": jnp.mean(infos.acceptance_rate)}
        point_kwargs = hmc_state._asdict()
        del(point_kwargs['beta'])
        point_kwargs["x"] = point_kwargs["position"]
        del(point_kwargs['position'])
        point = Point(**point_kwargs)

        transition_operator_state = transition_operator_state._replace(key=key)
        return point, transition_operator_state, info

    return TransitionOperator(init, step)
