from typing import Tuple, Callable, NamedTuple
from functools import partial

import chex
import jax.numpy as jnp
import jax.random
from jax.flatten_util import ravel_pytree


from fabjax.base import TransitionOperator, Point, get_intermediate_log_prob, get_grad_intermediate_log_prob, LogProbFn
from fabjax.mcmc.blackjax_hmc_rewrite import kernel as hmc_kernel, init as hmc_init


class HMCState(NamedTuple):
    key: chex.PRNGKey
    inverse_mass_maxtric: chex.Array


def build_blackjax_hmc(
                 dim: int,
                 n_outer_steps: int = 1,
                 n_inner_steps: int = 5,
                 init_step_size: float = 0.1,
                 alpha: float = 2.0,
) -> TransitionOperator:
    # We have an additional +1 to the dim for our dummy `change_detector` within `InternalMCMCPoint`

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

        key, subkey = jax.random.split(transition_operator_state.key)

        for i in range(n_outer_steps):
            key, subkey = jax.random.split(key)
            hmc_state, info = one_step(
                key,
                hmc_state,
                log_q_fn=log_q_fn,
                log_p_fn=log_p_fn,
                step_size=init_step_size,
                inverse_mass_matrix=transition_operator_state.inverse_mass_maxtric,
                num_integration_steps=n_inner_steps)

        info = {f"mean_acceptance_rate": jnp.mean(info.acceptance_rate)}
        point_kwargs = hmc_state._asdict()
        del(point_kwargs['beta'])
        point_kwargs["x"] = point_kwargs["position"]
        del(point_kwargs['position'])
        point = Point(**point_kwargs)
        return point, transition_operator_state, info

    return TransitionOperator(init, step)


if __name__ == '__main__':
    hmc_transition_operator = build_blackjax_hmc(dim=2)


    log_q_fn = lambda x: jnp.sum(x**2)
    log_p_fn = lambda x: jnp.sum(x**3)

    x = Point(jnp.ones((2,)), jnp.ones(()), jnp.ones(()), jnp.ones((2,)),
                          jnp.ones((2,)))

    key = jax.random.PRNGKey(0)

    hmc_state = hmc_transition_operator.init(key)

    x, hmc_state, info = hmc_transition_operator.step(x, hmc_state, 1.0, log_q_fn, log_p_fn)
    print(x)