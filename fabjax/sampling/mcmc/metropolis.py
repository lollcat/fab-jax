from typing import Tuple, NamedTuple, Optional
from functools import partial

import chex
import jax.numpy as jnp
import jax.random


from fabjax.sampling.base import TransitionOperator, Point, LogProbFn, get_intermediate_log_prob, create_point


class MetropolisState(NamedTuple):
    key: chex.PRNGKey
    step_size: chex.Array


def build_metropolis(
                 dim: int,
                 n_steps: int = 1,
                 init_step_size: float = 1.
) -> TransitionOperator:

    def init(key: chex.PRNGKey) -> MetropolisState:
        return MetropolisState(key, jnp.array(init_step_size*dim))


    def step(point: Point,
             transition_operator_state: MetropolisState,
             beta: chex.Array,
             alpha: float,
             log_q_fn: LogProbFn,
             log_p_fn: LogProbFn,
             ) -> \
            Tuple[Point, MetropolisState, dict]:

        chex.assert_rank(point.x, 2)
        batch_size = point.x.shape[0]

        def one_step(point: Point, key) -> Tuple[Point, dict]:
            chex.assert_rank(point.x, 1)
            key1, key2, key3, key4 = jax.random.split(key, 4)

            new_x = point.x + jax.random.normal(key1, shape=point.x.shape) * transition_operator_state.step_size
            new_point = create_point(new_x, log_q_fn=log_q_fn, log_p_fn=log_p_fn, with_grad=False)

            log_p_accept = get_intermediate_log_prob(new_point.log_q, new_point.log_p, beta=beta, alpha=alpha) -\
                           get_intermediate_log_prob(point.log_q, point.log_p, beta=beta, alpha=alpha)
            log_threshold = jax.random.exponential(key3)

            accept = (log_p_accept > log_threshold) & jnp.isfinite(new_point.log_q) & jnp.isfinite(new_point.log_p)
            point = jax.lax.cond(accept, lambda p_new, p: p_new, lambda p_new, p: p, new_point, point)
            info = {"p_accept": jnp.clip(jnp.exp(log_p_accept), a_max=1)}
            return point, info


        def scan_fn(body, xs):
            key = xs
            key_batch = jax.random.split(key, batch_size)
            point = body
            point, info = jax.vmap(one_step)(point, key_batch)
            info = jax.tree_map(jnp.mean, info)
            return point, info


        key, subkey = jax.random.split(transition_operator_state.key)
        point, infos = jax.lax.scan(scan_fn, point, jax.random.split(subkey, n_steps))

        # Info for logging
        info = jax.tree_map(jnp.mean, infos)
        transition_operator_state = transition_operator_state._replace(key=key)
        return point, transition_operator_state, info

    return TransitionOperator(init, step, uses_grad=False)

