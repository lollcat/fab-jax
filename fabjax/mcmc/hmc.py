from typing import Tuple, Callable, NamedTuple, Optional
from functools import partial

import chex
import jax.numpy as jnp
import jax.random
from blackjax.adaptation.step_size import dual_averaging_adaptation


from fabjax.base import TransitionOperator, Point, LogProbFn
from fabjax.mcmc.blackjax_hmc_rewrite import kernel as hmc_kernel, init as hmc_init


class HMCState(NamedTuple):
    key: chex.PRNGKey
    inverse_mass_maxtric: chex.Array
    adaption_state: Optional[chex.ArrayTree] = None


def build_blackjax_hmc(
                 dim: int,
                 n_outer_steps: int = 1,
                 n_inner_steps: int = 5,
                 init_step_size: float = 1e-4,
                 adapt_step_size: bool = True,
                 alpha: float = 2.0,
                 target_p_accept: float = 0.65
) -> TransitionOperator:
    # TODO: Use alpha as an argument.

    one_step = hmc_kernel(divergence_threshold=1000)

    if adapt_step_size:
        adapt_init, adapt_update, _ = dual_averaging_adaptation(target_p_accept, )


    def init(key: chex.PRNGKey) -> HMCState:
        inverse_mass_matrix = jnp.ones(dim)
        adaption_state = adapt_init(init_step_size) if adapt_step_size else None
        return HMCState(key, inverse_mass_matrix, adaption_state)

    def step(point: Point,
             transition_operator_state: HMCState,
             beta: chex.Array,
             log_q_fn: LogProbFn,
             log_p_fn: LogProbFn,
             ) -> \
            Tuple[Point, chex.ArrayTree, dict]:

        chex.assert_rank(point.x, 2)
        batch_size = point.x.shape[0]

        hmc_state = jax.vmap(hmc_init, in_axes=(0, None))(point, beta)

        def scan_fn(body, xs):
            key = xs
            key_batch = jax.random.split(key, batch_size)
            hmc_state, transition_operator_state = body
            step_size = jnp.exp(transition_operator_state.adaption_state.log_step_size) if adapt_step_size else \
                init_step_size
            step_fn_partial = partial(one_step,
                                      log_q_fn=log_q_fn,
                                        log_p_fn=log_p_fn,
                                        step_size=step_size,
                                        inverse_mass_matrix=transition_operator_state.inverse_mass_maxtric,
                                        num_integration_steps=n_inner_steps)
            hmc_state, info = jax.vmap(step_fn_partial)(key_batch, hmc_state)
            if adapt_step_size:
                new_adaption_state = adapt_update(transition_operator_state.adaption_state,
                                                  jnp.mean(info.acceptance_rate))
                transition_operator_state = transition_operator_state._replace(adaption_state=new_adaption_state)
            return (hmc_state, transition_operator_state), info


        key, subkey = jax.random.split(transition_operator_state.key)
        (hmc_state, transition_operator_state), infos = jax.lax.scan(
            scan_fn,
            (hmc_state, transition_operator_state),
            jax.random.split(subkey, n_outer_steps))

        info = {f"mean_acceptance_rate": jnp.mean(infos.acceptance_rate)}
        point_kwargs = hmc_state._asdict()
        del(point_kwargs['beta'])
        point_kwargs["x"] = point_kwargs["position"]
        del(point_kwargs['position'])
        point = Point(**point_kwargs)

        transition_operator_state = transition_operator_state._replace(key=key)
        return point, transition_operator_state, info

    return TransitionOperator(init, step)
