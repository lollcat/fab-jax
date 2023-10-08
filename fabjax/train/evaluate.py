from typing import Union, Tuple, Callable

import chex
import jax
import jax.numpy as jnp
from fabjax.sampling.smc import SequentialMonteCarloSampler
from fabjax.sampling.resampling import log_effective_sample_size
from fabjax.flow.flow import Flow, FlowParams
from fabjax.train.fab_with_buffer import TrainStateWithBuffer
from fabjax.train.fab_without_buffer import TrainStateNoBuffer


def setup_fab_eval_function(
                      flow: Flow,
                      ais: SequentialMonteCarloSampler,
                      log_p_x,
                      batch_size: int,
                      inner_batch_size: int
) -> Callable[[chex.ArrayTree, chex.PRNGKey], dict]:
    """Evaluate the ESS of the flow, and AIS. """
    assert ais.alpha == 1.  # Make sure target is set to p.
    assert ais.use_resampling is False  # Make sure we are doing AIS, not SMC.

    
    @jax.jit
    def eval_fn(state: Union[TrainStateNoBuffer, TrainStateWithBuffer], key: chex.PRNGKey):
        def log_q_fn(x: chex.Array) -> chex.Array:
            return flow.log_prob_apply(state.flow_params, x)
        
        def inner_fn(carry: None, xs: chex.PRNGKey) -> Tuple[None, Tuple[chex.Array, chex.Array]]:
            """Perform SMC forward pass and grab just the importance weights."""
            key = xs
            x0, log_q_flow = flow.sample_and_log_prob_apply(state.flow_params, key, (inner_batch_size,))
            point, log_w, smc_state, smc_info = ais.step(x0, state.smc_state, log_q_fn, log_p_x)
            log_w_flow = log_p_x(x0) - log_q_flow
            return None, (log_w_flow, log_w)
        
        # Run scan function.
        n_batches = (batch_size // inner_batch_size) + 1
        _, (log_w_flow, log_w_ais) = jax.lax.scan(inner_fn, init=None, xs=jax.random.split(key, n_batches))

        n_samples = n_batches * inner_batch_size
        assert n_samples == log_w_ais.flatten().shape[0]

        # Compute metrics
        info = {}
        info.update(eval_ess_flow=jnp.exp(log_effective_sample_size(log_w_flow.flatten())),
                    eval_ess_ais=jnp.exp(log_effective_sample_size(log_w_ais.flatten())),
                    log_z_flow=jax.nn.logsumexp(log_w_flow.flatten()) - jnp.log(n_samples),
                    log_z_ais=jax.nn.logsumexp(log_w_ais.flatten()) - jnp.log(n_samples)
                    )
        return info

    return eval_fn
