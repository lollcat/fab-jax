from typing import Union, Tuple, Callable, Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np
from fabjax.sampling.smc import SequentialMonteCarloSampler
from fabjax.sampling.resampling import log_effective_sample_size
from fabjax.flow.flow import Flow
from fabjax.train.fab_with_buffer import TrainStateWithBuffer
from fabjax.train.fab_without_buffer import TrainStateNoBuffer


def setup_fab_eval_function(
      flow: Flow,
      ais: SequentialMonteCarloSampler,
      log_p_x: Callable[[chex.Array], chex.Array],
      eval_n_samples: int,
      inner_batch_size: int,
      log_Z_true: Optional[float] =  None
) -> Callable[[chex.ArrayTree, chex.PRNGKey], dict]:
    """Evaluate the ESS of the flow, and AIS. """
    assert ais.alpha == 1.  # Make sure target is set to p.
    assert ais.use_resampling is False  # Make sure we are doing AIS, not SMC.


    def eval_fn(state: Union[TrainStateNoBuffer, TrainStateWithBuffer], key: chex.PRNGKey) -> dict:
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
        n_batches = np.ceil(eval_n_samples / inner_batch_size).astype(int)
        _, (log_w_flow, log_w_ais) = jax.lax.scan(inner_fn, init=None, xs=jax.random.split(key, n_batches))

        # Ensure correct number of samples used for estimate.
        log_w_ais, log_w_flow = log_w_ais.flatten()[:eval_n_samples], log_w_flow.flatten()[:eval_n_samples]

        # Compute estimate.
        log_z_flow = jax.nn.logsumexp(log_w_flow) - jnp.log(eval_n_samples)
        log_z_ais = jax.nn.logsumexp(log_w_ais) - jnp.log(eval_n_samples)

        # Compute metrics
        info = {}
        info.update(
            eval_ess_flow=jnp.exp(log_effective_sample_size(log_w_flow)),
            eval_ess_ais=jnp.exp(log_effective_sample_size(log_w_ais)),
                    )
        if log_Z_true is not None:
            info.update(abs_err_log_z_flow=jnp.abs(log_z_flow - log_Z_true),
                        abs_err_log_z_ais=jnp.abs(log_z_ais - log_Z_true))
        else:
            info.update(log_z_flow=log_z_flow, log_z_ais=log_z_ais)
        return info

    return eval_fn


def calculate_log_forward_ess(
        log_w: chex.Array,
        mask: Optional[chex.Array] = None,
        log_Z: Optional[float] = None
) -> chex.Array:

    """Calculate forward ess.
    log_w = p(x)/q(x), where x ~ p(x).
    This can be passed as the `further_fn` to `eacf.train.base.eval_fn`."""
    if mask is None:
        mask = jnp.ones_like(log_w)

    chex.assert_equal_shape((log_w, mask))
    log_w = jnp.where(mask, log_w, jnp.zeros_like(log_w))  # make sure log_w finite

    if log_Z is None:
        log_z_inv = jax.nn.logsumexp(-log_w, b=mask) - jnp.log(jnp.sum(mask))
    else:
        log_z_inv = - log_Z

    log_z_expectation_p_over_q = jax.nn.logsumexp(log_w, b=mask) - jnp.log(jnp.sum(mask))
    log_forward_ess = - log_z_inv - log_z_expectation_p_over_q
    return log_forward_ess
