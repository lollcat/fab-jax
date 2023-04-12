from typing import Callable, NamedTuple, Tuple

import chex
import jax.numpy as jnp
import jax.random
import optax

from fabjax.sampling.smc import SequentialMonteCarloSampler, SMCState
from fabjax.flow.flow import Flow, FlowParams
from fabjax.buffer import PrioritisedBuffer, PrioritisedBufferState

Params = chex.ArrayTree
LogProbFn = Callable[[chex.Array], chex.Array]
ParameterizedLogProbFn = Callable[[chex.ArrayTree, chex.Array], chex.Array]
Info = dict


def fab_loss_buffer_samples(
        params: FlowParams,
        x: chex.Array,
        log_q_old: chex.Array,
        alpha: chex.Array,
        log_q_fn_apply: ParameterizedLogProbFn) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
    """Estimate FAB loss with a batch of samples from the prioritized replay buffer."""
    chex.assert_rank(x, 2)
    chex.assert_rank(log_q_old, 1)

    log_q = log_q_fn_apply(params, x)
    log_w_adjust = (1 - alpha) * (jax.lax.stop_gradient(log_q) - log_q_old)
    chex.assert_equal_shape((log_q, log_w_adjust))
    return - jnp.mean(jnp.exp(log_w_adjust) * log_q), (log_w_adjust, log_q)


class TrainStateWithBuffer(NamedTuple):
    flow_params: FlowParams
    key: chex.PRNGKey
    opt_state: optax.OptState
    smc_state: SMCState
    buffer_state: PrioritisedBufferState


def build_fab_with_buffer_init_step_fns(
        flow: Flow,
        log_p_fn: LogProbFn,
        smc: SequentialMonteCarloSampler,
        buffer: PrioritisedBuffer,
        optimizer: optax.GradientTransformation,
        batch_size: int,
        n_updates_per_smc_forward_pass: int,
        alpha: float = 2.
):
    assert smc.alpha == alpha

    def init(key: chex.PRNGKey) -> TrainStateWithBuffer:
        """Initialise the flow, optimizer, SMC and buffer states."""
        key1, key2, key3, key4 = jax.random.split(key, 4)
        dummy_sample = jnp.zeros(flow.dim)
        flow_params = flow.init(key1, dummy_sample)
        opt_state = optimizer.init(flow_params)
        smc_state = smc.init(key2)

        # Now run multiple forward passes of SMC to fill the buffer. This also
        # tunes the SMC state in the process.
        def log_q_fn(x: chex.Array) -> chex.Array:
            return flow.log_prob_apply(flow_params, x)

        def body_fn(carry, xs):
            """fer."""
            smc_state = carry
            key = xs
            x0 = flow.sample_apply(flow_params, key, (batch_size,))
            chex.assert_rank(x0, 2)  # Currently written assuming x only has 1 event dimension.
            point, log_w, smc_state, smc_info = smc.step(x0, smc_state, log_q_fn, log_p_fn)
            return smc_state, (point.x, log_w, point.log_q)

        n_forward_pass = (buffer.min_lengtht_to_sample // batch_size) + 1
        smc_state, (x, log_w, log_q) = jax.lax.scan(body_fn, init=smc_state,
                                                    xs=jax.random.split(key4, n_forward_pass))

        buffer_state = buffer.init(jnp.reshape(x, (n_forward_pass*batch_size, flow.dim)),
                                               log_w.flatten(),
                                               log_q.flatten())

        return TrainStateWithBuffer(flow_params=flow_params, key=key3, opt_state=opt_state,
                                    smc_state=smc_state, buffer_state=buffer_state)

    def one_gradient_update(carry: Tuple[FlowParams, optax.OptState], xs: Tuple[chex.Array, chex.Array]):
        """Perform on update to the flow parameters with a batch of data from the buffer."""
        flow_params, opt_state = carry
        x, log_q_old = xs
        info = {}

        # Estimate loss and update flow params.
        (loss, (log_w_adjust, log_q)), grad = jax.value_and_grad(fab_loss_buffer_samples, has_aux=True)(
            flow_params, x, log_q_old, alpha, flow.log_prob_apply)
        updates, new_opt_state = optimizer.update(grad, opt_state, params=flow_params)
        new_params = optax.apply_updates(flow_params, updates)
        info.update(loss=loss)
        return (new_params, new_opt_state), (info, log_w_adjust, log_q)

    @jax.jit
    @chex.assert_max_traces(4)
    def step(state: TrainStateWithBuffer) -> Tuple[TrainStateWithBuffer, Info]:
        """Perform a single iteration of the FAB algorithm."""
        info = {}

        # Sample from buffer.
        key, subkey = jax.random.split(state.key)
        x_buffer, log_q_old_buffer, indices = buffer.sample_n_batches(subkey, state.buffer_state, batch_size,
                                                                      n_updates_per_smc_forward_pass)
        # Perform sgd steps on flow.
        (new_flow_params, new_opt_state), (infos, log_w_adjust, log_q_old) = jax.lax.scan(
            one_gradient_update, init=(state.flow_params, state.opt_state), xs=(x_buffer, log_q_old_buffer),
            length=n_updates_per_smc_forward_pass
        )
        # Adjust samples in the buffer.
        buffer_state = buffer.adjust(log_q=log_q_old.flatten(), log_w_adjustment=log_w_adjust.flatten(),
                                     indices=indices.flatten(),
                                     buffer_state=state.buffer_state)
        # Update info.
        for i in range(n_updates_per_smc_forward_pass):
            info.update(jax.tree_map(lambda x: x[i], infos))

        # Run smc and add samples to the buffer. Note this is done with the flow params before they were updated so that
        # this can occur in parallel (jax will do this after compilation).
        def log_q_fn(x: chex.Array) -> chex.Array:
            return flow.log_prob_apply(state.flow_params, x)

        key, subkey = jax.random.split(key)
        x0 = flow.sample_apply(state.flow_params, subkey, (batch_size,))
        chex.assert_rank(x0, 2)  # Currently written assuming x only has 1 event dimension.
        point, log_w, smc_state, smc_info = smc.step(x0, state.smc_state, log_q_fn, log_p_fn)
        info.update(smc_info)
        buffer_state = buffer.add(x=point.x, log_w=log_w, log_q=point.log_q, buffer_state=buffer_state)

        new_state = TrainStateWithBuffer(flow_params=new_flow_params, key=key, opt_state=new_opt_state,
                                         smc_state=smc_state, buffer_state=buffer_state)
        return new_state, info

    return init, step
