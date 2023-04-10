from typing import Callable, NamedTuple, Tuple

import chex
import jax.numpy as jnp
import jax.random
import optax

from fabjax.sampling.ais import AnnealedImportanceSampler, AISState
from fabjax.flow.flow import Flow, FlowParams

Params = chex.ArrayTree
LogProbFn = Callable[[chex.Array], chex.Array]
ParameterizedLogProbFn = Callable[[chex.ArrayTree, chex.Array], chex.Array]
Info = dict


def fab_loss_ais_samples(params, x: chex.Array, log_w: chex.Array, log_q_fn_apply: ParameterizedLogProbFn):
    """Estimate FAB loss with a batch of samples from AIS."""
    chex.assert_rank(log_w, 1)
    chex.assert_rank(x, 2)

    log_q = log_q_fn_apply(params, x)
    chex.assert_equal_shape((log_q, log_w))
    return - jnp.mean(jax.nn.softmax(log_w) * log_q)


class TrainStateNoBuffer(NamedTuple):
    flow_params: FlowParams
    key: chex.PRNGKey
    opt_state: optax.OptState
    ais_state: AISState


def build_fab_no_buffer_init_step_fns(flow: Flow, log_p_fn: LogProbFn,
                                      ais: AnnealedImportanceSampler, optimizer: optax.GradientTransformation,
                                      batch_size: int):

    def init(key: chex.PRNGKey) -> TrainStateNoBuffer:
        """Initialise the flow, optimizer and AIS states."""
        key1, key2, key3 = jax.random.split(key, 3)
        dummy_sample = jnp.zeros(flow.dim)
        flow_params = flow.init(key1, dummy_sample)
        opt_state = optimizer.init(flow_params)
        ais_state = ais.init(key2)
        return TrainStateNoBuffer(flow_params=flow_params, key=key3, opt_state=opt_state, ais_state=ais_state)

    @jax.jit
    @chex.assert_max_traces(4)
    def step(state: TrainStateNoBuffer) -> Tuple[TrainStateNoBuffer, Info]:
        key, subkey = jax.random.split(state.key)
        info = {}

        # Run Ais.
        def log_q_fn(x: chex.Array) -> chex.Array:
            return flow.log_prob_apply(state.flow_params, x)

        x0 = flow.sample_apply(state.flow_params, subkey, (batch_size,))
        point, log_w, ais_state, ais_info = ais.step(x0, state.ais_state, log_q_fn, log_p_fn)
        info.update(ais_info)

        # Estimate loss and update flow params.
        loss, grad = jax.value_and_grad(fab_loss_ais_samples)(state.flow_params, point.x, log_w, flow.log_prob_apply)
        updates, new_opt_state = optimizer.update(grad, state.opt_state, params=state.flow_params)
        new_params = optax.apply_updates(state.flow_params, updates)
        info.update(loss=loss)

        new_state = TrainStateNoBuffer(flow_params=new_params, key=key, opt_state=new_opt_state, ais_state=ais_state)
        return new_state, info

    return init, step
