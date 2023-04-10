from typing import NamedTuple, Tuple
from functools import partial

import chex
import jax.numpy as jnp
import jax
from typing import Callable

from fabjax.sampling.base import TransitionOperator, LogProbFn, create_point, Point, get_intermediate_log_prob
from fabjax.sampling.resampling import log_effective_sample_size
from fabjax.utils.jax_util import broadcasted_where


IntermediateLogProb = Callable[[chex.Array, int], chex.Array]


class AISState(NamedTuple):
    transition_operator_state: chex.ArrayTree
    key: chex.PRNGKey


AisStepFn = Callable[[chex.Array, AISState, LogProbFn, LogProbFn], Tuple[Point, chex.Array, AISState, dict]]

class AnnealedImportanceSampler(NamedTuple):
    init: Callable[[chex.PRNGKey], AISState]
    step: AisStepFn
    transition_operator: TransitionOperator
    betas: chex.Array
    alpha: float


def log_weight_contribution_point(point: Point, ais_step_index: int, betas: chex.Array, alpha: float):
    """AIS step index is between 0 and n_intermediate_distributions."""
    log_numerator = get_intermediate_log_prob(point.log_q, point.log_p, betas[ais_step_index + 1], alpha)
    log_denominator = get_intermediate_log_prob(point.log_q, point.log_p, betas[ais_step_index], alpha)
    return log_numerator - log_denominator


def ais_inner_transition(point: Point, log_w: chex.Array, trans_op_state: chex.Array, betas: chex.Array,
                        ais_step_index: int, transition_operator: TransitionOperator,
                        log_q_fn: LogProbFn, log_p_fn: LogProbFn, alpha: float) -> \
        Tuple[Tuple[Point, chex.Array], Tuple[chex.ArrayTree, dict]]:
    """Perform inner iteration of AIS, incrementing the log_w appropriately."""
    beta = betas[ais_step_index]

    new_point, trans_op_state, info = transition_operator.step(point, trans_op_state, beta, alpha, log_q_fn, log_p_fn)

    # Remove invalid samples.
    valid_samples = jnp.isfinite(new_point.log_q) & jnp.isfinite(new_point.log_p) & \
                    jnp.alltrue(jnp.isfinite(new_point.x), axis=-1)
    point = jax.tree_map(lambda a, b: broadcasted_where(valid_samples, a, b), new_point, point)

    log_w = log_w + log_weight_contribution_point(point, ais_step_index, betas, alpha)

    return (point, log_w), (trans_op_state, info)


def replace_invalid_samples_with_valid_ones(point: Point, key: chex.PRNGKey) -> Point:
    """Replace invalid samples in the point with valid ones (where valid ones are sampled uniformly)."""
    valid_samples = jnp.isfinite(point.log_q) & jnp.isfinite(point.log_p) & \
                    jnp.alltrue(jnp.isfinite(point.x), axis=-1)
    p = jnp.where(valid_samples, jnp.ones_like(valid_samples), jnp.zeros_like(valid_samples))
    indices = jax.random.choice(key, jnp.arange(valid_samples.shape[0]), p=p, shape=valid_samples.shape)
    alt_points = jax.tree_map(lambda x: x[indices], point)

    # Replace invalid samples with valid samples
    point = jax.tree_map(lambda a, b: broadcasted_where(valid_samples, a, b), point, alt_points)
    return point


def build_ais(
        transition_operator: TransitionOperator,
        n_intermediate_distributions: int,
        spacing_type: str = 'linear',
        alpha: float = 2.
              ):
    scan = True

    if spacing_type == "geometric":
        # rough heuristic, copying ratio used in example in AIS paper.
        # One quarter of Beta linearly spaced between 0 and 0.01
        n_intermediate_linspace_points = int(n_intermediate_distributions / 4)
        # The rest geometrically spaced between 0.01 and 1.0
        n_intermediate_geomspace_points = n_intermediate_distributions - \
                                          n_intermediate_linspace_points - 1
        betas = jnp.concatenate([jnp.linspace(0, 0.01, n_intermediate_linspace_points + 2)[:-1],
                                  jnp.geomspace(0.01, 1, n_intermediate_geomspace_points + 2)])
    elif spacing_type == "linear":
        betas = jnp.linspace(0.0, 1.0, n_intermediate_distributions + 2)
    else:
        raise NotImplementedError


    def init(key: chex.PRNGKey) -> chex.ArrayTree:
        key1, key2 = jax.random.split(key)
        trans_op_state = transition_operator.init(key1)
        trans_op_state = jax.tree_map(lambda x: jnp.repeat(x[None, ...], n_intermediate_distributions, axis=0),
                                      trans_op_state)
        return AISState(trans_op_state, key2)


    def step(x0: chex.Array, ais_state: AISState, log_q_fn: LogProbFn, log_p_fn: LogProbFn) -> \
            Tuple[Point, chex.Array, AISState, dict]:
        chex.assert_rank(x0, 2)  # [batch_size, dim]
        info = {}

        # Setup start of AIS chain.
        point0 = jax.vmap(
            partial(create_point, log_q_fn=log_q_fn, log_p_fn=log_p_fn, with_grad=transition_operator.uses_grad)
        )(x0)

        # Sometimes the flow produces nan samples - remove these.
        key, subkey = jax.random.split(ais_state.key)
        point0 = replace_invalid_samples_with_valid_ones(point0, key)

        log_w = log_weight_contribution_point(point0, 0, betas, alpha)

        # Run MCMC from point0 sampling from pi_0 to point_n generated by MCMC targetting pi_{n-1}.

        if scan:
            # Setup scan body function.
            def body_fn(carry: Point, xs: Tuple[chex.ArrayTree, int]) -> \
                    Tuple[Tuple[Point, chex.Array], Tuple[chex.ArrayTree, dict]]:
                point, log_w = carry
                trans_op_state, ais_step_index = xs
                (point, log_w), (trans_op_state, info) = ais_inner_transition(point, log_w, trans_op_state, betas,
                                                                              ais_step_index,
                                                                              transition_operator, log_q_fn, log_p_fn,
                                                                              alpha)
                return (point, log_w), (trans_op_state, info)

            # Run scan.
            per_step_inputs = (ais_state.transition_operator_state, jnp.arange(n_intermediate_distributions) + 1)
            scan_init = (point0, log_w)
            (point, log_w), (trans_op_states, infos) = jax.lax.scan(body_fn, init=scan_init, xs=per_step_inputs,
                                                                    length=n_intermediate_distributions)

        else:
            # Loop version. Useful for debugging.
            trans_op_states = []
            infos = []
            point = point0
            for ais_step_index in jnp.arange(n_intermediate_distributions) + 1:
                trans_op_state = jax.tree_map(lambda x: x[ais_step_index-1], ais_state.transition_operator_state)
                (point, log_w), (trans_op_state, info) = ais_inner_transition(point, log_w, trans_op_state, betas,
                                                                              ais_step_index,
                                                                              transition_operator, log_q_fn, log_p_fn,
                                                                              alpha)
                trans_op_states.append(trans_op_state)
                infos.append(info)
            trans_op_states = jax.tree_map(lambda *xs: jnp.stack(xs), *trans_op_states)
            infos = jax.tree_map(lambda *xs: jnp.stack(xs), *infos)


        chex.assert_trees_all_equal_structs(ais_state.transition_operator_state, trans_op_states)
        ais_state = AISState(transition_operator_state=trans_op_states, key=key)

        # Info for logging.
        log_ess_q_p = log_effective_sample_size(point0.log_p - point0.log_q)
        log_ess_ais = log_effective_sample_size(log_w)
        info.update(log_ess_q_p=log_ess_q_p, log_ess_ais=log_ess_ais,
                    ess_q_p=jnp.exp(log_ess_q_p), ess_ais=jnp.exp(log_ess_ais))
        for i in range(n_intermediate_distributions):
            info.update({f"dist{i+1}_" + key: value for key, value in jax.tree_map(lambda x: x[i], infos).items()})

        return point, log_w, ais_state, info


    return AnnealedImportanceSampler(init=init, step=step, betas=betas,
                                     transition_operator=transition_operator, alpha=alpha)

