"""Like SMC but with for loop to make debugging easier."""

from typing import Tuple
from functools import partial

import chex
import jax.numpy as jnp
import jax

from fabjax.sampling.base import TransitionOperator, LogProbFn, create_point, Point
from fabjax.sampling.resampling import log_effective_sample_size, optionally_resample
from fabjax.sampling.smc import SequentialMonteCarloSampler, SMCState, replace_invalid_samples_with_valid_ones,\
    get_intermediate_log_prob, broadcasted_where
    # log_weight_contribution_point, ais_inner_transition


def log_weight_contribution_point(point: Point, ais_step_index: int, betas: chex.Array, alpha: float):
    """Calculate a points contribution to the SMC log weights.
    AIS step index is between 0 and n_intermediate_distributions."""
    chex.assert_rank(betas, 1)
    chex.assert_rank(alpha, 0)

    log_numerator = get_intermediate_log_prob(
        log_q=point.log_q, log_p=point.log_p, beta=betas[ais_step_index + 1], alpha=alpha)
    log_denominator = get_intermediate_log_prob(
        log_q=point.log_q, log_p=point.log_p, beta=betas[ais_step_index], alpha=alpha)
    return log_numerator - log_denominator


def ais_inner_transition(point: Point, log_w: chex.Array, trans_op_state: chex.Array, betas: chex.Array,
                        ais_step_index: int, transition_operator: TransitionOperator,
                        log_q_fn: LogProbFn, log_p_fn: LogProbFn, alpha: float) -> \
        Tuple[Tuple[Point, chex.Array], Tuple[chex.ArrayTree, dict]]:
    """Perform inner iteration of AIS, incrementing the log_w appropriately."""
    chex.assert_rank(betas, 1)
    chex.assert_rank(alpha, 0)
    chex.assert_rank(point.x, 2)
    chex.assert_rank(log_w, 1)

    beta = betas[ais_step_index]

    new_point, trans_op_state, info = transition_operator.step(
        point=point, transition_operator_state=trans_op_state,
        beta=beta, alpha=alpha, log_q_fn=log_q_fn, log_p_fn=log_p_fn)

    # assert (new_point.log_q == log_q_fn(new_point.x)).all()
    # assert (new_point.log_p == log_p_fn(new_point.x)).all()

    # Remove invalid samples.
    valid_samples = jnp.isfinite(new_point.log_q) & jnp.isfinite(new_point.log_p) & \
                    jnp.alltrue(jnp.isfinite(new_point.x), axis=-1)
    info.update(n_valid_samples = jnp.sum(valid_samples))
    new_point = jax.tree_map(lambda a, b: broadcasted_where(valid_samples, a, b), new_point, point)

    log_w_increment = log_weight_contribution_point(new_point, ais_step_index, betas, alpha)
    chex.assert_equal_shape((log_w_increment, log_w))
    log_w = log_w + log_w_increment

    return (new_point, log_w), (trans_op_state, info)

def build_smc(
        transition_operator: TransitionOperator,
        n_intermediate_distributions: int,
        spacing_type: str = 'linear',
        alpha: float = 2.,
        use_resampling: bool = False,
        resampling_threshold: float = 0.3,
        verbose: bool = False
              ) -> SequentialMonteCarloSampler:
    if spacing_type == "geometric":
        # Rough heuristic, copying ratio used in example in AIS paper.
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


    def init(key: chex.PRNGKey) -> SMCState:
        """Initialise the state of the SMC sampler."""
        key1, key2 = jax.random.split(key)
        trans_op_state = transition_operator.init(key1)
        trans_op_state = jax.tree_map(lambda x: jnp.repeat(x[None, ...], n_intermediate_distributions, axis=0),
                                      trans_op_state)
        return SMCState(trans_op_state, key2)


    def step(x0: chex.Array, smc_state: SMCState, log_q_fn: LogProbFn, log_p_fn: LogProbFn) -> \
            Tuple[Point, chex.Array, SMCState, dict]:
        """
        Run the SMC forward pass.

        Args:
            x0: Samples from `q` for initialising the SMC chain.
            smc_state: State of the SMC sampler. Contains the parameters for the transition operator.
            log_q_fn: Log density of the base distribution (typically the flow being trained).
            log_p_fn: Log density of the target distribution that we wish to approximate with the distribution `q`.

        Returns:
            point: Final point in the SMC forward pass.
            log_w: Unnormalized log importance weights.
            smc_state: Updated SMC state.
            info: Info for diagnostics/logging.
        """
        chex.assert_rank(x0, 2)  # [batch_size, dim]
        info = {}

        # Setup start of AIS chain.
        point0 = jax.vmap(
            partial(create_point, log_q_fn=log_q_fn, log_p_fn=log_p_fn, with_grad=transition_operator.uses_grad)
        )(x0)

        # Sometimes the flow produces nan samples - remove these.
        key, subkey = jax.random.split(smc_state.key)
        point0 = replace_invalid_samples_with_valid_ones(point0, subkey)
        # chex.assert_trees_all_equal(replace_invalid_samples_with_valid_ones(point0, subkey).x, point0.x)

        log_w_init = log_weight_contribution_point(point0, 0, betas=betas, alpha=alpha)
        # log_w_init == log_p_fn(point0.x) - log_q_fn(point0.x)
        log_w = log_w_init

        point = point0
        key, subkey = jax.random.split(key)
        per_step_inputs = (jax.random.split(subkey, n_intermediate_distributions),
                           smc_state.transition_operator_state,
                           jnp.arange(n_intermediate_distributions) + 1)
        trans_op_states = []
        infos = []
        for i in jnp.arange(n_intermediate_distributions):
                info = {}
                key, trans_op_state, ais_step_index = jax.tree_map(lambda x: x[i], per_step_inputs)
                assert ais_step_index == (i + 1)
                if use_resampling:
                    point, log_w, log_ess = optionally_resample(key=key, log_weights=log_w, samples=point,
                                                       resample_threshold=resampling_threshold)
                    info.update(ess=jnp.exp(log_ess))
                (point, log_w), (trans_op_state, info_transition) = ais_inner_transition(
                    point=point, log_w=log_w, trans_op_state=trans_op_state, betas=betas,
                    ais_step_index=ais_step_index,
                    transition_operator=transition_operator, log_q_fn=log_q_fn, log_p_fn=log_p_fn,
                    alpha=alpha)
                info.update(info_transition)

                infos.append(info)
                trans_op_states.append(trans_op_state)
        if n_intermediate_distributions != 0:
            infos = jax.tree_map(lambda *x: jnp.stack(x), *infos)
            trans_op_states = jax.tree_map(lambda *x: jnp.stack(x), *trans_op_states)

            chex.assert_trees_all_equal_structs(smc_state.transition_operator_state, trans_op_states)
        smc_state = SMCState(transition_operator_state=trans_op_states, key=key)

        # Info for logging.
        if not verbose:
            for i in range(n_intermediate_distributions):
                info.update(
                    {f"dist{i + 1}_" + key: value for key, value in jax.tree_map(lambda x: x[i], infos).items()})
        log_ess_q_p = log_effective_sample_size(point0.log_p - point0.log_q)
        log_ess_ais = log_effective_sample_size(log_w)
        info.update(log_ess_q_p=log_ess_q_p, log_ess_smc_final=log_ess_ais,
                    ess_q_p=jnp.exp(log_ess_q_p), ess_smc_final=jnp.exp(log_ess_ais))


        is_finite = jnp.all(jnp.isfinite(point.x), axis=-1) & jnp.isfinite(log_w)
        info.update(n_finite_ais_samples=jnp.sum(is_finite))
        info.update(ais_max_abs_x=jnp.max(jnp.abs(point.x)))
        return point, log_w, smc_state, info


    return SequentialMonteCarloSampler(init=init, step=step, betas=betas,
                                       transition_operator=transition_operator,
                                       use_resampling=use_resampling,
                                       alpha=alpha)

