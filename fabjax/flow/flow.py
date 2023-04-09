from typing import NamedTuple, Callable, Tuple, Union, Any, Optional

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp

from fabjax.flow.distrax_with_extra import Extra, BijectorWithExtra

Params = hk.Params
LogProb = chex.Array
LogDet = chex.Array
Sample = chex.Array


class FlowRecipe(NamedTuple):
    """Defines input needed to create an instance of the `Flow` callables."""
    make_base: Callable[[], distrax.Distribution]
    make_bijector: Callable[[], BijectorWithExtra]
    n_layers: int
    config: Any
    dim: int
    compile_n_unroll: int = 2


class FlowParams(NamedTuple):
    base: Params
    bijector: Params


class AugmentedFlow(NamedTuple):
    init: Callable[[chex.PRNGKey], FlowParams]
    log_prob_apply: Callable[[FlowParams, Sample], LogProb]
    sample_and_log_prob_apply: Callable[[FlowParams, chex.PRNGKey, chex.Shape], Tuple[Sample, LogProb]]
    sample_apply: Callable[[FlowParams, chex.PRNGKey, chex.Shape], Sample]
    log_prob_with_extra_apply: Callable[[FlowParams, Sample], Tuple[LogProb, Extra]]
    sample_and_log_prob_with_extra_apply: Callable[[FlowParams, chex.PRNGKey, chex.Shape], Tuple[Sample, LogProb, Extra]]
    config: Any
    dim: int



def create_flow(recipe: FlowRecipe) -> AugmentedFlow:
    """Create a `Flow` given the provided definition. Allows for extra info to be passed forward in the flow, and
    is faster to compile than the distrax chain."""


    @hk.without_apply_rng
    @hk.transform
    def base_sample_fn(seed: chex.PRNGKey, sample_shape: chex.Shape) -> Sample:
        sample = recipe.make_base().sample(seed=seed, sample_shape=sample_shape)
        return sample

    @hk.without_apply_rng
    @hk.transform
    def base_log_prob_fn(sample: Sample) -> LogProb:
        return recipe.make_base().log_prob(value=sample)

    @hk.without_apply_rng
    @hk.transform
    def bijector_forward_and_log_det(x: Sample) -> Tuple[Sample, LogDet]:
        y, logdet = recipe.make_bijector().forward_and_log_det(x)
        return y, logdet

    @hk.without_apply_rng
    @hk.transform
    def bijector_inverse_and_log_det(y: Sample) -> Tuple[Sample, LogDet]:
        x, logdet = recipe.make_bijector().inverse_and_log_det(y)
        return x, logdet

    @hk.without_apply_rng
    @hk.transform
    def bijector_forward_and_log_det_with_extra(x: Sample) -> \
            Tuple[Sample, LogDet, Extra]:
        bijector = recipe.make_bijector()
        if isinstance(bijector, BijectorWithExtra):
            y, log_det, extra = bijector.forward_and_log_det_with_extra(x)
        else:
            y, log_det = bijector.forward_and_log_det(x)
            extra = Extra()
        extra.aux_info.update(mean_log_det=jnp.mean(log_det))
        extra.info_aggregator.update(mean_log_det=jnp.mean)
        return y, log_det, extra

    @hk.without_apply_rng
    @hk.transform
    def bijector_inverse_and_log_det_with_extra(y: Sample) -> \
            Tuple[Sample, LogDet, Extra]:
        bijector = recipe.make_bijector()
        if isinstance(bijector, BijectorWithExtra):
            x, log_det, extra = bijector.inverse_and_log_det_with_extra(y)
        else:
            x, log_det = bijector.inverse_and_log_det(y)
            extra = Extra()
        extra.aux_info.update(mean_log_det=jnp.mean(log_det))
        extra.info_aggregator.update(mean_log_det=jnp.mean)
        return x, log_det, extra


    def log_prob_apply(params: FlowParams, sample: Sample) -> LogProb:
        def scan_fn(carry, bijector_params):
            y, log_det_prev = carry
            x, log_det = bijector_inverse_and_log_det.apply(bijector_params, y)
            chex.assert_equal_shape((log_det_prev, log_det))
            return (x, log_det_prev + log_det), None

        log_prob_shape = sample.positions.shape[:-3]
        (x, log_det), _ = jax.lax.scan(scan_fn, init=(sample, jnp.zeros(log_prob_shape)),
                                       xs=params.bijector, reverse=True,
                                       unroll=recipe.compile_n_unroll)
        base_log_prob = base_log_prob_fn.apply(params.base, x)
        chex.assert_equal_shape((base_log_prob, log_det))
        return base_log_prob + log_det

    def log_prob_with_extra_apply(params: FlowParams, sample: Sample) -> Tuple[LogProb, Extra]:
        def scan_fn(carry, bijector_params):
            y, log_det_prev = carry
            x, log_det, extra = bijector_inverse_and_log_det_with_extra.apply(bijector_params, y)
            chex.assert_equal_shape((log_det_prev, log_det))
            return (x, log_det_prev + log_det), extra

        log_prob_shape = sample.positions.shape[:-3]
        (x, log_det), extra = jax.lax.scan(scan_fn, init=(sample, jnp.zeros(log_prob_shape)),
                                           xs=params.bijector,
                                           reverse=True, unroll=recipe.compile_n_unroll)
        base_log_prob = base_log_prob_fn.apply(params.base, x)
        chex.assert_equal_shape((base_log_prob, log_det))

        info = {}
        aggregators = {}
        for i in reversed(range(recipe.n_layers)):
          info.update({f"block{i}_" + key: val[i] for key, val in extra.aux_info.items()})
          aggregators.update({f"block{i}_" + key: val for key, val in extra.info_aggregator.items()})

        info.update(mean_base_log_prob=jnp.mean(base_log_prob))
        aggregators.update(mean_base_log_prob=jnp.mean)
        extra = Extra(aux_loss=extra.aux_loss, aux_info=info, info_aggregator=aggregators)

        return base_log_prob + log_det, extra

    def sample_and_log_prob_apply(params: FlowParams, key: chex.PRNGKey, shape: chex.Shape) -> Tuple[Sample, LogProb]:
        def scan_fn(carry, bijector_params):
            x, log_det_prev = carry
            y, log_det = bijector_forward_and_log_det.apply(bijector_params, x)
            chex.assert_equal_shape((log_det_prev, log_det))
            return (y, log_det_prev + log_det), None

        x = base_sample_fn.apply(params.base, key, shape)
        base_log_prob = base_log_prob_fn.apply(params.base, x)
        (y, log_det), _ = jax.lax.scan(scan_fn, init=(x, jnp.zeros(x.positions.shape[:-3])), xs=params.bijector,
                                       unroll=recipe.compile_n_unroll)
        chex.assert_equal_shape((base_log_prob, log_det))
        log_prob = base_log_prob - log_det
        return y, log_prob


    def sample_and_log_prob_with_extra_apply(params: FlowParams,
                                             key: chex.PRNGKey,
                                             shape: chex.Shape) -> Tuple[Sample, LogProb, Extra]:
        def scan_fn(carry, bijector_params):
            x, log_det_prev = carry
            y, log_det, extra = bijector_forward_and_log_det_with_extra.apply(bijector_params, x)
            chex.assert_equal_shape((log_det_prev, log_det))
            return (y, log_det_prev + log_det), extra

        x = base_sample_fn.apply(params.base, key, shape)
        base_log_prob = base_log_prob_fn.apply(params.base, x)
        (y, log_det), extra = jax.lax.scan(scan_fn, init=(x, jnp.zeros(x.positions.shape[:-3])), xs=params.bijector,
                                           unroll=recipe.compile_n_unroll)
        chex.assert_equal_shape((base_log_prob, log_det))
        log_prob = base_log_prob - log_det

        info = {}
        aggregators = {}
        for i in range(recipe.n_layers):
          info.update({f"block{i}_" + key: val[i] for key, val in extra.aux_info.items()})
          aggregators.update({f"block{i}_" + key: val for key, val in extra.info_aggregator.items()})
        info.update(mean_base_log_prob=jnp.mean(base_log_prob))
        aggregators.update(mean_base_log_prob=jnp.mean)
        extra = Extra(aux_loss=extra.aux_loss, aux_info=info, info_aggregator=aggregators)
        return y, log_prob, extra


    def init(seed: chex.PRNGKey, sample: Sample) -> FlowParams:
        # Check shapes.
        chex.assert_tree_shape_suffix(sample, (recipe.dim,))

        key1, key2 = jax.random.split(seed)
        params_base = base_log_prob_fn.init(key1, sample)
        params_bijector_single = bijector_inverse_and_log_det.init(key2, sample)
        params_bijectors = jax.tree_map(lambda x: jnp.repeat(x[None, ...], recipe.n_layers, axis=0),
                                        params_bijector_single)
        return FlowParams(base=params_base, bijector=params_bijectors)

    def sample_apply(*args, **kwargs):
        return sample_and_log_prob_apply(*args, **kwargs)[0]


    flow = AugmentedFlow(
        dim=recipe.dim,
        init=init,
        log_prob_apply=log_prob_apply,
        sample_and_log_prob_apply=sample_and_log_prob_apply,
        log_prob_with_extra_apply=log_prob_with_extra_apply,
        sample_and_log_prob_with_extra_apply=sample_and_log_prob_with_extra_apply,
        sample_apply=sample_apply,
        config=recipe.config
                        )
    return flow
