from typing import NamedTuple, Callable, Any

import chex
import distrax
import flax.linen as nn
import haiku as hk
import jax
import jax.numpy as jnp
from haiku import Transformed

from fabjax.flow.distrax_with_extra import Extra, BijectorWithExtra

Params = chex.ArrayTree
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
    use_flax: bool = True


class FlowParams(NamedTuple):
    base: Params
    bijector: Params


class Flow(NamedTuple):
    init: Callable[[chex.PRNGKey, Sample], FlowParams]
    log_prob_apply: Callable[[FlowParams, Sample], LogProb]
    sample_and_log_prob_apply: Callable[[FlowParams, chex.PRNGKey, chex.Shape], tuple[Sample, LogProb]]
    sample_apply: Callable[[FlowParams, chex.PRNGKey, chex.Shape], Sample]
    log_prob_with_extra_apply: Callable[[FlowParams, Sample], tuple[LogProb, Extra]]
    sample_and_log_prob_with_extra_apply: Callable[[FlowParams, chex.PRNGKey, chex.Shape], tuple[Sample, LogProb, Extra]]
    config: Any
    dim: int


class FlowForwardAndLogDet(nn.Module):
    bijector: BijectorWithExtra

    @nn.compact
    def __call__(self, x: chex.Array) -> tuple[chex.Array, LogDet]:
        return self.bijector.forward_and_log_det(x)

class FlowInverseAndLogDet(nn.Module):
    bijector: BijectorWithExtra

    @nn.compact
    def __call__(self, y: chex.Array) -> tuple[chex.Array, LogDet]:
        return self.bijector.inverse_and_log_det(y)


class FlowForwardAndLogDetWithExtra(nn.Module):
    bijector: BijectorWithExtra

    @nn.compact
    def __call__(self, x: chex.Array) -> tuple[chex.Array, LogDet, Extra]:
        if isinstance(self.bijector, BijectorWithExtra):
            y, log_det, extra = self.bijector.forward_and_log_det_with_extra(x)
        else:
            y, log_det = self.bijector.forward_and_log_det(x)
            extra = Extra()
        extra.aux_info.update(mean_log_det=jnp.mean(log_det))
        extra.info_aggregator.update(mean_log_det=jnp.mean)
        return y, log_det, extra

class FlowInverseAndLogDetWithExtra(nn.Module):
    bijector: BijectorWithExtra

    @nn.compact
    def __call__(self, y: chex.Array) -> tuple[chex.Array, LogDet, Extra]:
        if isinstance(self.bijector, BijectorWithExtra):
            x, log_det, extra = self.bijector.inverse_and_log_det_with_extra(y)
        else:
            x, log_det = self.bijector.inverse_and_log_det(y)
            extra = Extra()
        extra.aux_info.update(mean_log_det=jnp.mean(log_det))
        extra.info_aggregator.update(mean_log_det=jnp.mean)
        return x, log_det, extra

class BaseSampleFn(nn.Module):
    base: Any

    @nn.compact
    def __call__(self, seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> Sample:
        sample = self.base.sample(seed=seed, sample_shape=sample_shape)
        return sample

class BaseLogProbFn(nn.Module):
    base: Any

    @nn.compact
    def __call__(self, sample: Sample) -> LogProb:
        return self.base.log_prob(value=sample)


def setup_haiku_fns(recipe: FlowRecipe) -> tuple[
    Transformed, Transformed, Transformed, Transformed, Transformed, Transformed]:
    @hk.without_apply_rng
    @hk.transform
    def base_sample_fn(seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> Sample:
        return recipe.make_base().sample(seed=seed, sample_shape=sample_shape)

    @hk.without_apply_rng
    @hk.transform
    def base_log_prob_fn(sample: Sample) -> LogProb:
        return recipe.make_base().log_prob(value=sample)

    @hk.without_apply_rng
    @hk.transform
    def forward_and_log_det_single(x: Sample) -> tuple[Sample, LogDet]:
        y, logdet = recipe.make_bijector().forward_and_log_det(x)
        return y, logdet

    @hk.without_apply_rng
    @hk.transform
    def inverse_and_log_det_single(y: Sample) -> tuple[Sample, LogDet]:
        x, logdet = recipe.make_bijector().inverse_and_log_det(y)
        return x, logdet

    @hk.without_apply_rng
    @hk.transform
    def forward_and_log_det_with_extra_single(x: Sample) -> \
            tuple[Sample, LogDet, Extra]:
        bijector = recipe.make_bijector()
        if isinstance(bijector, BijectorWithExtra):
            y, log_det, extra = bijector.forward_and_log_det_with_extra(x)
        else:
            y, log_det = bijector.forward_and_log_det(x)
            extra = Extra()
        extra.aux_info.update(mean_log_det=jnp.mean(-log_det))
        extra.info_aggregator.update(mean_log_det=jnp.mean)
        return y, log_det, extra


    @hk.without_apply_rng
    @hk.transform
    def inverse_and_log_det_with_extra_single(y: Sample) -> \
            tuple[Sample, LogDet, Extra]:
        bijector = recipe.make_bijector()
        if isinstance(bijector, BijectorWithExtra):
            x, log_det, extra = bijector.inverse_and_log_det_with_extra(y)
        else:
            x, log_det = bijector.inverse_and_log_det(y)
            extra = Extra()
        extra.aux_info.update(mean_log_det=jnp.mean(log_det))
        extra.info_aggregator.update(mean_log_det=jnp.mean)
        return x, log_det, extra

    return (base_sample_fn,
            base_log_prob_fn,
            forward_and_log_det_single,
            inverse_and_log_det_single,
            forward_and_log_det_with_extra_single,
            inverse_and_log_det_with_extra_single
            )

def create_flow(recipe: FlowRecipe) -> Flow:
    """Create a `Flow` given the provided definition. Allows for extra info to be passed forward in the flow, and
    is faster to compile than the distrax chain."""

    if recipe.use_flax:
        bijector_block = recipe.make_bijector()
        base = recipe.make_base()
        base_sample_fn = BaseSampleFn(base=base)
        base_log_prob_fn = BaseLogProbFn(base=base)
        forward_and_log_det_single = FlowForwardAndLogDet(bijector=bijector_block)
        inverse_and_log_det_single = FlowInverseAndLogDet(bijector=bijector_block)
        forward_and_log_det_with_extra_single = FlowForwardAndLogDetWithExtra(bijector=bijector_block)
        inverse_and_log_det_with_extra_single = FlowInverseAndLogDetWithExtra(bijector=bijector_block)
    else:
        (base_sample_fn, base_log_prob_fn, forward_and_log_det_single, inverse_and_log_det_single,
         forward_and_log_det_with_extra_single, inverse_and_log_det_with_extra_single) = (
            setup_haiku_fns(recipe=recipe))

    def log_prob_apply(params: FlowParams, sample: Sample) -> LogProb:
        def scan_fn(carry, bijector_params):
            y, log_det_prev = carry
            x, log_det = inverse_and_log_det_single.apply(bijector_params, y)
            chex.assert_equal_shape((log_det_prev, log_det))
            return (x, log_det_prev + log_det), None

        log_prob_shape = sample.shape[:-1]
        (x, log_det), _ = jax.lax.scan(scan_fn, init=(sample, jnp.zeros(log_prob_shape)),
                                       xs=params.bijector, reverse=True,
                                       unroll=recipe.compile_n_unroll)
        base_log_prob = base_log_prob_fn.apply(params.base, x)
        chex.assert_equal_shape((base_log_prob, log_det))
        return base_log_prob + log_det

    def log_prob_with_extra_apply(params: FlowParams, sample: Sample) -> tuple[LogProb, Extra]:
        def scan_fn(carry, bijector_params):
            y, log_det_prev = carry
            x, log_det, extra = inverse_and_log_det_with_extra_single.apply(bijector_params, y)
            chex.assert_equal_shape((log_det_prev, log_det))
            return (x, log_det_prev + log_det), extra

        log_prob_shape = sample.shape[:-1]
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

    def sample_and_log_prob_apply(params: FlowParams, key: chex.PRNGKey, shape: chex.Shape) -> tuple[Sample, LogProb]:
        def scan_fn(carry, bijector_params):
            x, log_det_prev = carry
            y, log_det = forward_and_log_det_single.apply(bijector_params, x)
            chex.assert_equal_shape((log_det_prev, log_det))
            return (y, log_det_prev + log_det), None

        x = base_sample_fn.apply(params.base, key, shape)
        base_log_prob = base_log_prob_fn.apply(params.base, x)
        (y, log_det), _ = jax.lax.scan(scan_fn, init=(x, jnp.zeros(x.shape[:-1])), xs=params.bijector,
                                       unroll=recipe.compile_n_unroll)
        chex.assert_equal_shape((base_log_prob, log_det))
        log_prob = base_log_prob - log_det
        return y, log_prob


    def sample_and_log_prob_with_extra_apply(params: FlowParams,
                                             key: chex.PRNGKey,
                                             shape: chex.Shape) -> tuple[Sample, LogProb, Extra]:
        def scan_fn(carry, bijector_params):
            x, log_det_prev = carry
            y, log_det, extra = forward_and_log_det_with_extra_single.apply(bijector_params, x)
            chex.assert_equal_shape((log_det_prev, log_det))
            return (y, log_det_prev + log_det), extra

        x = base_sample_fn.apply(params.base, key, shape)
        base_log_prob = base_log_prob_fn.apply(params.base, x)
        (y, log_det), extra = jax.lax.scan(scan_fn, init=(x, jnp.zeros(x.shape[:-1])), xs=params.bijector,
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
        params_bijector_single = inverse_and_log_det_single.init(key2, sample)
        params_bijectors = jax.tree_map(lambda x: jnp.repeat(x[None, ...], recipe.n_layers, axis=0),
                                        params_bijector_single)
        return FlowParams(base=params_base, bijector=params_bijectors)

    def sample_apply(*args, **kwargs):
        return sample_and_log_prob_apply(*args, **kwargs)[0]


    bijector_block = Flow(
        dim=recipe.dim,
        init=init,
        log_prob_apply=log_prob_apply,
        sample_and_log_prob_apply=sample_and_log_prob_apply,
        log_prob_with_extra_apply=log_prob_with_extra_apply,
        sample_and_log_prob_with_extra_apply=sample_and_log_prob_with_extra_apply,
        sample_apply=sample_apply,
        config=recipe.config
                        )
    return bijector_block
