from typing import NamedTuple, Tuple, Iterable, Callable, Optional, Protocol

import jax.lax
import jax.numpy as jnp
import chex
from functools import partial

from fabjax.utils.jax_util import broadcasted_where

def sample_without_replacement(key: chex.Array, logits: chex.Array, n: int) -> chex.Array:
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    key1, key2 = jax.random.split(key)
    z = jax.random.gumbel(key=key1, shape=logits.shape)
    # vals, indices = jax.lax.approx_max_k(z + logits, n)
    vals, indices = jax.lax.top_k(z + logits, n)
    indices = jax.random.permutation(key2, indices)
    return indices


class Data(NamedTuple):
    """Log weights and samples generated by annealed importance sampling."""
    x: chex.Array
    log_w: chex.Array
    log_q_old: chex.Array

class PrioritisedBufferState(NamedTuple):
    """State of the buffer, storing the data and additional info needed for it's use."""
    data: Data
    is_full: jnp.bool_
    can_sample: jnp.bool_
    current_index: jnp.int32

class InitFn(Protocol):
    def __call__(self, x: chex.Array, log_w: chex.Array, log_q_old: chex.Array) -> PrioritisedBufferState:
        """Initialise the buffer state, by filling it above `min_sample_length`."""

class AddFn(Protocol):
    def __call__(self, x: chex.Array, log_w: chex.Array, log_q: chex.Array,
            buffer_state: PrioritisedBufferState) -> PrioritisedBufferState:
        """Update the buffer's state with a new batch of data."""

class SampleFn(Protocol):
    def __call__(self, key: chex.PRNGKey,
               buffer_state: PrioritisedBufferState,
               batch_size: int) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Sample a batch from the buffer in proportion to the log weights.
        Returns:
            x: Samples.
            log_q_old: Value of log_q when log_w was calculated.
            indices: Indices of samples for their location in the buffer state.
        """

class SampleNBatchesFn(Protocol):
    def __call__(self, key: chex.PRNGKey,
            buffer_state: PrioritisedBufferState,
            batch_size: int,
            n_batches: int) -> \
            Iterable[Tuple[chex.Array, chex.Array, chex.Array]]:
        """Returns dataset with n-batches on the leading axis. See `SampleFn`."""

class AdjustFn(Protocol):
    def __call__(self, log_q: chex.Array,
               log_w_adjustment: chex.Array,
               indices: chex.Array,
               buffer_state: PrioritisedBufferState) \
            -> PrioritisedBufferState:
        """Adjust log weights and log q to match new value of theta, this is typically performed
        over minibatches, rather than over the whole dataset at once."""


class PrioritisedBuffer(NamedTuple):
    init: InitFn
    add: AddFn
    sample: SampleFn
    sample_n_batches: SampleNBatchesFn
    adjust: AdjustFn
    min_lengtht_to_sample: int
    max_length: int


def build_prioritised_buffer(
        dim: int,
        max_length: int,
        min_length_to_sample: int,
        sample_with_replacement: bool = False
) -> PrioritisedBuffer:
    """
    Create replay buffer for batched sampling and adding of data.

    Args:
        dim: Dimension of x data.
        max_length: Maximum length of the buffer.
        min_length_to_sample: Minimum length of buffer required for sampling.
        sample_with_replacement: Whether to sample with replacement.

    The `max_length` and `min_sample_length` should be sufficiently long to prevent overfitting
    to the replay data. For example, if `min_sample_length` is equal to the
    sampling batch size, then we may overfit to the first batch of data, as we would update
    on it many times during the start of training.
    """
    assert min_length_to_sample <= max_length


    def init(x: chex.Array, log_w: chex.Array, log_q_old: chex.Array) -> PrioritisedBufferState:
        """
        Initialise the buffer state, by filling it above `min_sample_length`.
        """
        chex.assert_rank(x, 2)
        chex.assert_shape(x[0], (dim,))
        chex.assert_equal_shape((x[:, 0], log_w, log_q_old))
        n_samples = x.shape[0]
        assert n_samples >= min_length_to_sample, "Buffer requires at least `min_sample_length` samples for init."

        current_index = 0
        is_full = False  # whether the buffer is full
        can_sample = False  # whether the buffer is full enough to begin sampling
        # init data to have -inf log_w to prevent these values being sampled.
        data = Data(x=jnp.zeros((max_length, dim)) * float("nan"),
                    log_w=-jnp.ones(max_length, ) * float("inf"),
                    log_q_old=jnp.zeros(max_length,) * float("nan")
                    )
        buffer_state = PrioritisedBufferState(data=data, is_full=is_full, can_sample=can_sample,
                                              current_index=current_index)
        buffer_state = add(x, log_w, log_q_old, buffer_state)
        return buffer_state

    def add(x: chex.Array, log_w: chex.Array, log_q: chex.Array,
            buffer_state: PrioritisedBufferState) -> PrioritisedBufferState:
        """Update the buffer's state with a new batch of data."""
        chex.assert_rank(x, 2)
        chex.assert_equal_shape((x[0], buffer_state.data.x[0]))
        chex.assert_equal_shape((x[:, 0], log_w, log_q))
        batch_size = x.shape[0]
        valid_samples = jnp.isfinite(log_w) & jnp.all(jnp.isfinite(x), axis=-1) \
                        & jnp.isfinite(log_q)
        indices = (jnp.arange(batch_size) + buffer_state.current_index) % max_length

        # Remove invalid samples.
        x, log_w, log_q = jax.tree_map(
            lambda a, b: broadcasted_where(valid_samples, a, b),
            (x, log_w, log_q),
            (buffer_state.data.x[indices], buffer_state.data.log_w[indices],
             buffer_state.data.log_q_old[indices]))

        # Add valid samples to buffer (possibly overwriting old data).
        x = buffer_state.data.x.at[indices].set(x)
        log_w = buffer_state.data.log_w.at[indices].set(log_w)
        log_q = buffer_state.data.log_q_old.at[indices].set(log_q)

        # Keep track of index, and whether buffer is full.
        new_index = buffer_state.current_index + batch_size
        is_full = jax.lax.select(buffer_state.is_full, buffer_state.is_full,
                                 new_index >= max_length)
        can_sample = jax.lax.select(buffer_state.is_full, buffer_state.can_sample,
                                    new_index >= min_length_to_sample)
        current_index = new_index % max_length

        data = Data(x=x, log_w=log_w, log_q_old=log_q)
        state = PrioritisedBufferState(data=data,
                                       current_index=current_index,
                                       is_full=is_full,
                                       can_sample=can_sample)
        return state


    def sample(key: chex.PRNGKey,
               buffer_state: PrioritisedBufferState,
               batch_size: int) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Sample a batch from the buffer in proportion to the log weights.
        Returns:
            x: Samples.
            log_q_old: Value of log_q when log_w was calculated.
            indices: Indices of samples for their location in the buffer state.
        """
        assert batch_size <= min_length_to_sample, "Min length to sample must be greater than or equal to " \
                                                   "the batch size."
        # Get indices.
        if sample_with_replacement:
            indices = jax.random.categorical(key, buffer_state.data.log_w, shape=(batch_size,))
        else:
            key1, key2 = jax.random.split(key)
            indices = sample_without_replacement(key1, buffer_state.data.log_w, batch_size)
            indices = jax.random.permutation(key2, indices)

        return buffer_state.data.x[indices], buffer_state.data.log_q_old[indices], indices


    def sample_n_batches(
            key: chex.PRNGKey,
            buffer_state: PrioritisedBufferState,
            batch_size: int,
            n_batches: int) -> \
            Iterable[Tuple[chex.Array, chex.Array, chex.Array]]:
        """Returns dataset with n-batches on the leading axis."""
        x, log_q_old, indices = sample(key, buffer_state, batch_size*n_batches)
        dataset = jax.tree_map(lambda x: x.reshape((n_batches, batch_size, *x.shape[1:])),
                               (x, log_q_old, indices))
        return dataset

    def adjust(log_q: chex.Array,
               log_w_adjustment: chex.Array,
               indices: chex.Array,
               buffer_state: PrioritisedBufferState) \
            -> PrioritisedBufferState:
        """Adjust log weights and log q to match new value of theta, this is typically performed
        over minibatches, rather than over the whole dataset at once."""
        chex.assert_rank(log_q, 1)
        chex.assert_equal_shape((log_q, log_w_adjustment, indices))

        # prevent invalid adjustments
        valid_adjustment = jnp.isfinite(log_w_adjustment) & jnp.isfinite(log_q)
        log_w = buffer_state.data.log_w[indices] + log_w_adjustment
        # remove invalid samples
        log_w, log_q = jax.tree_map(
            partial(broadcasted_where, valid_adjustment),
            (log_w, log_q),
            (-jnp.ones_like(log_w)*jnp.inf, jnp.zeros_like(log_q)))

        # adjust log weights in buffer state
        log_w = buffer_state.data.log_w.at[indices].set(log_w)
        log_q_old = buffer_state.data.log_q_old.at[indices].set(log_q)
        new_data = Data(x=buffer_state.data.x, log_w=log_w, log_q_old=log_q_old)
        return PrioritisedBufferState(data=new_data, current_index=buffer_state.current_index,
                                      can_sample=buffer_state.can_sample,
                                      is_full=buffer_state.is_full)

    return PrioritisedBuffer(init=init,
                             add=add,
                             adjust=adjust,
                             sample=sample,
                             sample_n_batches=sample_n_batches,
                             min_lengtht_to_sample=min_length_to_sample,
                             max_length=max_length)
