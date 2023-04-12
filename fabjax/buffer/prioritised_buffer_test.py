import jax.numpy as jnp
import jax

from fabjax.buffer.prioritised_buffer import build_prioritised_buffer


def test_prioritised_buffer_does_not_smoke():
    """Check that the buffer runs without raising any errors."""
    dim = 6
    batch_size = 3
    n_batches_total_length = 10
    length = n_batches_total_length * batch_size
    min_sample_length = int(length * 0.5)
    rng_key = jax.random.PRNGKey(0)

    x_init, log_w_init, low_q_init = jax.random.normal(rng_key, (min_sample_length, dim)), \
        jnp.zeros(min_sample_length), jnp.zeros(min_sample_length)
    buffer = build_prioritised_buffer(dim, length, min_sample_length)
    buffer_state = buffer.init(x_init, log_w_init, low_q_init)
    for i in range(100):
        buffer_state = buffer.add(jax.random.normal(rng_key, (batch_size, dim)),
                                  jnp.zeros(batch_size), jnp.zeros(
                batch_size),
                                  buffer_state)
        rng_key, subkey = jax.random.split(rng_key)
        x, log_q_old, indices = buffer.sample(subkey, buffer_state, batch_size)
        buffer_state = buffer.adjust(jnp.ones(batch_size,), jnp.ones(batch_size,), indices, buffer_state)

    x, log_q_old, indices = buffer.sample_n_batches(rng_key, buffer_state, 12, 2)
    assert (x[1, 1] == buffer_state.data.x[indices[1, 1]]).all()
