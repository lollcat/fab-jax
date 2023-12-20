"""Taken from https://github.com/google-deepmind/annealed_flow_transport."""
from typing import List

import jax.numpy as jnp
import chex
import jax
import matplotlib.pyplot as plt

from fabjax.targets.base import Target

Array = chex.Array


def phi_four_log_density(x: Array,
                         mass_squared: Array,
                         bare_coupling: Array) -> Array:
    """Evaluate the phi_four_log_density.

    Args:
        x: Array of size (L_x, L_y)- values on 2D lattice.
        mass_squared: Scalar representing bare mass squared.
        bare_coupling: Scare representing bare coupling.

    Returns:
        Scalar corresponding to log_density.
    """
    chex.assert_rank(x, 2)
    chex.assert_rank(mass_squared, 0)
    chex.assert_rank(bare_coupling, 0)
    mass_term = mass_squared * jnp.sum(jnp.square(x))
    quadratic_term = bare_coupling * jnp.sum(jnp.power(x, 4))
    roll_x_plus = jnp.roll(x, shift=1, axis=0)
    roll_x_minus = jnp.roll(x, shift=-1, axis=0)
    roll_y_plus = jnp.roll(x, shift=1, axis=1)
    roll_y_minus = jnp.roll(x, shift=-1, axis=1)
    # D'alembertian operator acting on field phi.
    dalembert_phi = 4.*x - roll_x_plus - roll_x_minus - roll_y_plus - roll_y_minus
    kinetic_term = jnp.sum(x * dalembert_phi)
    action_density = kinetic_term + mass_term + quadratic_term
    return -action_density


class PhiFourTheory(Target):
    """Log density for phi four field theory in two dimensions."""
    # TODO: Normalizing constant?

    def __init__(self,
                 mass_squared: Array = -4.75,
                 bare_coupling: Array = 5.1,
                 num_grid_per_dim: int = 14) -> None:
        super().__init__(dim=num_grid_per_dim**2, log_Z=None, can_sample=False, n_plots=0,
                         n_model_samples_eval=1000, n_target_samples_eval=2000)
        self._num_grid_per_dim = num_grid_per_dim
        self._mass_squared = mass_squared
        self._bare_coupling = bare_coupling

    def reshape_and_call(self, x: Array) -> Array:
        return phi_four_log_density(jnp.reshape(x, (self._num_grid_per_dim,
                                                    self._num_grid_per_dim)),
                                    self._mass_squared,
                                    self._bare_coupling)

    def log_prob(self, x: Array) -> Array:
        if x.ndim == 1:
            return self.reshape_and_call(x)
        else:
            assert x.ndim == 2
            return jax.vmap(self.reshape_and_call)(x)

    def visualise(self,
                  samples: chex.Array,
                  axes: List[plt.Axes],
                  ) -> None:
        return None


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (14*14,))
    target = PhiFourTheory()

    print(target.log_prob(x))
    print(target.log_prob(x) + 1)
    print(target.log_prob(x) + 2)
