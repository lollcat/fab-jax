import jax.numpy as jnp
import jax
import chex
import matplotlib.pyplot as plt

from fabjax.targets.base import Target
from fabjax.utils.plot import plot_marginal_pair, plot_contours_2D

class Energy:
    """
    https://zenodo.org/record/3242635#.YNna8uhKjIW
    """
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def _energy(self, x):
        raise NotImplementedError()

    def energy(self, x, temperature=None):
        assert x.shape[-1] == self._dim, "`x` does not match `dim`"
        if temperature is None:
            temperature = 1.
        return self._energy(x) / temperature

    def force(self, x, temperature=None):
        e_func = lambda x: jnp.sum(self.energy(x, temperature=temperature))
        return -jax.grad(e_func)(x)


class DoubleWellEnergy(Energy):
    def __init__(self):
        dim = 2
        a = -0.5
        b = -6.0
        c = 1.
        super().__init__(dim)
        self._a = a
        self._b = b
        self._c = c

    def _energy(self, x):
        d = x[:, [0]]
        v = x[:, 1:]
        e1 = self._a * d + self._b * d**2 + self._c * d**4
        e2 = jnp.sum(0.5 * v**2, axis=-1, keepdims=True)
        return e1 + e2

    def log_prob(self, x):
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, axis=0)
        return jnp.squeeze(-self.energy(x))

    @property
    def log_Z(self):
        log_Z_dim0 = jnp.log(11784.50927)
        log_Z_dim1 = 0.5 * jnp.log(2 * jnp.pi)
        return log_Z_dim0 + log_Z_dim1


class ManyWellEnergy(Target):
    def __init__(self, dim: int = 32):

        assert dim % 2 == 0
        self.n_wells = dim // 2
        self.double_well_energy = DoubleWellEnergy()

        log_Z = self.double_well_energy.log_Z * self.n_wells
        # TODO: Add rejection sampling samples.
        super().__init__(dim=dim, log_Z=log_Z, can_sample=False, n_plots=1,
                         n_model_samples_eval=1000, n_target_samples_eval=1000)

        self.centre = 1.7
        self.max_dim_for_all_modes = 40  # otherwise we get memory issues on huuuuge test set
        if self.dim < self.max_dim_for_all_modes:
            dim_1_vals_grid = jnp.meshgrid(*[jnp.array([-self.centre, self.centre])for _ in
                                              range(self.n_wells)])
            dim_1_vals = jnp.stack([dim.flatten() for dim in dim_1_vals_grid], axis=-1)
            n_modes = 2**self.n_wells
            assert n_modes == dim_1_vals.shape[0]
            test_set = jnp.zeros((n_modes, dim))
            test_set = test_set.at[:, jnp.arange(dim) % 2 == 0].set(dim_1_vals)
            self.test_set = test_set
        else:
            raise NotImplementedError("still need to implement this")

        self.shallow_well_bounds = [-1.75, -1.65]
        self.deep_well_bounds = [1.7, 1.8]
        self._plot_bound = 3.

    def log_prob(self, x):
        return jnp.sum(jnp.stack([self.double_well_energy.log_prob(x[..., i*2:i*2+2]) for i in range(
                self.n_wells)], axis=-1), axis=-1)

    def log_prob_2D(self, x):
        """Marginal 2D pdf - useful for plotting."""
        return self.double_well_energy.log_prob(x)

    def visualise(
            self,
            samples: chex.Array,
            axes: list[plt.Axes],
             ) -> None:
        """Visualise samples from the model."""
        assert len(axes) == self.n_plots

        ax = axes[0]
        plot_contours_2D(self.log_prob_2D, ax, bound=self._plot_bound, levels=20)
        plot_marginal_pair(samples, ax, bounds=(-self._plot_bound, self._plot_bound))

