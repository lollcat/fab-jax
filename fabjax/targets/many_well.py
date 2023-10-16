from typing import Callable, Tuple

from warnings import warn

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import chex
import distrax

from fabjax.targets.base import Target, LogProbFn
from fabjax.utils.plot import plot_marginal_pair, plot_contours_2D
from fabjax.train.evaluate import calculate_log_forward_ess
from fabjax.sampling.rejection_sampling import rejection_sampling

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


    def sample_first_dimension(self, key: chex.Array, n: int) -> chex.Array:
        # see fab.sampling.rejection_sampling_test.py
        if self._a == -0.5 and self._b == -6 and self._c == 1.0:
            # Define target.
            def target_log_prob(x):
                return -x ** 4 + 6 * x ** 2 + 1 / 2 * x

            TARGET_Z = 11784.50927

            # Define proposal params
            component_mix = jnp.array([0.2, 0.8])
            means = jnp.array([-1.7, 1.7])
            scales = jnp.array([0.5, 0.5])

            # Define proposal
            mix = distrax.Categorical(component_mix)
            com = distrax.Normal(means, scales)

            proposal = distrax.MixtureSameFamily(
                mixture_distribution=mix,
                components_distribution=com)

            k = TARGET_Z * 3

            samples = rejection_sampling(n_samples=n, proposal=proposal,
                                         target_log_prob_fn=target_log_prob, k=k, key=key)
            return samples
        else:
            raise NotImplementedError


    def sample(self, key: chex.PRNGKey, shape: chex.Shape):
        if self._a == -0.5 and self._b == -6 and self._c == 1.0:
            assert len(shape) == 1
            key1, key2 = jax.random.split(key=key)
            dim1_samples = self.sample_first_dimension(key=key1, n=shape[0])
            dim2_samples = distrax.Normal(
                jnp.array(0.0),
                jnp.array(1.0)).sample(seed=key2, sample_shape=shape)
            return jnp.stack([dim1_samples, dim2_samples], axis=-1)
        else:
            raise NotImplementedError



class ManyWellEnergy(Target):
    def __init__(self, dim: int = 32):

        assert dim % 2 == 0
        self.n_wells = dim // 2
        self.double_well_energy = DoubleWellEnergy()

        log_Z = self.double_well_energy.log_Z * self.n_wells
        super().__init__(dim=dim, log_Z=log_Z, can_sample=False, n_plots=1,
                         n_model_samples_eval=2000, n_target_samples_eval=10000)

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
            self.modes_test_set = test_set
        else:
            raise NotImplementedError("still need to implement this")

        self.shallow_well_bounds = [-1.75, -1.65]
        self.deep_well_bounds = [1.7, 1.8]
        self._plot_bound = 3.

        self.double_well_samples = self.double_well_energy.sample(
            key=jax.random.PRNGKey(0), shape=(int(1e6),))

        if self.n_target_samples_eval < self.modes_test_set.shape[0]:
            warn("Evaluation occuring on subset of the modes test set.")

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

    def get_eval_samples(self, key: chex.PRNGKey, n: int) -> Tuple[chex.Array, chex.Array]:
        key1, key2 = jax.random.split(key)
        dw_sample_indices = jax.random.randint(
            minval=0, maxval=self.double_well_samples.shape[0],
            key=key1, shape=(n*self.n_wells,))
        dw_samples = self.double_well_samples[dw_sample_indices]
        samples_p = jnp.reshape(dw_samples, (-1, self.dim))

        if n < self.modes_test_set.shape[0]:
            mode_sample_indices = jax.random.choice(
                a=jnp.arange(self.modes_test_set.shape[0]),
                key=key2, shape=(n,), replace=False)
            samples_modes = self.modes_test_set[mode_sample_indices]
        else:
            samples_modes = self.modes_test_set
        return samples_p, samples_modes

    def evaluate(self,
                 model_log_prob_fn: LogProbFn,
                 model_sample_and_log_prob_fn: Callable[[chex.PRNGKey, chex.Shape], Tuple[chex.Array, chex.Array]],
                 key: chex.PRNGKey,
                 ) -> dict:
        """Evaluate a model. Note that reverse ESS will be estimated separately, so should not be estimated here."""

        info = {}

        # Evaluate on (close to exact) samples from target.
        samples_p, samples_modes = self.get_eval_samples(key, self.n_target_samples_eval)
        log_prob_q = model_log_prob_fn(samples_p)
        log_prob_p = self.log_prob(samples_p)
        log_w = log_prob_p - log_prob_q
        log_forward_ess = calculate_log_forward_ess(log_w, log_Z=self.log_Z)
        info.update(log_lik=jnp.mean(log_prob_q),
                    log_forward_ess=log_forward_ess,
                    forward_ess=jnp.exp(log_forward_ess))

        log_prob_modes = model_log_prob_fn(samples_modes)
        info.update(log_prob_modes=jnp.mean(log_prob_modes))
        return info


if __name__ == '__main__':
    target = ManyWellEnergy(dim=8)
    key1 = jax.random.PRNGKey(0)
    samples_p, samples_modes = target.get_eval_samples(key1, 400)

    fig, axs = plt.subplots()
    target.visualise(samples_p, [axs])
    plt.show()
