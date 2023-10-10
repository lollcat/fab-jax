import jax.numpy as jnp
import jax


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
    def __init__(self, dim, a=-0.5, b=-6.0, c=1.):
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


class ManyWellEnergy:
    # TODO: Add problem to Target abstraction

    def __init__(self, dim: int = 4, *args, **kwargs):
        assert dim % 2 == 0
        self.n_wells = dim // 2
        self.double_well_energy = DoubleWellEnergy(dim=2, *args, **kwargs)
        self.dim = dim

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

    def log_prob(self, x):
        return jnp.sum(jnp.stack([self.double_well_energy.log_prob(x[..., i*2:i*2+2]) for i in range(
                self.n_wells)], axis=-1), axis=-1)

    def log_prob_2D(self, x):
        """Marginal 2D pdf - useful for plotting."""
        return self.double_well_energy.log_prob(x)


def setup_manywell_evaluator(many_well: ManyWellEnergy,
                             flow: HaikuDistribution) -> Evaluator:
    # TODO:
    # test_set_folder = "datasets/manywell.np"
    #
    # def create_test_set():
    #     log_prob_2D = ManyWellEnergy(dim=2).log_prob_2D
    #     hmc_transition_operator = HamiltoneanMonteCarloTFP(n_intermediate_distributions=1)
    #     def step(carry, xs):
    #         key = xs
    #         x, transition_operator_state = carry
    #         x_new, transition_operator_state, \
    #         info = hmc_transition_operator.run(
    #             key,  transition_operator_state=transition_operator_state,
    #             x=x, transition_target_log_prob=log_prob_2D, i=jnp.array(0))
    #         return x_new, transition_operator_state

    def evaluate(outer_batch_size, inner_batch_size, state: State):
        test_set_log_prob = flow.log_prob.apply(
            state.learnt_distribution_params, many_well.test_set)
        info = {"test_set_mean_log_prob": jnp.mean(test_set_log_prob)}
        return info
    return evaluate


if __name__ == '__main__':
    dim = 2
    energy = ManyWellEnergy(dim=2)
    x = jax.random.normal(jax.random.PRNGKey(42), shape=(3, dim))
    print(energy.log_prob(x))

    import itertools
    import matplotlib.pyplot as plt
    from fabjax.utils.plotting import plot_3D
    import numpy as np
    bound = -3
    n_points = 200
    x_points_dim1 = np.linspace(-bound, bound, n_points)
    x_points_dim2 = np.linspace(-bound, bound, n_points)
    x_points = np.array(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_probs = energy.log_prob(x_points)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot_3D(x_points, np.exp(log_probs), n_points, ax, title="log p(x)")
    plt.show()
