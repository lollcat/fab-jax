import jax.numpy as jnp
import jax
import distrax
import matplotlib.pyplot as plt

from fabjax.sampling.mcmc.hmc import build_blackjax_hmc
from fabjax.sampling.base import create_point
from fabjax.utils.plot import plot_contours_2D, plot_marginal_pair


def test_hmc_produces_good_samples():

    dim = 2
    beta = 1.  # Equivalent to setting target to p.
    key = jax.random.PRNGKey(0)
    batch_size = 1000
    n_outer_steps = 100
    target_p_accept = 0.65
    alpha = 2.

    hmc_transition_operator = build_blackjax_hmc(dim=2, n_outer_steps=n_outer_steps,
                                                 init_step_size=1e-4)

    loc_q = jnp.zeros((dim,))
    dist_q = distrax.MultivariateNormalDiag(loc_q, jnp.ones((dim,)))

    loc_p = jnp.zeros((dim,)) + 3
    scale_p = jnp.ones((dim,)) * 0.5
    dist_p = distrax.MultivariateNormalDiag(loc_p, scale_p)

    # Initialise MCMC chain.
    positions = dist_q.sample(seed=key, sample_shape=(batch_size,))
    points = jax.vmap(create_point, in_axes=(0, None, None))(positions, dist_q.log_prob, dist_p.log_prob)


    hmc_state = hmc_transition_operator.init(key)

    # Run MCMC chain.
    x_new, hmc_state, info = hmc_transition_operator.step(points, hmc_state, beta, alpha,
                                                          dist_q.log_prob, dist_p.log_prob)

    # Visualise samples.
    bound = 10
    fig, axs = plt.subplots(1, 2)
    plot_contours_2D(dist_q.log_prob, axs[0], bound=bound)
    plot_marginal_pair(x_new.x, axs[0], bounds=(-bound, bound), alpha=0.2)
    plot_contours_2D(dist_p.log_prob, axs[1], bound=bound)
    plot_marginal_pair(x_new.x, axs[1], bounds=(-bound, bound), alpha=0.2)
    axs[0].set_title("samples vs log prob q contours")
    axs[1].set_title("samples vs log prob p contours")
    plt.tight_layout()
    plt.show()

    # assert_trees_all_different(x_new.x, positions)
    # Check some metrics are reasonable.
    x_mean = jnp.mean(x_new.x, axis=0)

    assert ((x_mean - loc_p)**2 < (x_mean - loc_q)).all()

    assert (0.6 < info['mean_acceptance_rate']) and (info['mean_acceptance_rate'] < 0.7)



