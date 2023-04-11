import jax.numpy as jnp
import jax
import distrax
import matplotlib.pyplot as plt

from molboil.utils.loggers import ListLogger
from molboil.utils.plotting import plot_history

from fabjax.sampling.mcmc.metropolis import build_metropolis
from fabjax.sampling.base import create_point
from fabjax.utils.plot import plot_contours_2D, plot_marginal_pair


def test_metropolis_produces_good_samples():

    dim = 2
    beta = 1.  # Equivalent to setting target to p.
    key = jax.random.PRNGKey(0)
    batch_size = 1000
    n_outer_steps = 100
    alpha = 2.

    metropolis_trans_op = build_metropolis(dim=2, n_steps=n_outer_steps, init_step_size=1., tune_step_size=True)

    loc_q = jnp.zeros((dim,))
    dist_q = distrax.MultivariateNormalDiag(loc_q, jnp.ones((dim,)))

    loc_p = jnp.zeros((dim,)) + 3
    scale_p = jnp.ones((dim,)) * 0.5
    dist_p = distrax.MultivariateNormalDiag(loc_p, scale_p)

    # Initialise MCMC chain.
    positions = dist_q.sample(seed=key, sample_shape=(batch_size,))
    points = jax.vmap(create_point, in_axes=(0, None, None, None))(positions, dist_q.log_prob, dist_p.log_prob, False)


    trans_op_state = metropolis_trans_op.init(key)

    logger = ListLogger()

    # Run MCMC chain.
    for i in range(50):
        x_new, trans_op_state, info = metropolis_trans_op.step(points, trans_op_state, beta, alpha,
                                                              dist_q.log_prob, dist_p.log_prob)
        logger.write(info)

    plot_history(logger.history)
    plt.show()

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

    print(info)



