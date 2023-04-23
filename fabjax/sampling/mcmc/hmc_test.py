import jax.numpy as jnp
import jax
import distrax
import matplotlib.pyplot as plt
from tqdm import tqdm

from fabjax.sampling.mcmc.hmc import build_blackjax_hmc
from fabjax.sampling.base import create_point
from fabjax.utils.plot import plot_contours_2D, plot_marginal_pair, plot_history
from fabjax.utils.logging import ListLogger


def test_hmc_produces_good_samples():

    dim = 2
    beta = 0.5  # Equivalent to setting target to p.
    key = jax.random.PRNGKey(0)
    batch_size = 1000
    n_outer_steps = 100
    target_p_accept = 0.65
    alpha = 2.

    hmc_transition_operator = build_blackjax_hmc(dim=2, n_outer_steps=n_outer_steps,
                                                 init_step_size=1e-4, adapt_step_size=True)
    trans_op_state = hmc_transition_operator.init(key)

    loc_q = jnp.zeros((dim,))
    dist_q = distrax.MultivariateNormalDiag(loc_q, jnp.ones((dim,)))

    loc_p = jnp.zeros((dim,)) + 3
    scale_p = jnp.ones((dim,))*0.5
    dist_p = distrax.MultivariateNormalDiag(loc_p, scale_p)
    logger = ListLogger()

    # Run MCMC chain.
    for i in tqdm(range(50)):
        # Initialise MCMC chain.
        key, subkey = jax.random.split(key)
        positions = dist_q.sample(seed=subkey, sample_shape=(batch_size,))
        points = jax.vmap(create_point, in_axes=(0, None, None, None))(positions, dist_q.log_prob, dist_p.log_prob,
                                                                       True)
        x_new, trans_op_state, info = hmc_transition_operator.step(
                                                              point=points,
                                                              transition_operator_state=trans_op_state,
                                                              beta=beta,
                                                              alpha=alpha,
                                                              log_q_fn=dist_q.log_prob,
                                                              log_p_fn=dist_p.log_prob)
        logger.write(info)

    plot_history(logger.history)
    plt.show()

    # Visualise samples.
    bound = 10
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4))
    plot_contours_2D(dist_q.log_prob, axs[0], bound=bound)
    plot_marginal_pair(x_new.x, axs[0], bounds=(-bound, bound), alpha=0.2)
    plot_contours_2D(dist_p.log_prob, axs[1], bound=bound)
    plot_marginal_pair(x_new.x, axs[1], bounds=(-bound, bound), alpha=0.2)
    plot_contours_2D(dist_p.log_prob, axs[2], bound=bound)
    plot_marginal_pair(dist_p.sample(seed=key, sample_shape=(batch_size,)),
                       axs[2], bounds=(-bound, bound), alpha=0.2)
    axs[0].set_title("samples vs log prob q contours")
    axs[1].set_title("samples vs log prob p contours")
    axs[2].set_title("p samples vs p contours")
    plt.tight_layout()
    plt.show()

    # assert_trees_all_different(x_new.x, positions)
    # Check some metrics are reasonable.
    x_mean = jnp.mean(x_new.x, axis=0)

    assert ((x_mean - loc_p)**2 < (x_mean - loc_q)).all()

    assert (0.6 < info['mean_acceptance_rate']) and (info['mean_acceptance_rate'] < 0.7)



