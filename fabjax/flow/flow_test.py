import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from fabjax.flow.build_flow import build_flow, FlowDistConfig
from fabjax.utils.plot import plot_marginal_pair


def test_flow_does_not_smoke():
    batch_size = 32
    key = jax.random.PRNGKey(0)
    config = FlowDistConfig(dim=2,
                            n_layers=4,
                            conditioner_mlp_units=(16,),
                            identity_init=False
                            )

    flow = build_flow(config)
    params = flow.init(key, jnp.zeros((1, config.dim)))

    key, subkey = jax.random.split(key)
    x, log_prob = flow.sample_and_log_prob_apply(params, key, (batch_size,))
    log_prob_ = flow.log_prob_apply(params, x)

    chex.assert_trees_all_close(log_prob_, log_prob)

    plot_marginal_pair(x)
    plt.show()
