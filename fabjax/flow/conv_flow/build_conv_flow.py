from typing import List

import distrax
import chex

from fabjax.flow.conv_flow.flow import ConvAffineCouplingStack, ConvAffineConfig
from fabjax.flow.conv_flow.aft_translator import FlowAFT

def build_conv_bijector(
    num_elem: int,
    num_coupling_layers: int = 2,
    conv_kernel_shape: List[int] = [3, 3],
    conv_num_middle_layers: int = 1,
    conv_num_middle_channels: int = 10,
    is_torus: bool = True,
    identity_init: bool = True,
) -> distrax.Bijector:

    config = ConvAffineConfig(num_elem=num_elem,
                              conv_kernel_shape=conv_kernel_shape,
                              conv_num_middle_layers=conv_num_middle_layers,
                              is_torus=is_torus,
                              conv_num_middle_channels=conv_num_middle_channels,
                              identity_init=identity_init,
                              num_coupling_layers=num_coupling_layers
                              )
    def build_flow_aft():
        return ConvAffineCouplingStack(config)
    bijector = FlowAFT(build_flow_aft)

    return bijector


if __name__ == '__main__':
    import haiku as hk
    import jax

    num_elem = 14 * 14

    @hk.without_apply_rng
    @hk.transform
    def forward_and_log_det_single(x):
        bijector = build_conv_bijector(
            num_elem=num_elem,
            conv_kernel_shape=[3, 3],
            conv_num_middle_layers=3,
            is_torus=True,
            identity_init=False
        )
        y, logdet = bijector.forward_and_log_det(x)
        return y, logdet

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key=key, shape=(num_elem,))

    params = forward_and_log_det_single.init(key, x)
    y, log_det = forward_and_log_det_single.apply(params, x)

    chex.assert_shape(y, (num_elem,))
    chex.assert_shape(log_det, ())
    assert (log_det != 0.0).all()

    # Check translation equivariance?
    y_trans, log_det_trans = forward_and_log_det_single.apply(params, x + 1.)
    print(y - y_trans)