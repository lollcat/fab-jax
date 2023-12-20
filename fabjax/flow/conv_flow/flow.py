"""Taken from https://github.com/google-deepmind/annealed_flow_transport."""

from typing import Tuple, Callable, NamedTuple

import abc

import chex
import jax.numpy as jnp
import jax
import numpy as np
import haiku as hk
import ml_collections


Array = chex.Array
NpArray = np.ndarray
ConfigDict = ml_collections.ConfigDict
Samples = chex.ArrayTree
SampleShape = chex.Shape


class ConfigurableFlow(hk.Module, abc.ABC):
  """Abstract base clase for configurable normalizing flows.

  This is the interface expected by all flow based algorithms called in train.py
  """

  def __init__(self, config: ConfigDict):
    super().__init__()
    self._check_configuration(config)
    self._config = config

  def _check_input(self, x: Samples):
    chex.assert_rank(x, 2)

  def _check_outputs(self, x: Samples, transformed_x: Samples,
                     log_abs_det_jac: Array):
    chex.assert_rank(x, 2)
    chex.assert_equal_shape([x, transformed_x])
    num_batch = x.shape[0]
    chex.assert_shape(log_abs_det_jac, (num_batch,))

  def _check_members_types(self, config: ConfigDict, expected_members_types):
    for elem, elem_type in expected_members_types:
      if elem not in config:
        raise ValueError('Flow config element not found: ', elem)
      if not isinstance(config[elem], elem_type):
        msg = 'Flow config element '+elem+' is not of type '+str(elem_type)
        raise TypeError(msg)

  def __call__(self, x: Samples) -> Tuple[Samples, Array]:
    """Call transform_and_log abs_det_jac with automatic shape checking.

    This calls transform_and_log_abs_det_jac which needs to be implemented
    in derived classes.

    Args:
      x: input samples to flow.
    Returns:
      output samples and (num_batch,) log abs det Jacobian.
    """
    self._check_input(x)
    vmapped = hk.vmap(self.transform_and_log_abs_det_jac, split_rng=False)
    output, log_abs_det_jac = vmapped(x)
    self._check_outputs(x, output, log_abs_det_jac)
    return output, log_abs_det_jac

  def inverse(self, x: Samples) -> Tuple[Samples, Array]:
    """Call transform_and_log abs_det_jac with automatic shape checking.

    This calls transform_and_log_abs_det_jac which needs to be implemented
    in derived classes.

    Args:
      x: input to flow
    Returns:
      output and (num_batch,) log abs det Jacobian.
    """
    self._check_input(x)
    vmapped = hk.vmap(self.inv_transform_and_log_abs_det_jac, split_rng=False)
    output, log_abs_det_jac = vmapped(x)
    self._check_outputs(x, output, log_abs_det_jac)
    return output, log_abs_det_jac

  @abc.abstractmethod
  def transform_and_log_abs_det_jac(self, x: Array) -> Tuple[Array, Array]:
    """Transform x through the flow and compute log abs determinant of Jacobian.

    Args:
      x: (num_dim,) input to the flow.
    Returns:
      Array size (num_dim,) containing output and Scalar log abs det Jacobian.
    """

  def inv_transform_and_log_abs_det_jac(self, x: Array) -> Tuple[Array, Array]:
    """Transform x through inverse and compute log abs determinant of Jacobian.

    Args:
      x: (num_dim,) input to the flow.
    Returns:
      Array size (num_dim,) containing output and Scalar log abs det Jacobian.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def _check_configuration(self, config: ConfigDict):
    """Check the configuration includes the necessary fields.

    Will typically raise Assertion like errors.

    Args:
      config: A ConfigDict include the fields required by the flow.
    """


def expand_periodic_dim(x: Array, num_extra_vals: int):
  if num_extra_vals == 0:
    return x
  first = x[-num_extra_vals:, :]
  last = x[:num_extra_vals, :]
  return jnp.vstack([first, x, last])

def pad_periodic_2d(x: Array, kernel_shape) -> Array:
  """Pad x to be have the required extra terms at the edges."""
  assert len(kernel_shape) == 2
  chex.assert_rank(x, 2)
  # this code is unbatched
  # we require that kernel shape has odd rows/cols.
  is_even = False
  for elem in kernel_shape:
    is_even = is_even or (elem % 2 == 0)
  if is_even:
    raise ValueError('kernel_shape is assumed to have odd rows and cols')
  # calculate num extra rows/cols each side.
  num_extra_row = (kernel_shape[0] - 1) // 2
  num_extra_col = (kernel_shape[1] -1) // 2
  row_expanded_x = expand_periodic_dim(x,
                                       num_extra_row)
  col_expanded_x = expand_periodic_dim(row_expanded_x.T,
                                       num_extra_col).T
  return col_expanded_x

def batch_pad_periodic_2d(x: Array, kernel_shape) -> Array:
  assert len(kernel_shape) == 2
  chex.assert_rank(x, 4)
  batch_func = jax.vmap(pad_periodic_2d, in_axes=(0, None))
  batch_channel_func = jax.vmap(batch_func, in_axes=(3, None), out_axes=3)
  return batch_channel_func(x, kernel_shape)


class Conv2DTorus(hk.Conv2D):
  """Convolution in 2D with periodic boundary conditions.

  Strides are ignored and this is not checked.
  kernel_shapes is a tuple (a, b) where a and b are odd positive integers.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, padding='VALID', **kwargs)

  def __call__(self, x: Array) -> Array:
    padded_x = batch_pad_periodic_2d(x, self.kernel_shape)
    return super().__call__(padded_x)



class FullyConvolutionalNetwork(hk.Module):
  """A fully convolutional network with ResNet middle layers."""

  def __init__(self,
               num_middle_channels: int = 5,
               num_middle_layers: int = 2,
               num_final_channels: int = 2,
               kernel_shape: Tuple[int] = (3, 3),
               zero_final: bool = True,
               is_torus: bool = False):  # pytype: disable=annotation-type-mismatch
    super().__init__()
    self._num_middle_channels = num_middle_channels
    self._num_middle_layers = num_middle_layers
    self._num_final_channels = num_final_channels
    self._kernel_shape = kernel_shape
    self._zero_final = zero_final
    self._is_torus = is_torus

  def __call__(self,
               x: Array):
    """Call the residual network on x.

    Args:
      x: is of shape (length_a, length_b)
    Returns:
      Array of shape (length_a, length_b, num_channels[-1])
    """
    chex.assert_rank(x, 2)
    length_a, length_b = jnp.shape(x)
    non_linearity = jax.nn.relu
    if self._is_torus:
      conv_two_d = Conv2DTorus
    else:
      conv_two_d = hk.Conv2D
    # Cast to batch size of one and one channel in last index.
    representation = x[None, :, :, None]

    for middle_layer_index in range(self._num_middle_layers):
      if middle_layer_index == 0:
        representation = conv_two_d(
            output_channels=self._num_middle_channels,
            stride=1,
            kernel_shape=self._kernel_shape,
            with_bias=True)(representation)
        representation = non_linearity(representation)
      else:
        conv_result = conv_two_d(
            output_channels=self._num_middle_channels,
            stride=1,
            kernel_shape=self._kernel_shape,
            with_bias=True)(representation)
        representation = representation + non_linearity(conv_result)
    if self._zero_final:
      representation = conv_two_d(
          output_channels=self._num_final_channels,
          stride=1,
          kernel_shape=self._kernel_shape,
          with_bias=True,
          w_init=jnp.zeros,
          b_init=jnp.zeros)(representation)
    else:
      representation = conv_two_d(
          output_channels=self._num_final_channels,
          stride=1,
          kernel_shape=self._kernel_shape,
          with_bias=True)(representation)
    chex.assert_shape(representation,
                      [1, length_a, length_b, self._num_final_channels])
    # Remove extraneous batch index of size 1.
    representation = representation[0, :, :, :]
    return representation



class CouplingLayer(object):
  """A generic coupling layer.

  Takes the following functions as inputs.
  1) A conditioner network mapping from event_shape->event_shape + (num_params,)
  2) Mask of shape event_shape.
  3) transformer A map from event_shape -> event_shape that acts elementwise on
  the terms to give a diagonal Jacobian expressed as shape event_shape and in
  abs-log space.
  It is parameterised by parameters of shape params_shape.

  """

  def __init__(self, conditioner_network: Callable[[Array], Array], mask: Array,
               transformer):
    self._conditioner_network = conditioner_network
    self._mask = mask
    self._transformer = transformer

  def __call__(self, x):
    """Transform x with coupling layer.

    Args:
      x: event_shape Array.
    Returns:
      output_x: event_shape Array corresponding to the output.
      log_abs_det: scalar corresponding to the log abs det Jacobian.
    """
    mask_complement = 1. - self._mask
    masked_x = x * self._mask
    chex.assert_equal_shape([masked_x, x])
    transformer_params = self._conditioner_network(masked_x)
    transformed_x, log_abs_dets = self._transformer(transformer_params, x)
    output_x = masked_x + mask_complement * transformed_x
    chex.assert_equal_shape([transformed_x,
                             output_x,
                             x,
                             log_abs_dets])
    log_abs_det = jnp.sum(log_abs_dets * mask_complement)
    return output_x, log_abs_det

  def inverse(self, y):
    """Transform y with inverse coupling layer.

    Args:
      y: event_shape Array.
    Returns:
      output_y: event_shape Array corresponding to the output.
      log_abs_det: scalar corresponding to the log abs det Jacobian.
    """
    mask_complement = 1. - self._mask
    masked_y = y * self._mask
    chex.assert_equal_shape([masked_y, y])
    transformer_params = self._conditioner_network(masked_y)
    transformed_y, log_abs_dets = self._transformer.inverse(transformer_params,
                                                            y)
    output_y = masked_y + mask_complement * transformed_y
    chex.assert_equal_shape([transformed_y,
                             output_y,
                             y,
                             log_abs_dets])
    log_abs_det = jnp.sum(log_abs_dets * mask_complement)
    return output_y, log_abs_det


def affine_transformation(params: Array,
                          x: Array) -> Tuple[Array, Array]:
  shift = params[0]
  # Assuming params start as zero adding 1 to scale gives identity transform.
  scale = params[1] + 1.
  output = x * scale + shift
  return output, jnp.log(jnp.abs(scale))


def inverse_affine_transformation(params: Array,
                                  y: Array) -> Tuple[Array, Array]:
  shift = params[0]
  # Assuming params start as zero adding 1 to scale gives identity transform.
  scale = params[1] + 1.
  output = (y - shift) / scale
  return output, -1.*jnp.log(jnp.abs(scale))


class AffineTransformer:

  def __call__(self, params: Array, x: Array) -> Tuple[Array, Array]:
    vectorized_affine = jnp.vectorize(affine_transformation,
                                      signature='(k),()->(),()')
    return vectorized_affine(params, x)

  def inverse(self, params: Array, y: Array) -> Tuple[Array, Array]:
    vectorized_affine = jnp.vectorize(inverse_affine_transformation,
                                      signature='(k),()->(),()')
    return vectorized_affine(params, y)


class ConvAffineCoupling(CouplingLayer):
  """A convolutional affine coupling layer."""

  def __init__(self,
               mask: Array,
               conv_num_middle_channels: int = 5,
               conv_num_middle_layers: int = 2,
               conv_kernel_shape: Tuple[int] = (3, 3),
               identity_init: bool = True,
               is_torus: bool = False):  # pytype: disable=annotation-type-mismatch
    
    conv_net = FullyConvolutionalNetwork(
        num_middle_channels=conv_num_middle_channels,
        num_middle_layers=conv_num_middle_layers,
        num_final_channels=2,
        kernel_shape=conv_kernel_shape,
        zero_final=identity_init,
        is_torus=is_torus)
    vectorized_affine = AffineTransformer()

    super().__init__(conv_net,
                     mask,
                     vectorized_affine)


def get_checkerboard_mask(overall_shape: Tuple[int, int],
                          period: int):
  range_a = jnp.arange(overall_shape[0])
  range_b = jnp.arange(overall_shape[1])
  def modulo_func(index_a, index_b):
    return jnp.mod(index_a+index_b+period, 2)
  func = lambda y: jax.vmap(modulo_func, in_axes=[0, None])(range_a, y)
  vals = func(range_b)
  chex.assert_shape(vals, overall_shape)
  return vals


class ConvAffineConfig(NamedTuple):
    num_elem: int
    num_coupling_layers: int
    conv_kernel_shape: list[int]
    conv_num_middle_layers: int
    conv_num_middle_channels: int
    is_torus: bool
    identity_init: bool


class ConvAffineCouplingStack(ConfigurableFlow):
  """A stack of convolutional affine coupling layers."""

  def __init__(self, config: ConvAffineConfig):
    super().__init__(config=ConfigDict(config._asdict()))
    num_elem = config.num_elem
    num_grid_per_dim = int(np.sqrt(num_elem))
    assert num_grid_per_dim * num_grid_per_dim == num_elem
    self._true_shape = (num_grid_per_dim, num_grid_per_dim)
    self._coupling_layers = []
    for index in range(self._config.num_coupling_layers):
      mask = get_checkerboard_mask(self._true_shape, index)
      coupling_layer = ConvAffineCoupling(
          mask,
          conv_kernel_shape=self._config.conv_kernel_shape,
          conv_num_middle_layers=self._config.conv_num_middle_layers,
          conv_num_middle_channels=self._config.conv_num_middle_channels,
          is_torus=self._config.is_torus,
          identity_init=self._config.identity_init
      )
      self._coupling_layers.append(coupling_layer)

  def _check_configuration(self, config: ConvAffineConfig):
    expected_members_types = [
        ('conv_kernel_shape', list),
        ('conv_num_middle_layers', int),
        ('conv_num_middle_channels', int),
        ('is_torus', bool),
        ('identity_init', bool)
    ]

    self._check_members_types(config, expected_members_types)

  def transform_and_log_abs_det_jac(self, x: Array) -> Tuple[Array, Array]:
    reshaped_x = jnp.reshape(x, self._true_shape)
    transformed_x = reshaped_x
    log_abs_det = 0.
    for index in range(self._config.num_coupling_layers):
      coupling_layer = self._coupling_layers[index]
      transformed_x, log_det_increment = coupling_layer(transformed_x)
      chex.assert_equal_shape([transformed_x, reshaped_x])
      log_abs_det += log_det_increment
    restored_x = jnp.reshape(transformed_x, x.shape)
    return restored_x, log_abs_det

  def inv_transform_and_log_abs_det_jac(self, x: Array) -> tuple[Array, Array]:
    reshaped_x = jnp.reshape(x, self._true_shape)
    transformed_x = reshaped_x
    log_abs_det = 0.
    for index in range(self._config.num_coupling_layers-1, -1, -1):
      coupling_layer = self._coupling_layers[index]
      transformed_x, log_det_increment = coupling_layer.inverse(transformed_x)
      chex.assert_equal_shape([transformed_x, reshaped_x])
      log_abs_det += log_det_increment
    restored_x = jnp.reshape(transformed_x, x.shape)
    return restored_x, log_abs_det