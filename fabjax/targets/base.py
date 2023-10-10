from typing import Tuple, Callable, List, Optional, Union

import abc

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from fabjax.train.evaluate import calculate_log_forward_ess

LogProbFn = Callable[[chex.Array], chex.Array]


class Target(abc.ABC):
    """Abstraction of target distribution that allows our training and evaluation scripts to be generic."""
    def __init__(self,
                 dim: int,
                 log_Z: Optional[float],
                 can_sample: bool,
                 n_plots: int,
                 n_model_samples_eval: int,
                 n_target_samples_eval: Optional[int],
                 ):
        self._n_model_samples_eval = n_model_samples_eval
        self._n_target_samples_eval = n_target_samples_eval
        self._dim = dim
        self._log_Z = log_Z
        self._n_plots = n_plots
        self._can_sample = can_sample

    @property
    def dim(self) -> int:
        """Dimensionality of the problem."""
        return self._dim

    @property
    def n_plots(self) -> int:
        """Number of matplotlib axes that samples are visualized on."""
        return self._n_plots

    @property
    def can_sample(self) -> bool:
        """Whether the target may be sampled form."""
        return self._can_sample

    @property
    def log_Z(self) -> Union[int, None]:
        """Log normalizing constant if available."""
        return self._log_Z

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        raise NotImplemented

    @abc.abstractmethod
    def log_prob(self, value: chex.Array) -> chex.Array:
        """(Possibly unnormalized) target probability density."""

    def evaluate(self,
                 model_log_prob_fn: LogProbFn,
                 model_sample_and_log_prob_fn: Callable[[chex.PRNGKey, chex.Shape], Tuple[chex.Array, chex.Array]],
                 key: chex.PRNGKey,
                 ) -> dict:
        """Evaluate a model. Note that reverse ESS will be estimated separately, so should not be estimated here."""
        key1, key2 = jax.random.split(key)

        info = {}

        if self.can_sample:
            assert self.n_target_samples_eval is not None
            samples_p = self.sample(key1, (self.n_target_samples_eval,))
            log_prob_q = model_log_prob_fn(samples_p)
            log_prob_p = self.log_prob(samples_p)
            log_w = log_prob_p - log_prob_q
            log_forward_ess = calculate_log_forward_ess(log_w, log_Z=self.log_Z)
            info.update(log_lik=jnp.mean(log_prob_q),
                        log_forward_ess=log_forward_ess,
                        forward_ess=jnp.exp(log_forward_ess))
        return info


    @abc.abstractmethod
    def visualise(self,
                  samples: chex.Array,
                  axes: List[plt.Axes],
                  ) -> None:
        """Visualise samples from the model."""
