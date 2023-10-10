from typing import Tuple, Callable

import abc

import chex

from fabjax.sampling.base import LogProbFn

class Target(abc.ABC):
    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        raise NotImplemented

    @abc.abstractmethod
    def log_prob(self, value: chex.Array) -> chex.Array:
        raise NotImplemented

    @abc.abstractmethod
    def evaluate(self,
                 model_log_prob_fn: LogProbFn,
                 model_sample_and_log_prob_fn: Callable[[chex.PRNGKey, chex.Shape], Tuple[chex.Array, chex.Array]],
                 key: chex.PRNGKey,
                 ) -> dict:
        """Evaluate a model."""
        raise NotImplemented


    @abc.abstractmethod
    def visualise(self,
                 model_log_prob_fn: LogProbFn,
                 model_sample_fn: Callable[[chex.PRNGKey, chex.Shape], chex.Array],
                 key: chex.PRNGKey,
                 ) -> dict:
        """Visualise samples from the model."""
        raise NotImplemented
