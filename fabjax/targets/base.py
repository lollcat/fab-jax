from typing import Tuple, Callable, List, Optional, Union

import abc

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from fabjax.train.evaluate import calculate_log_forward_ess

LogProbFn = Callable[[chex.Array], chex.Array]

class Target(abc.ABC):
    def __init__(self,
                 n_model_samples_eval: int,
                 n_target_samples_eval: Optional[int] = None,
                 ):
        self.n_model_samples_eval = n_model_samples_eval
        self.n_target_samples_eval = n_target_samples_eval

    @property
    def can_sample(self) -> bool:
        """Whether the target may be sampled form."""
        return False

    @property
    def log_Z(self) -> Union[int, None]:
        """Normalizing constant."""
        return None

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        raise NotImplemented

    @abc.abstractmethod
    def log_prob(self, value: chex.Array) -> chex.Array:
        raise NotImplemented

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
                 model_log_prob_fn: LogProbFn,
                 model_sample_fn: Callable[[chex.PRNGKey, chex.Shape], chex.Array],
                 key: chex.PRNGKey,
                 ) -> List[plt.Figure]:
        """Visualise samples from the model."""
        return []
