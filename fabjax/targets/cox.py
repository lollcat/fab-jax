from typing import List

import os.path as osp
import pathlib

import chex
import jax.numpy as jnp
import matplotlib.pyplot as plt

from fabjax.targets.base import Target
from fabjax.targets.cox_utils import Cox



class CoxDist(Target):
    def __init__(self, num_bins_per_dim: int = 40):
        fcsv = osp.join(pathlib.Path(__file__).parent.resolve(), "df_pines.csv")
        self.num_bins_per_dim = num_bins_per_dim
        self.cox = Cox(fcsv, num_bins_per_dim, use_whitened=False)

        dim = int(num_bins_per_dim ** 2)

        super().__init__(dim=dim, log_Z=self.gt_logz(), can_sample=False, n_plots=0,
                         n_model_samples_eval=1000, n_target_samples_eval=1000)

    def gt_logz(self):
        if self.num_bins_per_dim == 40:
            return 501.806  # from long run SMC
        elif self.num_bins_per_dim == 32:
            return 503.3939
        else:
            return None

    def log_prob(self, x: chex.Array):
        batched = x.ndim == 2
        if not batched:
            x = x[None, ]
        log_prob = self.cox.evaluate_log_density(x)
        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)
        return log_prob

    def visualise(self,
                  samples: chex.Array,
                  axes: List[plt.Axes],
                  ) -> None:
        return None

