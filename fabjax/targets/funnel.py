import jax.numpy as jnp
import distrax

class FunnelSet():
    """
    x0 ~ N(0, 3^2), xi | x0 ~ N(0, exp(x0)), i = 1, ..., 9
    """
    def __init__(self, dim):
        super().__init__()
        self.data_ndim = dim

        self.dist_dominant = distrax.Normal(jnp.array([0.0]), jnp.array([3.0]))
        self.mean_other = jnp.zeros(dim - 1, dtype=float)
        self.cov_eye = jnp.eye(dim - 1).reshape((1, dim - 1, dim - 1))

    def gt_logz(self):
        return 0.

    def energy(self, x):
        return -self.funner_log_pdf(x)

    def funner_log_pdf(self, x):
        dominant_x = x[:, 0]
        log_density_dominant = self.dist_dominant.log_prob(dominant_x)  # (B, )
        # log_density_other = self._dist_other(dominant_x).log_prob(x[:, 1:])  # (B, )

        log_sigma = 0.5 * x[:, 0:1]
        sigma2 = jnp.exp(x[:, 0:1])
        neglog_density_other = 0.5*jnp.log(2*jnp.pi) + log_sigma + 0.5 * x[:, 1:] ** 2 / sigma2
        log_density_other = jnp.sum(-neglog_density_other, dim=-1)
        return log_density_dominant + log_density_other

    def sample(self, batch_size):
        dominant_x = self.dist_dominant.sample((batch_size,))  # (B,1)
        x_others = self._dist_other(dominant_x).sample()  # (B, dim-1)
        return jnp.hstack([dominant_x, x_others])

    def _dist_other(self, dominant_x):
        variance_other = jnp.exp(dominant_x)
        cov_other = variance_other.view(-1, 1, 1) * self.cov_eye
        # use covariance matrix, not std
        return distrax.MultivariateNormalFullCovariance(self.mean_other, cov_other)
