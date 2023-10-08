import jax.numpy as jnp
import distrax


"""
2-D Guassian mixture
https://github.com/zdhNarsil/Diffusion-Generative-Flow-Samplers/blob/main/target/distribution/gm.py
"""
class GaussianMixture2D:
    def __init__(self, scale=0.5477222):
        super().__init__()
        mean_ls = [
            [-5., -5.], [-5., 0.], [-5., 5.],
            [0., -5.], [0., 0.], [0., 5.],
            [5., -5.], [5., 0.], [5., 5.],
        ]
        nmode = len(mean_ls)
        mean = jnp.stack([jnp.array(xy) for xy in mean_ls])
        comp = distrax.Independent(
            distrax.Normal(loc=mean, scale=jnp.ones_like(mean)*scale),
            reinterpreted_batch_ndims=1
        )
        mix = distrax.Categorical(logits=jnp.ones(nmode))
        self.gmm = distrax.MixtureSameFamily(mixture_distribution=mix,
                                             components_distribution=comp)

    def log_prob(self, x):
        log_prob = self.gmm.log_prob(x)
        return log_prob


    def sample(self, seed, sample_shape):
        return self.gmm.sample(seed=seed, sample_shape=sample_shape)

    # def viz_pdf(self, fsave="ou-density.png"):
    #     x = torch.linspace(-8, 8, 100).cuda()
    #     y = torch.linspace(-8, 8, 100).cuda()
    #     X, Y = torch.meshgrid(x, y)
    #     x = torch.stack([X.flatten(), Y.flatten()], dim=1) #?
    #
    #     density = self.unnorm_pdf(x)
    #     # x, pdf = as_numpy([x, density])
    #     x, pdf = torch.from_numpy(x), torch.from_numpy(density)
    #
    #     fig, axs = plt.subplots(1, 1, figsize=(1 * 7, 1 * 7))
    #     axs.plot(x, pdf)
    #
    #     # plt.contourf(X, Y, density, levels=20, cmap='viridis')
    #     # plt.colorbar()
    #     # plt.xlabel('x')
    #     # plt.ylabel('y')
    #     # plt.title('2D Function Plot')
    #
    #     fig.savefig(fsave)
    #     plt.close(fig)
    #
    # def __getitem__(self, idx):
    #     del idx
    #     return self.data[0]
