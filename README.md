# Flow Annealed Importance Sampling Bootstrap (FAB) with Jax
[[FAB paper]](https://arxiv.org/abs/2208.01893)

See `experiments` for training runs on various common problems using FAB.

## Install
```shell
pip install -e .
```

## Key tips
 - Please reach out to us if you would like us to help apply FAB to a problem of interest!
 - To pick hyperparameters begin with the defaults inside `experiments/configs` - these should give a solid starting point.
The most important hyper-parameters to tune are the number of iterations, the batch size, the number of intermediate distributions, the flow architecture, the MCMC transition operator, the learning rate schedule. 
 - For FAB to work well we need SMC to preform reasonably well, where by reasonble we just mean that it produces samples that are
better than samples from the flow by a noticeable margin.
If applying FAB to a new problem, make sure that transition operator works well (e.g. has a well tuned step size).
Having good plotting tools for visualising samples from the flow and SMC can be very helpful for diagnosing performance.
 - For getting started with a new problem we recommend starting with a small toy version of the problem, getting that to work
well, and then to move onto more challenging versions of the problem.


## Library
Key components of FAB:
- `sampling`: Running SMC with a trainable base q, and target p, targetting the optimal distribution for estimating alpha-divergence.
   - `metropolis`: Propose step by adding Gaussian noise to sample and then accept/reject. Includes step size tuning.
   - `hmc`: Hamiltonean Monte Carlo. Simple step size tuning.
- `buffer`: Prioritised replay buffer. 

these are written to be self-contained such that they can be easily ported into an existing code base.

Additionally, we have
 - `flow`: Create minimal real-nvp/spline normalizing flow for the gmm problem (using distrax).
 - `targets`: Target distributions to be fit.
 - `train`: Training script for fab (not modular, but can be copy and pasted and adapted).


## Experiments
Current problems include `cox`, `funnel` `gmm_v0` `gmm_v1` and `many_well`.


## Related Libraries
- [Annealed Flow Transport](https://github.com/google-deepmind/annealed_flow_transport/tree/master): CRAFT algorithm, and many target densities that were used in this library.
- [Diffusion Generative Flow Samples](https://github.com/zdhNarsil/Diffusion-Generative-Flow-Samplers): Sampling from unnormalized targets using diffusion models.
- [fab-torch](https://github.com/lollcat/fab-torch): Original FAB implementation, contains code for alanine dipeptide problem.


## Citation

If you use this code in your research, please cite it as:

> Laurence I. Midgley, Vincent Stimper, Gregor N. C. Simm, Bernhard Schölkopf, José Miguel Hernández-Lobato.
> Flow Annealed Importance Sampling Bootstrap. The Eleventh International Conference on Learning Representations. 2023.

**Bibtex**

```
@inproceedings{
midgley2023flow,
title={Flow Annealed Importance Sampling Bootstrap},
author={Laurence Illing Midgley and Vincent Stimper and Gregor N. C. Simm and Bernhard Sch{\"o}lkopf and Jos{\'e} Miguel Hern{\'a}ndez-Lobato},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=XCTVFJwS9LJ}
}
```