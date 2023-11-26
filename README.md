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
- `sampling`: Running AIS/SMC with a trainable base q, and target p, targetting the optimal distribution for estimating alpha-divergence.
  - `Transition operators`: Currently the below MCMC transition operators are implemented.
     - `metropolis`: Propose step by adding Gaussian noise to sample and then accept/reject. Includes step size tuning.
     - `hmc`: Hamiltonean Monte Carlo. Simple step size tuning.
  - `point_is_valid_fn`: Provides users ability to specify which points are valid (invalid points are rejected within AIS/SMC). This can improve the efficiency of MCMC, and training stability.
    - `default_point_is_valid_fn`: Default setting. Rejects points with Nan values, or NaN density under the base/target.
    - `point_is_valid_if_in_bounds_fn`: Allows specification of bounds for a problem. Points that fall outside the bounds are rejected.
    - Write your own: The `point_is_valid_fn` is flexible to any criterion - so custom problem specific versions can be easily implemented.
- `buffer`: Prioritised replay buffer. 

these are written to be self-contained such that they can be easily ported into an existing code base.

Additionally, we have
 - `flow`: Create minimal real-nvp/spline normalizing flow for the gmm problem (using distrax).
 - `targets`: Target distributions to be fit.
 - `train`: Training script for fab (not modular, but can be copy and pasted and adapted).


## Experiments
Current problems include `cox`, `funnel` `gmm_v0` `gmm_v1` and `many_well`.

These problems may be run using the command
```shell
python experiments/gmm_v0.py 
```
Additionally we have a quickstart notebook:

<a href="https://colab.research.google.com/github/lollcat/fab-jax/blob/master/experiments/fabjax_quickstart.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


We use the WANDB logger for training. 
If you have a WANDB account then simply change the config inside `experiments/config/{problem_name}.yaml` 
to match your WANDB project. Alternatively a `list_logger` or `pandas_logger` is available if you do not 
use WANDB (the list logger is used inside the Quickstart notebook). 



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