# Flow Annealed Importance Sampling Bootstrap with Jax


# Library
Key components of FAB:
- sampling: Running AIS with a trainable base q, and target p, targetting the optimal distribution for estimating alpha-divergence.
- prioritised buffer (# TODO)
these can be used separately (e.g. the AIS sampler is general and can be used without this libraries specific implementation
of a flow.)

Additionally, we have
 - flow: create normalizing flows (using distrax)
 - targets: energy functions for training
 - train: training script for fab (not modular, but can be copy and pasted and adapted)


# TODO:
 - prioritised buffer
 - make nice target distribution abstractions