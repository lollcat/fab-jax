# Flow Annealed Importance Sampling Bootstrap with Jax



# Key tips
 - For FAB to work well we need reasonbly performing AIS, where by reasonble we just mean that AIS produces samples that are
better than samples from the flow by a noticable margin.
If applying FAB to a new problem, make sure that AIS


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