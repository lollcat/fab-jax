# Flow Annealed Importance Sampling Bootstrap with Jax


# Library
Key components:
- sampling: Running AIS with a trainable base q, and target p, targetting the optimal distribution for estimating alpha-divergence.
- flow: create normalizing flows (using distrax)
- prioritised buffer (# TODO)

these can be used separately (e.g. the AIS sampler is general and can be used without this libraries specific implementation
of a flow.)