# Flow Annealed Importance Sampling Bootstrap (FAB) with Jax
See [`examples/gmm.py`] for a simple example of running FAB.
This library is currently being built - I will be steadily adding more features and improving the documentation.

# Key tips
 - For FAB to work well we need SMC to preform reasonably well, where by reasonble we just mean that it produces samples that are
better than samples from the flow by a noticeable margin.
If applying FAB to a new problem, make sure that transition operator has a well tuned step size.
Having good plotting tools for visualising samples from the flow and SMC can be very helpful for diagnosing performance.
 - For getting started with a new problem we recommend starting with a small toy version of the problem, getting that to work
well, and then to move onto more challenging versions of the problem. 


# Library
Key components of FAB:
- `sampling`: Running SMC with a trainable base q, and target p, targetting the optimal distribution for estimating alpha-divergence.
   - `metropolis`: Propose step by adding Gaussian noise to sample and then accept/reject. Includes step size tuning.
   - `hmc`: Hamiltonean Monte Carlo. Simple step size tuning.
- `buffer`: Prioritised replay buffer. 

these are written to be self-contained such that they can be easily ported into an existing code base.

Additionally, we have
 - `flow`: Create minimal realnvp normalizing flow for the gmm problem (using distrax).
 - `targets`: energy functions for training
 - `train`: training script for fab (not modular, but can be copy and pasted and adapted)


# TODO:
 - Improve stability. By making the GMM problem very hard we can break training, which is useful for finding 
the most unstable parts of the code. 
 - Use sum-tree for prioritised buffer implementation. Would be nice to do everything with log probs.
 - Make nice target distribution abstractions. 
 - Could use exponential moving average for flow params used in SMC (i.e. target network).
 - Add `jaxtyping`. 