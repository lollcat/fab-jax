from jax.flatten_util import ravel_pytree


def assert_trees_all_different(tree1, tree2):
    assert (ravel_pytree(tree1)[0] != ravel_pytree(tree2)[0]).all()