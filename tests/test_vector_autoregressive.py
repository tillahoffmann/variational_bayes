import itertools as it
import pytest
import variational_bayes as vb
import numpy as np


@pytest.mark.parametrize('num_nodes, order', it.product(
    [37, 13],
    [1, 2, 3],
))
def test_pack_unpack_coefficients_roundtrip(num_nodes, order):
    adjacency = np.random.normal(0, 1, (num_nodes, num_nodes, order))
    bias = np.random.normal(0, 1, num_nodes)
    packed = vb.pack_coefficients(adjacency, bias)
    unpacked_adjacency, unpacked_bias = vb.unpack_coefficients(packed)
    np.testing.assert_allclose(unpacked_adjacency, adjacency)
    np.testing.assert_allclose(unpacked_bias, bias)


@pytest.mark.parametrize('num_nodes, order', it.product(
    [37, 13],
    [1, 2, 3],
))
def test_pack_unpack_coefficient_var_roundtrip(num_nodes, order):
    adjacency_var = np.random.normal(0, 1, (num_nodes, num_nodes, order, order))
    bias_var = np.random.normal(0, 1, num_nodes)
    packed = vb.pack_coefficient_var(adjacency_var, bias_var)
    assert packed.shape == (num_nodes, num_nodes * order + 1, num_nodes * order + 1)

    unpacked_adjacency_var, unpacked_bias_var = vb.unpack_coefficient_var(packed)
    np.testing.assert_allclose(unpacked_adjacency_var, adjacency_var)
    np.testing.assert_allclose(unpacked_bias_var, bias_var)

    # Check that most of the matrix is zero
    assert np.sum(packed == 0) == packed.size - adjacency_var.size - bias_var.size
