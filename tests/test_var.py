import itertools as it
import pytest
import variational_bayes as vb
import numpy as np
import scipy.stats


@pytest.fixture(params=[37, 13])
def num_nodes(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def order(request):
    return request.param


def test_pack_unpack_coefficients_roundtrip(num_nodes, order):
    adjacency = np.random.normal(0, 1, (num_nodes, num_nodes, order))
    bias = np.random.normal(0, 1, num_nodes)
    packed = vb.pack_coefficients(adjacency, bias)
    unpacked_adjacency, unpacked_bias = vb.unpack_coefficients(packed)
    np.testing.assert_allclose(unpacked_adjacency, adjacency)
    np.testing.assert_allclose(unpacked_bias, bias)


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


def test_diag_adjacency(num_nodes, order):
    adjacency = np.random.normal(0, 1, (num_nodes, num_nodes, order))
    diag = vb.VARDiagAdjacencyDistribution(adjacency)
    for i in range(num_nodes):
        np.testing.assert_allclose(diag.mean[i], adjacency[i, i])
        np.testing.assert_allclose(diag.outer[i], adjacency[i, i, :, None] * adjacency[i, i, None])


def test_diag_adjacency_roundtrip(num_nodes, order):
    idx = np.diag_indices(num_nodes)
    precision = scipy.stats.wishart.rvs(order + 1, np.eye(order), (num_nodes, num_nodes))
    if precision.ndim != 4:
        precision = vb.pad_dims(precision, 4)
    adjacency = vb.MultiNormalDistribution(
        np.random.normal(0, 1, (num_nodes, num_nodes, order)),
        precision
    )
    diag = vb.VARDiagAdjacencyDistribution(adjacency)
    # Evaluate natural parameters for a multinormal distribution
    natural_parameters = {
        'mean': np.einsum('...ij,...j', np.linalg.inv(diag.cov), diag.mean),
        'outer': -0.5 * np.linalg.inv(diag.cov)
    }
    actual_natural_parameters = diag.transform_natural_parameters(None, natural_parameters)
    expected_natural_parameters = adjacency.natural_parameters(None, None)
    for key, expected in expected_natural_parameters.items():
        np.testing.assert_allclose(actual_natural_parameters[key][idx], expected[idx], err_msg=key)


@pytest.fixture
def coefficients(num_nodes, order):
    return vb.MultiNormalDistribution(
        np.random.normal(0, 1, (num_nodes, num_nodes * order + 1)),
        np.ones((num_nodes, 1, 1)) * np.eye(num_nodes * order + 1)
    )


def test_var_bias_distribution(coefficients, num_nodes):
    dist = vb.VARBiasDistribution(coefficients)
    np.testing.assert_allclose(dist.mean, coefficients.mean[..., 0])

    _ = dist.transform_natural_parameters(dist, {
        'mean': np.random.normal(0, 1, num_nodes),
        'square': np.random.gamma(1, 1, num_nodes)
    })


def test_var_adjacency_distribution(coefficients, num_nodes, order):
    dist = vb.VARAdjacencyDistribution(coefficients)
    shape = (num_nodes, num_nodes, order)
    np.testing.assert_allclose(dist.mean, coefficients.mean[..., 1:].reshape(shape))

    _ = dist.transform_natural_parameters(dist, {
        'mean': np.random.normal(0, 1, (num_nodes, num_nodes, order)),
        'outer': np.random.gamma(1, 1, (num_nodes, num_nodes, order, order))
    })


@pytest.mark.parametrize('num_groups', [1, 2, 3])
def test_var_model(num_nodes, order, num_groups):
    x = np.random.normal(0, 1, (1000, num_nodes))
    model = vb.var_model(x, order, num_groups)
    elbos, _ = model.update(1)
    assert elbos[1] > elbos[0]
