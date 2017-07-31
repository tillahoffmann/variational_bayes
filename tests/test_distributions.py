import pickle
import numpy as np
import scipy.stats
import pytest
import variational_bayes as vb


@pytest.fixture(params=[(), (10, ), (2, 3)])
def batch_shape(request):
    return request.param


@pytest.fixture(params=[
    vb.NormalDistribution,
    vb.GammaDistribution,
    vb.CategoricalDistribution,
    vb.MultiNormalDistribution,
    vb.WishartDistribution,
    vb.BetaDistribution,
    vb.BernoulliDistribution,
    vb.DirichletDistribution,
])
def distribution(request, batch_shape):
    cls = request.param
    if cls is vb.NormalDistribution:
        return cls(np.random.normal(0, 1, batch_shape), np.random.gamma(1, 1, batch_shape))
    elif cls is vb.GammaDistribution:
        return cls(np.random.gamma(1, 1, batch_shape), np.random.gamma(1, 1, batch_shape))
    elif cls is vb.CategoricalDistribution:
        return cls(np.random.dirichlet(np.ones(3), batch_shape))
    elif cls is vb.MultiNormalDistribution:
        return cls(np.random.normal(0, 1, batch_shape + (3,)),
                   scipy.stats.wishart.rvs(3, np.eye(3), size=batch_shape or 1))
    elif cls is vb.WishartDistribution:
        return cls(3 + np.random.gamma(1, 1, batch_shape),
                   scipy.stats.wishart.rvs(3, np.eye(3), size=batch_shape or 1))
    elif cls is vb.BetaDistribution:
        return cls(np.random.gamma(5, 1, batch_shape), np.random.gamma(5, 1, batch_shape))
    elif cls is vb.BernoulliDistribution:
        return cls(np.random.uniform(0, 1, batch_shape))
    elif cls is vb.DirichletDistribution:
        return cls(np.random.gamma(5, 1, batch_shape + (3,)))
    else:
        raise KeyError(cls)


@pytest.fixture
def scipy_distribution(batch_shape, distribution):
    if batch_shape:
        pytest.skip("batch_shape is not empty: %s" % [batch_shape])
    elif isinstance(distribution, vb.NormalDistribution):
        return scipy.stats.norm(distribution._mean, 1 / np.sqrt(distribution._precision))
    elif isinstance(distribution, vb.GammaDistribution):
        return scipy.stats.gamma(distribution._shape, scale=1.0 / distribution._scale)
    elif isinstance(distribution, vb.CategoricalDistribution):
        return scipy.stats.multinomial(1, distribution._proba)
    elif isinstance(distribution, vb.MultiNormalDistribution):
        return scipy.stats.multivariate_normal(distribution._mean,
                                               np.linalg.inv(distribution._precision))
    elif isinstance(distribution, vb.WishartDistribution):
        return scipy.stats.wishart(np.asscalar(distribution._shape),
                                   np.linalg.inv(distribution._scale))
    elif isinstance(distribution, vb.BetaDistribution):
        return scipy.stats.beta(distribution._a, distribution._b)
    elif isinstance(distribution, vb.BernoulliDistribution):
        return scipy.stats.bernoulli(distribution._proba)
    elif isinstance(distribution, vb.DirichletDistribution):
        return scipy.stats.dirichlet(distribution._alpha)
    else:
        raise KeyError(distribution)


def _get_or_call(x):
    return x() if callable(x) else x


def _scipy_statistic(dist, statistic):
    if hasattr(dist, statistic):
        return _get_or_call(getattr(dist, statistic))
    elif statistic == 'var' and hasattr(dist, 'cov'):
        return np.diag(_get_or_call(getattr(dist, 'cov')))
    elif statistic == 'std':
        return np.sqrt(_scipy_statistic(dist, 'var'))
    else:
        num_samples = 1000
        x = dist.rvs(num_samples)
        x = vb.evaluate_statistic(x, statistic)
        return {
            'mean': np.mean(x, axis=0),
            'std': np.std(x, axis=0) / np.sqrt(num_samples - 1)
        }


def test_finite_statistics(distribution):
    for statistic in distribution.statistics:
        if statistic == 'interaction' and distribution._proba.ndim != 2:
            continue
        value = vb.evaluate_statistic(distribution, statistic)
        assert np.all(np.isfinite(value))


def test_interaction_statistic():
    num_nodes = 20
    num_groups = 3
    q = vb.CategoricalDistribution(np.ones((num_nodes, num_groups)) / num_groups)
    actual = q.interaction
    # Check the shpae
    assert actual.shape == (num_nodes, num_nodes, num_groups, num_groups)  # pylint: disable=E1101

    # Simulate samples and average
    samples = np.random.randint(0, num_groups, (1000, num_nodes))
    samples = [vb.evaluate_statistic(vb.onehot(x, num_groups), 'interaction') for x in samples]
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0) / np.sqrt(len(samples) - 1)
    z = (actual - mean) / std
    # Ensure both the mean and the computed statistic are zero where the std is zero
    assert np.all(mean[std == 0] == 0)
    assert np.all(actual[std == 0] == 0)
    # Remove the nans
    z[np.isnan(z)] = 0
    np.testing.assert_array_less(np.abs(z), 5)


def test_compare_statistics(distribution, scipy_distribution):
    """Test that all statistics match the scipy implementation."""
    for statistic in distribution.statistics:
        if statistic == 'interaction':
            continue
        actual = vb.evaluate_statistic(distribution, statistic)
        assert np.all(np.isfinite(actual)), "non-finite statistic"
        desired = _scipy_statistic(scipy_distribution, statistic)
        if not isinstance(desired, dict):
            np.testing.assert_allclose(actual, desired, err_msg="statistic '%s' of %s does not "
                                       "match the scipy implementation" % (statistic, distribution))
        else:
            mean, std = desired['mean'], desired['std']
            z = (mean - actual) / std
            np.testing.assert_array_less(z, 5, "statistic '%s' of %s does not match the scipy "
                                         "implementation at five sigma: expected %s +- %s but got "
                                         "%s" % (statistic, distribution, mean, std, actual))


def test_log_proba(distribution, scipy_distribution):
    x = scipy_distribution.rvs()
    likelihood = distribution.likelihood(x)
    # For some reason, scipy can't handle its own RVs as input
    if isinstance(distribution, vb.DirichletDistribution):
        x = x[0]

    actual = likelihood.evaluate()

    desired = None
    for method in ['logpdf', 'logpmf']:
        if hasattr(scipy_distribution, method):
            try:
                desired = getattr(scipy_distribution, method)(x)
                break
            except AttributeError:
                pass

    assert desired is not None, 'cannot evaluate proba of scipy distribution'

    np.testing.assert_allclose(actual, desired, err_msg="failed to reproduce log proba for %s" %
                               distribution)


def test_natural_parameters_roundtrip(distribution):
    # Get the natural parameters
    likelihood = distribution.likelihood(distribution)
    natural_parameters = likelihood.natural_parameters(distribution)

    # Check the exact shape
    for key, value in natural_parameters.items():
        assert value.shape == getattr(distribution, key).shape, "shape of %s does not " \
            "match" % key

    # Reconstruct the distribution
    reconstructed = distribution.__class__.from_natural_parameters(natural_parameters)

    assert set(reconstructed.parameters) == set(distribution.parameters), \
        "reconstructed %s has different parameters" % reconstructed

    for key, value in reconstructed.parameters.items():
        np.testing.assert_allclose(value, distribution.parameters[key],
                                   err_msg="failed to reconstruct %s of %s" % (key, distribution))


def test_natural_parameters(distribution):
    likelihood = distribution.likelihood(distribution)
    for variable in list(distribution.parameters.values()):
        try:
            natural_parameters = likelihood.natural_parameters(variable)

            # Validate the natural parameters
            for key, value in natural_parameters.items():
                assert value is not None, "%s is None" % key
                assert np.all(np.isfinite(value)), "%s is not finite" % key
                # The leading dimensions must match the batch shape of the distribution
                ndim = min(value.ndim, distribution.mean.ndim)
                assert value.shape[:ndim] == distribution.mean.shape[:ndim], "%s does not match " \
                    "leading shape" % key
        except NotImplementedError:
            pass


@pytest.mark.parametrize('extra_dims', [0, 1, 2])
def test_aggregate_natural_parameters(distribution, extra_dims):
    likelihood = distribution.likelihood(distribution)
    # Get the natural parameters
    natural_parameters = likelihood.natural_parameters(distribution)
    # Add extra dimensions to the natural parameters
    x = {key : np.reshape(value, (1,) * extra_dims + value.shape)
         for key, value in natural_parameters.items()}

    actual = distribution.aggregate_natural_parameters([x])

    for key, value in natural_parameters.items():
        np.testing.assert_allclose(actual[key], value, err_msg='failed to aggregate %s' % key)


def test_reshape():
    # Generate data
    x = np.random.normal(0, 1, (100, 5, 5))
    # Define factors and likelihood (using reshape)
    q = vb.NormalDistribution(np.random.normal(0, 1, 25), np.random.gamma(1, 1, 25))
    newshape = (5, 5)
    reshaped = vb.ReshapedDistribution(q, newshape)

    # Check the statistics
    for statistic in q.statistics:
        actual = vb.s(reshaped, statistic)
        desired = vb.s(q, statistic).reshape(newshape)
        np.testing.assert_allclose(actual, desired)

    # Check the natural parameters
    likelihood = vb.NormalDistribution(reshaped, 1).likelihood(x)
    natural_parameters = likelihood.natural_parameters(q)
    natural_parameters = q.aggregate_natural_parameters([natural_parameters])
    np.testing.assert_allclose(natural_parameters['mean'], np.sum(x, axis=0).ravel(),
                               err_msg="'mean' natural parameter does not match")
    np.testing.assert_allclose(natural_parameters['square'], - 0.5 * 100,
                               err_msg="'square' natural parameter does not match")


def test_pickle(distribution):
    reconstructed = pickle.loads(pickle.dumps(distribution))
    np.testing.assert_allclose(distribution.mean, reconstructed.mean)


def test_copy():
    d = vb.NormalDistribution(np.random.normal(0, 1, 10), np.ones(10))
    c = d.copy()
    d.parameters['precision'] *= 10
    np.testing.assert_allclose(d.mean, c.mean)
    np.testing.assert_equal(d._precision, 10)
    np.testing.assert_equal(c._precision, 1)
