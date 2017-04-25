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
    else:
        raise KeyError(cls)


@pytest.fixture
def scipy_distribution(batch_shape, distribution):
    if batch_shape != tuple():
        pytest.skip()
    elif isinstance(distribution, vb.NormalDistribution):
        return scipy.stats.norm(distribution['mean'], 1 / np.sqrt(distribution['precision']))
    elif isinstance(distribution, vb.GammaDistribution):
        return scipy.stats.gamma(distribution['shape'], scale=1.0 / distribution['scale'])
    elif isinstance(distribution, vb.CategoricalDistribution):
        return scipy.stats.multinomial(1, distribution['proba'])
    elif isinstance(distribution, vb.MultiNormalDistribution):
        return scipy.stats.multivariate_normal(distribution['mean'],
                                               np.linalg.inv(distribution['precision']))
    elif isinstance(distribution, vb.WishartDistribution):
        return scipy.stats.wishart(distribution['shape'], np.linalg.inv(distribution['scale']))
    else:
        raise KeyError(distribution)


def test_evaluate_statistics(distribution):
    """Test that all statistics are finite."""
    for statistic in distribution.statistics:
        value = getattr(distribution, statistic)
        assert np.all(np.isfinite(value)), "statistic '%s' of %s is not finite" % \
            (statistic, distribution)


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


def test_compare_statistics(distribution, scipy_distribution):
    """Test that all statistics match the scipy implementation."""
    for statistic in distribution.statistics:
        actual = getattr(distribution, statistic)
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
    actual = distribution.log_proba(x)
    if hasattr(scipy_distribution, 'logpdf'):
        desired = scipy_distribution.logpdf(x)
    elif hasattr(scipy_distribution, 'logpmf'):
        desired = scipy_distribution.logpmf(x)
    else:
        raise RuntimeError('cannot evaluate proba of scipy distribution')

    np.testing.assert_allclose(actual, desired, err_msg="failed to reproduce log proba for %s" %
                               distribution)
