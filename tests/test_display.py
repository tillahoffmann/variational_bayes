import numpy as np
import variational_bayes as vb
import pytest


@pytest.fixture(params=[(), (10, ), (3, 5)])
def distribution(request):
    shape = request.param
    return vb.NormalDistribution(np.random.normal(0, 1, shape),
                                 np.random.gamma(1, 1, shape))


def test_plot_proba(distribution):
    vb.plot_proba(distribution, reference=distribution['mean'])


def test_plot_comparison(distribution):
    vb.plot_comparison(distribution, distribution['mean'])
