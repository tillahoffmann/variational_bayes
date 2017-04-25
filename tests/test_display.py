import numpy as np
import variational_bayes as vb
import pytest


@pytest.mark.parametrize('shape', [(), (10, ), (3, 5)])
def test_plot_proba(shape):
    dist = vb.NormalDistribution(np.random.normal(0, 1, shape),
                                 np.random.gamma(1, 1, shape))

    vb.plot_proba(dist)
