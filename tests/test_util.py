import numpy as np
import pytest
import variational_bayes as vb


@pytest.mark.parametrize('shape', [10, (3, 7)])
def test_softmax(shape):
    x = np.random.normal(0, 1, shape)
    proba = vb.softmax(x)
    np.testing.assert_array_less(0, proba)
    np.testing.assert_allclose(np.sum(proba, axis=-1), 1)

