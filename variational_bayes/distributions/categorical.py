import operator
import numpy as np

from .util import Distribution, statistic, s, Likelihood
from ..util import softmax


class CategoricalLikelihood(Likelihood):
    @staticmethod
    def evaluate(x, proba):  # pylint: disable=W0221
        return np.einsum('...i,...i', s(x, 1), s(proba, 'log'))

    @staticmethod
    def natural_parameters(variable, x, proba):  # pylint: disable=W0221
        if variable == 'x':
            return {
                'mean': s(proba, 'log')
            }
        elif variable == 'proba':
            return {
                'log': s(x, 1)
            }
        else:
            raise KeyError(variable)


class CategoricalDistribution(Distribution):
    """
    Vector categorical distribution.

    Parameters
    ----------
    proba : np.ndarray
        vector of probabilities
    """
    sample_ndim = 1
    _likelihood = CategoricalLikelihood

    def __init__(self, proba):
        super(CategoricalDistribution, self).__init__(proba=proba)

    @statistic
    def mean(self):
        return self._proba

    @statistic
    def square(self):
        return self._proba

    @statistic
    def var(self):
        return self._proba * (1 - self._proba)

    @statistic
    def cov(self):
        _cov = - self._proba[..., None] * self._proba[..., None, :]
        i = np.arange(_cov.shape[-1])
        _cov[..., i, i] += self._proba
        return _cov

    @statistic
    def entropy(self):
        summands = np.log(np.where(self._proba > 0, self._proba, 1.0))
        return - np.sum(self._proba * summands, axis=-1)

    @classmethod
    def from_natural_parameters(cls, natural_parameters):
        return cls(softmax(natural_parameters['mean']))

    def assert_valid_parameters(self):
        np.testing.utils.assert_array_compare(operator.__le__, 0, self._proba,
                                              "proba must be non-negative")
        np.testing.assert_allclose(np.sum(self._proba, axis=-1), 1, err_msg='proba must sum to one')
