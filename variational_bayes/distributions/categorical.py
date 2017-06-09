import operator
import numpy as np

from .distribution import Distribution, statistic, s, is_dependent
from ..util import softmax


class CategoricalDistribution(Distribution):
    """
    Vector categorical distribution.

    Parameters
    ----------
    proba : np.ndarray
        vector of probabilities
    """
    sample_ndim = 1

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
    def interaction(self):
        assert self.proba.ndim == 2, "interaction statistic is only defined for 2D probability matrix"
        # Compute the outer product across both
        zz = np.einsum('ik,jl->ijkl', self.mean, self.mean)
        # Add the covariance to the diagonal terms
        zz[np.diag_indices(self.proba.shape[0])] += self.cov
        return zz

    @statistic
    def entropy(self):
        summands = np.log(np.where(self._proba > 0, self._proba, 1.0))
        return - np.sum(self._proba * summands, axis=-1)

    @staticmethod
    def canonical_parameters(natural_parameters):
        return {
            'proba': softmax(natural_parameters.pop('mean'))
        }

    def assert_valid_parameters(self):
        proba = s(self._proba, 1)
        np.testing.utils.assert_array_compare(operator.__le__, 0, proba,
                                              "proba must be non-negative")
        np.testing.assert_allclose(np.sum(proba, axis=-1), 1, err_msg='proba must sum to one')

    def log_proba(self, x):
        return np.einsum('...i,...i', s(x, 1), s(self._proba, 'log'))

    def natural_parameters(self, x, variable):
        if is_dependent(x, variable):
            return {
                'mean': s(self._proba, 'log')
            }
        elif is_dependent(self._proba, variable):
            return {
                'log': s(x, 1)
            }
