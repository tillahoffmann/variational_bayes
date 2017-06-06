import numpy as np

from .distribution import Distribution, statistic
from ..util import diag, is_positive_definite


class MultiNormalDistribution(Distribution):
    """
    Vector normal distribution.
    """
    sample_ndim = 1

    def __init__(self, mean, precision):
        super(MultiNormalDistribution, self).__init__(mean=mean, precision=precision)

    @statistic
    def mean(self):
        return self._mean

    @statistic
    def cov(self):
        return np.linalg.inv(self._precision)

    @statistic
    def var(self):
        return diag(self.cov)

    @statistic
    def entropy(self):
        p = np.shape(self.mean)[-1]
        return 0.5 * p * (np.log(2 * np.pi) + 1) - 0.5 * np.linalg.slogdet(self._precision)[1]

    @statistic
    def outer(self):
        return self.mean[..., None] * self.mean[..., None, :] + self.cov

    @staticmethod
    def canonical_parameters(natural_parameters):
        precision = - 2 * natural_parameters['outer']
        cov = np.linalg.inv(precision)
        mean = np.einsum('...ij,...j', cov, natural_parameters['mean'])
        return {
            'mean': mean,
            'precision': precision,
        }

    def assert_valid_parameters(self):
        assert self._mean.ndim > 0, "the mean must be at least one-dimensional"
        assert self._precision.ndim == self._mean.ndim + 1, "dimensionality of the precision must " \
            "be one larger than the dimensionality of the mean"
        assert self._precision.shape[-1] == self._precision.shape[-2], "last two dimensions of the " \
            "precision must be equal"
        assert np.all(np.isfinite(self._mean)), "mean must be finite"
        assert is_positive_definite(self._precision), "precision must be positive definite"
