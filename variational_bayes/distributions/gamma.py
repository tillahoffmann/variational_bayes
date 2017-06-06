import numpy as np
import scipy.special

from .distribution import Distribution, statistic


class GammaDistribution(Distribution):
    """
    Univariate gamma distribution.

    Parameters
    ----------
    shape : np.ndarray
        shape parameter
    scale : np.ndarray
        scale parameter
    """
    sample_ndim = 0

    def __init__(self, shape, scale):
        super(GammaDistribution, self).__init__(shape=shape, scale=scale)

    @statistic
    def mean(self):
        return self._shape / self._scale

    @statistic
    def var(self):
        return self._shape / np.square(self._scale)

    @statistic
    def entropy(self):
        return self._shape - np.log(self._scale) + scipy.special.gammaln(self._shape) + \
                (1 - self._shape) * scipy.special.digamma(self._shape)

    @statistic
    def log(self):
        return scipy.special.digamma(self._shape) - np.log(self._scale)

    @staticmethod
    def canonical_parameters(natural_parameters):
        return {
            'shape': natural_parameters['log'] + 1,
            'scale': -natural_parameters['mean']
        }

    def assert_valid_parameters(self):
        assert self._shape.shape == self._scale.shape, "shape of shape and scale must match"
        np.testing.assert_array_less(0, self._shape, "shape must be positive")
        np.testing.assert_array_less(0, self._scale, "scale must be positive")
