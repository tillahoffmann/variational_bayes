import numpy as np
import scipy.special

from .distribution import Distribution, statistic, s, assert_constant, is_dependent


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
        assert_constant(shape, scale)
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
            'shape': natural_parameters.pop('log') + 1,
            'scale': -natural_parameters.pop('mean')
        }

    def assert_valid_parameters(self):
        assert self._shape.shape == self._scale.shape, "shape of shape and scale must match"
        np.testing.assert_array_less(0, self._shape, "shape must be positive")
        np.testing.assert_array_less(0, self._scale, "scale must be positive")

    def log_proba(self, x):
        return self._shape * np.log(self._scale) + (self._shape - 1.0) * s(x, 'log') - \
            self._scale * s(x, 1) - scipy.special.gammaln(self._shape)

    def natural_parameters(self, x, variable):
        if is_dependent(x, variable):
            return {
                'log': self._shape - 1.0,
                'mean': - self._scale
            }
        elif is_dependent(self._shape, variable):
            raise NotImplementedError('shape')
        elif is_dependent(self._scale, variable):
            raise NotImplementedError('scale')
