import operator
import numpy as np
from scipy.special import multigammaln

from .util import Distribution, statistic, s, assert_constant, Likelihood
from ..util import multidigamma, diag


class WishartLikelihood(Likelihood):
    @staticmethod
    def evaluate(x, shape, scale):   # pylint: disable=W0221
        assert_constant(shape)
        assert_constant(scale)
        p = scale.shape[-1]
        return 0.5 * s(x, 'logdet') * (shape - p - 1.0) - 0.5 * \
            np.sum(scale * s(x, 1), axis=(-1, -2)) - 0.5 * shape * p * np.log(2) - \
            multigammaln(0.5 * shape, p) + 0.5 * shape * s(scale, 'logdet')

    @staticmethod
    def natural_parameters(variable, x, shape, scale):  # pylint: disable=W0221
        if variable == 'x':
            assert_constant(scale)
            return {
                'logdet': 0.5 * (shape - scale.shape[-1] - 1),
                'mean': - 0.5 * scale,
            }
        elif variable in ('shape', 'scale'):
            raise NotImplementedError(variable)
        else:
            raise KeyError(variable)


class WishartDistribution(Distribution):
    """
    Matrix Wishart distribution.
    """
    sample_ndim = 1
    likelihood = WishartLikelihood

    def __init__(self, shape, scale):
        super(WishartDistribution, self).__init__(shape=shape, scale=scale)

    @statistic
    def _inv_scale(self):
        return np.linalg.inv(self._scale)

    @statistic
    def _logdet_scale(self):
        return np.linalg.slogdet(self._scale)[1]

    @statistic
    def mean(self):
        return self._shape[..., None, None] * self._inv_scale

    @statistic
    def entropy(self):
        p = self._scale.shape[-1]
        return 0.5 * p * (p + 1) * np.log(2) + multigammaln(0.5 * self._shape, p) - \
            0.5 * (self._shape - p - 1) * multidigamma(0.5 * self._shape, p) + 0.5 * self._shape * \
            p - 0.5 * (p + 1) * self._logdet_scale

    @statistic
    def var(self):
        _diag = diag(self._inv_scale)
        return self._shape[..., None, None] * \
            (np.square(self._inv_scale) + _diag[..., None, :] * _diag[..., :, None])

    @statistic
    def logdet(self):
        p = self._scale.shape[-1]
        return multidigamma(0.5 * self._shape, p) + p * np.log(2) - self._logdet_scale

    @classmethod
    def from_natural_parameters(cls, natural_parameters):
        p = natural_parameters['mean'].shape[-1]
        shape = 2 * natural_parameters['logdet'] + p + 1
        scale = - 2 * natural_parameters['mean']
        return cls(shape, scale)

    def assert_valid_parameters(self):
        assert np.ndim(self._shape) + 2 == np.ndim(self._scale), "scale parameter must have " \
            "dimensionality two larger than shape parameter"
        np.testing.utils.assert_array_compare(operator.__le__, self._scale.shape[-1], self._shape,
                                              "shape must not be smaller than the dimensionality "
                                              "of the matrix")
        # TODO: assert positive-definite scale
