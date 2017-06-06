import numpy as np

from scipy.special import multigammaln

from .likelihood import Likelihood
from ..distributions import s, assert_constant


class WishartLikelihood(Likelihood):
    def __init__(self, x, shape, scale):
        assert_constant(shape, scale)
        super(WishartLikelihood, self).__init__(x=x, shape=shape, scale=scale)

    def evaluate(self):
        p = self._scale.shape[-1]
        return 0.5 * s(self._x, 'logdet') * (self._shape - p - 1.0) - 0.5 * \
            np.sum(self._scale * s(self._x, 1), axis=(-1, -2)) - 0.5 * self._shape * p * np.log(2) - \
            multigammaln(0.5 * self._shape, p) + 0.5 * self._shape * s(self._scale, 'logdet')

    def natural_parameters(self, variable):
        if variable == 'x':
            p = self._scale.shape[-1]
            return {
                'logdet': 0.5 * (self._shape - p - 1),
                'mean': - 0.5 * self._scale,
            }
        elif variable in ('shape', 'scale'):
            raise NotImplementedError(variable)
        else:
            raise KeyError(variable)
