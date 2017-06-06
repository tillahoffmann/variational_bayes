import numpy as np
import scipy.special

from .likelihood import Likelihood
from ..distributions import s, assert_constant


class GammaLikelihood(Likelihood):
    def __init__(self, x, shape, scale):
        assert_constant(shape)
        assert_constant(scale)
        super(GammaLikelihood, self).__init__(x=x, shape=shape, scale=scale)

    def evaluate(self):
        return self._shape * np.log(self._scale) + (self._shape - 1.0) * s(self._x, 'log') - \
            self._scale * s(self._x, 1) - scipy.special.gammaln(self._shape)

    def natural_parameters(self, variable):
        if variable == 'x':
            return {
                'log': self._shape - 1.0,
                'mean': - self._scale
            }
        elif variable in ('shape', 'scale'):
            raise NotImplementedError(variable)
        else:
            raise KeyError(variable)
