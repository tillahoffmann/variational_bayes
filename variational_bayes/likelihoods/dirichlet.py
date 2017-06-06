import numpy as np
from scipy.special import gammaln

from .likelihood import Likelihood
from ..distributions import s, assert_constant


class DirichletLikelihood(Likelihood):
    def __init__(self, x, alpha):
        assert_constant(alpha)
        super(DirichletLikelihood, self).__init__(x=x, alpha=alpha)

    def evaluate(self):
        return gammaln(np.sum(self._alpha, axis=-1)) - np.sum(gammaln(self._alpha), axis=-1) + \
            np.sum((self._alpha - 1) * s(self._x, 'log'), axis=-1)

    def natural_parameters(self, variable):
        if variable == 'x':
            return {
                'log': self._alpha - 1
            }
        elif variable == 'alpha':
            raise NotImplementedError(variable)
        else:
            raise KeyError(variable)
