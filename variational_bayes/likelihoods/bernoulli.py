import numpy as np

from .likelihood import Likelihood
from ..distributions import s

class BernoulliLikelihood(Likelihood):
    def __init__(self, x, proba):
        super(BernoulliLikelihood, self).__init__(x=x, proba=proba)

    def evaluate(self):
        return s(self._x, 1) * s(self._proba, 'log') + (1 - s(self._x, 1)) * s(self._proba, 'log1m')

    def natural_parameters(self, variable):
        if variable == 'x':
            return {
                'mean': s(self._proba, 'log') - s(self._proba, 'log1m')
            }
        elif variable == 'proba':
            ones = np.ones(np.broadcast(s(self._x, 1), s(self._proba, 1)).shape)
            return {
                'log': s(self._x, 1) * ones,
                'log1m': 1 - s(self._x, 1) * ones
            }
        else:
            raise KeyError(variable)
