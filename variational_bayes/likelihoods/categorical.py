import numpy as np

from .likelihood import Likelihood
from ..distributions import s

class CategoricalLikelihood(Likelihood):
    def __init__(self, x, proba):
        super(CategoricalLikelihood, self).__init__(x=x, proba=proba)

    def evaluate(self):
        return np.einsum('...i,...i', s(self._x, 1), s(self._proba, 'log'))

    def natural_parameters(self, variable):
        if variable == 'x':
            return {
                'mean': s(self._proba, 'log')
            }
        elif variable == 'proba':
            return {
                'log': s(self._x, 1)
            }
        else:
            raise KeyError(variable)
