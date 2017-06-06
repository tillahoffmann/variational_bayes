from scipy.special import gammaln

from .likelihood import Likelihood
from ..distributions import s, assert_constant


class BetaLikelihood(Likelihood):
    def __init__(self, x, a, b):
        assert_constant(a, b)
        super(BetaLikelihood, self).__init__(x=x, a=a, b=b)

    def evaluate(self):
        return gammaln(self._a + self._b) - gammaln(self._a) - gammaln(self._b) + \
            (self._a - 1) * s(self._x, 'log') + (self._b - 1) * s(self._x, 'log1m')

    def natural_parameters(self, variable):
        if variable == 'x':
            return {
                'log': self._a - 1,
                'log1m': self._b - 1,
            }
        elif variable in ('a', 'b'):
            raise NotImplementedError(variable)
        else:
            raise KeyError(variable)
