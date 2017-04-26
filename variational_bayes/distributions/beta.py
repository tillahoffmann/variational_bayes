import numpy as np
from scipy.special import digamma, gammaln

from .util import Distribution, Likelihood, s, statistic, assert_constant


class BetaLikelihood(Likelihood):
    def __init__(self, x, a, b):
        super(BetaLikelihood, self).__init__(x=x, a=a, b=b)

    @staticmethod
    def evaluate(x, a, b):   # pylint: disable=W0221
        assert_constant(a, b)
        return gammaln(a + b) - gammaln(a) - gammaln(b) + \
            (a - 1) * s(x, 'log') + (b - 1) * s(x, 'log1m')

    @staticmethod
    def natural_parameters(variable, x, a, b):   # pylint: disable=W0221
        if variable == 'x':
            return {
                'log': a - 1,
                'log1m': b - 1,
            }
        elif variable in ('a', 'b'):
            raise NotImplementedError(variable)
        else:
            raise KeyError(variable)


class BetaDistribution(Distribution):
    likelihood = BetaLikelihood
    sample_ndim = 0

    def __init__(self, a, b):
        super(BetaDistribution, self).__init__(a=a, b=b)

    @statistic
    def mean(self):
        return self._a / self._total

    @statistic
    def var(self):
        return self._a * self._b / (np.square(self._total) * (self._total + 1))

    @statistic
    def _total(self):
        return self._a + self._b

    @statistic
    def entropy(self):
        return gammaln(self._a) + gammaln(self._b) - gammaln(self._total) + (1 - self._a) * \
            digamma(self._a) + (1 - self._b) * digamma(self._b) + (self._total - 2) * \
            digamma(self._total)

    @statistic
    def log(self):
        return digamma(self._a) - digamma(self._total)

    @statistic
    def log1m(self):
        return digamma(self._b) - digamma(self._total)

    def assert_valid_parameters(self):
        np.testing.assert_array_less(0, self._a, "first shape parameter must be positive")
        np.testing.assert_array_less(0, self._b, "second shape parameter must be positive")

    @staticmethod
    def canonical_parameters(natural_parameters):
        return {
            'a': natural_parameters['log'] + 1,
            'b': natural_parameters['log1m'] + 1
        }
