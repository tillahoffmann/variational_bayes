import numpy as np
from scipy.special import digamma, gammaln

from .distribution import Distribution, statistic, s, assert_constant


class BetaDistribution(Distribution):
    sample_ndim = 0

    def __init__(self, a, b):
        assert_constant(a, b)
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

    def log_proba(self, x):
        return gammaln(self._a + self._b) - gammaln(self._a) - gammaln(self._b) + \
            (self._a - 1) * s(x, 'log') + (self._b - 1) * s(x, 'log1m')

    def natural_parameters(self, x, variable):
        if variable == 'x':
            return {
                'log': self._a - 1,
                'log1m': self._b - 1,
            }
        elif variable in ('a', 'b'):
            raise NotImplementedError(variable)
        else:
            raise KeyError(variable)
