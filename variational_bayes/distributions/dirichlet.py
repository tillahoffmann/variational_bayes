import numpy as np
from scipy.special import digamma, gammaln

from .distribution import Distribution, statistic, s, assert_constant, is_dependent


class DirichletDistribution(Distribution):
    sample_ndim = 1

    def __init__(self, alpha):
        assert_constant(alpha)
        super(DirichletDistribution, self).__init__(alpha=alpha)

    @statistic
    def mean(self):
        return self._alpha / self._total

    @statistic
    def var(self):
        return self._alpha * (self._total - self._alpha) / \
            (np.square(self._total) * (self._total + 1))

    @statistic
    def _total(self):
        return np.sum(self._alpha, axis=-1, keepdims=True)

    @statistic
    def entropy(self):
        p = self._alpha.shape[-1]
        _total = self._total[..., 0]
        return np.sum(gammaln(self._alpha), axis=-1) - gammaln(_total) + (_total - p) * \
            digamma(_total) + np.sum((1 - self._alpha) * digamma(self._alpha), axis=-1)

    @statistic
    def log(self):
        return digamma(self._alpha) - digamma(self._total)

    def assert_valid_parameters(self):
        np.testing.assert_array_less(0, self._alpha, "concentration parameter must be positive")

    @staticmethod
    def canonical_parameters(natural_parameters):
        return {
            'alpha': natural_parameters.pop('log') + 1
        }

    def log_proba(self, x):
        return gammaln(np.sum(self._alpha, axis=-1)) - np.sum(gammaln(self._alpha), axis=-1) + \
            np.sum((self._alpha - 1) * s(x, 'log'), axis=-1)

    def natural_parameters(self, x, variable):
        if is_dependent(x, variable):
            return {
                'log': self._alpha - 1
            }
        elif is_dependent(self._alpha, variable):
            raise NotImplementedError('alpha')
