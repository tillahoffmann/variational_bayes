import numpy as np
from scipy.special import digamma, gammaln

from .util import Distribution, Likelihood, s, statistic, assert_constant


class DirichletLikelihood(Likelihood):
    @staticmethod
    def evaluate(x, alpha):   # pylint: disable=W0221
        assert_constant(alpha)
        return gammaln(np.sum(alpha, axis=-1)) - np.sum(gammaln(alpha), axis=-1) + \
            np.sum((alpha - 1) * s(x, 'log'), axis=-1)

    @staticmethod
    def natural_parameters(variable, x, alpha):   # pylint: disable=W0221
        assert_constant(alpha)
        if variable == 'x':
            return {
                'log': alpha - 1
            }
        elif variable == 'alpha':
            raise NotImplementedError(variable)
        else:
            raise KeyError(variable)


class DirichletDistribution(Distribution):
    likelihood = DirichletLikelihood
    sample_ndim = 1

    def __init__(self, alpha):
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

    @classmethod
    def from_natural_parameters(cls, natural_parameters):
        return DirichletDistribution(natural_parameters['log'] + 1)
