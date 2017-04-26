import operator
import numpy as np
from scipy.special import expit

from .util import Distribution, Likelihood, s, statistic, assert_constant
from ..util import softmax, safe_log


class BernoulliLikelihood(Likelihood):
    def __init__(self, x, proba):
        super(BernoulliLikelihood, self).__init__(x=x, proba=proba)

    @staticmethod
    def evaluate(x, proba):   # pylint: disable=W0221
        return s(x, 1) * s(proba, 'log') + (1 - s(x, 1)) * s(proba, 'log1m')

    @staticmethod
    def natural_parameters(variable, x, proba):   # pylint: disable=W0221
        if variable == 'x':
            return {
                'mean': s(proba, 'log') - s(proba, 'log1m')
            }
        elif variable == 'proba':
            return {
                'log': s(x, 1),
                'log1m': 1 - s(x, 1)
            }
        else:
            raise KeyError(variable)


class BernoulliDistribution(Distribution):
    likelihood = BernoulliLikelihood
    sample_ndim = 0

    def __init__(self, proba):
        super(BernoulliDistribution, self).__init__(proba=proba)

    @statistic
    def mean(self):
        return self._proba

    @statistic
    def var(self):
        return self._proba * (1 - self._proba)

    @statistic
    def entropy(self):
        return - self._proba * safe_log(self._proba) - (1 - self._proba) * safe_log(1 - self._proba)

    def assert_valid_parameters(self):
        np.testing.utils.assert_array_compare(operator.__le__, 0, self._proba,
                                              "probability must be non-negative")
        np.testing.utils.assert_array_compare(operator.__le__, self._proba, 1,
                                              "probability must be <= 1")

    @staticmethod
    def canonical_parameters(natural_parameters):
        return {
            'proba': expit(natural_parameters['mean'])
        }
