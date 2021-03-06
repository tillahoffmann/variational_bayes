import operator
import numpy as np
from scipy.special import expit

from .distribution import Distribution, statistic, s, is_dependent
from ..util import safe_log


class BernoulliDistribution(Distribution):
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
        proba = s(self._proba, 1)
        np.testing.utils.assert_array_compare(operator.__le__, 0, proba,
                                              "probability must be non-negative")
        np.testing.utils.assert_array_compare(operator.__le__, proba, 1,
                                              "probability must be <= 1")

    @staticmethod
    def canonical_parameters(natural_parameters):
        return {
            'proba': expit(natural_parameters.pop('mean'))
        }

    def log_proba(self, x):
        return s(x, 1) * s(self._proba, 'log') + (1 - s(x, 1)) * s(self._proba, 'log1m')

    def natural_parameters(self, x, variable):
        if is_dependent(x, variable):
            return {
                'mean': s(self._proba, 'log') - s(self._proba, 'log1m')
            }
        elif is_dependent(self._proba, variable):
            ones = np.ones(np.broadcast(s(x, 1), s(self._proba, 1)).shape)
            return {
                'log': s(x, 1) * ones,
                'log1m': 1 - s(x, 1) * ones
            }
