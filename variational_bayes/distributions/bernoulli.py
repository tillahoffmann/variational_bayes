import operator
import numpy as np
from scipy.special import expit

from .distribution import Distribution, statistic
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
        np.testing.utils.assert_array_compare(operator.__le__, 0, self._proba,
                                              "probability must be non-negative")
        np.testing.utils.assert_array_compare(operator.__le__, self._proba, 1,
                                              "probability must be <= 1")

    @staticmethod
    def canonical_parameters(natural_parameters):
        return {
            'proba': expit(natural_parameters['mean'])
        }
