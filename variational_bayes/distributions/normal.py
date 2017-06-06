import operator
import numpy as np

from .distribution import Distribution, statistic


class NormalDistribution(Distribution):
    """
    Univariate normal distribution.

    Parameters
    ----------
    mean : np.ndarray
        mean of the distribution
    precision : np.ndarray
        precision or inverse variance of the distribution
    """
    sample_ndim = 0

    def __init__(self, mean, precision):
        super(NormalDistribution, self).__init__(mean=mean, precision=precision)

    @statistic
    def mean(self):
        return self._mean

    @statistic
    def var(self):
        return 1.0 / self._precision

    @statistic
    def entropy(self):
        return 0.5 * (np.log(2 * np.pi) + 1 - np.log(self._precision))

    @staticmethod
    def canonical_parameters(natural_parameters):
        precision = - 2 * natural_parameters['square']
        mean = natural_parameters['mean'] / precision
        return {
            'mean': mean,
            'precision': precision,
        }

    def assert_valid_parameters(self):
        assert np.all(np.isfinite(self._mean)), "mean must be finite"
        np.testing.utils.assert_array_compare(operator.__le__, 0, self._precision,
                                              "precision must be non-negative")
        assert np.shape(self._mean) == np.shape(self._precision), "shape of mean and precision " \
            "must match"
