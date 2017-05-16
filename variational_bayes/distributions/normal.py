import operator
import numpy as np

from .distribution import Distribution, s, statistic
from .likelihood import Likelihood


class NormalLikelihood(Likelihood):
    def __init__(self, x, mean, precision):
        super(NormalLikelihood, self).__init__(x=x, mean=mean, precision=precision)

    @staticmethod
    def evaluate(x, mean, precision):  # pylint: disable=W0221
        return 0.5 * (s(precision, 'log') - np.log(2 * np.pi) - s(precision, 1) * (
            s(x, 2) - 2 * s(x, 1) * s(mean, 1) + s(mean, 2)
        ))

    @staticmethod
    def natural_parameters(variable, x, mean, precision):  # pylint: disable=W0221
        # Get an object of ones with the correct broadcasted shape
        ones = np.ones(np.broadcast(s(x, 1), s(mean, 1)).shape)

        if variable == 'x':
            return {
                'mean': s(precision, 1) * s(mean, 1) * ones,
                'square': - 0.5 * s(precision, 1) * ones
            }
        elif variable == 'mean':
            return {
                'mean': s(precision, 1) * s(x, 1) * ones,
                'square': - 0.5 * s(precision, 1) * ones
            }
        elif variable == 'precision':
            return {
                'log': 0.5 * ones,
                'mean': - 0.5 * (s(x, 2) - 2 * s(x, 1) * s(mean, 1) + s(mean, 2))
            }
        else:
            raise KeyError(variable)


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
    likelihood = NormalLikelihood

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
